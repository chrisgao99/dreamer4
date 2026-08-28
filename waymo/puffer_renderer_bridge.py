"""Subprocess bridge between the Dreamer game and PufferDrive rendering.

Dreamer and PufferDrive intentionally run in different Python environments.
This module keeps their boundary small: requests are UTF-8 JSON frames and a
render response is a raw JPEG frame.  Every frame is prefixed by one unsigned
32-bit little-endian payload length.

The worker owns the Raylib context.  Dreamer remains authoritative for agent
motion and continues to read the focus-centred NPZ dataset.
"""

from __future__ import annotations

import csv
import json
import math
import os
import select
import shlex
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROTOCOL_VERSION = 1
_FRAME_HEADER = struct.Struct("<I")
_DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class PufferBridgeError(RuntimeError):
    """Raised when the renderer worker cannot satisfy a request."""


def _normalise_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def local_to_world_pose(
    xy: np.ndarray,
    yaw: np.ndarray,
    velocity_xy: np.ndarray,
    *,
    origin_xy: np.ndarray,
    origin_heading: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform focus-centred NPZ poses into the raw Waymo world frame.

    Args:
        xy: ``(..., 2)`` local positions.
        yaw: ``(...)`` local headings in radians.
        velocity_xy: ``(..., 2)`` local velocity vectors.
        origin_xy: World position used when the NPZ was centred.
        origin_heading: World heading used when the NPZ was centred.

    Returns:
        World-frame ``(xy, yaw, velocity_xy)`` arrays.  Inputs are not mutated.
    """

    xy_array = np.asarray(xy, dtype=np.float32)
    yaw_array = np.asarray(yaw, dtype=np.float32)
    velocity_array = np.asarray(velocity_xy, dtype=np.float32)
    origin = np.asarray(origin_xy, dtype=np.float32)
    if xy_array.shape[-1:] != (2,):
        raise ValueError(f"xy must end in dimension 2, got {xy_array.shape}")
    if velocity_array.shape != xy_array.shape:
        raise ValueError(
            f"velocity_xy must have shape {xy_array.shape}, got {velocity_array.shape}"
        )
    if yaw_array.shape != xy_array.shape[:-1]:
        raise ValueError(f"yaw must have shape {xy_array.shape[:-1]}, got {yaw_array.shape}")
    if origin.shape != (2,):
        raise ValueError(f"origin_xy must have shape (2,), got {origin.shape}")

    heading = float(origin_heading)
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    world_xy = xy_array @ rotation.T + origin
    world_velocity = velocity_array @ rotation.T
    unwrapped_yaw = yaw_array.astype(np.float64) + heading
    world_yaw = np.arctan2(np.sin(unwrapped_yaw), np.cos(unwrapped_yaw)).astype(
        np.float32
    )
    return world_xy.astype(np.float32), world_yaw, world_velocity.astype(np.float32)


@dataclass(frozen=True)
class PufferSceneReference:
    """Preconverted Puffer scene and metadata needed by the renderer worker."""

    scenario_id: str
    npz_path: Path
    focus_track_id: int
    puffer_map_dir: Path | None = None
    puffer_bin_path: Path | None = None
    scenario_pb_path: Path | None = None


@dataclass(frozen=True)
class _SceneAssets:
    scenario_id: str
    puffer_map_dir: Path | None
    puffer_bin_path: Path | None
    scenario_pb_path: Path | None


class ScenarioManifest:
    """Resolve a Dreamer NPZ view to its cached raw Waymo Scenario proto."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        *,
        scenario_cache_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve(strict=False)
            if manifest_path
            else None
        )
        self.scenario_cache_dir = (
            Path(scenario_cache_dir).expanduser().resolve(strict=False)
            if scenario_cache_dir
            else None
        )
        self._by_npz_path: dict[str, _SceneAssets] = {}
        self._by_scenario_focus: dict[tuple[str, int], _SceneAssets] = {}
        self._by_scenario: dict[str, _SceneAssets] = {}
        self._ambiguous_scenarios: set[str] = set()
        if self.manifest_path is not None:
            self._load_csv(self.manifest_path)

    @staticmethod
    def _row_path(value: str, manifest_path: Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        return path.resolve(strict=False)

    @staticmethod
    def _insert_unique(
        mapping: dict[Any, _SceneAssets],
        key: Any,
        value: _SceneAssets,
        label: str,
    ) -> None:
        previous = mapping.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"Conflicting {label} entries for {key!r}: {previous} vs {value}")
        mapping[key] = value

    def _load_csv(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Puffer scenario manifest not found: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if "scenario_id" not in fields:
                raise ValueError(f"Puffer manifest {path} is missing column: scenario_id")
            asset_fields = {"puffer_map_dir", "puffer_bin_path", "scenario_pb_path"}
            if not fields.intersection(asset_fields):
                raise ValueError(
                    f"Puffer manifest {path} must contain one of {sorted(asset_fields)}"
                )
            for row_number, row in enumerate(reader, start=2):
                scenario_id = str(row.get("scenario_id", "")).strip()
                if not scenario_id:
                    raise ValueError(
                        f"Puffer manifest {path}:{row_number} has an empty scenario_id"
                    )
                values = {
                    field: str(row.get(field, "")).strip() for field in asset_fields
                }
                assets = _SceneAssets(
                    scenario_id=scenario_id,
                    puffer_map_dir=(
                        self._row_path(values["puffer_map_dir"], path)
                        if values["puffer_map_dir"]
                        else None
                    ),
                    puffer_bin_path=(
                        self._row_path(values["puffer_bin_path"], path)
                        if values["puffer_bin_path"]
                        else None
                    ),
                    scenario_pb_path=(
                        self._row_path(values["scenario_pb_path"], path)
                        if values["scenario_pb_path"]
                        else None
                    ),
                )
                # A converter may intentionally emit one Puffer map per focus
                # view.  In that case exact NPZ and scenario/focus lookup stay
                # valid, while bare scenario lookup is deliberately disabled.
                if scenario_id not in self._ambiguous_scenarios:
                    previous_assets = self._by_scenario.get(scenario_id)
                    if previous_assets is None:
                        self._by_scenario[scenario_id] = assets
                    elif previous_assets != assets:
                        self._by_scenario.pop(scenario_id, None)
                        self._ambiguous_scenarios.add(scenario_id)

                focus_value = str(row.get("focus_track_id", "")).strip()
                if focus_value:
                    self._insert_unique(
                        self._by_scenario_focus,
                        (scenario_id, int(focus_value)),
                        assets,
                        "scenario/focus",
                    )

                npz_value = str(row.get("npz_path", "")).strip()
                if npz_value:
                    npz_path = self._row_path(npz_value, path)
                    self._insert_unique(
                        self._by_npz_path,
                        _normalise_path(npz_path),
                        assets,
                        "NPZ path",
                    )

    def resolve(
        self,
        *,
        scenario_id: str,
        npz_path: str | Path,
        focus_track_id: int,
        require_file: bool = True,
    ) -> PufferSceneReference:
        """Resolve with exact-view, scenario/focus, scenario, then cache fallback."""

        scenario_id = str(scenario_id)
        normalised_npz = _normalise_path(npz_path)
        assets = self._by_npz_path.get(normalised_npz)
        if assets is None:
            assets = self._by_scenario_focus.get(
                (scenario_id, int(focus_track_id))
            )
        if assets is None:
            assets = self._by_scenario.get(scenario_id)
        if assets is None and self.scenario_cache_dir is not None:
            assets = _SceneAssets(
                scenario_id=scenario_id,
                puffer_map_dir=None,
                puffer_bin_path=None,
                scenario_pb_path=self.scenario_cache_dir / f"{scenario_id}.pb",
            )
        if assets is None:
            raise KeyError(
                f"No Puffer scene mapping for scenario={scenario_id!r}, NPZ={normalised_npz}"
            )
        if assets.scenario_id != scenario_id:
            raise ValueError(
                f"Puffer manifest scenario mismatch for NPZ {normalised_npz}: "
                f"requested {scenario_id!r}, mapped {assets.scenario_id!r}"
            )
        if require_file:
            if assets.puffer_bin_path is not None and not assets.puffer_bin_path.is_file():
                raise FileNotFoundError(
                    f"Mapped Puffer binary does not exist for {scenario_id}: "
                    f"{assets.puffer_bin_path}"
                )
            if assets.puffer_map_dir is not None and not assets.puffer_map_dir.is_dir():
                raise FileNotFoundError(
                    f"Mapped Puffer map directory does not exist for {scenario_id}: "
                    f"{assets.puffer_map_dir}"
                )
            if assets.puffer_bin_path is None and assets.puffer_map_dir is None:
                diagnostic = (
                    f" Raw Scenario: {assets.scenario_pb_path}."
                    if assets.scenario_pb_path is not None
                    else ""
                )
                raise FileNotFoundError(
                    f"No preconverted Puffer map for {scenario_id}. Run the offline "
                    f"Scenario-to-Puffer conversion and use its manifest.{diagnostic}"
                )
        return PufferSceneReference(
            scenario_id=scenario_id,
            npz_path=Path(normalised_npz),
            focus_track_id=int(focus_track_id),
            puffer_map_dir=assets.puffer_map_dir,
            puffer_bin_path=assets.puffer_bin_path,
            scenario_pb_path=assets.scenario_pb_path,
        )

    @property
    def mapped_npz_paths(self) -> frozenset[str]:
        """Canonical NPZ paths with an explicit converted render asset."""

        return frozenset(self._by_npz_path)


@dataclass(frozen=True)
class PufferFrameState:
    """One externally authored world-frame state sent to PufferDrive."""

    step: int
    agent_ids: np.ndarray
    agent_types: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    velocity_xy: np.ndarray
    valid: np.ndarray
    source_time_index: int | None = None

    def as_request(self, scenario: PufferSceneReference) -> dict[str, Any]:
        ids = np.asarray(self.agent_ids, dtype=np.int64)
        types = np.asarray(self.agent_types, dtype=np.int64)
        xy = np.asarray(self.xy, dtype=np.float32)
        yaw = np.asarray(self.yaw, dtype=np.float32)
        velocity = np.asarray(self.velocity_xy, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        count = len(ids)
        expected_vector = (count,)
        if types.shape != expected_vector or yaw.shape != expected_vector:
            raise ValueError("agent_types and yaw must match agent_ids")
        if valid.shape != expected_vector:
            raise ValueError("valid must match agent_ids")
        if xy.shape != (count, 2) or velocity.shape != (count, 2):
            raise ValueError("xy and velocity_xy must have shape (num_agents, 2)")

        finite = (
            np.isfinite(xy).all(axis=1)
            & np.isfinite(yaw)
            & np.isfinite(velocity).all(axis=1)
        )
        valid = valid & finite
        safe_xy = np.where(np.isfinite(xy), xy, 0.0)
        safe_yaw = np.where(np.isfinite(yaw), yaw, 0.0)
        safe_velocity = np.where(np.isfinite(velocity), velocity, 0.0)
        request = {
            "type": "render_frame",
            "protocol_version": PROTOCOL_VERSION,
            "scenario_id": scenario.scenario_id,
            "step": int(self.step),
            "focus_track_id": int(scenario.focus_track_id),
            "agent_ids": ids.tolist(),
            "agent_types": types.tolist(),
            "x": safe_xy[:, 0].tolist(),
            "y": safe_xy[:, 1].tolist(),
            "yaw": safe_yaw.tolist(),
            "vx": safe_velocity[:, 0].tolist(),
            "vy": safe_velocity[:, 1].tolist(),
            "valid": valid.tolist(),
            # Dreamer NPZs do not retain z.  The worker obtains it from the
            # loaded Scenario rather than incorrectly forcing world z=0.
            "preserve_scene_z": True,
        }
        if self.source_time_index is not None:
            # Keep the source index in the protocol for diagnostics and a
            # future elevation-aware renderer.  The current native Puffer
            # target style draws roads/actors on fixed display planes.
            request["source_time_index"] = int(self.source_time_index)
        return request


class PufferRendererClient:
    """Synchronous client for a single long-lived Puffer renderer worker."""

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        width: int,
        height: int,
        view_mode: str = "AGENT_PERSP",
        jpeg_quality: int = 90,
        timeout_s: float = 15.0,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
        environment: dict[str, str] | None = None,
    ) -> None:
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv:
            raise ValueError("Puffer worker command cannot be empty")
        self.command = tuple(str(value) for value in argv)
        self.width = int(width)
        self.height = int(height)
        self.view_mode = str(view_mode)
        self.jpeg_quality = int(jpeg_quality)
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)
        self.environment = (
            None
            if environment is None
            else {str(key): str(value) for key, value in environment.items()}
        )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Puffer render dimensions must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("Puffer JPEG quality must be in [1, 100]")
        if self.timeout_s <= 0:
            raise ValueError("Puffer worker timeout must be positive")
        self._process: subprocess.Popen[bytes] | None = None
        self._loaded_scene_key: tuple[str, str, int] | None = None
        self._lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def _start(self) -> subprocess.Popen[bytes]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        process_environment = None
        if self.environment is not None:
            process_environment = os.environ.copy()
            process_environment.update(self.environment)
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Inherit stderr so diagnostics are visible and cannot fill a pipe.
            stderr=None,
            bufsize=0,
            env=process_environment,
        )
        self._loaded_scene_key = None
        return self._process

    def _read_exact(self, process: subprocess.Popen[bytes], size: int) -> bytes:
        if process.stdout is None:
            raise PufferBridgeError("Puffer worker stdout is unavailable")
        descriptor = process.stdout.fileno()
        deadline = time.monotonic() + self.timeout_s
        chunks: list[bytes] = []
        remaining = int(size)
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise PufferBridgeError(
                    f"Timed out after {self.timeout_s:g}s waiting for Puffer worker"
                )
            readable, _, _ = select.select([descriptor], [], [], timeout)
            if not readable:
                raise PufferBridgeError(
                    f"Timed out after {self.timeout_s:g}s waiting for Puffer worker"
                )
            chunk = os.read(descriptor, remaining)
            if not chunk:
                exit_code = process.poll()
                detail = "closed stdout" if exit_code is None else f"exited with code {exit_code}"
                raise PufferBridgeError(f"Puffer worker {detail}")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_frame(self, process: subprocess.Popen[bytes]) -> bytes:
        header = self._read_exact(process, _FRAME_HEADER.size)
        (payload_size,) = _FRAME_HEADER.unpack(header)
        if payload_size <= 0 or payload_size > self.max_response_bytes:
            raise PufferBridgeError(f"Invalid Puffer response size: {payload_size}")
        return self._read_exact(process, payload_size)

    @staticmethod
    def _decode_json_payload(payload: bytes, *, request_type: str) -> dict[str, Any]:
        try:
            response = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PufferBridgeError(
                f"Puffer worker returned a non-JSON response to {request_type}"
            ) from error
        if not isinstance(response, dict):
            raise PufferBridgeError(f"Invalid Puffer response to {request_type}: {response!r}")
        if not bool(response.get("ok", False)):
            raise PufferBridgeError(
                str(response.get("error") or f"Puffer {request_type} request failed")
            )
        return response

    def _exchange(self, request: dict[str, Any]) -> bytes:
        process = self._start()
        if process.stdin is None:
            raise PufferBridgeError("Puffer worker stdin is unavailable")
        try:
            payload = json.dumps(
                request, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            process.stdin.write(_FRAME_HEADER.pack(len(payload)))
            process.stdin.write(payload)
            process.stdin.flush()
            return self._read_frame(process)
        except (BrokenPipeError, OSError, ValueError, PufferBridgeError) as error:
            self._stop_process()
            if isinstance(error, PufferBridgeError):
                raise
            raise PufferBridgeError(f"Puffer worker communication failed: {error}") from error

    def _load_scene(self, scene: PufferSceneReference) -> None:
        render_source = scene.puffer_bin_path or scene.puffer_map_dir
        if render_source is None:
            raise PufferBridgeError(
                f"Scenario {scene.scenario_id} has no preconverted Puffer render asset"
            )
        key = (
            scene.scenario_id,
            _normalise_path(render_source),
            int(scene.focus_track_id),
        )
        if key == self._loaded_scene_key:
            return
        request = {
            "type": "load_scene",
            "protocol_version": PROTOCOL_VERSION,
            "scenario_id": scene.scenario_id,
            "npz_path": str(scene.npz_path),
            "focus_track_id": int(scene.focus_track_id),
            "width": self.width,
            "height": self.height,
            "view_mode": self.view_mode,
            "jpeg_quality": self.jpeg_quality,
        }
        if scene.puffer_map_dir is not None:
            request["puffer_map_dir"] = str(scene.puffer_map_dir)
        if scene.puffer_bin_path is not None:
            request["puffer_bin_path"] = str(scene.puffer_bin_path)
        if scene.scenario_pb_path is not None:
            # Diagnostics/provenance only.  Runtime conversion is intentionally
            # kept out of the latency-sensitive renderer worker.
            request["scenario_pb_path"] = str(scene.scenario_pb_path)
        response = self._exchange(request)
        self._decode_json_payload(response, request_type="load_scene")
        self._loaded_scene_key = key

    def render(self, scene: PufferSceneReference, frame: PufferFrameState) -> bytes:
        """Load ``scene`` if necessary, inject ``frame``, and return JPEG bytes."""

        with self._lock:
            self._load_scene(scene)
            response = self._exchange(frame.as_request(scene))
            if response.startswith(b"{"):
                # Errors and status responses use JSON framing.  A successful
                # render response is the raw JPEG payload itself.
                self._decode_json_payload(response, request_type="render_frame")
                raise PufferBridgeError("Puffer worker returned status without a JPEG")
            if not response.startswith(b"\xff\xd8\xff"):
                raise PufferBridgeError("Puffer worker response is not a JPEG frame")
            return response

    def _stop_process(self) -> None:
        process, self._process = self._process, None
        self._loaded_scene_key = None
        if process is None:
            return

        # Give the worker a chance to close Raylib and the Xvfb child it owns.
        # Closing stdin and immediately sending SIGTERM skips that cleanup and
        # can leave a display process/socket behind after every game session.
        if process.poll() is None and process.stdin is not None:
            try:
                payload = json.dumps(
                    {"type": "close", "protocol_version": PROTOCOL_VERSION},
                    separators=(",", ":"),
                ).encode("utf-8")
                process.stdin.write(_FRAME_HEADER.pack(len(payload)))
                process.stdin.write(payload)
                process.stdin.flush()
                process.wait(timeout=2.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._stop_process()

    def __enter__(self) -> "PufferRendererClient":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
