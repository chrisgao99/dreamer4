#!/usr/bin/env python3
"""Build view-specific PufferDrive scenes for Dreamer Waymo NPZ files.

Dreamer consumes focus-local, filtered NPZ views, while PufferDrive consumes a
binary scene containing world-coordinate actors and map geometry.  This tool
joins the two representations by ``scenario_id`` and actor track ID:

* the original WOMD ``Scenario`` proto supplies the full 3D map, actor sizes,
  elevations, and original trajectories;
* the NPZ supplies the exact masked actor set and the focus actor ordering;
* PufferDrive's own JSON-to-binary converter produces ``map_000.bin``.

Each output directory is a single-map PufferDrive ``map_dir``.  This avoids the
simulator's random numbered-map selection and makes the generated manifest an
unambiguous runtime join table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUFFERDRIVE_ROOT = REPO_ROOT.parent / "PufferDrive"
DEFAULT_SCENARIO_CACHE_DIR = (
    REPO_ROOT / "waymo/cache/wosac_internal_val_scenarios/scenarios"
)
DEFAULT_SCENARIO_MANIFEST = (
    REPO_ROOT / "waymo/cache/wosac_internal_val_scenarios/eval_manifest.csv"
)

PUFFER_TRAJECTORY_LENGTH = 91
INVALID_POSITION = -10_000.0
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

OBJECT_TYPE_NAMES = {
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}

TRAFFIC_LIGHT_STATE_NAMES = {
    0: "unknown",
    1: "arrow_stop",
    2: "arrow_caution",
    3: "arrow_go",
    4: "stop",
    5: "caution",
    6: "go",
    7: "flashing_stop",
    8: "flashing_caution",
}


@dataclass(frozen=True)
class NpzView:
    path: Path
    scenario_id: str
    focus_track_id: int
    agent_ids: tuple[int, ...]
    ego_origin_xy: tuple[float, float] | None
    ego_heading: float | None


@dataclass(frozen=True)
class ScenarioManifestEntry:
    scenario_id: str
    scenario_pb_path: Path


@dataclass(frozen=True)
class BuiltScene:
    data: dict[str, Any]
    selected_source_track_indices: tuple[int, ...]
    original_sdc_track_index: int
    original_sdc_track_id: int | None


def _scalar(array: np.ndarray, key: str, path: Path) -> Any:
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError(
            f"{path}: expected scalar NPZ field {key!r}, got shape {value.shape}"
        )
    return value.reshape(-1)[0].item()


def _text_scalar(array: np.ndarray, key: str, path: Path) -> str:
    value = _scalar(array, key, path)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = str(value)
    if not result:
        raise ValueError(f"{path}: NPZ field {key!r} is empty")
    return result


def _optional_origin(data: Mapping[str, np.ndarray], path: Path) -> tuple[float, float] | None:
    if "ego_origin_xy" not in data:
        return None
    origin = np.asarray(data["ego_origin_xy"], dtype=np.float64).reshape(-1)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError(
            f"{path}: ego_origin_xy must contain two finite values, got {origin}"
        )
    return float(origin[0]), float(origin[1])


def _optional_heading(data: Mapping[str, np.ndarray], path: Path) -> float | None:
    if "ego_heading" not in data:
        return None
    heading = float(_scalar(data["ego_heading"], "ego_heading", path))
    if not math.isfinite(heading):
        raise ValueError(f"{path}: ego_heading must be finite, got {heading}")
    return heading


def load_npz_view(path: Path) -> NpzView:
    """Load the static join fields from a Dreamer vector NPZ."""

    path = path.resolve()
    try:
        with np.load(path, allow_pickle=False) as data:
            required = ("scenario_id", "agent_ids", "agent_mask")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"missing fields {missing}")

            scenario_id = _text_scalar(data["scenario_id"], "scenario_id", path)
            all_ids = np.asarray(data["agent_ids"], dtype=np.int64).reshape(-1)
            mask = np.asarray(data["agent_mask"], dtype=bool).reshape(-1)
            if all_ids.shape != mask.shape:
                raise ValueError(
                    f"{path}: agent_ids shape {all_ids.shape} does not match "
                    f"agent_mask shape {mask.shape}"
                )
            masked_ids = [int(value) for value in all_ids[mask]]
            if not masked_ids:
                raise ValueError(f"{path}: agent_mask selects no actors")
            if len(masked_ids) != len(set(masked_ids)):
                raise ValueError(f"{path}: masked agent_ids are not unique: {masked_ids}")

            if "focus_track_id" in data:
                focus_track_id = int(
                    _scalar(data["focus_track_id"], "focus_track_id", path)
                )
                if focus_track_id not in masked_ids:
                    raise ValueError(
                        f"{path}: focus_track_id={focus_track_id} is not selected by "
                        f"agent_mask; selected IDs={masked_ids}"
                    )
            else:
                # Older filtered NPZs establish the same invariant through slot 0.
                focus_track_id = masked_ids[0]

            ordered_ids = [focus_track_id]
            ordered_ids.extend(value for value in masked_ids if value != focus_track_id)
            origin = _optional_origin(data, path)
            heading = _optional_heading(data, path)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(str(path)):
            raise
        raise type(exc)(f"{path}: failed to load Dreamer NPZ: {exc}") from exc

    return NpzView(
        path=path,
        scenario_id=scenario_id,
        focus_track_id=focus_track_id,
        agent_ids=tuple(ordered_ids),
        ego_origin_xy=origin,
        ego_heading=heading,
    )


def _resolve_csv_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_scenario_manifest(
    manifest_path: Path,
) -> dict[Path, ScenarioManifestEntry]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Scenario manifest does not exist: {manifest_path}")

    result: dict[Path, ScenarioManifestEntry] = {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"npz_path", "scenario_id", "scenario_pb_path"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{manifest_path}: missing required columns {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            npz_path = _resolve_csv_path(row["npz_path"], manifest_path)
            entry = ScenarioManifestEntry(
                scenario_id=str(row["scenario_id"]),
                scenario_pb_path=_resolve_csv_path(
                    row["scenario_pb_path"], manifest_path
                ),
            )
            previous = result.get(npz_path)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"{manifest_path}:{row_number}: conflicting entries for {npz_path}: "
                    f"{previous} vs {entry}"
                )
            result[npz_path] = entry
    return result


class ScenarioResolver:
    """Resolve a cached serialized Scenario, with an exact-manifest fallback."""

    def __init__(
        self,
        cache_dir: Path | None,
        manifest_path: Path | None,
    ) -> None:
        self.cache_dir = cache_dir.resolve() if cache_dir is not None else None
        self.manifest_path = (
            manifest_path.resolve() if manifest_path is not None else None
        )
        self.by_npz = (
            load_scenario_manifest(self.manifest_path)
            if self.manifest_path is not None
            else {}
        )

    def resolve(self, view: NpzView) -> Path:
        attempted: list[Path] = []
        entry = self.by_npz.get(view.path.resolve())
        if entry is not None:
            if entry.scenario_id != view.scenario_id:
                raise ValueError(
                    f"{view.path}: scenario_id={view.scenario_id!r} conflicts with "
                    f"{self.manifest_path}: scenario_id={entry.scenario_id!r}"
                )
            attempted.append(entry.scenario_pb_path)
            if entry.scenario_pb_path.is_file():
                return entry.scenario_pb_path

        if self.cache_dir is not None:
            direct = self.cache_dir / f"{view.scenario_id}.pb"
            attempted.append(direct)
            if direct.is_file():
                return direct.resolve()

        attempted_text = "\n".join(f"  - {path}" for path in attempted)
        if not attempted_text:
            attempted_text = "  - no scenario manifest or cache directory configured"
        raise FileNotFoundError(
            f"No cached WOMD Scenario proto for NPZ {view.path}\n"
            f"scenario_id={view.scenario_id}\n"
            f"Attempted:\n{attempted_text}\n"
            "Create the cache with waymo/evaluation/prepare_wosac_scenario_cache.py "
            "or pass --scenario-manifest/--scenario-cache-dir."
        )


def load_scenario_proto(path: Path) -> Any:
    """Load a serialized WOMD Scenario without importing TensorFlow."""

    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "waymo_open_dataset is required to parse cached Scenario protos. "
            "Run this CLI in an environment that provides both Waymo protos and "
            "PufferDrive (for this workspace: /p/yufeng/.conda/envs/puffd/bin/python)."
        ) from exc

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise OSError(f"Failed to read Scenario proto {path}: {exc}") from exc
    scenario = scenario_pb2.Scenario()
    try:
        scenario.ParseFromString(payload)
    except Exception as exc:
        raise ValueError(f"Failed to parse Scenario proto {path}: {exc}") from exc
    return scenario


def _checked_int32(value: Any, label: str) -> int:
    result = int(value)
    if result < INT32_MIN or result > INT32_MAX:
        raise ValueError(
            f"{label}={result} does not fit PufferDrive's signed int32 binary field"
        )
    return result


def _wrap_heading(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _point_dict(point: Any) -> dict[str, float]:
    return {
        "x": float(point.x),
        "y": float(point.y),
        "z": float(getattr(point, "z", 0.0)),
    }


def _track_to_object(track: Any, scenario_id: str) -> dict[str, Any]:
    track_id = _checked_int32(track.id, f"{scenario_id} track id")
    object_type = int(track.object_type)
    if object_type not in OBJECT_TYPE_NAMES:
        raise ValueError(
            f"{scenario_id}: selected track id={track_id} has unsupported Waymo "
            f"object_type={object_type}; PufferDrive supports vehicle, pedestrian, "
            "and cyclist only"
        )

    states = list(track.states)
    if len(states) != PUFFER_TRAJECTORY_LENGTH:
        raise ValueError(
            f"{scenario_id}: selected track id={track_id} has {len(states)} states; "
            f"PufferDrive's binary writer requires {PUFFER_TRAJECTORY_LENGTH}"
        )
    valid_states = [state for state in states if bool(state.valid)]
    if not valid_states:
        raise ValueError(f"{scenario_id}: selected track id={track_id} has no valid state")
    final_state = valid_states[-1]

    positions = []
    velocities = []
    headings = []
    valids = []
    for state in states:
        valid = bool(state.valid)
        valids.append(valid)
        if valid:
            positions.append(
                {
                    "x": float(state.center_x),
                    "y": float(state.center_y),
                    "z": float(state.center_z),
                }
            )
            velocities.append(
                {
                    "x": float(state.velocity_x),
                    "y": float(state.velocity_y),
                    "z": float(getattr(state, "velocity_z", 0.0)),
                }
            )
            headings.append(_wrap_heading(float(state.heading)))
        else:
            positions.append(
                {
                    "x": INVALID_POSITION,
                    "y": INVALID_POSITION,
                    "z": INVALID_POSITION,
                }
            )
            velocities.append(
                {
                    "x": INVALID_POSITION,
                    "y": INVALID_POSITION,
                    "z": INVALID_POSITION,
                }
            )
            headings.append(INVALID_POSITION)

    return {
        "position": positions,
        "width": float(final_state.width),
        "length": float(final_state.length),
        "height": float(final_state.height),
        "heading": headings,
        "velocity": velocities,
        "valid": valids,
        "goalPosition": {
            "x": float(final_state.center_x),
            "y": float(final_state.center_y),
            "z": float(final_state.center_z),
        },
        "type": OBJECT_TYPE_NAMES[object_type],
        "id": track_id,
        # Dreamer's externally supplied state is authoritative at runtime.
        "mark_as_expert": False,
    }


def _map_element_id(feature_name: str, payload: Any) -> int:
    if feature_name == "lane":
        lane_type = int(payload.type)
        return lane_type if 0 <= lane_type <= 3 else -1
    if feature_name == "road_line":
        line_type = int(payload.type)
        return line_type + 5 if 0 <= line_type <= 8 else -1
    if feature_name == "road_edge":
        edge_type = int(payload.type)
        return edge_type + 14 if 0 <= edge_type <= 2 else -1
    return {
        "stop_sign": 17,
        "crosswalk": 18,
        "speed_bump": 19,
        "driveway": 20,
    }.get(feature_name, -1)


def _map_feature_to_road(feature: Any, scenario_id: str) -> dict[str, Any]:
    feature_name = feature.WhichOneof("feature_data")
    if not feature_name:
        raise ValueError(f"{scenario_id}: map feature id={feature.id} has no feature_data")
    payload = getattr(feature, feature_name)
    if feature_name == "stop_sign":
        points = [payload.position]
    elif feature_name in {"crosswalk", "speed_bump", "driveway"}:
        points = payload.polygon
    else:
        if not hasattr(payload, "polyline"):
            raise ValueError(
                f"{scenario_id}: unsupported map feature {feature_name!r} id={feature.id}"
            )
        points = payload.polyline
    return {
        "geometry": [_point_dict(point) for point in points],
        "type": feature_name,
        "map_element_id": _map_element_id(feature_name, payload),
        "id": _checked_int32(feature.id, f"{scenario_id} map feature id"),
    }


def _traffic_lights(scenario: Any) -> dict[str, dict[str, list[Any]]]:
    lights: defaultdict[int, dict[str, list[Any]]] = defaultdict(
        lambda: {"state": [], "x": [], "y": [], "z": [], "time_index": []}
    )
    for time_index, dynamic_state in enumerate(scenario.dynamic_map_states):
        for lane_state in dynamic_state.lane_states:
            lane_id = int(lane_state.lane)
            light = lights[lane_id]
            light["state"].append(
                TRAFFIC_LIGHT_STATE_NAMES.get(int(lane_state.state), "unknown")
            )
            light["x"].append(float(lane_state.stop_point.x))
            light["y"].append(float(lane_state.stop_point.y))
            light["z"].append(float(lane_state.stop_point.z))
            light["time_index"].append(time_index)
    # JSON object keys are strings; making that explicit keeps tests and diffs stable.
    return {str(key): value for key, value in lights.items()}


def build_puffer_scene(scenario: Any, view: NpzView) -> BuiltScene:
    """Create a Puffer-compatible scene with exactly the NPZ's masked actors."""

    scenario_id = str(scenario.scenario_id)
    if scenario_id != view.scenario_id:
        raise ValueError(
            f"Scenario proto/NPZ mismatch: proto={scenario_id!r}, "
            f"NPZ={view.scenario_id!r} ({view.path})"
        )
    encoded_id = scenario_id.encode("utf-8")
    if len(encoded_id) > 16:
        raise ValueError(
            f"{scenario_id}: UTF-8 scenario ID is {len(encoded_id)} bytes, but "
            "PufferDrive stores at most 16 bytes"
        )

    tracks = list(scenario.tracks)
    track_by_id: dict[int, tuple[int, Any]] = {}
    for source_index, track in enumerate(tracks):
        track_id = int(track.id)
        if track_id in track_by_id:
            raise ValueError(f"{scenario_id}: duplicate raw track id={track_id}")
        track_by_id[track_id] = (source_index, track)

    missing_ids = [track_id for track_id in view.agent_ids if track_id not in track_by_id]
    if missing_ids:
        raise KeyError(
            f"{scenario_id}: NPZ actor IDs absent from raw Scenario: {missing_ids}"
        )

    selected_source_indices = tuple(track_by_id[value][0] for value in view.agent_ids)
    objects = [
        _track_to_object(track_by_id[track_id][1], scenario_id)
        for track_id in view.agent_ids
    ]
    if not objects or int(objects[0]["id"]) != view.focus_track_id:
        raise AssertionError("focus actor ordering invariant was not established")

    source_to_selected = {
        source_index: selected_index
        for selected_index, source_index in enumerate(selected_source_indices)
    }
    tracks_to_predict = []
    for prediction in scenario.tracks_to_predict:
        source_index = int(prediction.track_index)
        if source_index < 0 or source_index >= len(tracks):
            raise ValueError(
                f"{scenario_id}: tracks_to_predict contains invalid source index "
                f"{source_index} for {len(tracks)} tracks"
            )
        if source_index in source_to_selected:
            tracks_to_predict.append(
                {
                    "track_index": source_to_selected[source_index],
                    "difficulty": int(prediction.difficulty),
                }
            )

    selected_ids = set(view.agent_ids)
    selected_ooi_ids = [
        int(track_id)
        for track_id in scenario.objects_of_interest
        if int(track_id) in selected_ids
    ]

    original_sdc_index = int(scenario.sdc_track_index)
    original_sdc_id: int | None = None
    if 0 <= original_sdc_index < len(tracks):
        original_sdc_id = int(tracks[original_sdc_index].id)

    metadata = {
        # The view focus, not the original Waymo SDC, must be Puffer slot zero.
        "sdc_track_index": 0,
        "objects_of_interest": selected_ooi_ids,
        "tracks_to_predict": tracks_to_predict,
        # Extra fields are ignored by Puffer's writer but make the JSON auditable.
        "focus_track_id": view.focus_track_id,
        "selected_track_ids": list(view.agent_ids),
        "selected_source_track_indices": list(selected_source_indices),
        "source_sdc_track_index": original_sdc_index,
        "source_sdc_track_id": original_sdc_id,
    }
    data = {
        "name": f"{scenario_id}_focus_{view.focus_track_id}",
        "scenario_id": scenario_id,
        "objects": objects,
        "roads": [
            _map_feature_to_road(feature, scenario_id)
            for feature in scenario.map_features
        ],
        "tl_states": _traffic_lights(scenario),
        "metadata": metadata,
    }
    return BuiltScene(
        data=data,
        selected_source_track_indices=selected_source_indices,
        original_sdc_track_index=original_sdc_index,
        original_sdc_track_id=original_sdc_id,
    )


def load_pufferdrive_json_converter(
    pufferdrive_root: Path,
) -> Callable[[str, int, str], None]:
    """Import PufferDrive's canonical ``load_map`` JSON-to-BIN function."""

    root = pufferdrive_root.resolve()
    drive_path = root / "pufferlib/ocean/drive/drive.py"
    if not drive_path.is_file():
        raise FileNotFoundError(
            f"PufferDrive converter not found at {drive_path}; "
            "pass --pufferdrive-root"
        )
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("pufferlib.ocean.drive.drive")
    except Exception as exc:
        raise RuntimeError(
            f"Could not import PufferDrive from {root}: {exc}. Run this command "
            "in the PufferDrive environment (workspace default: "
            "/p/yufeng/.conda/envs/puffd/bin/python)."
        ) from exc
    module_path = Path(module.__file__).resolve()
    if root not in module_path.parents:
        raise RuntimeError(
            f"Imported PufferDrive converter from {module_path}, not requested root {root}"
        )
    return module.load_map


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError(f"Cannot make a safe output name from {value!r}")
    return result


def view_key(view: NpzView) -> str:
    return f"{_safe_component(view.scenario_id)}__focus_{view.focus_track_id}"


def _write_view_outputs(
    view_dir: Path,
    scene: Mapping[str, Any],
    converter: Callable[[str, int, str], None],
    overwrite: bool,
) -> tuple[Path, Path]:
    json_path = view_dir / "scene.json"
    binary_path = view_dir / "map_000.bin"
    existing = [path for path in (json_path, binary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already exists (pass --overwrite to replace exact view files): "
            + ", ".join(str(path) for path in existing)
        )
    view_dir.mkdir(parents=True, exist_ok=True)

    json_fd, json_tmp_name = tempfile.mkstemp(
        prefix=".scene.", suffix=".json.tmp", dir=view_dir
    )
    os.close(json_fd)
    bin_fd, bin_tmp_name = tempfile.mkstemp(
        prefix=".map_000.", suffix=".bin.tmp", dir=view_dir
    )
    os.close(bin_fd)
    json_tmp = Path(json_tmp_name)
    bin_tmp = Path(bin_tmp_name)
    try:
        with json_tmp.open("w") as handle:
            json.dump(scene, handle, separators=(",", ":"))
            handle.write("\n")
        converter(str(json_tmp), 0, str(bin_tmp))
        if not bin_tmp.is_file() or bin_tmp.stat().st_size == 0:
            raise RuntimeError(
                f"PufferDrive converter did not create a non-empty binary at {bin_tmp}"
            )
        os.replace(json_tmp, json_path)
        os.replace(bin_tmp, binary_path)
    finally:
        json_tmp.unlink(missing_ok=True)
        bin_tmp.unlink(missing_ok=True)
    return json_path.resolve(), binary_path.resolve()


def _manifest_value(value: float | None) -> str | float:
    return "" if value is None else value


def convert_view(
    view: NpzView,
    scenario_pb_path: Path,
    views_root: Path,
    converter: Callable[[str, int, str], None],
    overwrite: bool,
    scenario_loader: Callable[[Path], Any] = load_scenario_proto,
) -> dict[str, Any]:
    scenario = scenario_loader(scenario_pb_path)
    built = build_puffer_scene(scenario, view)
    key = view_key(view)
    output_dir = views_root / key
    json_path, binary_path = _write_view_outputs(
        output_dir, built.data, converter, overwrite
    )
    origin_x = view.ego_origin_xy[0] if view.ego_origin_xy is not None else None
    origin_y = view.ego_origin_xy[1] if view.ego_origin_xy is not None else None
    return {
        "view_key": key,
        "scenario_id": view.scenario_id,
        "focus_track_id": view.focus_track_id,
        "npz_path": str(view.path),
        "scenario_pb_path": str(scenario_pb_path.resolve()),
        "puffer_map_dir": str(output_dir.resolve()),
        "puffer_json_path": str(json_path),
        "puffer_bin_path": str(binary_path),
        "num_agents": len(view.agent_ids),
        "ordered_agent_ids": ";".join(str(value) for value in view.agent_ids),
        "ordered_source_track_indices": ";".join(
            str(value) for value in built.selected_source_track_indices
        ),
        "ordered_agent_id_to_puffer_index": json.dumps(
            {str(track_id): index for index, track_id in enumerate(view.agent_ids)},
            separators=(",", ":"),
        ),
        "ego_origin_x": _manifest_value(origin_x),
        "ego_origin_y": _manifest_value(origin_y),
        "ego_heading": _manifest_value(view.ego_heading),
        "original_sdc_track_index": built.original_sdc_track_index,
        "original_sdc_track_id": (
            "" if built.original_sdc_track_id is None else built.original_sdc_track_id
        ),
        "puffer_sdc_track_index": 0,
        "num_tracks_to_predict": len(
            built.data["metadata"]["tracks_to_predict"]
        ),
        "num_roads": len(built.data["roads"]),
        "trajectory_length": PUFFER_TRAJECTORY_LENGTH,
    }


def _write_manifest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty conversion manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def expand_npz_inputs(inputs: Iterable[str]) -> list[Path]:
    """Expand files, directories, and quoted glob patterns deterministically."""

    paths: list[Path] = []
    for value in inputs:
        candidate = Path(value)
        if candidate.is_file():
            paths.append(candidate.resolve())
        elif candidate.is_dir():
            paths.extend(path.resolve() for path in candidate.rglob("*.npz"))
        elif glob.has_magic(value):
            matches = [Path(path) for path in glob.glob(value, recursive=True)]
            files = [path.resolve() for path in matches if path.is_file()]
            if not files:
                raise FileNotFoundError(f"NPZ pattern matched no files: {value}")
            paths.extend(files)
        else:
            raise FileNotFoundError(f"NPZ input does not exist: {candidate}")

    unique = sorted(set(paths), key=str)
    if not unique:
        raise ValueError("No NPZ files were selected")
    non_npz = [path for path in unique if path.suffix.lower() != ".npz"]
    if non_npz:
        raise ValueError(f"Inputs must be .npz files: {non_npz}")
    return unique


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    npz_paths = expand_npz_inputs(args.npz)
    views = [load_npz_view(path) for path in npz_paths]
    keys: dict[str, Path] = {}
    for view in views:
        key = view_key(view)
        previous = keys.get(key)
        if previous is not None and previous != view.path:
            raise ValueError(
                f"Two NPZ files map to the same output view_key={key}: "
                f"{previous} and {view.path}"
            )
        keys[key] = view.path

    manifest_arg = getattr(args, "scenario_manifest", None)
    if manifest_arg is not None:
        scenario_manifest = Path(manifest_arg)
    elif DEFAULT_SCENARIO_MANIFEST.is_file():
        scenario_manifest = DEFAULT_SCENARIO_MANIFEST
    else:
        scenario_manifest = None
    cache_arg = getattr(args, "scenario_cache_dir", None)
    scenario_cache_dir = Path(cache_arg) if cache_arg is not None else None
    resolver = ScenarioResolver(scenario_cache_dir, scenario_manifest)
    scenario_paths = [resolver.resolve(view) for view in views]

    output_root = Path(args.output_dir).resolve()
    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}; pass --overwrite to replace it"
        )
    converter = load_pufferdrive_json_converter(Path(args.pufferdrive_root))
    views_root = output_root / "views"

    rows = []
    for index, (view, scenario_path) in enumerate(
        zip(views, scenario_paths), start=1
    ):
        row = convert_view(
            view=view,
            scenario_pb_path=scenario_path,
            views_root=views_root,
            converter=converter,
            overwrite=bool(args.overwrite),
        )
        rows.append(row)
        print(
            f"[{index}/{len(views)}] {row['view_key']} agents={row['num_agents']} "
            f"roads={row['num_roads']}",
            flush=True,
        )

    _write_manifest(rows, manifest_path)
    print(
        json.dumps(
            {
                "views": len(rows),
                "unique_scenarios": len({row["scenario_id"] for row in rows}),
                "manifest": str(manifest_path),
                "output_dir": str(output_root),
            },
            indent=2,
        ),
        flush=True,
    )
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Dreamer Waymo NPZ views plus cached raw Scenario protos "
            "into view-specific PufferDrive map directories."
        )
    )
    parser.add_argument(
        "npz",
        nargs="+",
        help="One or more NPZ files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root; each view is written under views/<scenario>__focus_<id>.",
    )
    parser.add_argument(
        "--scenario-cache-dir",
        default=str(DEFAULT_SCENARIO_CACHE_DIR),
        help="Directory containing <scenario_id>.pb cached Scenario protos.",
    )
    parser.add_argument(
        "--scenario-manifest",
        default=None,
        help=(
            "Optional CSV with npz_path, scenario_id, and scenario_pb_path. "
            f"Defaults to {DEFAULT_SCENARIO_MANIFEST} when it exists, then falls "
            "back to --scenario-cache-dir."
        ),
    )
    parser.add_argument(
        "--pufferdrive-root",
        default=str(DEFAULT_PUFFERDRIVE_ROOT),
        help="PufferDrive checkout containing pufferlib/ocean/drive/drive.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing scene.json, map_000.bin, and manifest.csv files.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
