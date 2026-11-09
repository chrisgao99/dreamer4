#!/usr/bin/env python3
"""Render complete Waymo NPZ histories in PufferDrive without a world model.

This module deliberately has no Torch, tokenizer, checkpoint, action-builder,
or dynamics imports. Every rendered actor state comes directly from the NPZ;
per-frame elevation is sent from the matching converted ``scene.json`` sidecar.
The current native Puffer renderer still draws actors on fixed display planes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.evaluation.puffer_video_export import (  # noqa: E402
    FfmpegJpegWriter,
    load_subset_records,
    overlay_label,
    preflight_headless_dependencies,
)
from waymo.puffer_renderer_bridge import (  # noqa: E402
    PufferFrameState,
    PufferRendererClient,
    PufferSceneReference,
    ScenarioManifest,
    local_to_world_pose,
)


DEFAULT_SUBSET_MANIFEST = (
    WAYMO_ROOT / "evaluation/val_random128_seed0_manifest.json"
)
DEFAULT_PUFFER_MANIFEST = (
    WAYMO_ROOT / "cache/pufferdrive_val128_seed0/manifest.csv"
)
DEFAULT_PUFFER_SCENARIO_CACHE = (
    WAYMO_ROOT / "cache/wosac_internal_val_scenarios/scenarios"
)
DEFAULT_PUFFER_WORKER_COMMAND = " ".join(
    (
        "/p/yufeng/.conda/envs/puffd/bin/python",
        "-u",
        str(REPO_ROOT.parent / "PufferDrive/puffer_renderer_worker.py"),
    )
)


@dataclass(frozen=True)
class GroundTruthFrames:
    xy: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    valid: np.ndarray

    @property
    def count(self) -> int:
        return int(self.xy.shape[0])


@dataclass(frozen=True)
class GroundTruthScene:
    scenario_id: str
    focus_track_id: int
    npz_path: Path
    agent_ids: np.ndarray
    agent_types: np.ndarray
    agent_mask: np.ndarray
    ego_origin_xy: np.ndarray
    ego_heading: float
    frames: GroundTruthFrames
    puffer_scene: PufferSceneReference


def output_video_name(record: dict[str, Any], *, total_frames: int) -> str:
    return (
        f"sample_{int(record['sample_order']):03d}_"
        f"scene_{int(record['dataset_index'])}_"
        f"{record['scenario_id']}_focus_{int(record['focus_track_id'])}_"
        f"ground_truth_{int(total_frames)}f.mp4"
    )


def _agents_to_tkf(agents: np.ndarray, num_agents: int) -> np.ndarray:
    if agents.ndim != 3 or agents.shape[-1] < 8:
        raise ValueError(f"Expected agents shape (K,T,8+) or (T,K,8+), got {agents.shape}")
    if agents.shape[0] == num_agents:
        return np.transpose(agents, (1, 0, 2))
    if agents.shape[1] == num_agents:
        return agents
    raise ValueError(
        f"Neither agents axis matches agent_mask K={num_agents}: {agents.shape}"
    )


def _load_scene_z(
    puffer_scene: PufferSceneReference,
    *,
    agent_ids: np.ndarray,
    agent_mask: np.ndarray,
    total_frames: int,
) -> np.ndarray:
    map_dir = puffer_scene.puffer_map_dir
    if map_dir is None:
        raise FileNotFoundError(
            f"Ground-truth z requires puffer_map_dir for {puffer_scene.scenario_id}"
        )
    scene_json = map_dir / "scene.json"
    if not scene_json.is_file():
        raise FileNotFoundError(f"Converted Puffer scene sidecar is missing: {scene_json}")
    payload = json.loads(scene_json.read_text(encoding="utf-8"))
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"Puffer sidecar has no objects list: {scene_json}")
    by_id: dict[int, dict[str, Any]] = {}
    for obj in objects:
        object_id = int(obj["id"])
        if object_id in by_id:
            raise ValueError(f"Duplicate object id {object_id} in {scene_json}")
        by_id[object_id] = obj

    z = np.zeros((total_frames, len(agent_ids)), dtype=np.float32)
    for slot in np.flatnonzero(agent_mask & (agent_ids >= 0)):
        object_id = int(agent_ids[slot])
        if object_id not in by_id:
            raise KeyError(f"Agent id {object_id} is missing from {scene_json}")
        positions = by_id[object_id].get("position")
        if not isinstance(positions, list) or len(positions) != total_frames:
            raise ValueError(
                f"Object {object_id} must have {total_frames} positions; "
                f"got {len(positions) if isinstance(positions, list) else type(positions)}"
            )
        z[:, slot] = np.asarray([float(point["z"]) for point in positions], dtype=np.float32)
    return z


def load_ground_truth_scene(
    record: dict[str, Any],
    manifest: ScenarioManifest,
) -> GroundTruthScene:
    """Load all recorded states directly, with no action/model operations."""

    npz_path = Path(str(record["path"])).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Ground-truth NPZ does not exist: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        agents = np.asarray(data["agents"], dtype=np.float32)
        agent_mask = np.asarray(data["agent_mask"], dtype=bool)
        agent_ids = np.asarray(data["agent_ids"], dtype=np.int64)
        ego_origin_xy = np.asarray(data["ego_origin_xy"], dtype=np.float32)
        ego_heading = float(np.asarray(data["ego_heading"]).item())
        scenario_id = str(np.asarray(data["scenario_id"]).item())
        focus_track_id = int(np.asarray(data["focus_track_id"]).item())

    num_agents = int(agent_mask.shape[0])
    agents_tkf = _agents_to_tkf(agents, num_agents)
    if agent_ids.shape != (num_agents,):
        raise ValueError(
            f"agent_ids must have shape {(num_agents,)}, got {agent_ids.shape}"
        )
    if ego_origin_xy.shape != (2,):
        raise ValueError(f"ego_origin_xy must have shape (2,), got {ego_origin_xy.shape}")
    if scenario_id != str(record["scenario_id"]):
        raise ValueError(
            f"Scenario mismatch: subset={record['scenario_id']} npz={scenario_id}"
        )
    if focus_track_id != int(record["focus_track_id"]):
        raise ValueError(
            f"Focus mismatch: subset={record['focus_track_id']} npz={focus_track_id}"
        )
    if not agent_mask[0] or int(agent_ids[0]) != focus_track_id:
        raise ValueError(
            f"Focus slot mismatch: slot0 id={int(agent_ids[0])}, expected {focus_track_id}"
        )

    valid = (agents_tkf[..., 5] > 0.5) & agent_mask[None, :]
    if not bool(valid[:, 0].all()):
        invalid = np.flatnonzero(~valid[:, 0]).tolist()
        raise ValueError(
            f"Focus car is not valid throughout the full replay; frames={invalid}"
        )
    agent_types = np.rint(agents_tkf[0, :, 7]).astype(np.int64)
    if int(agent_types[0]) != 1:
        raise ValueError(f"Focus slot is not a vehicle: type={int(agent_types[0])}")

    puffer_scene = manifest.resolve(
        scenario_id=scenario_id,
        npz_path=npz_path,
        focus_track_id=focus_track_id,
    )
    total_frames = int(agents_tkf.shape[0])
    z = _load_scene_z(
        puffer_scene,
        agent_ids=agent_ids,
        agent_mask=agent_mask,
        total_frames=total_frames,
    )
    return GroundTruthScene(
        scenario_id=scenario_id,
        focus_track_id=focus_track_id,
        npz_path=npz_path,
        agent_ids=agent_ids,
        agent_types=agent_types,
        agent_mask=agent_mask,
        ego_origin_xy=ego_origin_xy,
        ego_heading=ego_heading,
        frames=GroundTruthFrames(
            xy=agents_tkf[..., 0:2].copy(),
            z=z,
            yaw=agents_tkf[..., 6].copy(),
            velocity=agents_tkf[..., 3:5].copy(),
            valid=valid,
        ),
        puffer_scene=puffer_scene,
    )


def build_puffer_frame(scene: GroundTruthScene, frame_index: int) -> PufferFrameState:
    selected = scene.agent_mask & (scene.agent_ids >= 0)
    world_xy, world_yaw, world_velocity = local_to_world_pose(
        scene.frames.xy[frame_index, selected],
        scene.frames.yaw[frame_index, selected],
        scene.frames.velocity[frame_index, selected],
        origin_xy=scene.ego_origin_xy,
        origin_heading=scene.ego_heading,
    )
    return PufferFrameState(
        step=int(frame_index),
        agent_ids=scene.agent_ids[selected],
        agent_types=scene.agent_types[selected],
        xy=world_xy,
        z=scene.frames.z[frame_index, selected],
        yaw=world_yaw,
        velocity_xy=world_velocity,
        valid=scene.frames.valid[frame_index, selected],
        source_time_index=int(frame_index),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-manifest",
        "--selection-manifest",
        dest="subset_manifest",
        default=str(DEFAULT_SUBSET_MANIFEST),
    )
    parser.add_argument(
        "--puffer-manifest",
        "--puffer-scenario-manifest",
        dest="puffer_manifest",
        default=str(DEFAULT_PUFFER_MANIFEST),
    )
    parser.add_argument(
        "--puffer-scenario-cache",
        default=str(DEFAULT_PUFFER_SCENARIO_CACHE),
    )
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--puffer-width", type=int, default=1280)
    parser.add_argument("--puffer-height", type=int, default=720)
    parser.add_argument("--puffer-view-mode", default="AGENT_PERSP")
    parser.add_argument("--puffer-timeout", type=float, default=120.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--puffer-worker-command",
        default=DEFAULT_PUFFER_WORKER_COMMAND,
    )
    parser.add_argument(
        "--puffer-use-inherited-display",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.puffer_manifest:
        parser.error("--puffer-manifest is required")
    if args.sample_start < 0:
        parser.error("--sample-start must be non-negative")
    if args.num_scenes < 1:
        parser.error("--num-scenes must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.puffer_width <= 0 or args.puffer_height <= 0:
        parser.error("Puffer dimensions must be positive")
    if args.puffer_timeout <= 0:
        parser.error("--puffer-timeout must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")


def render(args: argparse.Namespace) -> Path:
    preflight_headless_dependencies(
        ffmpeg=args.ffmpeg,
        use_inherited_display=bool(args.puffer_use_inherited_display),
    )
    records = load_subset_records(
        args.subset_manifest,
        sample_start=int(args.sample_start),
        num_scenes=int(args.num_scenes),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ScenarioManifest(
        args.puffer_manifest,
        scenario_cache_dir=args.puffer_scenario_cache,
    )
    renderer = PufferRendererClient(
        args.puffer_worker_command,
        width=int(args.puffer_width),
        height=int(args.puffer_height),
        view_mode=args.puffer_view_mode,
        jpeg_quality=int(args.jpeg_quality),
        timeout_s=float(args.puffer_timeout),
        environment=(
            {"PUFFER_USE_INHERITED_DISPLAY": "1"}
            if args.puffer_use_inherited_display
            else None
        ),
    )
    results: list[dict[str, Any]] = []
    started = time.time()
    try:
        for batch_index, record in enumerate(records, start=1):
            scene = load_ground_truth_scene(record, manifest)
            output_path = output_dir / output_video_name(
                record, total_frames=scene.frames.count
            )
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}; pass --overwrite to replace it"
                )
            print(
                f"[ground truth {batch_index}/{len(records)}] dataset "
                f"#{int(record['dataset_index'])} scenario={scene.scenario_id} "
                f"focus={scene.focus_track_id} frames={scene.frames.count}",
                flush=True,
            )
            scene_started = time.time()
            first_jpeg = renderer.render(scene.puffer_scene, build_puffer_frame(scene, 0))
            with FfmpegJpegWriter(
                output_path,
                ffmpeg=args.ffmpeg,
                fps=float(args.fps),
                overwrite=bool(args.overwrite),
            ) as writer:
                for frame_index in range(scene.frames.count):
                    jpeg = (
                        first_jpeg
                        if frame_index == 0
                        else renderer.render(
                            scene.puffer_scene,
                            build_puffer_frame(scene, frame_index),
                        )
                    )
                    if args.overlay:
                        jpeg = overlay_label(
                            jpeg,
                            (
                                f"sample #{int(record['sample_order'])} | "
                                f"dataset #{int(record['dataset_index'])} | "
                                f"scenario {scene.scenario_id} | focus {scene.focus_track_id} | "
                                f"GROUND TRUTH {frame_index + 1}/{scene.frames.count}"
                            ),
                        )
                    writer.write(jpeg)
                    if (
                        (frame_index + 1) % 10 == 0
                        or frame_index + 1 == scene.frames.count
                    ):
                        print(
                            f"[ground truth {batch_index}/{len(records)}] rendered "
                            f"{frame_index + 1}/{scene.frames.count} frames",
                            flush=True,
                        )

            seconds = time.time() - scene_started
            print(f"[saved] {output_path} ({seconds:.1f}s)", flush=True)
            results.append(
                {
                    "sample_order": int(record["sample_order"]),
                    "dataset_index": int(record["dataset_index"]),
                    "scenario_id": scene.scenario_id,
                    "focus_track_id": scene.focus_track_id,
                    "npz_path": str(scene.npz_path),
                    "output_mp4": str(output_path),
                    "replayed_frames": scene.frames.count,
                    "generated_frames": 0,
                    "fps": float(args.fps),
                    "seconds": seconds,
                }
            )
    finally:
        renderer.close()

    summary_path = output_dir / "summary.json"
    summary = {
        "mode": "ground_truth_replay",
        "trajectory_source": "recorded_npz_with_scene_json_z",
        "world_model_used": False,
        "tokenizer_used": False,
        "actions_used": False,
        "action_source": None,
        "generated_frames": 0,
        "subset_manifest": str(Path(args.subset_manifest).expanduser().resolve()),
        "puffer_manifest": str(Path(args.puffer_manifest).expanduser().resolve()),
        "num_videos": len(results),
        "elapsed_seconds": time.time() - started,
        "videos": results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[complete] ground-truth videos={len(results)} summary={summary_path}",
        flush=True,
    )
    return summary_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
