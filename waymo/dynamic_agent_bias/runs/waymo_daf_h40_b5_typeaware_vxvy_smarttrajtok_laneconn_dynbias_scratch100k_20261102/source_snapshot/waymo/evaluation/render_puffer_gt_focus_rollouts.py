#!/usr/bin/env python3
"""Render offline Dreamer rollouts driven by recorded focus-car actions.

Each output video contains an exact recorded context followed by an
autoregressive world-model rollout.  The controlled focus car follows the
action sequence computed from its NPZ trajectory; all other actor states in
the rollout are decoded from Dreamer.  PufferDrive is used only as the 3D
renderer and runs in its external RGB-array worker.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.evaluation.puffer_video_export import (  # noqa: E402
    FfmpegJpegWriter,
    load_subset_records,
    overlay_label,
    preflight_headless_dependencies,
)
from waymo.interactive_world_model_game import (  # noqa: E402
    SessionState,
    WaymoInteractiveServer,
    build_parser as build_game_parser,
)
from waymo.puffer_renderer_bridge import (  # noqa: E402
    PufferFrameState,
    local_to_world_pose,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


DEFAULT_SUBSET_MANIFEST = (
    WAYMO_ROOT / "evaluation/val_random128_seed0_manifest.json"
)
DEFAULT_PUFFER_MANIFEST = (
    WAYMO_ROOT / "cache/pufferdrive_val128_seed0/manifest.csv"
)


@dataclass(frozen=True)
class RolloutFrames:
    xy: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    valid: np.ndarray

    @property
    def count(self) -> int:
        return int(self.xy.shape[0])


def resolve_dataset_index(
    record: dict[str, Any],
    dataset_paths: list[str] | tuple[str, ...],
) -> int:
    """Resolve by exact NPZ path and verify the recorded dataset index."""

    wanted = str(Path(str(record["path"])).expanduser().resolve(strict=False))
    by_path = {
        str(Path(path).expanduser().resolve(strict=False)): index
        for index, path in enumerate(dataset_paths)
    }
    if wanted not in by_path:
        raise KeyError(f"Subset NPZ is not present in the checkpoint dataset: {wanted}")
    actual = int(by_path[wanted])
    recorded = int(record["dataset_index"])
    if actual != recorded:
        raise ValueError(
            f"Dataset index mismatch for {wanted}: subset={recorded}, loaded={actual}"
        )
    return actual


def output_video_name(
    record: dict[str, Any],
    *,
    context_frames: int,
    unroll_steps: int,
) -> str:
    return (
        f"sample_{int(record['sample_order']):03d}_"
        f"scene_{int(record['dataset_index'])}_"
        f"{record['scenario_id']}_focus_{int(record['focus_track_id'])}_"
        f"ctx{int(context_frames)}_u{int(unroll_steps)}.mp4"
    )


def _cpu_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


@torch.inference_mode()
def generate_gt_focus_rollout(
    server: WaymoInteractiveServer,
    state: SessionState,
    *,
    unroll_steps: int,
    seed: int,
) -> RolloutFrames:
    """Generate other actors while conditioning on recorded focus actions."""

    context_frames = len(state.context_focus)
    total_frames = context_frames + int(unroll_steps)
    start = int(state.context_start_frame)
    end = start + total_frames
    available = int(state.base_batch["lights"].shape[1])
    if end > available:
        raise ValueError(
            f"GT focus actions require frames [{start}, {end}), but the NPZ has "
            f"only {available} frames"
        )

    # Build once on the original full timeline.  Building on a future slice
    # would incorrectly zero the first delta-XY/delta-yaw transition.
    full_actions, full_action_mask, action_slots = wm.build_ego_action_features(
        state.base_batch, server.model_args
    )
    if full_actions is None or full_action_mask is None or action_slots is None:
        raise RuntimeError("The checkpoint did not produce focus-agent actions")
    if int(action_slots[0].item()) != 0:
        raise ValueError(
            f"Expected focus action slot 0, got {int(action_slots[0].item())}"
        )

    actions = full_actions[:, start:end].clone()
    action_mask = full_action_mask[:, start:end].clone()
    # Match the exact context representation already encoded by the
    # interactive loader, including its first-frame action convention.
    actions[:, :context_frames] = torch.stack(state.action_history, dim=0)[None]
    action_mask[:, :context_frames] = torch.stack(
        state.action_mask_history, dim=0
    )[None]

    context_packed = torch.stack(state.z_history, dim=0)[None]
    if int(context_packed.shape[1]) != context_frames:
        raise ValueError(
            f"Expected {context_frames} context latents, got {context_packed.shape[1]}"
        )
    # The sampler only reads the first context_frames values from z_gt_packed;
    # zero padding supplies the requested total length without encoding future
    # GT frames and leaking them through tokenizer temporal attention.
    padding = torch.zeros(
        (1, int(unroll_steps), *context_packed.shape[2:]),
        device=context_packed.device,
        dtype=context_packed.dtype,
    )
    z_stub = torch.cat([context_packed, padding], dim=1)

    wm.seed_everything(int(seed))
    z_pred_packed = wm.sample_autoregressive_packed_sequence(
        wm.unwrap_model(server.dynamics),
        z_gt_packed=z_stub,
        actions=actions,
        act_mask=action_mask,
        map_tokens=state.map_tokens,
        map_mask=state.map_mask,
        ctx_length=context_frames,
        horizon=int(unroll_steps),
        k_max=int(server.model_args.k_max),
        sched=server.schedule,
        max_rollout_window=server.model_rollout_window,
    )
    z_pred = wm.unpack_spatial_to_bottleneck(
        z_pred_packed, k=int(server.model_args.packing_factor)
    )

    agents_btkf = wm.agents_to_btkf(
        state.base_batch["agents"], state.base_batch["agent_mask"]
    )[0, start:end]
    gt = _cpu_numpy(agents_btkf)
    gt_valid = (gt[..., 5] > 0.5) & state.agent_mask[None, :]
    if int(round(float(gt[context_frames - 1, 0, 7]))) != 1:
        raise ValueError("Focus slot 0 is not a vehicle at the context handoff")
    if not bool(gt_valid[:, 0].all()):
        invalid = np.flatnonzero(~gt_valid[:, 0]).tolist()
        raise ValueError(
            f"Focus car is not valid through the requested video; local frames={invalid}"
        )

    num_agents = len(state.agent_ids)
    xy = np.zeros((total_frames, num_agents, 2), dtype=np.float32)
    yaw = np.zeros((total_frames, num_agents), dtype=np.float32)
    velocity = np.zeros((total_frames, num_agents, 2), dtype=np.float32)
    valid = np.zeros((total_frames, num_agents), dtype=bool)
    # The first context frames are recorded history for every actor.
    xy[:context_frames] = gt[:context_frames, :, 0:2]
    yaw[:context_frames] = gt[:context_frames, :, 6]
    velocity[:context_frames] = gt[:context_frames, :, 3:5]
    valid[:context_frames] = gt_valid[:context_frames]

    # Match the interactive game exactly: at every generated frame decode the
    # trailing --decode-window latents and retain only the newest output.  A
    # one-shot chunk32/stride30 decode would reset its temporal history near
    # frames 32 and 62 and could introduce visible discontinuities.
    decode_window = max(1, int(server.args.decode_window))
    for frame_index in range(context_frames, total_frames):
        decode_start = max(0, frame_index + 1 - decode_window)
        z_window = z_pred[:, decode_start : frame_index + 1]
        decode_batch = server._decode_batch(
            state, decode_start, int(z_window.shape[1])
        )
        decoder_kwargs: dict[str, torch.Tensor] = {}
        if getattr(server.tokenizer.decoder, "attend_map", False):
            if state.map_tokens is None or state.map_mask is None:
                raise RuntimeError(
                    "Tokenizer decoder requires map memory, but none was encoded"
                )
            decoder_kwargs = {
                "encoder_map_tokens": state.map_tokens,
                "encoder_map_mask": state.map_mask,
            }
        decoded = server.tokenizer.decoder(
            z_window,
            agent_mask=decode_batch["agent_mask"],
            light_mask=decode_batch["light_mask"],
            **decoder_kwargs,
        )
        pred_xy = decoder_agent_xy(
            decoded,
            agent_xy_loss=str(server.model_args.agent_xy_loss),
            agent_xy_parameterization=str(
                server.model_args.agent_xy_parameterization
            ),
            anchor_xy=None,
        )[0, -1]
        continuous = decoded.agent_continuous[0, -1]
        xy[frame_index] = _cpu_numpy(pred_xy)
        yaw[frame_index] = _cpu_numpy(
            torch.atan2(continuous[:, 5], continuous[:, 6])
        )
        velocity[frame_index] = _cpu_numpy(continuous[:, 3:5])
        frame_valid = (
            torch.sigmoid(decoded.agent_valid_logits[0, -1])
            >= float(server.args.valid_threshold)
        ).detach().cpu().numpy()
        valid[frame_index] = frame_valid & state.agent_mask

    # Slot 0 follows the exact recorded focus action trajectory.  Dreamer
    # supplies the reactions of all remaining actor slots.
    xy[:, 0] = gt[:, 0, 0:2]
    yaw[:, 0] = gt[:, 0, 6]
    velocity[:, 0] = gt[:, 0, 3:5]
    valid[:, 0] = gt_valid[:, 0]

    if xy.shape != (total_frames, len(state.agent_ids), 2):
        raise ValueError(f"Unexpected decoded XY shape: {xy.shape}")
    return RolloutFrames(xy=xy, yaw=yaw, velocity=velocity, valid=valid)


def build_puffer_frame(
    state: SessionState,
    rollout: RolloutFrames,
    frame_index: int,
) -> PufferFrameState:
    selected = state.agent_mask & (state.agent_ids >= 0)
    world_xy, world_yaw, world_velocity = local_to_world_pose(
        rollout.xy[frame_index, selected],
        rollout.yaw[frame_index, selected],
        rollout.velocity[frame_index, selected],
        origin_xy=state.ego_origin_xy,
        origin_heading=state.ego_heading,
    )
    return PufferFrameState(
        step=int(frame_index),
        agent_ids=state.agent_ids[selected],
        agent_types=state.agent_types[selected],
        xy=world_xy,
        yaw=world_yaw,
        velocity_xy=world_velocity,
        valid=rollout.valid[frame_index, selected],
        source_time_index=state.context_start_frame + int(frame_index),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_game_parser()
    parser.description = (
        "Render headless PufferDrive MP4s from recorded focus-car actions."
    )
    parser.set_defaults(
        renderer="puffer",
        puffer_strict=True,
        puffer_manifest=str(DEFAULT_PUFFER_MANIFEST),
        context_frames=11,
        max_steps=80,
        fps=10.0,
        puffer_timeout=120.0,
        autoplay=False,
    )
    for action in parser._actions:
        if action.dest == "max_steps":
            action.help = (
                "Generated model frames after recorded context replay. Default: 80. "
                "Recorded focus actions require enough future NPZ frames."
            )
        elif action.dest == "fps":
            action.help = "Output MP4 frames per second. Default: 10."
    parser.add_argument(
        "--subset-manifest",
        "--selection-manifest",
        dest="subset_manifest",
        default=str(DEFAULT_SUBSET_MANIFEST),
        help="JSON manifest whose sample_order defines 'first N'.",
    )
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay subset/dataset/scenario/focus/progress identity on each frame.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.renderer != "puffer":
        parser.error("this offline renderer requires --renderer puffer")
    if not args.puffer_manifest:
        parser.error("--puffer-manifest is required")
    if args.context_frames < 2:
        parser.error("--context-frames must be at least 2")
    if args.max_steps < 1:
        parser.error("--unroll-steps must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.num_scenes < 1:
        parser.error("--num-scenes must be positive")


def render_model_rollouts(args: argparse.Namespace) -> Path:
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
    server = WaymoInteractiveServer(args)
    dataset_paths = list(server.dataset.paths)
    results: list[dict[str, Any]] = []
    started = time.time()

    try:
        for batch_index, record in enumerate(records, start=1):
            dataset_index = resolve_dataset_index(record, dataset_paths)
            state = server._load_scene(dataset_index)
            if state.puffer_scene is None or server.puffer_renderer is None:
                raise RuntimeError(
                    f"No Puffer render asset for dataset scene #{dataset_index}"
                )
            if state.scenario_id != str(record["scenario_id"]):
                raise ValueError(
                    f"Scenario mismatch: subset={record['scenario_id']} loaded={state.scenario_id}"
                )
            if int(state.agent_ids[0]) != int(record["focus_track_id"]):
                raise ValueError(
                    f"Focus mismatch: subset={record['focus_track_id']} loaded={state.agent_ids[0]}"
                )

            scene_seed = int(args.seed) + int(record["sample_order"])
            print(
                f"[batch {batch_index}/{len(records)}] generating dataset "
                f"#{dataset_index} scenario={state.scenario_id} "
                f"focus={int(state.agent_ids[0])} seed={scene_seed}",
                flush=True,
            )
            scene_started = time.time()
            output_path = output_dir / output_video_name(
                record,
                context_frames=int(args.context_frames),
                unroll_steps=int(args.max_steps),
            )
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}; pass --overwrite to replace it"
                )
            # Fail on a broken map, Raylib, or Xvfb before paying for the
            # 80-step model rollout. This exact context frame is reused below.
            first_jpeg = server.puffer_renderer.render(
                state.puffer_scene, server._puffer_frame_state(state)
            )
            print(
                f"[batch {batch_index}/{len(records)}] headless Puffer preflight passed",
                flush=True,
            )
            rollout = generate_gt_focus_rollout(
                server,
                state,
                unroll_steps=int(args.max_steps),
                seed=scene_seed,
            )
            with FfmpegJpegWriter(
                output_path,
                ffmpeg=args.ffmpeg,
                fps=float(args.fps),
                overwrite=bool(args.overwrite),
            ) as writer:
                for frame_index in range(rollout.count):
                    if frame_index == 0:
                        jpeg = first_jpeg
                    else:
                        frame = build_puffer_frame(state, rollout, frame_index)
                        jpeg = server.puffer_renderer.render(state.puffer_scene, frame)
                    if args.overlay:
                        if frame_index < int(args.context_frames):
                            progress = (
                                f"CONTEXT {frame_index + 1}/{int(args.context_frames)}"
                            )
                        else:
                            progress = (
                                f"ROLLOUT {frame_index - int(args.context_frames) + 1}/"
                                f"{int(args.max_steps)}"
                            )
                        jpeg = overlay_label(
                            jpeg,
                            (
                                f"sample #{int(record['sample_order'])} | "
                                f"dataset #{dataset_index} | scenario {state.scenario_id} | "
                                f"focus {int(state.agent_ids[0])} | {progress}"
                            ),
                        )
                    writer.write(jpeg)
                    if (frame_index + 1) % 10 == 0 or frame_index + 1 == rollout.count:
                        print(
                            f"[batch {batch_index}/{len(records)}] rendered "
                            f"{frame_index + 1}/{rollout.count} frames",
                            flush=True,
                        )

            seconds = time.time() - scene_started
            print(f"[saved] {output_path} ({seconds:.1f}s)", flush=True)
            results.append(
                {
                    "sample_order": int(record["sample_order"]),
                    "dataset_index": dataset_index,
                    "scenario_id": state.scenario_id,
                    "focus_track_id": int(state.agent_ids[0]),
                    "npz_path": state.scene_path,
                    "output_mp4": str(output_path),
                    "context_frames": int(args.context_frames),
                    "generated_frames": int(args.max_steps),
                    "total_frames": rollout.count,
                    "fps": float(args.fps),
                    "seed": scene_seed,
                    "seconds": seconds,
                }
            )
            del rollout, state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        server.close()

    summary_path = output_dir / "summary.json"
    summary = {
        "checkpoint_profile": server.checkpoint_profile,
        "checkpoint_path": str(server.checkpoint_path),
        "subset_manifest": str(Path(args.subset_manifest).expanduser().resolve()),
        "puffer_manifest": str(Path(args.puffer_manifest).expanduser().resolve()),
        "action_source": "recorded focus car; [dx,dy,dyaw,speed,vx,vy,valid]",
        "context_frames": int(args.context_frames),
        "generated_frames": int(args.max_steps),
        "num_videos": len(results),
        "elapsed_seconds": time.time() - started,
        "videos": results,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[complete] videos={len(results)} summary={summary_path}", flush=True)
    return summary_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    render_model_rollouts(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
