#!/usr/bin/env python3
"""Interactive browser game for an action-conditioned Waymo world model.

The default episode first replays recorded Waymo frames 1 through 11.  The
checkpoint's 11-token rollout window then retains recorded frames 2 through 11
as the ten-token past and appends one query token to predict frame 12.  The 2D
renderer generates 80 interactive frames by default; both PufferDrive 3D
frontends generate 150.

The human controls the focus agent (slot 0).  Each simulation tick converts
the held arrow keys into the raw action representation used during training:

    [delta_x, delta_y, delta_yaw, speed, velocity_x, velocity_y, valid]

The commanded focus state is integrated exactly and supplied to the dynamics
model.  The decoded world-model state supplies all other agents.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from aiohttp import WSMsgType, web
from PIL import Image, ImageDraw, ImageFont

WAYMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.puffer_renderer_bridge import (  # noqa: E402
    PufferBridgeError,
    PufferFrameState,
    PufferRendererClient,
    PufferSceneReference,
    ScenarioManifest,
    local_to_world_pose,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


WORLD_MODEL_CHECKPOINT_DIR = (
    REPO_ROOT
    / "waymo/checkpoints/waymo_wm_original_stmlayer_3_stage"
    / "waymo_wm_stage1best_mon8_fullmotion_physproxy_h30best30k_ctx1_h90_d1_chunk32s30_30k"
)
WORLD_MODEL_CHECKPOINT_PROFILES = {
    # User-facing names intentionally describe the two rollout variants, not
    # the checkpoint's internal step metadata (the finetuned h30 file records
    # step=0).
    "h30": WORLD_MODEL_CHECKPOINT_DIR / "best_multisample_finetuned.pt",
    "h90": WORLD_MODEL_CHECKPOINT_DIR / "step_00027000.pt",
}
DEFAULT_CHECKPOINT_PROFILE = "h90"
DEFAULT_CONTEXT_FRAMES = 11
DEFAULT_2D_ROLLOUT_STEPS = 80
DEFAULT_3D_ROLLOUT_STEPS = 150
# Backward-compatible name for callers that mean the original 2D default.
DEFAULT_ROLLOUT_STEPS = DEFAULT_2D_ROLLOUT_STEPS
DEFAULT_HTML = WAYMO_ROOT / "interactive_world_model_game.html"
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


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


@dataclass(frozen=True)
class ControlConfig:
    dt: float = 0.1
    acceleration_mps2: float = 5.0
    braking_mps2: float = 8.0
    yaw_rate_deg_s: float = 45.0
    min_speed_mps: float = 0.0
    max_speed_mps: float = 30.0


@dataclass(frozen=True)
class AnalogControl:
    """Normalised local controller input.

    Positive steering turns left (counter-clockwise); throttle and brake are
    independently normalised to ``[0, 1]``.  Keeping this representation
    device-independent lets wheel-specific axis conventions stay in the local
    frontend.
    """

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0


@dataclass
class FocusState:
    x: float
    y: float
    speed: float
    yaw: float

    @property
    def velocity(self) -> tuple[float, float]:
        return self.speed * math.cos(self.yaw), self.speed * math.sin(self.yaw)


def context_frame_bounds(
    start_frame: int,
    total_steps: int,
    context_frames: int,
) -> tuple[int, int]:
    """Return the recorded half-open interval used for the initial replay."""

    start = int(start_frame)
    total = int(total_steps)
    count = int(context_frames)
    if start < 0:
        raise ValueError(f"start-frame must be non-negative, got {start}")
    if count < 2:
        raise ValueError(f"context-frames must be at least 2, got {count}")
    end = start + count
    if end > total:
        raise ValueError(
            f"Recorded context [{start}, {end}) exceeds the available {total} frames"
        )
    return start, end


def prediction_context_bounds(
    context_start: int,
    context_frames: int,
    max_rollout_window: int,
) -> tuple[int, int]:
    """Return recorded frames retained for the first prediction.

    ``max_rollout_window`` includes the noisy prediction token, so a window of
    11 retains ten past frames.  With the default replay interval ``[0, 11)``,
    this returns ``[1, 11)``: human-readable Waymo frames 2 through 11.
    """

    start = int(context_start)
    count = int(context_frames)
    window = int(max_rollout_window)
    end = start + count
    past_keep = count if window <= 0 else min(count, max(1, window - 1))
    return end - past_keep, end


def resolve_rollout_steps(renderer: str, max_steps: int | None) -> int:
    """Resolve the default horizon without overriding an explicit CLI value."""

    if max_steps is not None:
        return int(max_steps)
    if str(renderer) == "puffer":
        return DEFAULT_3D_ROLLOUT_STEPS
    return DEFAULT_2D_ROLLOUT_STEPS


def initial_focus_action(
    state: FocusState,
    velocity_xy: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Build the time-zero action expected by ``build_ego_action_features``."""
    vx, vy = state.velocity if velocity_xy is None else velocity_xy
    action = torch.zeros(16, dtype=torch.float32)
    action[3:7] = torch.tensor([state.speed, vx, vy, 1.0], dtype=torch.float32)
    return action


def integrate_focus_control(
    state: FocusState,
    keys_down: Iterable[str],
    config: ControlConfig,
) -> tuple[FocusState, torch.Tensor]:
    """Advance the commanded focus state and return its native 16-D action."""
    keys = set(keys_down)
    throttle = float("ArrowUp" in keys) - float("ArrowDown" in keys)
    steering = float("ArrowLeft" in keys) - float("ArrowRight" in keys)

    speed = float(
        np.clip(
            state.speed + throttle * config.acceleration_mps2 * config.dt,
            config.min_speed_mps,
            config.max_speed_mps,
        )
    )
    delta_yaw = math.radians(config.yaw_rate_deg_s) * config.dt * steering
    yaw = wrap_angle(state.yaw + delta_yaw)
    vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)
    delta_x, delta_y = vx * config.dt, vy * config.dt
    next_state = FocusState(
        x=state.x + delta_x,
        y=state.y + delta_y,
        speed=speed,
        yaw=yaw,
    )

    action = torch.zeros(16, dtype=torch.float32)
    action[:7] = torch.tensor(
        [delta_x, delta_y, delta_yaw, speed, vx, vy, 1.0],
        dtype=torch.float32,
    )
    return next_state, action


def integrate_focus_analog_control(
    state: FocusState,
    control: AnalogControl,
    config: ControlConfig,
) -> tuple[FocusState, torch.Tensor]:
    """Advance the focus state from continuous wheel and pedal inputs."""

    steering = float(np.clip(control.steering, -1.0, 1.0))
    throttle = float(np.clip(control.throttle, 0.0, 1.0))
    brake = float(np.clip(control.brake, 0.0, 1.0))
    acceleration = (
        throttle * config.acceleration_mps2 - brake * config.braking_mps2
    )
    speed = float(
        np.clip(
            state.speed + acceleration * config.dt,
            config.min_speed_mps,
            config.max_speed_mps,
        )
    )
    delta_yaw = math.radians(config.yaw_rate_deg_s) * config.dt * steering
    yaw = wrap_angle(state.yaw + delta_yaw)
    vx, vy = speed * math.cos(yaw), speed * math.sin(yaw)
    delta_x, delta_y = vx * config.dt, vy * config.dt
    next_state = FocusState(
        x=state.x + delta_x,
        y=state.y + delta_y,
        speed=speed,
        yaw=yaw,
    )

    action = torch.zeros(16, dtype=torch.float32)
    action[:7] = torch.tensor(
        [delta_x, delta_y, delta_yaw, speed, vx, vy, 1.0],
        dtype=torch.float32,
    )
    return next_state, action


def _checkpoint_arg_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, argparse.Namespace):
        return dict(vars(value))
    if isinstance(value, dict):
        return dict(value)
    return {}


def _model_args_from_checkpoint(checkpoint: dict[str, Any]) -> SimpleNamespace:
    # Populate newer optional arguments that may not have existed when an older
    # checkpoint was written, then let checkpoint metadata remain authoritative.
    defaults = vars(
        wm.build_argparser().parse_args(
            ["--data_dir", "unused", "--tokenizer_ckpt", "unused"]
        )
    )
    defaults.update(_checkpoint_arg_dict(checkpoint.get("args", {})))
    return SimpleNamespace(**defaults)


def _first_path(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return None if value is None else str(value)


def _scenario_id(item: dict[str, Any], path: str) -> str:
    value = item.get("scenario_id", "")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value) or Path(path).stem


def resolve_world_model_checkpoint(
    checkpoint_profile: str,
    world_model_ckpt: str | Path | None,
) -> tuple[Path, str]:
    """Resolve a named interactive checkpoint or an explicit path override."""

    if world_model_ckpt:
        return Path(world_model_ckpt).expanduser().resolve(), "custom"
    try:
        path = WORLD_MODEL_CHECKPOINT_PROFILES[str(checkpoint_profile)]
    except KeyError as error:
        choices = ", ".join(sorted(WORLD_MODEL_CHECKPOINT_PROFILES))
        raise ValueError(
            f"Unknown checkpoint profile {checkpoint_profile!r}; choose one of: {choices}"
        ) from error
    return path.expanduser().resolve(), str(checkpoint_profile)


def _to_cpu_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _agent_color(agent_type: int, *, focus: bool = False) -> tuple[int, int, int]:
    if focus:
        return (78, 232, 140)
    return {
        1: (74, 158, 255),
        2: (234, 92, 190),
        3: (255, 190, 52),
    }.get(int(agent_type), (190, 195, 205))


def _lighten(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(int(c + (255 - c) * amount) for c in color)


def _world_to_pixel(
    points: np.ndarray,
    *,
    center_xy: np.ndarray,
    radius_m: float,
    canvas_size: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    scale = canvas_size / (2.0 * float(radius_m))
    px = canvas_size * 0.5 + (points[..., 0] - center_xy[0]) * scale
    py = canvas_size * 0.5 - (points[..., 1] - center_xy[1]) * scale
    return np.stack([px, py], axis=-1)


def _draw_agent(
    draw: ImageDraw.ImageDraw,
    xy_px: np.ndarray,
    yaw: float,
    color: tuple[int, int, int],
    *,
    focus: bool,
) -> None:
    x, y = float(xy_px[0]), float(xy_px[1])
    # Image y points down, hence the negative sine.
    direction = np.asarray([math.cos(yaw), -math.sin(yaw)], dtype=np.float32)
    side = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    length = 13.0 if focus else 10.0
    width = 7.0 if focus else 6.0
    center = np.asarray([x, y], dtype=np.float32)
    polygon = [
        center + direction * length,
        center - direction * length * 0.65 + side * width,
        center - direction * length * 0.65 - side * width,
    ]
    draw.polygon([tuple(map(float, point)) for point in polygon], fill=color, outline=(15, 18, 25))
    if focus:
        r = 11.0
        draw.ellipse((x - r, y - r, x + r, y + r), outline=_lighten(color, 0.35), width=2)


@dataclass
class SessionState:
    scene_index: int
    scenario_id: str
    scene_path: str
    puffer_scene: PufferSceneReference | None
    base_batch: dict[str, Any]
    map_tokens: torch.Tensor | None
    map_mask: torch.Tensor | None
    map_polylines: np.ndarray
    map_point_mask: np.ndarray
    agent_mask: np.ndarray
    agent_ids: np.ndarray
    agent_types: np.ndarray
    ego_origin_xy: np.ndarray
    ego_heading: float
    context_start_frame: int
    focus: FocusState
    z_history: list[torch.Tensor]
    action_history: list[torch.Tensor]
    action_mask_history: list[torch.Tensor]
    context_focus: list[FocusState]
    context_world: np.ndarray
    context_valid: np.ndarray
    context_yaw: np.ndarray
    context_velocity: np.ndarray
    world_history: list[np.ndarray]
    valid_history: list[np.ndarray]
    yaw_history: list[np.ndarray]
    velocity_history: list[np.ndarray]
    keys_down: set[str] = field(default_factory=set)
    paused: bool = True
    step_once: bool = False
    reset_requested: bool = False
    new_scene_requested: bool = False
    replay_index: int = 0
    step: int = 0
    cached_jpeg: bytes | None = None
    cached_frame_id: int = -1
    last_inference_ms: float = 0.0
    renderer_name: str = "2d"
    renderer_error: str | None = None
    puffer_disabled: bool = False


def scene_identity(state: SessionState) -> dict[str, Any]:
    """Return a stable, reportable identifier for the current focus view."""

    focus_track_id = int(state.agent_ids[0])
    scene_file = Path(state.scene_path).name
    label = (
        f"scene #{state.scene_index} | scenario {state.scenario_id} | "
        f"focus {focus_track_id}"
    )
    return {
        "scene_index": int(state.scene_index),
        "scenario_id": str(state.scenario_id),
        "focus_track_id": focus_track_id,
        "scene_file": scene_file,
        "scene_label": label,
    }


class WaymoInteractiveServer:
    def __init__(self, args: argparse.Namespace):
        args.max_steps = resolve_rollout_steps(args.renderer, args.max_steps)
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {args.device!r} was requested, but CUDA is unavailable. "
                "Run on a GPU node or pass --device cpu for a slow functional test."
            )
        wm.seed_everything(args.seed)
        self.random = random.Random(args.seed)
        self.infer_lock = asyncio.Lock()
        self.control = ControlConfig(
            dt=args.sim_dt,
            acceleration_mps2=args.acceleration,
            braking_mps2=args.braking,
            yaw_rate_deg_s=args.yaw_rate,
            min_speed_mps=args.min_speed,
            max_speed_mps=args.max_speed,
        )
        self.context_frames = int(args.context_frames)
        if self.context_frames < 2:
            raise ValueError(
                f"context-frames must be at least 2, got {self.context_frames}"
            )
        if int(args.max_steps) < 1:
            raise ValueError(f"max-steps must be positive, got {args.max_steps}")

        checkpoint_path, checkpoint_profile = resolve_world_model_checkpoint(
            args.checkpoint_profile,
            args.world_model_ckpt,
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"World-model checkpoint for profile {checkpoint_profile!r} not found: "
                f"{checkpoint_path}"
            )
        self.checkpoint_profile = checkpoint_profile
        self.checkpoint_path = checkpoint_path
        print(
            f"[load] world model profile={checkpoint_profile}: {checkpoint_path}",
            flush=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        self.checkpoint = checkpoint
        self.model_args = _model_args_from_checkpoint(checkpoint)
        self._validate_checkpoint_contract()
        self.model_rollout_window = int(self.model_args.max_rollout_window)
        self.model_context_frames = (
            self.context_frames
            if self.model_rollout_window <= 0
            else min(self.context_frames, max(1, self.model_rollout_window - 1))
        )

        tokenizer_path = args.tokenizer_ckpt or _first_path(self.model_args.tokenizer_ckpt)
        if not tokenizer_path:
            raise ValueError("No tokenizer checkpoint was provided or recorded in the world-model checkpoint")
        tokenizer_path = str(Path(tokenizer_path).expanduser().resolve())
        print(f"[load] tokenizer: {tokenizer_path}", flush=True)
        self.tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(tokenizer_path, self.device)
        if isinstance(self.tokenizer, wm.FrozenWaymoFocusTokenizer):
            raise ValueError("The interactive Waymo game requires the vector tokenizer")

        n_latents = int(tok_args.get("n_latents", self.tokenizer.decoder.n_latents))
        d_bottleneck = int(tok_args.get("d_bottleneck", self.tokenizer.decoder.up_proj.in_features))
        packing_factor = int(self.model_args.packing_factor)
        if n_latents % packing_factor:
            raise ValueError(f"n_latents={n_latents} is not divisible by packing_factor={packing_factor}")
        self.model_args.n_spatial = n_latents // packing_factor
        self.model_args.d_spatial = d_bottleneck * packing_factor
        self.d_bottleneck = d_bottleneck

        self.dynamics = base_eval.build_dynamics(
            self.model_args,
            d_bottleneck,
            self.device,
            map_memory_dim=(
                wm.tokenizer_map_memory_dim(self.tokenizer)
                if self.model_args.dynamics_attend_map
                else None
            ),
        )
        base_eval.load_dynamics_state(self.dynamics, str(checkpoint_path), ckpt=checkpoint)
        self.dynamics.eval()
        # Training checkpoints also contain a large optimizer state.  Retain
        # only small metadata after loading the dynamics weights so the game
        # does not keep the optimizer/state-dict mapping resident in CPU RAM.
        self.checkpoint = {
            key: checkpoint.get(key)
            for key in ("format", "step", "epoch")
            if key in checkpoint
        }
        del checkpoint
        gc.collect()
        self.schedule = wm.make_tau_schedule(
            k_max=int(self.model_args.k_max),
            schedule="shortcut",
            d=float(args.eval_d),
        )
        if int(self.schedule["K"]) != 1:
            print(
                f"[warning] eval_d={args.eval_d:g} uses {self.schedule['K']} solver passes per frame; "
                "D1 (--eval-d 1) is recommended for interactive speed.",
                flush=True,
            )

        data_dir = args.data_dir or _first_path(self.model_args.val_data_dir)
        if not data_dir:
            raise ValueError("No --data-dir was provided and checkpoint args contain no val_data_dir")
        self.dataset = wm.WaymoVectorDataset(str(Path(data_dir).expanduser().resolve()))
        if not (0 <= args.scene_index < len(self.dataset)):
            raise IndexError(f"scene-index must be in [0, {len(self.dataset) - 1}]")
        self.initial_scene_index = int(args.scene_index)

        self.puffer_manifest: ScenarioManifest | None = None
        self.puffer_renderer: PufferRendererClient | None = None
        self.puffer_scene_indices: tuple[int, ...] = ()
        self.scene_queue: tuple[int, ...] = ()
        self._reported_puffer_errors: set[str] = set()
        if args.renderer == "puffer":
            if args.puffer_manifest:
                try:
                    self.puffer_manifest = ScenarioManifest(
                        args.puffer_manifest,
                        scenario_cache_dir=args.puffer_scenario_cache,
                    )
                except (FileNotFoundError, ValueError) as error:
                    if args.puffer_strict:
                        raise
                    print(
                        f"[puffer warning] {error}; trying the raw Scenario cache "
                        "for diagnostics only",
                        flush=True,
                    )
                    self.puffer_manifest = ScenarioManifest(
                        scenario_cache_dir=args.puffer_scenario_cache
                    )
            else:
                self.puffer_manifest = ScenarioManifest(
                    scenario_cache_dir=args.puffer_scenario_cache
                )
            self.puffer_renderer = PufferRendererClient(
                args.puffer_worker_command,
                width=args.puffer_width,
                height=args.puffer_height,
                view_mode=args.puffer_view_mode,
                jpeg_quality=args.jpeg_quality,
                timeout_s=args.puffer_timeout,
                environment=(
                    {"PUFFER_USE_INHERITED_DISPLAY": "1"}
                    if args.puffer_use_inherited_display
                    else None
                ),
            )
            if self.puffer_manifest is not None:
                mapped_paths = self.puffer_manifest.mapped_npz_paths
                mapped_scene_indices = tuple(
                    index
                    for index, path in enumerate(self.dataset.paths)
                    if str(Path(path).expanduser().resolve(strict=False)) in mapped_paths
                )
                self.puffer_scene_indices = tuple(
                    index
                    for index in mapped_scene_indices
                    if self._has_continuous_puffer_focus_context(index)
                )
                skipped = len(mapped_scene_indices) - len(self.puffer_scene_indices)
                if skipped:
                    print(
                        f"[puffer] skipped {skipped} mapped views whose focus agent "
                        f"is absent during the {self.context_frames}-frame chase-camera replay",
                        flush=True,
                    )
                if mapped_scene_indices and not self.puffer_scene_indices:
                    raise ValueError(
                        "No mapped Puffer scene has a continuously valid focus agent "
                        f"through the {self.context_frames}-frame context replay"
                    )
                if (
                    self.puffer_scene_indices
                    and self.initial_scene_index not in self.puffer_scene_indices
                ):
                    replacement = int(self.puffer_scene_indices[0])
                    print(
                        f"[puffer] scene-index {self.initial_scene_index} is not "
                        f"renderable for this manifest/context; starting at {replacement}",
                        flush=True,
                    )
                    self.initial_scene_index = replacement

        requested_scene_queue = tuple(
            int(index) for index in (getattr(args, "scene_queue", None) or ())
        )
        if requested_scene_queue:
            available_scene_indices: tuple[int, ...] | range
            if args.renderer == "puffer" and self.puffer_scene_indices:
                available_scene_indices = self.puffer_scene_indices
            else:
                available_scene_indices = range(len(self.dataset))
            available_set = set(available_scene_indices)
            unavailable = [
                index for index in requested_scene_queue if index not in available_set
            ]
            if unavailable:
                raise ValueError(
                    "Queued scene indices are not available to the active renderer: "
                    + ", ".join(str(index) for index in unavailable)
                )
            requested_set = set(requested_scene_queue)
            self.scene_queue = requested_scene_queue + tuple(
                int(index)
                for index in available_scene_indices
                if index not in requested_set
            )
            print(
                "[scene queue] priority: "
                + " -> ".join(str(index) for index in requested_scene_queue),
                flush=True,
            )

        self.html = Path(args.html).read_text(encoding="utf-8")
        print(
            f"[ready] checkpoint step={int(self.checkpoint.get('step', -1))} "
            f"profile={self.checkpoint_profile} "
            f"dataset scenes={len(self.dataset)} device={self.device} "
            f"dtype={next(self.dynamics.parameters()).dtype} renderer={args.renderer} "
            f"replay={self.context_frames} model_context={self.model_context_frames} "
            f"rollout={int(args.max_steps)}",
            flush=True,
        )

    def _validate_checkpoint_contract(self) -> None:
        problems = []
        if self.checkpoint.get("format") == base_eval.MOTION_LATENT_V1_FORMAT:
            problems.append("checkpoint is MotionLatent V1, not the legacy latent world model")
        if not bool(getattr(self.model_args, "use_ego_actions", False)):
            problems.append("checkpoint was not trained with ego actions")
        if str(getattr(self.model_args, "ego_action_source", "")) != "focus":
            problems.append("ego_action_source is not 'focus'")
        if str(getattr(self.model_args, "ego_action_normalization", "")) != "raw":
            problems.append("ego_action_normalization is not 'raw'")
        if str(getattr(self.model_args, "agent_xy_parameterization", "")) != "absolute":
            problems.append("agent_xy_parameterization is not 'absolute'")
        if problems:
            raise ValueError("Incompatible checkpoint: " + "; ".join(problems))

    def _has_continuous_puffer_focus_context(self, scene_index: int) -> bool:
        """Whether Puffer's chase camera can follow slot 0 for the full replay."""

        dataset_paths = getattr(self.dataset, "paths", None)
        if dataset_paths is not None:
            # Avoid loading each scene's much larger map arrays while filtering
            # a conversion manifest at startup.
            with np.load(dataset_paths[int(scene_index)], allow_pickle=False) as data:
                agents = np.asarray(data["agents"])
                agent_mask = np.asarray(data["agent_mask"], dtype=bool)
                total_steps = int(data["lights"].shape[0])
        else:
            # Small synthetic datasets used by tests need not expose paths.
            item = self.dataset[int(scene_index)]
            agents = np.asarray(item["agents"])
            agent_mask = np.asarray(item["agent_mask"], dtype=bool)
            total_steps = int(item["lights"].shape[0])
        try:
            start, end = context_frame_bounds(
                self.args.start_frame,
                total_steps,
                self.context_frames,
            )
        except ValueError:
            return False
        if agents.ndim != 3 or agent_mask.ndim != 1:
            return False
        if agents.shape[0] == agent_mask.shape[0]:
            focus_valid = agents[0, start:end, 5] > 0.5
        elif agents.shape[1] == agent_mask.shape[0]:
            focus_valid = agents[start:end, 0, 5] > 0.5
        else:
            return False
        return bool(agent_mask[0]) and bool(focus_valid.all())

    def _pick_new_scene_index(self, current: int) -> int:
        scene_queue = getattr(self, "scene_queue", ())
        if scene_queue:
            try:
                current_position = scene_queue.index(int(current))
            except ValueError:
                return int(scene_queue[0])
            return int(scene_queue[(current_position + 1) % len(scene_queue)])

        candidates: tuple[int, ...] | range
        if self.args.renderer == "puffer" and self.puffer_scene_indices:
            candidates = self.puffer_scene_indices
        else:
            candidates = range(len(self.dataset))
        if not candidates:
            return current
        if len(candidates) == 1:
            return int(candidates[0])
        candidate = current
        while candidate == current:
            candidate = self.random.choice(candidates)
        return candidate

    @torch.inference_mode()
    def _load_scene(self, scene_index: int) -> SessionState:
        item = self.dataset[int(scene_index)]
        batch = wm.move_batch(wm._collate([item]), self.device)
        total_steps = int(batch["lights"].shape[1])
        start_frame, context_end = context_frame_bounds(
            self.args.start_frame,
            total_steps,
            self.context_frames,
        )
        context = wm.slice_future_batch(batch, start_frame, context_end)
        z_context, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            self.tokenizer,
            context,
            self.model_args,
            return_map=bool(self.model_args.dynamics_attend_map),
        )
        z_context_packed = wm.pack_bottleneck_to_spatial(
            z_context,
            n_spatial=int(self.model_args.n_spatial),
            k=int(self.model_args.packing_factor),
        )[0].detach()
        context_actions, context_action_masks, _action_slots = (
            wm.build_ego_action_features(context, self.model_args)
        )
        if context_actions is None or context_action_masks is None:
            raise RuntimeError("Interactive context requires focus-agent action features")
        if int(z_context_packed.shape[0]) != self.context_frames:
            raise ValueError(
                f"Tokenizer returned {z_context_packed.shape[0]} context frames; "
                f"expected {self.context_frames}"
            )

        agents = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])[0]
        context_agents = _to_cpu_numpy(agents[start_frame:context_end]).copy()
        agent_mask = batch["agent_mask"][0].detach().cpu().numpy().astype(bool)
        context_valid = (context_agents[..., 5] > 0.5) & agent_mask[None, :]
        valid_focus_frames = np.flatnonzero(context_valid[:, 0])
        if not context_valid[-1, 0]:
            raise ValueError(
                f"Focus slot is invalid at the frame-{context_end} control handoff"
            )
        # A focus track can enter the scene during the recorded context.  The
        # tokenizer/action masks retain its true validity.  For the 2D camera
        # readout only, borrow the nearest valid pose instead of jumping to a
        # padded zero row while the actor is absent.  Puffer scenes with such
        # gaps are excluded above because its chase camera requires a valid
        # focus actor on every displayed frame.
        context_focus = []
        for frame_index in range(self.context_frames):
            source_index = frame_index
            if not context_valid[frame_index, 0]:
                source_index = int(
                    valid_focus_frames[
                        np.argmin(np.abs(valid_focus_frames - frame_index))
                    ]
                )
            focus_row = context_agents[source_index, 0]
            context_focus.append(
                FocusState(
                    x=float(focus_row[0]),
                    y=float(focus_row[1]),
                    speed=max(0.0, float(focus_row[2])),
                    yaw=float(focus_row[6]),
                )
            )
        context_world = context_agents[..., 0:2].copy()
        context_yaw = context_agents[..., 6].copy()
        context_velocity = context_agents[..., 3:5].copy()
        focus = context_focus[0]

        agent_types = np.rint(context_agents[-1, :, 7]).astype(np.int64)
        if not agent_mask[0]:
            raise ValueError("Focus slot 0 is not selected by agent_mask")
        if int(context_actions.shape[1]) != self.context_frames:
            raise ValueError(
                f"Action builder returned {context_actions.shape[1]} context frames; "
                f"expected {self.context_frames}"
            )
        scene_path = str(item.get("path", self.dataset.paths[int(scene_index)]))
        scenario_id = _scenario_id(item, scene_path)
        agent_ids = batch["agent_ids"][0].detach().cpu().numpy()
        focus_track_id = int(agent_ids[0])
        if "focus_track_id" in item:
            recorded_focus_id = int(torch.as_tensor(item["focus_track_id"]).item())
            if recorded_focus_id != focus_track_id:
                raise ValueError(
                    f"Focus track mismatch for {scene_path}: slot 0={focus_track_id}, "
                    f"metadata={recorded_focus_id}"
                )
        ego_origin_xy = _to_cpu_numpy(batch["ego_origin_xy"][0])
        ego_heading = float(batch["ego_heading"][0].detach().cpu())
        puffer_scene = None
        renderer_error = None
        if self.puffer_manifest is not None:
            try:
                puffer_scene = self.puffer_manifest.resolve(
                    scenario_id=scenario_id,
                    npz_path=scene_path,
                    focus_track_id=focus_track_id,
                )
            except (FileNotFoundError, KeyError, ValueError) as error:
                if self.args.puffer_strict:
                    raise
                renderer_error = str(error)
        state = SessionState(
            scene_index=int(scene_index),
            scenario_id=scenario_id,
            scene_path=scene_path,
            puffer_scene=puffer_scene,
            base_batch=batch,
            map_tokens=map_tokens,
            map_mask=map_mask,
            map_polylines=_to_cpu_numpy(batch["map_polylines"][0]),
            map_point_mask=batch["map_mask"][0].detach().cpu().numpy().astype(bool),
            agent_mask=agent_mask,
            agent_ids=agent_ids,
            agent_types=agent_types,
            ego_origin_xy=ego_origin_xy,
            ego_heading=ego_heading,
            context_start_frame=start_frame,
            focus=focus,
            z_history=[value for value in z_context_packed.unbind(dim=0)],
            action_history=[
                value.detach() for value in context_actions[0].unbind(dim=0)
            ],
            action_mask_history=[
                value.detach() for value in context_action_masks[0].unbind(dim=0)
            ],
            context_focus=context_focus,
            context_world=context_world,
            context_valid=context_valid,
            context_yaw=context_yaw,
            context_velocity=context_velocity,
            world_history=[context_world[0].copy()],
            valid_history=[context_valid[0].copy()],
            yaw_history=[context_yaw[0].copy()],
            velocity_history=[context_velocity[0].copy()],
            paused=not self.args.autoplay,
            renderer_error=renderer_error,
        )
        identity = scene_identity(state)
        print(
            f"[scene] {identity['scene_label']} | npz {identity['scene_file']}",
            flush=True,
        )
        return state

    def new_session(self) -> SessionState:
        return self._load_scene(self.initial_scene_index)

    def _reset_session(self, state: SessionState, *, new_scene: bool) -> None:
        index = self._pick_new_scene_index(state.scene_index) if new_scene else state.scene_index
        replacement = self._load_scene(index)
        state.__dict__.clear()
        state.__dict__.update(replacement.__dict__)

    @staticmethod
    def _replay_pending(state: SessionState) -> bool:
        return state.replay_index < len(state.context_focus) - 1

    @staticmethod
    def _display_frame_id(state: SessionState) -> int:
        """Zero-based timeline offset, including replay and generated frames."""

        return int(state.replay_index) + int(state.step)

    def _advance_replay(self, state: SessionState) -> None:
        """Display the next recorded context frame without running the model."""

        if not self._replay_pending(state):
            return
        state.replay_index += 1
        index = int(state.replay_index)
        state.focus = state.context_focus[index]
        state.world_history.append(state.context_world[index].copy())
        state.valid_history.append(state.context_valid[index].copy())
        state.yaw_history.append(state.context_yaw[index].copy())
        state.velocity_history.append(state.context_velocity[index].copy())
        state.last_inference_ms = 0.0

    def _decode_batch(
        self,
        state: SessionState,
        timeline_start: int,
        time_steps: int,
    ) -> dict[str, Any]:
        # Only masks and static map are consumed by the latent-only decoder.
        # Recorded context masks are preserved; generated positions use the
        # handoff frame's masks so decoding can continue beyond frame 91.
        base = state.base_batch
        context_start = int(state.context_start_frame)
        context_end = context_start + len(state.context_focus)
        agents_btkf = wm.agents_to_btkf(base["agents"], base["agent_mask"])
        context_agents = agents_btkf[:, context_start:context_end]
        context_lights = base["lights"][:, context_start:context_end]
        context_light_mask = base["light_mask"][:, context_start:context_end]
        timeline_indices = torch.arange(
            int(timeline_start),
            int(timeline_start) + int(time_steps),
            device=context_lights.device,
            dtype=torch.long,
        ).clamp(max=len(state.context_focus) - 1)
        return {
            **base,
            "agents": context_agents.index_select(1, timeline_indices),
            "lights": context_lights.index_select(1, timeline_indices),
            "light_mask": context_light_mask.index_select(1, timeline_indices),
        }

    @torch.inference_mode()
    def _advance(
        self,
        state: SessionState,
        analog_control: AnalogControl | None = None,
    ) -> None:
        if analog_control is None:
            next_focus, action = integrate_focus_control(
                state.focus, set(state.keys_down), self.control
            )
        else:
            next_focus, action = integrate_focus_analog_control(
                state.focus, analog_control, self.control
            )
        state.action_history.append(action.to(self.device))
        action_mask = torch.zeros_like(action, device=self.device)
        action_mask[:7] = 1.0
        state.action_mask_history.append(action_mask)

        past = torch.stack(state.z_history, dim=0).unsqueeze(0)
        actions = torch.stack(state.action_history, dim=0).unsqueeze(0)
        act_mask = torch.stack(state.action_mask_history, dim=0).unsqueeze(0)
        z_next = wm.sample_one_timestep_packed(
            self.dynamics,
            past_packed=past,
            actions_seq=actions,
            act_mask_seq=act_mask,
            map_tokens=state.map_tokens,
            map_mask=state.map_mask,
            k_max=int(self.model_args.k_max),
            sched=self.schedule,
            max_rollout_window=self.model_rollout_window,
        )
        state.z_history.append(z_next[0].detach())
        state.focus = next_focus
        state.step += 1

        decode_window = max(1, int(self.args.decode_window))
        decode_start = max(0, len(state.z_history) - decode_window)
        z_packed = torch.stack(state.z_history[decode_start:], dim=0).unsqueeze(0)
        z = wm.unpack_spatial_to_bottleneck(z_packed, k=int(self.model_args.packing_factor))
        decode_batch = self._decode_batch(state, decode_start, int(z.shape[1]))
        decoder_kwargs: dict[str, torch.Tensor] = {}
        if getattr(self.tokenizer.decoder, "attend_map", False):
            if state.map_tokens is None or state.map_mask is None:
                raise RuntimeError("Tokenizer decoder requires map memory, but none was encoded")
            decoder_kwargs = {
                "encoder_map_tokens": state.map_tokens,
                "encoder_map_mask": state.map_mask,
            }
        decoded = self.tokenizer.decoder(
            z,
            agent_mask=decode_batch["agent_mask"],
            light_mask=decode_batch["light_mask"],
            **decoder_kwargs,
        )
        pred_xy = decoder_agent_xy(
            decoded,
            agent_xy_loss=str(self.model_args.agent_xy_loss),
            agent_xy_parameterization=str(self.model_args.agent_xy_parameterization),
            anchor_xy=None,
        )[0, -1]
        continuous = decoded.agent_continuous[0, -1]
        xy = _to_cpu_numpy(pred_xy)
        yaw = _to_cpu_numpy(torch.atan2(continuous[:, 5], continuous[:, 6]))
        velocity = _to_cpu_numpy(continuous[:, 3:5])
        valid = (
            torch.sigmoid(decoded.agent_valid_logits[0, -1]) >= float(self.args.valid_threshold)
        ).detach().cpu().numpy()
        valid &= state.agent_mask

        # Slot 0 is controlled exactly.  The latent dynamics still sees this
        # same state through the action, so the other slots can react to it.
        xy[0] = np.asarray([state.focus.x, state.focus.y], dtype=np.float32)
        yaw[0] = state.focus.yaw
        velocity[0] = np.asarray(state.focus.velocity, dtype=np.float32)
        valid[0] = True
        state.world_history.append(xy)
        state.yaw_history.append(yaw)
        state.velocity_history.append(velocity)
        state.valid_history.append(valid)

    def _render_2d(self, state: SessionState) -> bytes:
        size = int(self.args.canvas_size)
        image = Image.new("RGB", (size, size), (17, 22, 31))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        center = np.asarray([state.focus.x, state.focus.y], dtype=np.float32)
        radius = float(self.args.view_radius)

        # A light metric grid makes speed and model drift easier to judge.
        grid_step = 10.0
        lo_x = math.floor((center[0] - radius) / grid_step) * grid_step
        hi_x = center[0] + radius
        lo_y = math.floor((center[1] - radius) / grid_step) * grid_step
        hi_y = center[1] + radius
        value = lo_x
        while value <= hi_x:
            pts = _world_to_pixel(
                np.asarray([[value, center[1] - radius], [value, center[1] + radius]]),
                center_xy=center,
                radius_m=radius,
                canvas_size=size,
            )
            draw.line([tuple(pts[0]), tuple(pts[1])], fill=(29, 38, 51), width=1)
            value += grid_step
        value = lo_y
        while value <= hi_y:
            pts = _world_to_pixel(
                np.asarray([[center[0] - radius, value], [center[0] + radius, value]]),
                center_xy=center,
                radius_m=radius,
                canvas_size=size,
            )
            draw.line([tuple(pts[0]), tuple(pts[1])], fill=(29, 38, 51), width=1)
            value += grid_step

        for polyline, mask in zip(state.map_polylines, state.map_point_mask):
            points = polyline[mask, :2]
            if len(points) < 2:
                continue
            if (
                points[:, 0].max() < center[0] - radius
                or points[:, 0].min() > center[0] + radius
                or points[:, 1].max() < center[1] - radius
                or points[:, 1].min() > center[1] + radius
            ):
                continue
            pixels = _world_to_pixel(points, center_xy=center, radius_m=radius, canvas_size=size)
            draw.line([tuple(map(float, point)) for point in pixels], fill=(93, 104, 119), width=2)

        trail_length = max(1, int(self.args.trail_length))
        history_start = max(0, len(state.world_history) - trail_length)
        for slot in range(len(state.agent_mask)):
            if not state.agent_mask[slot]:
                continue
            trail = []
            for xy, valid in zip(state.world_history[history_start:], state.valid_history[history_start:]):
                if valid[slot] and np.isfinite(xy[slot]).all():
                    trail.append(xy[slot])
            if len(trail) >= 2:
                color = _agent_color(int(state.agent_types[slot]), focus=slot == 0)
                pixels = _world_to_pixel(
                    np.asarray(trail), center_xy=center, radius_m=radius, canvas_size=size
                )
                draw.line(
                    [tuple(map(float, point)) for point in pixels],
                    fill=tuple(int(c * 0.65) for c in color),
                    width=4 if slot == 0 else 2,
                )

        current_xy = state.world_history[-1]
        current_yaw = state.yaw_history[-1]
        current_valid = state.valid_history[-1]
        # Draw focus last so it is always visible.
        order = [i for i in range(len(state.agent_mask)) if i != 0] + [0]
        for slot in order:
            if not (state.agent_mask[slot] and current_valid[slot] and np.isfinite(current_xy[slot]).all()):
                continue
            pixel = _world_to_pixel(
                current_xy[slot], center_xy=center, radius_m=radius, canvas_size=size
            )
            if not (-20 <= pixel[0] <= size + 20 and -20 <= pixel[1] <= size + 20):
                continue
            color = _agent_color(int(state.agent_types[slot]), focus=slot == 0)
            _draw_agent(draw, pixel, float(current_yaw[slot]), color, focus=slot == 0)
            if slot == 0:
                draw.text((float(pixel[0] + 14), float(pixel[1] - 18)), "YOU", fill=color, font=font)

        draw.rectangle((0, 0, size, 28), fill=(8, 11, 17))
        title = scene_identity(state)["scene_label"]
        draw.text((10, 9), title, fill=(255, 205, 90), font=font)
        progress = (
            f"CONTEXT {state.replay_index + 1}/{len(state.context_focus)}"
            if state.step == 0
            else f"ROLLOUT {state.step}/{self.args.max_steps}"
        )
        progress_box = draw.textbbox((0, 0), progress, font=font)
        draw.text(
            (size - (progress_box[2] - progress_box[0]) - 10, 9),
            progress,
            fill=(78, 232, 140),
            font=font,
        )
        if state.paused:
            text = "PAUSED"
            box = draw.textbbox((0, 0), text, font=font)
            width = box[2] - box[0]
            draw.rounded_rectangle(
                (size / 2 - width / 2 - 12, 42, size / 2 + width / 2 + 12, 68),
                radius=6,
                fill=(8, 11, 17),
                outline=(190, 197, 208),
            )
            draw.text((size / 2 - width / 2, 51), text, fill=(245, 245, 245), font=font)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(self.args.jpeg_quality), optimize=True)
        return buffer.getvalue()

    def _puffer_frame_state(self, state: SessionState) -> PufferFrameState:
        selected = state.agent_mask & (state.agent_ids >= 0)
        local_xy = state.world_history[-1][selected]
        local_yaw = state.yaw_history[-1][selected]
        local_velocity = state.velocity_history[-1][selected]
        world_xy, world_yaw, world_velocity = local_to_world_pose(
            local_xy,
            local_yaw,
            local_velocity,
            origin_xy=state.ego_origin_xy,
            origin_heading=state.ego_heading,
        )
        display_frame_id = self._display_frame_id(state)
        return PufferFrameState(
            step=display_frame_id,
            agent_ids=state.agent_ids[selected],
            agent_types=state.agent_types[selected],
            xy=world_xy,
            yaw=world_yaw,
            velocity_xy=world_velocity,
            valid=state.valid_history[-1][selected],
            source_time_index=state.context_start_frame + display_frame_id,
        )

    def _report_puffer_error(self, message: str) -> None:
        if message in self._reported_puffer_errors:
            return
        self._reported_puffer_errors.add(message)
        print(f"[puffer warning] {message}; using the 2D renderer", flush=True)

    def _render(self, state: SessionState) -> bytes:
        if self.args.renderer != "puffer":
            state.renderer_name = "2d"
            return self._render_2d(state)
        if state.puffer_disabled:
            state.renderer_name = "2d-fallback"
            return self._render_2d(state)
        if self.puffer_renderer is None or state.puffer_scene is None:
            message = state.renderer_error or "Puffer renderer or scene mapping is unavailable"
            state.renderer_error = message
            state.renderer_name = "2d-fallback"
            state.puffer_disabled = True
            if self.args.puffer_strict:
                raise PufferBridgeError(message)
            self._report_puffer_error(message)
            return self._render_2d(state)

        try:
            jpeg = self.puffer_renderer.render(
                state.puffer_scene, self._puffer_frame_state(state)
            )
        except (OSError, ValueError, PufferBridgeError) as error:
            message = str(error)
            state.renderer_error = message
            state.renderer_name = "2d-fallback"
            state.puffer_disabled = True
            if self.args.puffer_strict:
                raise
            self._report_puffer_error(message)
            return self._render_2d(state)

        state.renderer_name = "puffer"
        state.renderer_error = None
        return jpeg

    def _status(self, state: SessionState) -> dict[str, Any]:
        replay_steps = len(state.context_focus)
        phase = "context_replay" if state.step == 0 else "rollout"
        model_context_start, model_context_end = prediction_context_bounds(
            state.context_start_frame,
            replay_steps,
            self.model_rollout_window,
        )
        progress_text = (
            f"context {state.replay_index + 1}/{replay_steps}"
            if phase == "context_replay"
            else f"rollout {state.step}/{self.args.max_steps}"
        )
        identity = scene_identity(state)
        return {
            "type": "status",
            "paused": state.paused,
            "phase": phase,
            "replay_step": state.replay_index + 1,
            "replay_steps": replay_steps,
            "step": state.step,
            "max_steps": int(self.args.max_steps),
            "timeline_frame": (
                state.context_start_frame + self._display_frame_id(state) + 1
            ),
            "model_context_start": model_context_start + 1,
            "model_context_end": model_context_end,
            "speed": state.focus.speed,
            "heading_deg": math.degrees(state.focus.yaw),
            "x": state.focus.x,
            "y": state.focus.y,
            **identity,
            "inference_ms": state.last_inference_ms,
            "renderer": state.renderer_name,
            "checkpoint_profile": self.checkpoint_profile,
            "renderer_error": state.renderer_error,
            "text": (
                f"{identity['scene_label']} · ckpt {self.checkpoint_profile} · "
                f"{progress_text} · model context "
                f"{model_context_start + 1}–{model_context_end} · "
                f"speed {state.focus.speed:.1f} m/s · "
                f"heading {math.degrees(state.focus.yaw):+.0f}° · "
                f"model {state.last_inference_ms:.0f} ms · "
                f"renderer {state.renderer_name}"
            ),
        }

    def _tick_sync(
        self,
        state: SessionState,
        analog_control: AnalogControl | None = None,
    ) -> tuple[bytes | None, dict[str, Any]]:
        first_frame = state.cached_jpeg is None
        did_reset = False
        if state.reset_requested or state.new_scene_requested:
            self._reset_session(state, new_scene=state.new_scene_requested)
            first_frame = True
            did_reset = True
        has_remaining = self._replay_pending(state) or state.step < int(self.args.max_steps)
        should_advance = (
            not first_frame
            and not did_reset
            and (not state.paused or state.step_once)
            and has_remaining
        )
        if should_advance:
            if self._replay_pending(state):
                self._advance_replay(state)
            else:
                start = time.perf_counter()
                self._advance(state, analog_control=analog_control)
                state.last_inference_ms = (time.perf_counter() - start) * 1000.0
            state.step_once = False
            if not self._replay_pending(state) and state.step >= int(self.args.max_steps):
                state.paused = True
        frame_id = self._display_frame_id(state)
        jpeg = None
        if state.cached_jpeg is None or state.cached_frame_id != frame_id:
            state.cached_jpeg = self._render(state)
            state.cached_frame_id = frame_id
            jpeg = state.cached_jpeg
        return jpeg, self._status(state)

    async def index(self, _request: web.Request) -> web.Response:
        return web.Response(text=self.html, content_type="text/html")

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "device": str(self.device),
                "checkpoint_step": int(self.checkpoint.get("step", -1)),
                "checkpoint_profile": self.checkpoint_profile,
                "checkpoint_path": str(self.checkpoint_path),
                "dataset_size": len(self.dataset),
                "renderer": self.args.renderer,
                "context_replay_frames": self.context_frames,
                "model_context_frames": self.model_context_frames,
                "rollout_steps": int(self.args.max_steps),
                "puffer_worker_pid": (
                    None if self.puffer_renderer is None else self.puffer_renderer.pid
                ),
            }
        )

    def close(self) -> None:
        if self.puffer_renderer is not None:
            self.puffer_renderer.close()

    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024)
        await websocket.prepare(request)
        async with self.infer_lock:
            state = await asyncio.to_thread(self.new_session)

        async def receive_loop() -> None:
            async for message in websocket:
                if message.type != WSMsgType.TEXT:
                    if message.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                        return
                    continue
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    continue
                kind = payload.get("type")
                key = str(payload.get("key", ""))
                if kind == "keydown":
                    if key in {"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"}:
                        state.keys_down.add(key)
                    elif key == "Space":
                        state.paused = not state.paused
                    elif key in {"r", "R"}:
                        state.reset_requested = True
                    elif key in {"n", "N"}:
                        state.new_scene_requested = True
                    elif key in {".", ">"}:
                        state.step_once = True
                    elif key in {"q", "Q", "Escape"}:
                        await websocket.close()
                        return
                elif kind == "keyup":
                    state.keys_down.discard(key)
                elif kind == "toggle_pause":
                    state.paused = not state.paused
                elif kind == "step":
                    state.step_once = True
                elif kind == "reset":
                    state.reset_requested = True
                elif kind == "new_scene":
                    state.new_scene_requested = True
                elif kind == "disconnect":
                    await websocket.close()
                    return

        async def send_loop() -> None:
            period = 1.0 / max(float(self.args.fps), 0.1)
            next_tick = time.monotonic()
            while not websocket.closed:
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                next_tick = max(next_tick + period, time.monotonic())
                async with self.infer_lock:
                    jpeg, status = await asyncio.to_thread(self._tick_sync, state)
                if websocket.closed:
                    return
                await websocket.send_str(json.dumps(status))
                if jpeg is not None:
                    await websocket.send_bytes(jpeg)

        receiver = asyncio.create_task(receive_loop())
        sender = asyncio.create_task(send_loop())
        _done, pending = await asyncio.wait(
            {receiver, sender}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        return websocket


class InteractiveArgumentParser(argparse.ArgumentParser):
    """Argument parser that applies renderer-specific rollout defaults."""

    def parse_known_args(
        self,
        args: Any = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        parsed, extras = super().parse_known_args(args=args, namespace=namespace)
        if hasattr(parsed, "renderer") and hasattr(parsed, "max_steps"):
            parsed.max_steps = resolve_rollout_steps(
                parsed.renderer,
                parsed.max_steps,
            )
        return parsed, extras


def build_parser() -> argparse.ArgumentParser:
    parser = InteractiveArgumentParser(
        description="Play an action-conditioned Waymo world model in a browser."
    )
    parser.add_argument(
        "--checkpoint-profile",
        "--ckpt-profile",
        "--ckpt",
        choices=tuple(WORLD_MODEL_CHECKPOINT_PROFILES),
        default=DEFAULT_CHECKPOINT_PROFILE,
        help=(
            "Named interactive world-model checkpoint. h30 uses "
            "best_multisample_finetuned.pt; h90 uses step_00027000.pt."
        ),
    )
    parser.add_argument(
        "--world-model-ckpt",
        default=None,
        help="Explicit checkpoint path; when provided, overrides --checkpoint-profile.",
    )
    parser.add_argument("--tokenizer-ckpt", default=None, help="Defaults to the path saved in the WM checkpoint.")
    parser.add_argument("--data-dir", default=None, help="Defaults to checkpoint val_data_dir.")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based first recorded replay frame. Default 0 replays Waymo frames 1 through 11.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=DEFAULT_CONTEXT_FRAMES,
        help=(
            "Number of recorded frames to replay before interaction. With the "
            "checkpoint's 11-token rollout window, 11 displayed frames retain "
            "frames 2 through 11 for the first prediction."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--fps", type=float, default=10.0, help="Real-time model ticks per second.")
    parser.add_argument("--sim-dt", type=float, default=0.1, help="Waymo simulation seconds per model tick.")
    parser.add_argument("--acceleration", type=float, default=5.0, help="Up/down acceleration magnitude in m/s^2.")
    parser.add_argument(
        "--braking",
        type=float,
        default=8.0,
        help="Maximum analog brake deceleration in m/s^2.",
    )
    parser.add_argument("--yaw-rate", type=float, default=45.0, help="Left/right yaw rate in degrees/s.")
    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=30.0)
    parser.add_argument(
        "--unroll-steps",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=None,
        help=(
            "Generated interactive steps after context replay. Defaults to 80 "
            "for --renderer 2d and 150 for --renderer puffer. --max-steps is "
            "kept as a backward-compatible alias."
        ),
    )
    parser.add_argument("--eval-d", type=float, default=1.0, help="Shortcut step size; 1.0 gives one model pass/frame.")
    parser.add_argument("--decode-window", type=int, default=32)
    parser.add_argument("--valid-threshold", type=float, default=0.5)
    parser.add_argument("--autoplay", action="store_true")

    parser.add_argument("--canvas-size", type=int, default=720)
    parser.add_argument("--view-radius", type=float, default=55.0)
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--renderer",
        choices=("2d", "puffer"),
        default="2d",
        help="Frame renderer. The default keeps the original in-process 2D view.",
    )
    parser.add_argument(
        "--puffer-manifest",
        "--puffer-scenario-manifest",
        dest="puffer_manifest",
        default=None,
        help=(
            "Offline conversion manifest containing npz_path and puffer_map_dir "
            "or puffer_bin_path. Required for actual Puffer rendering."
        ),
    )
    parser.add_argument(
        "--puffer-scenario-cache",
        default=str(DEFAULT_PUFFER_SCENARIO_CACHE),
        help="Raw <scenario_id>.pb cache used only for provenance/error diagnostics.",
    )
    parser.add_argument(
        "--puffer-worker-command",
        default=DEFAULT_PUFFER_WORKER_COMMAND,
        help="Command for the length-prefixed Puffer renderer subprocess.",
    )
    parser.add_argument("--puffer-width", type=int, default=1280)
    parser.add_argument("--puffer-height", type=int, default=720)
    parser.add_argument("--puffer-view-mode", default="AGENT_PERSP")
    parser.add_argument("--puffer-timeout", type=float, default=15.0)
    parser.add_argument(
        "--puffer-use-inherited-display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the current X11 DISPLAY for the hidden Puffer render context. "
            "Intended for a verified local desktop; headless mode owns Xvfb instead."
        ),
    )
    parser.add_argument(
        "--puffer-strict",
        action="store_true",
        help="Abort on a missing Puffer asset or worker error instead of using 2D fallback.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--html", default=str(DEFAULT_HTML))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = WaymoInteractiveServer(args)
    app = web.Application()
    app.router.add_get("/", server.index)
    app.router.add_get("/health", server.health)
    app.router.add_get("/ws", server.ws_handler)

    async def close_renderer(_app: web.Application) -> None:
        await asyncio.to_thread(server.close)

    app.on_cleanup.append(close_renderer)
    print(f"[web] open http://{args.host}:{args.port}", flush=True)
    if args.host in {"127.0.0.1", "localhost"}:
        print(
            f"[web] remote machine: ssh -L {args.port}:127.0.0.1:{args.port} USER@HOST",
            flush=True,
        )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
