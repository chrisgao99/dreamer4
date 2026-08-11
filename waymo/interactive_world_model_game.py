#!/usr/bin/env python3
"""Interactive browser game for an action-conditioned Waymo world model.

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


DEFAULT_WORLD_MODEL_CKPT = (
    REPO_ROOT
    / "waymo/checkpoints/waymo_wm_original_stmlayer_3_stage"
    / "waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k"
    / "step_00040000.pt"
)
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
    yaw_rate_deg_s: float = 45.0
    min_speed_mps: float = 0.0
    max_speed_mps: float = 30.0


@dataclass
class FocusState:
    x: float
    y: float
    speed: float
    yaw: float

    @property
    def velocity(self) -> tuple[float, float]:
        return self.speed * math.cos(self.yaw), self.speed * math.sin(self.yaw)


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
    focus: FocusState
    z_history: list[torch.Tensor]
    action_history: list[torch.Tensor]
    world_history: list[np.ndarray]
    valid_history: list[np.ndarray]
    yaw_history: list[np.ndarray]
    velocity_history: list[np.ndarray]
    keys_down: set[str] = field(default_factory=set)
    paused: bool = True
    step_once: bool = False
    reset_requested: bool = False
    new_scene_requested: bool = False
    step: int = 0
    cached_jpeg: bytes | None = None
    cached_frame_id: int = -1
    last_inference_ms: float = 0.0
    renderer_name: str = "2d"
    renderer_error: str | None = None
    puffer_disabled: bool = False


class WaymoInteractiveServer:
    def __init__(self, args: argparse.Namespace):
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
            yaw_rate_deg_s=args.yaw_rate,
            min_speed_mps=args.min_speed,
            max_speed_mps=args.max_speed,
        )

        checkpoint_path = Path(args.world_model_ckpt).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"World-model checkpoint not found: {checkpoint_path}")
        print(f"[load] world model: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        self.checkpoint = checkpoint
        self.model_args = _model_args_from_checkpoint(checkpoint)
        self._validate_checkpoint_contract()

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
            )

        self.html = Path(args.html).read_text(encoding="utf-8")
        print(
            f"[ready] checkpoint step={int(self.checkpoint.get('step', -1))} "
            f"dataset scenes={len(self.dataset)} device={self.device} "
            f"dtype={next(self.dynamics.parameters()).dtype} renderer={args.renderer}",
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

    def _pick_new_scene_index(self, current: int) -> int:
        if len(self.dataset) <= 1:
            return current
        candidate = current
        while candidate == current:
            candidate = self.random.randrange(len(self.dataset))
        return candidate

    @torch.inference_mode()
    def _load_scene(self, scene_index: int) -> SessionState:
        item = self.dataset[int(scene_index)]
        batch = wm.move_batch(wm._collate([item]), self.device)
        total_steps = int(batch["lights"].shape[1])
        start_frame = min(max(0, int(self.args.start_frame)), total_steps - 1)
        context = wm.slice_future_batch(batch, start_frame, start_frame + 1)
        z0, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            self.tokenizer,
            context,
            self.model_args,
            return_map=bool(self.model_args.dynamics_attend_map),
        )
        z0_packed = wm.pack_bottleneck_to_spatial(
            z0,
            n_spatial=int(self.model_args.n_spatial),
            k=int(self.model_args.packing_factor),
        )[0, 0].detach()

        agents = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])[0]
        initial_agents = _to_cpu_numpy(agents[start_frame])
        focus_row = initial_agents[0]
        if focus_row[5] <= 0.5:
            raise ValueError(f"Focus slot is invalid at start frame {start_frame}")
        focus = FocusState(
            x=float(focus_row[0]),
            y=float(focus_row[1]),
            speed=max(0.0, float(focus_row[2])),
            yaw=float(focus_row[6]),
        )
        initial_velocity = (float(focus_row[3]), float(focus_row[4]))
        initial_agents[0, 0:5] = np.asarray(
            [focus.x, focus.y, focus.speed, *focus.velocity], dtype=np.float32
        )
        initial_agents[0, 6] = focus.yaw

        agent_mask = batch["agent_mask"][0].detach().cpu().numpy().astype(bool)
        agent_types = np.rint(initial_agents[:, 7]).astype(np.int64)
        valid = (initial_agents[:, 5] > 0.5) & agent_mask
        yaw = initial_agents[:, 6].copy()
        velocity = initial_agents[:, 3:5].copy()
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
        return SessionState(
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
            focus=focus,
            z_history=[z0_packed],
            action_history=[initial_focus_action(focus, initial_velocity).to(self.device)],
            world_history=[initial_agents[:, 0:2].copy()],
            valid_history=[valid],
            yaw_history=[yaw],
            velocity_history=[velocity],
            paused=not self.args.autoplay,
            renderer_error=renderer_error,
        )

    def new_session(self) -> SessionState:
        return self._load_scene(self.initial_scene_index)

    def _reset_session(self, state: SessionState, *, new_scene: bool) -> None:
        index = self._pick_new_scene_index(state.scene_index) if new_scene else state.scene_index
        replacement = self._load_scene(index)
        state.__dict__.clear()
        state.__dict__.update(replacement.__dict__)

    def _decode_batch(self, state: SessionState, time_steps: int) -> dict[str, Any]:
        # Only masks and static map are consumed by the latent-only decoder.
        # Repeating the start-frame mask also lets a session continue beyond the
        # recorded 91 frames, although --max-steps defaults to the trained H90.
        base = state.base_batch
        light_mask0 = base["light_mask"][:, self.args.start_frame : self.args.start_frame + 1]
        lights0 = base["lights"][:, self.args.start_frame : self.args.start_frame + 1]
        agents_btkf = wm.agents_to_btkf(base["agents"], base["agent_mask"])
        agents0 = agents_btkf[:, self.args.start_frame : self.args.start_frame + 1]
        return {
            **base,
            "agents": agents0.expand(-1, time_steps, -1, -1),
            "lights": lights0.expand(-1, time_steps, -1, -1),
            "light_mask": light_mask0.expand(-1, time_steps, -1),
        }

    @torch.inference_mode()
    def _advance(self, state: SessionState) -> None:
        next_focus, action = integrate_focus_control(state.focus, set(state.keys_down), self.control)
        state.action_history.append(action.to(self.device))

        past = torch.stack(state.z_history, dim=0).unsqueeze(0)
        actions = torch.stack(state.action_history, dim=0).unsqueeze(0)
        act_mask = torch.zeros_like(actions)
        act_mask[..., :7] = 1.0
        z_next = wm.sample_one_timestep_packed(
            self.dynamics,
            past_packed=past,
            actions_seq=actions,
            act_mask_seq=act_mask,
            map_tokens=state.map_tokens,
            map_mask=state.map_mask,
            k_max=int(self.model_args.k_max),
            sched=self.schedule,
            max_rollout_window=int(self.model_args.max_rollout_window),
        )
        state.z_history.append(z_next[0].detach())
        state.focus = next_focus
        state.step += 1

        decode_window = max(1, int(self.args.decode_window))
        decode_start = max(0, len(state.z_history) - decode_window)
        z_packed = torch.stack(state.z_history[decode_start:], dim=0).unsqueeze(0)
        z = wm.unpack_spatial_to_bottleneck(z_packed, k=int(self.model_args.packing_factor))
        decode_batch = self._decode_batch(state, int(z.shape[1]))
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
        title = f"scene {state.scene_index}  |  {state.scenario_id[:48]}"
        draw.text((10, 9), title, fill=(222, 228, 238), font=font)
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
        return PufferFrameState(
            step=state.step,
            agent_ids=state.agent_ids[selected],
            agent_types=state.agent_types[selected],
            xy=world_xy,
            yaw=world_yaw,
            velocity_xy=world_velocity,
            valid=state.valid_history[-1][selected],
            source_time_index=int(self.args.start_frame) + state.step,
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
        return {
            "type": "status",
            "paused": state.paused,
            "step": state.step,
            "max_steps": int(self.args.max_steps),
            "speed": state.focus.speed,
            "heading_deg": math.degrees(state.focus.yaw),
            "x": state.focus.x,
            "y": state.focus.y,
            "scene_index": state.scene_index,
            "scenario_id": state.scenario_id,
            "inference_ms": state.last_inference_ms,
            "renderer": state.renderer_name,
            "renderer_error": state.renderer_error,
            "text": (
                f"step {state.step}/{self.args.max_steps} · "
                f"speed {state.focus.speed:.1f} m/s · "
                f"heading {math.degrees(state.focus.yaw):+.0f}° · "
                f"model {state.last_inference_ms:.0f} ms · "
                f"renderer {state.renderer_name}"
            ),
        }

    def _tick_sync(self, state: SessionState) -> tuple[bytes | None, dict[str, Any]]:
        if state.reset_requested or state.new_scene_requested:
            self._reset_session(state, new_scene=state.new_scene_requested)
        should_advance = (not state.paused or state.step_once) and state.step < int(self.args.max_steps)
        if should_advance:
            start = time.perf_counter()
            self._advance(state)
            state.last_inference_ms = (time.perf_counter() - start) * 1000.0
            state.step_once = False
            if state.step >= int(self.args.max_steps):
                state.paused = True
        frame_id = state.step
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
                "dataset_size": len(self.dataset),
                "renderer": self.args.renderer,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play an action-conditioned Waymo world model in a browser."
    )
    parser.add_argument("--world-model-ckpt", default=str(DEFAULT_WORLD_MODEL_CKPT))
    parser.add_argument("--tokenizer-ckpt", default=None, help="Defaults to the path saved in the WM checkpoint.")
    parser.add_argument("--data-dir", default=None, help="Defaults to checkpoint val_data_dir.")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Seed frame in the recorded scene. Frame 0 matches this checkpoint's ctx1/H90 training protocol.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--fps", type=float, default=10.0, help="Real-time model ticks per second.")
    parser.add_argument("--sim-dt", type=float, default=0.1, help="Waymo simulation seconds per model tick.")
    parser.add_argument("--acceleration", type=float, default=5.0, help="Up/down acceleration magnitude in m/s^2.")
    parser.add_argument("--yaw-rate", type=float, default=45.0, help="Left/right yaw rate in degrees/s.")
    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=90)
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
