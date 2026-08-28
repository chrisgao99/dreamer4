#!/usr/bin/env python3
"""Local PufferDrive game controlled by a wheel/joystick or the keyboard.

Pygame owns the visible desktop window and input devices. Dreamer
inference and the existing PufferDrive RGB worker run on one background thread,
so SDL events continue to be pumped while CUDA inference is in progress.

Default Fanatec mapping (matching ``yf_metadrive/code/pygame_test.py``):

* axis 0: wheel, left negative and right positive
* axis 2: throttle, released +1 and fully pressed -1
* axis 5: brake, released +1 and fully pressed -1

The third pedal (axis 1) is intentionally never read.

With no compatible wheel, the default ``--input-device auto`` mode falls back
to keyboard control: Up/W throttle, Down/S brake, and Left/A or Right/D steer.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WAYMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WAYMO_ROOT.parent
for _path in (REPO_ROOT, WAYMO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from waymo.interactive_world_model_game import (  # noqa: E402
    AnalogControl,
    WaymoInteractiveServer,
    build_parser as build_game_parser,
)


DEFAULT_LOCAL_MANIFEST = WAYMO_ROOT / "cache/pufferdrive_static_smoke/manifest.csv"


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def rescaled_axis_deadzone(value: float, deadzone: float) -> float:
    """Apply a symmetric deadzone while preserving a full ``[-1, 1]`` range."""

    deadzone = float(deadzone)
    if not 0.0 <= deadzone < 1.0:
        raise ValueError(f"deadzone must be in [0, 1), got {deadzone}")
    value = _clamp(value, -1.0, 1.0)
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


def released_high_pedal(value: float, deadzone: float) -> float:
    """Convert a ``+1 released / -1 pressed`` pedal axis to ``[0, 1]``."""

    deadzone = float(deadzone)
    if not 0.0 <= deadzone < 1.0:
        raise ValueError(f"pedal deadzone must be in [0, 1), got {deadzone}")
    pressure = (1.0 - _clamp(value, -1.0, 1.0)) * 0.5
    if pressure <= deadzone:
        return 0.0
    return _clamp((pressure - deadzone) / (1.0 - deadzone), 0.0, 1.0)


def fanatec_axes_to_control(
    wheel_axis: float,
    throttle_axis: float,
    brake_axis: float,
    *,
    steering_deadzone: float = 0.02,
    pedal_deadzone: float = 0.02,
    steering_gain: float = 1.0,
    invert_steering: bool = True,
) -> AnalogControl:
    """Map raw Fanatec axes to the game's device-independent controls.

    The game uses positive steering for a left/CCW turn, whereas the tested
    Fanatec axis is positive to the right.  ``invert_steering`` therefore
    defaults to true.
    """

    steering = rescaled_axis_deadzone(wheel_axis, steering_deadzone)
    steering *= float(steering_gain)
    if invert_steering:
        steering = -steering
    return AnalogControl(
        steering=_clamp(steering, -1.0, 1.0),
        throttle=released_high_pedal(throttle_axis, pedal_deadzone),
        brake=released_high_pedal(brake_axis, pedal_deadzone),
    )


def keyboard_to_control(
    *,
    left: bool = False,
    right: bool = False,
    throttle: bool = False,
    brake: bool = False,
) -> AnalogControl:
    """Map held arrow/WASD keys to the same normalized analog interface."""

    return AnalogControl(
        steering=float(bool(left)) - float(bool(right)),
        throttle=float(bool(throttle)),
        brake=float(bool(brake)),
    )


@dataclass(frozen=True)
class WheelSample:
    control: AnalogControl
    raw_wheel: float
    raw_throttle: float
    raw_brake: float
    sampled_at: float
    source: str = "wheel"

    @classmethod
    def neutral(
        cls,
        sampled_at: float | None = None,
        *,
        source: str = "neutral",
    ) -> "WheelSample":
        return cls(
            control=AnalogControl(),
            raw_wheel=0.0,
            raw_throttle=1.0,
            raw_brake=1.0,
            sampled_at=time.monotonic() if sampled_at is None else float(sampled_at),
            source=source,
        )


def read_keyboard_sample(pygame_module: Any) -> WheelSample:
    """Read held arrow/WASD keys from Pygame's current keyboard state."""

    if hasattr(pygame_module.key, "get_focused") and not pygame_module.key.get_focused():
        return WheelSample.neutral(source="keyboard")
    pressed = pygame_module.key.get_pressed()
    control = keyboard_to_control(
        left=bool(pressed[pygame_module.K_LEFT] or pressed[pygame_module.K_a]),
        right=bool(pressed[pygame_module.K_RIGHT] or pressed[pygame_module.K_d]),
        throttle=bool(pressed[pygame_module.K_UP] or pressed[pygame_module.K_w]),
        brake=bool(pressed[pygame_module.K_DOWN] or pressed[pygame_module.K_s]),
    )
    # Preserve the wheel/pedal raw conventions in the diagnostic overlay.
    return WheelSample(
        control=control,
        raw_wheel=-control.steering,
        raw_throttle=1.0 - 2.0 * control.throttle,
        raw_brake=1.0 - 2.0 * control.brake,
        sampled_at=time.monotonic(),
        source="keyboard",
    )


def select_input_sample(
    input_device: str,
    *,
    keyboard_sample: WheelSample,
    wheel_sample: WheelSample | None,
) -> WheelSample:
    """Choose keyboard/wheel input without blending two control sources."""

    keyboard_active = keyboard_sample.control != AnalogControl()
    if input_device == "keyboard":
        return keyboard_sample
    if input_device == "auto" and (keyboard_active or wheel_sample is None):
        return keyboard_sample
    if wheel_sample is None:
        raise RuntimeError("Wheel/joystick input was requested but is unavailable")
    return wheel_sample


class LatestWheelSample:
    """Thread-safe latest-value channel; old inputs are never queued."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample = WheelSample.neutral()

    def set(self, sample: WheelSample) -> None:
        with self._lock:
            self._sample = sample

    def get_control(
        self,
        *,
        max_age_s: float,
        now: float | None = None,
    ) -> AnalogControl:
        with self._lock:
            sample = self._sample
        current_time = time.monotonic() if now is None else float(now)
        if current_time - sample.sampled_at > max_age_s:
            return AnalogControl()
        return sample.control


class FanatecWheel:
    """Small checked wrapper around one Pygame joystick."""

    def __init__(self, pygame_module: Any, args: argparse.Namespace):
        self.pygame = pygame_module
        self.steering_axis = int(args.steering_axis)
        self.throttle_axis = int(args.throttle_axis)
        self.brake_axis = int(args.brake_axis)
        required_axes = max(self.steering_axis, self.throttle_axis, self.brake_axis) + 1
        if min(self.steering_axis, self.throttle_axis, self.brake_axis) < 0:
            raise ValueError("joystick axis indices must be non-negative")

        count = int(pygame_module.joystick.get_count())
        if count <= 0:
            raise RuntimeError("No joystick or wheel was detected by Pygame/SDL")

        inventory: list[tuple[int, str, int]] = []
        for index in range(count):
            device = pygame_module.joystick.Joystick(index)
            device.init()
            inventory.append((index, str(device.get_name()), int(device.get_numaxes())))
            device.quit()
        print("[fanatec] SDL devices:", flush=True)
        for index, name, axes in inventory:
            print(f"  index={index} axes={axes} name={name!r}", flush=True)

        if args.joystick_index is not None:
            selected_index = int(args.joystick_index)
            if not 0 <= selected_index < count:
                raise ValueError(
                    f"--joystick-index must be in [0, {count - 1}], got {selected_index}"
                )
        else:
            compatible = [item for item in inventory if item[2] >= required_axes]
            if not compatible:
                raise RuntimeError(
                    f"No SDL device exposes the required {required_axes} axes; "
                    "use --joystick-index and axis override flags after checking pygame_test.py"
                )

            def score(item: tuple[int, str, int]) -> tuple[int, int, int]:
                index, name, axes = item
                lowered = name.lower()
                value = 0
                if "fanatec" in lowered:
                    value += 100
                if "wheel" in lowered:
                    value += 30
                if "pedal" in lowered:
                    value -= 200
                return value, axes, -index

            selected_index = max(compatible, key=score)[0]

        self.joystick = pygame_module.joystick.Joystick(selected_index)
        self.joystick.init()
        if int(self.joystick.get_numaxes()) < required_axes:
            raise RuntimeError(
                f"Selected joystick index {selected_index} exposes only "
                f"{self.joystick.get_numaxes()} axes; need {required_axes}"
            )
        self.index = selected_index
        self.name = str(self.joystick.get_name())
        self.instance_id = (
            int(self.joystick.get_instance_id())
            if hasattr(self.joystick, "get_instance_id")
            else selected_index
        )
        print(
            f"[fanatec] using index={self.index} instance={self.instance_id} "
            f"axes={self.joystick.get_numaxes()} name={self.name!r}",
            flush=True,
        )

        self.steering_deadzone = float(args.steering_deadzone)
        self.pedal_deadzone = float(args.pedal_deadzone)
        self.steering_gain = float(args.steering_gain)
        self.invert_steering = bool(args.invert_steering)

    def read(self) -> WheelSample:
        if not self.joystick.get_init():
            raise RuntimeError("Fanatec wheel is no longer initialized")
        raw_wheel = float(self.joystick.get_axis(self.steering_axis))
        raw_throttle = float(self.joystick.get_axis(self.throttle_axis))
        raw_brake = float(self.joystick.get_axis(self.brake_axis))
        return WheelSample(
            control=fanatec_axes_to_control(
                raw_wheel,
                raw_throttle,
                raw_brake,
                steering_deadzone=self.steering_deadzone,
                pedal_deadzone=self.pedal_deadzone,
                steering_gain=self.steering_gain,
                invert_steering=self.invert_steering,
            ),
            raw_wheel=raw_wheel,
            raw_throttle=raw_throttle,
            raw_brake=raw_brake,
            sampled_at=time.monotonic(),
            source="wheel",
        )

    def close(self) -> None:
        try:
            if self.joystick.get_init():
                self.joystick.quit()
        except Exception:
            # SDL may invalidate the object before emitting JOYDEVICEREMOVED.
            pass


@dataclass(frozen=True)
class GameUpdate:
    kind: str
    message: str = ""
    jpeg: bytes | None = None
    status: dict[str, Any] | None = None


class LocalGameWorker(threading.Thread):
    """Own all Dreamer, CUDA, SessionState, and Puffer worker operations."""

    def __init__(
        self,
        args: argparse.Namespace,
        latest_input: LatestWheelSample,
        commands: "queue.Queue[str]",
        updates: "queue.Queue[GameUpdate]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="dreamer-puffer-worker", daemon=True)
        self.args = args
        self.latest_input = latest_input
        self.commands = commands
        self.updates = updates
        self.stop_event = stop_event

    def _publish(self, update: GameUpdate) -> None:
        self.updates.put(update)

    def _apply_commands(self, state: Any) -> None:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            if command == "toggle_pause":
                state.paused = not state.paused
            elif command == "step":
                state.step_once = True
            elif command == "reset":
                state.reset_requested = True
            elif command == "new_scene":
                state.new_scene_requested = True

    def run(self) -> None:
        server: WaymoInteractiveServer | None = None
        try:
            self._publish(GameUpdate("loading", "Loading Dreamer checkpoints..."))
            server = WaymoInteractiveServer(self.args)
            if self.stop_event.is_set():
                return
            self._publish(GameUpdate("loading", "Loading the selected Waymo scene..."))
            state = server.new_session()
            if self.stop_event.is_set():
                return
            state.paused = not bool(self.args.autoplay)
            initial_jpeg = server._render(state)
            state.cached_jpeg = initial_jpeg
            state.cached_frame_id = server._display_frame_id(state)
            self._publish(
                GameUpdate(
                    "frame",
                    jpeg=initial_jpeg,
                    status=server._status(state),
                )
            )

            period = 1.0 / max(float(self.args.fps), 0.1)
            next_tick = time.monotonic() + period
            while not self.stop_event.is_set():
                delay = next_tick - time.monotonic()
                if delay > 0.0 and self.stop_event.wait(delay):
                    break
                # Never run catch-up bursts after a slow inference/render.
                next_tick = max(next_tick + period, time.monotonic() + period)
                self._apply_commands(state)
                control = self.latest_input.get_control(
                    max_age_s=float(self.args.input_timeout)
                )
                jpeg, status = server._tick_sync(
                    state,
                    analog_control=control,
                )
                self._publish(GameUpdate("frame", jpeg=jpeg, status=status))
        except BaseException as error:
            if not isinstance(error, (KeyboardInterrupt, SystemExit)):
                detail = "".join(traceback.format_exception(error))
                self._publish(GameUpdate("error", str(error), status={"traceback": detail}))
        finally:
            try:
                if server is not None:
                    server.close()
            finally:
                self._publish(GameUpdate("stopped"))


def _decode_jpeg(pygame_module: Any, jpeg: bytes) -> Any:
    return pygame_module.image.load(io.BytesIO(jpeg), "frame.jpg").convert()


def _blit_aspect_fit(pygame_module: Any, screen: Any, image: Any) -> None:
    screen_width, screen_height = screen.get_size()
    image_width, image_height = image.get_size()
    scale = min(screen_width / image_width, screen_height / image_height)
    target_size = (
        max(1, round(image_width * scale)),
        max(1, round(image_height * scale)),
    )
    rendered = (
        image
        if target_size == image.get_size()
        else pygame_module.transform.smoothscale(image, target_size)
    )
    destination = (
        (screen_width - target_size[0]) // 2,
        (screen_height - target_size[1]) // 2,
    )
    screen.blit(rendered, destination)


def _draw_bar(
    pygame_module: Any,
    screen: Any,
    rect: Any,
    value: float,
    color: tuple[int, int, int],
    *,
    centered: bool = False,
) -> None:
    pygame_module.draw.rect(screen, (35, 40, 48), rect, border_radius=3)
    pygame_module.draw.rect(screen, (105, 112, 122), rect, width=1, border_radius=3)
    if centered:
        center_x = rect.x + rect.width // 2
        end_x = center_x + round(_clamp(value, -1.0, 1.0) * rect.width * 0.5)
        fill = pygame_module.Rect(min(center_x, end_x), rect.y, abs(end_x - center_x), rect.height)
        pygame_module.draw.line(
            screen, (190, 195, 202), (center_x, rect.y), (center_x, rect.bottom), 1
        )
    else:
        fill = pygame_module.Rect(
            rect.x,
            rect.y,
            round(rect.width * _clamp(value, 0.0, 1.0)),
            rect.height,
        )
    if fill.width > 0:
        pygame_module.draw.rect(screen, color, fill, border_radius=3)


def _draw_overlay(
    pygame_module: Any,
    screen: Any,
    font: Any,
    sample: WheelSample,
    status: dict[str, Any] | None,
    input_name: str,
    message: str,
) -> None:
    width, _height = screen.get_size()
    panel = pygame_module.Surface((min(width, 760), 144), pygame_module.SRCALPHA)
    panel.fill((5, 8, 13, 205))
    screen.blit(panel, (10, 10))

    status = status or {}
    step = status.get("step", "-")
    max_steps = status.get("max_steps", "-")
    if status.get("phase") == "context_replay":
        progress = (
            f"context {status.get('replay_step', '-')}/"
            f"{status.get('replay_steps', '-')}"
        )
    else:
        progress = f"rollout {step}/{max_steps}"
    speed = float(status.get("speed", 0.0))
    inference_ms = float(status.get("inference_ms", 0.0))
    checkpoint_profile = str(status.get("checkpoint_profile", "-"))
    paused = bool(status.get("paused", False))
    headline = (
        f"{input_name}  |  ckpt {checkpoint_profile}  |  "
        f"{progress}  |  {speed:.1f} m/s  |  "
        f"model {inference_ms:.0f} ms"
    )
    if paused:
        headline += "  |  PAUSED"
    screen.blit(font.render(headline, True, (238, 241, 246)), (22, 18))

    # Display steering in physical screen direction: right is visually right.
    # Internally the action convention is left-positive, hence the minus sign.
    labels = (
        ("STEER", -sample.control.steering, (75, 185, 255), True),
        ("THROTTLE", sample.control.throttle, (75, 220, 125), False),
        ("BRAKE", sample.control.brake, (255, 95, 85), False),
    )
    for row, (label, value, color, centered) in enumerate(labels):
        y = 46 + row * 22
        screen.blit(font.render(f"{label:<8}", True, (215, 220, 228)), (22, y))
        rect = pygame_module.Rect(116, y + 2, 220, 14)
        _draw_bar(pygame_module, screen, rect, float(value), color, centered=centered)
        screen.blit(font.render(f"{float(value):+.2f}" if centered else f"{float(value):.2f}", True, color), (346, y))

    if sample.source == "keyboard":
        raw = "keyboard: Up/W throttle  Down/S brake  Left/A Right/D steer"
    else:
        raw = (
            f"raw axes: wheel={sample.raw_wheel:+.3f}  "
            f"throttle={sample.raw_throttle:+.3f}  brake={sample.raw_brake:+.3f}"
        )
    screen.blit(font.render(raw, True, (178, 185, 196)), (430, 47))
    help_text = "Drive Arrows/WASD  Space pause  R reset  N scene  Q/Esc quit"
    screen.blit(font.render(help_text, True, (178, 185, 196)), (430, 71))
    if message:
        screen.blit(font.render(message, True, (255, 205, 90)), (430, 95))
    scene_label = str(status.get("scene_label", "scene loading..."))
    screen.blit(font.render(scene_label, True, (255, 205, 90)), (22, 116))


def _import_pygame() -> Any:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError(
            "Pygame is required for the local wheel/keyboard frontend. Install it in "
            "the dreamer4 environment with: "
            "/p/yufeng/.conda/envs/dreamer4/bin/python -m pip install pygame"
        ) from error
    return pygame


def connect_optional_wheel(
    pygame_module: Any,
    args: argparse.Namespace,
) -> FanatecWheel | None:
    """Open a wheel, or return ``None`` for keyboard/automatic fallback."""

    if args.input_device == "keyboard":
        print("[input] keyboard mode selected", flush=True)
        return None
    try:
        return FanatecWheel(pygame_module, args)
    except RuntimeError as error:
        if args.input_device == "auto":
            print(f"[input] {error}; falling back to keyboard", flush=True)
            return None
        raise


def run_local_game(args: argparse.Namespace) -> int:
    pygame = _import_pygame()
    pygame.display.init()
    pygame.font.init()
    pygame.joystick.init()
    flags = pygame.FULLSCREEN if args.fullscreen else pygame.RESIZABLE
    initial_size = (int(args.window_width), int(args.window_height))
    screen = pygame.display.set_mode(initial_size, flags)
    pygame.display.set_caption("Dreamer4 + PufferDrive Local Control")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    wheel: FanatecWheel | None = None
    worker: LocalGameWorker | None = None
    stop_event = threading.Event()
    commands: queue.Queue[str] = queue.Queue()
    updates: queue.Queue[GameUpdate] = queue.Queue()
    latest_input = LatestWheelSample()
    latest_sample = WheelSample.neutral()
    latest_frame = None
    latest_status: dict[str, Any] | None = None
    input_name = "Keyboard"
    message = "Selecting wheel or keyboard input..."
    exit_code = 0

    def fall_back_to_keyboard(reason: object) -> None:
        nonlocal wheel, input_name, latest_sample
        print(f"[input] {reason}; keyboard active", flush=True)
        # Clear a possibly nonzero last wheel throttle before the next sample.
        latest_sample = WheelSample.neutral(source="keyboard")
        latest_input.set(latest_sample)
        if wheel is not None:
            wheel.close()
        wheel = None
        input_name = "Keyboard (Arrows/WASD)"

    try:
        wheel = connect_optional_wheel(pygame, args)
        input_name = wheel.name if wheel is not None else "Keyboard (Arrows/WASD)"
        keyboard_sample = read_keyboard_sample(pygame)
        try:
            wheel_sample = wheel.read() if wheel is not None else None
        except Exception as error:
            if args.input_device == "auto":
                fall_back_to_keyboard(f"wheel read failed: {error}")
                wheel_sample = None
            else:
                raise
        latest_sample = select_input_sample(
            args.input_device,
            keyboard_sample=keyboard_sample,
            wheel_sample=wheel_sample,
        )
        if latest_sample.source == "keyboard":
            input_name = "Keyboard (Arrows/WASD)"
        latest_input.set(latest_sample)
        worker = LocalGameWorker(
            args,
            latest_input,
            commands,
            updates,
            stop_event,
        )
        worker.start()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        commands.put("toggle_pause")
                    elif event.key == pygame.K_PERIOD:
                        commands.put("step")
                    elif event.key == pygame.K_r:
                        commands.put("reset")
                    elif event.key == pygame.K_n:
                        commands.put("new_scene")
                elif (
                    event.type == getattr(pygame, "JOYDEVICEREMOVED", -1)
                    and wheel is not None
                    and int(getattr(event, "instance_id", -1)) == wheel.instance_id
                ):
                    if args.input_device == "auto":
                        fall_back_to_keyboard("wheel disconnected")
                    else:
                        raise RuntimeError("Wheel or joystick was disconnected")

            keyboard_sample = read_keyboard_sample(pygame)
            try:
                wheel_sample = wheel.read() if wheel is not None else None
            except Exception as error:
                if args.input_device == "auto":
                    fall_back_to_keyboard(f"wheel read failed: {error}")
                    wheel_sample = None
                else:
                    raise
            latest_sample = select_input_sample(
                args.input_device,
                keyboard_sample=keyboard_sample,
                wheel_sample=wheel_sample,
            )
            input_name = (
                "Keyboard (Arrows/WASD)"
                if latest_sample.source == "keyboard"
                else (wheel.name if wheel is not None else "Wheel/Joystick")
            )
            latest_input.set(latest_sample)

            fatal_error: GameUpdate | None = None
            while True:
                try:
                    update = updates.get_nowait()
                except queue.Empty:
                    break
                if update.kind == "loading":
                    message = update.message
                elif update.kind == "frame":
                    if update.jpeg is not None:
                        latest_frame = _decode_jpeg(pygame, update.jpeg)
                    if update.status is not None:
                        latest_status = update.status
                    message = ""
                elif update.kind == "error":
                    fatal_error = update
                elif (
                    update.kind == "stopped"
                    and not stop_event.is_set()
                    and fatal_error is None
                ):
                    fatal_error = GameUpdate("error", "Dreamer/Puffer worker stopped unexpectedly")

            screen.fill((8, 11, 17))
            if latest_frame is not None:
                _blit_aspect_fit(pygame, screen, latest_frame)
            else:
                loading = font.render(message or "Loading...", True, (235, 238, 243))
                screen.blit(
                    loading,
                    (
                        (screen.get_width() - loading.get_width()) // 2,
                        (screen.get_height() - loading.get_height()) // 2,
                    ),
                )
            _draw_overlay(
                pygame,
                screen,
                font,
                latest_sample,
                latest_status,
                input_name,
                message,
            )
            pygame.display.flip()

            if fatal_error is not None:
                print(f"[local game error] {fatal_error.message}", file=sys.stderr)
                if fatal_error.status and fatal_error.status.get("traceback"):
                    print(fatal_error.status["traceback"], file=sys.stderr)
                exit_code = 1
                running = False
            clock.tick(max(10, int(args.ui_fps)))
    except KeyboardInterrupt:
        # Ctrl-C is an ordinary local-game exit, like Q or closing the window.
        exit_code = 0
    except Exception as error:
        print(f"[local game error] {error}", file=sys.stderr)
        exit_code = 1
    finally:
        # A stale input must never survive shutdown while inference finishes.
        latest_input.set(WheelSample.neutral())
        stop_event.set()
        if worker is not None:
            worker.join(timeout=float(args.puffer_timeout) + 5.0)
            if worker.is_alive():
                print(
                    "[local game warning] inference worker did not stop before timeout",
                    file=sys.stderr,
                )
                exit_code = 1
        if wheel is not None:
            wheel.close()
        pygame.joystick.quit()
        pygame.font.quit()
        pygame.display.quit()
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = build_game_parser()
    parser.description = "Run Dreamer4/PufferDrive locally with wheel/joystick or keyboard."
    parser.set_defaults(
        renderer="puffer",
        autoplay=True,
        puffer_strict=True,
        puffer_timeout=120.0,
        puffer_use_inherited_display=True,
        puffer_manifest=(
            str(DEFAULT_LOCAL_MANIFEST) if DEFAULT_LOCAL_MANIFEST.is_file() else None
        ),
    )
    parser.add_argument(
        "--input-device",
        choices=("auto", "wheel", "joystick", "keyboard"),
        default="auto",
        help=(
            "Control source. auto uses a compatible wheel/joystick when present "
            "and otherwise falls back to keyboard."
        ),
    )
    parser.add_argument(
        "--joystick-index",
        type=int,
        default=None,
        help="SDL joystick index. Default: prefer a compatible Fanatec/wheel device.",
    )
    parser.add_argument("--steering-axis", type=int, default=0)
    parser.add_argument("--throttle-axis", type=int, default=2)
    parser.add_argument("--brake-axis", type=int, default=5)
    parser.add_argument("--steering-deadzone", type=float, default=0.02)
    parser.add_argument("--pedal-deadzone", type=float, default=0.02)
    parser.add_argument("--steering-gain", type=float, default=1.0)
    parser.add_argument(
        "--invert-steering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert Fanatec right-positive axis to the game's left-positive convention.",
    )
    parser.add_argument(
        "--input-timeout",
        type=float,
        default=0.5,
        help="Use neutral controls if no fresh input sample arrives for this many seconds.",
    )
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--ui-fps", type=int, default=60)
    parser.add_argument("--fullscreen", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.renderer != "puffer":
        parser.error("the local Fanatec game requires --renderer puffer")
    if not args.puffer_manifest:
        parser.error(
            "--puffer-manifest is required; run prepare_pufferdrive_static_scenes.py first"
        )
    if not Path(args.puffer_manifest).expanduser().is_file():
        parser.error(f"Puffer manifest does not exist: {args.puffer_manifest}")
    if args.window_width <= 0 or args.window_height <= 0:
        parser.error("window dimensions must be positive")
    if args.ui_fps <= 0:
        parser.error("--ui-fps must be positive")
    if args.input_timeout <= 0:
        parser.error("--input-timeout must be positive")
    if not 0.0 <= args.steering_deadzone < 1.0:
        parser.error("--steering-deadzone must be in [0, 1)")
    if not 0.0 <= args.pedal_deadzone < 1.0:
        parser.error("--pedal-deadzone must be in [0, 1)")
    if args.steering_gain <= 0:
        parser.error("--steering-gain must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    return run_local_game(args)


if __name__ == "__main__":
    raise SystemExit(main())
