import inspect
import unittest
from unittest import mock

from waymo.interactive_world_model_game import AnalogControl
from waymo.local_puffer_fanatec_game import (
    LatestWheelSample,
    WheelSample,
    _draw_overlay,
    build_parser,
    connect_optional_wheel,
    fanatec_axes_to_control,
    keyboard_to_control,
    released_high_pedal,
    rescaled_axis_deadzone,
    select_input_sample,
)


class FanatecAxisMappingTest(unittest.TestCase):
    def test_released_pedals_are_zero_and_fully_pressed_are_one(self):
        self.assertEqual(released_high_pedal(1.0, 0.02), 0.0)
        self.assertEqual(released_high_pedal(-1.0, 0.02), 1.0)

    def test_small_pedal_noise_is_removed_and_remaining_range_rescaled(self):
        self.assertEqual(released_high_pedal(0.99, 0.02), 0.0)
        expected = (0.5 - 0.02) / 0.98
        self.assertAlmostEqual(released_high_pedal(0.0, 0.02), expected)

    def test_default_mapping_inverts_right_positive_fanatec_wheel(self):
        control = fanatec_axes_to_control(
            wheel_axis=0.51,
            throttle_axis=-1.0,
            brake_axis=1.0,
            steering_deadzone=0.02,
        )
        self.assertLess(control.steering, 0.0)  # Negative is a right turn.
        self.assertEqual(control.throttle, 1.0)
        self.assertEqual(control.brake, 0.0)

    def test_no_inversion_and_gain_are_supported_for_other_firmware_modes(self):
        control = fanatec_axes_to_control(
            0.5,
            1.0,
            1.0,
            steering_deadzone=0.0,
            steering_gain=0.5,
            invert_steering=False,
        )
        self.assertAlmostEqual(control.steering, 0.25)

    def test_axis_deadzone_preserves_full_range(self):
        self.assertEqual(rescaled_axis_deadzone(0.01, 0.02), 0.0)
        self.assertEqual(rescaled_axis_deadzone(-1.0, 0.02), -1.0)
        self.assertEqual(rescaled_axis_deadzone(1.0, 0.02), 1.0)


class LatestWheelSampleTest(unittest.TestCase):
    def test_stale_sample_fails_to_neutral(self):
        latest = LatestWheelSample()
        latest.set(
            WheelSample(
                control=AnalogControl(steering=0.5, throttle=1.0, brake=0.0),
                raw_wheel=-0.5,
                raw_throttle=-1.0,
                raw_brake=1.0,
                sampled_at=10.0,
            )
        )
        self.assertEqual(
            latest.get_control(max_age_s=0.5, now=10.4),
            AnalogControl(steering=0.5, throttle=1.0, brake=0.0),
        )
        self.assertEqual(
            latest.get_control(max_age_s=0.5, now=10.6),
            AnalogControl(),
        )


class KeyboardInputTest(unittest.TestCase):
    def test_arrow_or_wasd_controls_map_to_full_digital_input(self):
        self.assertEqual(
            keyboard_to_control(left=True, throttle=True),
            AnalogControl(steering=1.0, throttle=1.0, brake=0.0),
        )
        self.assertEqual(
            keyboard_to_control(right=True, brake=True),
            AnalogControl(steering=-1.0, throttle=0.0, brake=1.0),
        )

    def test_opposite_steering_keys_cancel(self):
        self.assertEqual(
            keyboard_to_control(left=True, right=True),
            AnalogControl(),
        )

    def test_auto_uses_keyboard_without_wheel_and_keyboard_overrides_wheel(self):
        keyboard = WheelSample(
            AnalogControl(throttle=1.0), 0.0, -1.0, 1.0, 10.0, "keyboard"
        )
        wheel = WheelSample(
            AnalogControl(steering=0.25), -0.25, 1.0, 1.0, 10.0, "wheel"
        )
        self.assertIs(
            select_input_sample("auto", keyboard_sample=keyboard, wheel_sample=None),
            keyboard,
        )
        self.assertIs(
            select_input_sample("auto", keyboard_sample=keyboard, wheel_sample=wheel),
            keyboard,
        )

    def test_auto_uses_wheel_while_keyboard_is_idle(self):
        keyboard = WheelSample.neutral(10.0, source="keyboard")
        wheel = WheelSample(
            AnalogControl(steering=0.25), -0.25, 1.0, 1.0, 10.0, "wheel"
        )
        self.assertIs(
            select_input_sample("auto", keyboard_sample=keyboard, wheel_sample=wheel),
            wheel,
        )


class OptionalWheelConnectionTest(unittest.TestCase):
    class EmptyJoystickAPI:
        @staticmethod
        def get_count():
            return 0

    class FakePygame:
        joystick = None

    def setUp(self):
        self.FakePygame.joystick = self.EmptyJoystickAPI()

    def test_auto_falls_back_when_sdl_detects_no_joystick(self):
        args = build_parser().parse_args(["--input-device", "auto"])
        self.assertIsNone(connect_optional_wheel(self.FakePygame, args))

    def test_keyboard_mode_does_not_probe_joysticks(self):
        args = build_parser().parse_args(["--input-device", "keyboard"])
        with mock.patch(
            "waymo.local_puffer_fanatec_game.FanatecWheel",
            side_effect=AssertionError("wheel should not be opened"),
        ):
            self.assertIsNone(connect_optional_wheel(self.FakePygame, args))

    def test_forced_wheel_preserves_clear_missing_device_error(self):
        args = build_parser().parse_args(["--input-device", "wheel"])
        with self.assertRaisesRegex(RuntimeError, "No joystick or wheel"):
            connect_optional_wheel(self.FakePygame, args)


class LocalCheckpointProfileTest(unittest.TestCase):
    def test_local_3d_parser_inherits_checkpoint_flag(self):
        args = build_parser().parse_args(["--ckpt", "h30"])
        self.assertEqual(args.checkpoint_profile, "h30")
        self.assertEqual(args.renderer, "puffer")

    def test_local_3d_defaults_to_h90(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.checkpoint_profile, "h90")
        self.assertEqual(args.input_device, "auto")

    def test_local_3d_inherits_context11_future150_protocol(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.renderer, "puffer")
        self.assertEqual(args.context_frames, 11)
        self.assertEqual(args.max_steps, 150)

    def test_local_3d_explicit_max_steps_overrides_default(self):
        args = build_parser().parse_args(["--max-steps", "42"])
        self.assertEqual(args.max_steps, 42)

    def test_local_3d_accepts_unroll_steps_alias(self):
        args = build_parser().parse_args(["--unroll-steps", "80"])
        self.assertEqual(args.max_steps, 80)

    def test_local_overlay_consumes_shared_scene_label(self):
        self.assertIn('status.get("scene_label"', inspect.getsource(_draw_overlay))


if __name__ == "__main__":
    unittest.main()
