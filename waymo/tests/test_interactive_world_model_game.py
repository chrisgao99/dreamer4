import math
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from waymo.interactive_world_model_game import (
    AnalogControl,
    ControlConfig,
    DEFAULT_CHECKPOINT_PROFILE,
    DEFAULT_CONTEXT_FRAMES,
    DEFAULT_2D_ROLLOUT_STEPS,
    DEFAULT_3D_ROLLOUT_STEPS,
    DEFAULT_ROLLOUT_STEPS,
    FocusState,
    WaymoInteractiveServer,
    WORLD_MODEL_CHECKPOINT_PROFILES,
    build_parser,
    context_frame_bounds,
    initial_focus_action,
    integrate_focus_analog_control,
    integrate_focus_control,
    prediction_context_bounds,
    resolve_world_model_checkpoint,
    scene_identity,
    wm,
)


class InteractiveControlTest(unittest.TestCase):
    def setUp(self):
        self.config = ControlConfig(
            dt=0.1,
            acceleration_mps2=5.0,
            braking_mps2=8.0,
            yaw_rate_deg_s=45.0,
            min_speed_mps=0.0,
            max_speed_mps=30.0,
        )

    def test_initial_action_has_absolute_motion_but_zero_deltas(self):
        action = initial_focus_action(FocusState(x=2.0, y=3.0, speed=10.0, yaw=math.pi / 2))
        self.assertEqual(tuple(action.shape), (16,))
        self.assertTrue(action[:3].eq(0).all())
        self.assertAlmostEqual(float(action[3]), 10.0)
        self.assertAlmostEqual(float(action[4]), 0.0, places=5)
        self.assertAlmostEqual(float(action[5]), 10.0, places=5)
        self.assertEqual(float(action[6]), 1.0)

    def test_up_and_left_create_native_raw_action(self):
        state = FocusState(x=0.0, y=0.0, speed=10.0, yaw=0.0)
        next_state, action = integrate_focus_control(
            state, {"ArrowUp", "ArrowLeft"}, self.config
        )
        expected_yaw = math.radians(4.5)
        self.assertAlmostEqual(next_state.speed, 10.5)
        self.assertAlmostEqual(next_state.yaw, expected_yaw)
        self.assertAlmostEqual(float(action[2]), expected_yaw, places=6)
        self.assertAlmostEqual(float(action[3]), 10.5)
        self.assertAlmostEqual(float(action[0]), float(action[4]) * 0.1, places=6)
        self.assertAlmostEqual(float(action[1]), float(action[5]) * 0.1, places=6)
        self.assertAlmostEqual(next_state.x, float(action[0]), places=6)
        self.assertAlmostEqual(next_state.y, float(action[1]), places=6)

    def test_speed_is_clamped_and_neutral_keeps_moving(self):
        state = FocusState(x=1.0, y=2.0, speed=0.1, yaw=0.0)
        stopped, stop_action = integrate_focus_control(state, {"ArrowDown"}, self.config)
        self.assertEqual(stopped.speed, 0.0)
        self.assertEqual(float(stop_action[0]), 0.0)
        moving, moving_action = integrate_focus_control(
            FocusState(x=1.0, y=2.0, speed=3.0, yaw=0.0), set(), self.config
        )
        self.assertAlmostEqual(moving.x, 1.3, places=6)
        self.assertAlmostEqual(float(moving_action[3]), 3.0)

    def test_analog_throttle_brake_and_steering(self):
        state = FocusState(x=0.0, y=0.0, speed=10.0, yaw=0.0)
        accelerated, action = integrate_focus_analog_control(
            state,
            AnalogControl(steering=0.5, throttle=0.8, brake=0.25),
            self.config,
        )
        # (0.8 * 5 - 0.25 * 8) * 0.1 = +0.2 m/s.
        self.assertAlmostEqual(accelerated.speed, 10.2)
        self.assertAlmostEqual(accelerated.yaw, math.radians(2.25))
        self.assertAlmostEqual(float(action[2]), math.radians(2.25), places=6)
        self.assertAlmostEqual(float(action[3]), 10.2, places=6)

    def test_analog_values_are_clamped_and_brake_cannot_reverse(self):
        state = FocusState(x=1.0, y=2.0, speed=0.2, yaw=0.0)
        stopped, action = integrate_focus_analog_control(
            state,
            AnalogControl(steering=-5.0, throttle=-1.0, brake=3.0),
            self.config,
        )
        self.assertEqual(stopped.speed, 0.0)
        self.assertAlmostEqual(stopped.yaw, math.radians(-4.5))
        self.assertEqual(float(action[0]), 0.0)
        self.assertEqual(float(action[1]), 0.0)

    def test_action_matches_training_feature_builder(self):
        initial = FocusState(x=0.0, y=0.0, speed=10.0, yaw=0.0)
        controlled, action = integrate_focus_control(
            initial, {"ArrowUp", "ArrowLeft"}, self.config
        )
        initial_vx, initial_vy = initial.velocity
        controlled_vx, controlled_vy = controlled.velocity
        agents = torch.tensor(
            [
                [
                    [[initial.x, initial.y, initial.speed, initial_vx, initial_vy, 1.0, initial.yaw, 1.0]],
                    [[controlled.x, controlled.y, controlled.speed, controlled_vx, controlled_vy, 1.0, controlled.yaw, 1.0]],
                ]
            ],
            dtype=torch.float32,
        )
        expected, mask, slots = wm.build_ego_action_features(
            {"agents": agents, "agent_mask": torch.ones(1, 1, dtype=torch.bool)},
            SimpleNamespace(
                use_ego_actions=True,
                ego_action_source="focus",
                ego_action_normalization="raw",
            ),
        )
        torch.testing.assert_close(initial_focus_action(initial), expected[0, 0])
        torch.testing.assert_close(action, expected[0, 1])
        self.assertTrue(mask[0, :, :7].eq(1).all())
        self.assertEqual(int(slots[0]), 0)


class ConvertedSceneSelectionTest(unittest.TestCase):
    def make_server(self, converted):
        server = WaymoInteractiveServer.__new__(WaymoInteractiveServer)
        server.args = SimpleNamespace(renderer="puffer")
        server.puffer_scene_indices = tuple(converted)
        server.dataset = range(20)
        server.random = random.Random(3)
        return server

    def test_new_scene_is_limited_to_converted_npz_views(self):
        server = self.make_server((2, 5, 9))
        selected = server._pick_new_scene_index(5)
        self.assertIn(selected, (2, 9))

    def test_single_converted_view_is_reloaded(self):
        server = self.make_server((7,))
        self.assertEqual(server._pick_new_scene_index(7), 7)
        self.assertEqual(server._pick_new_scene_index(3), 7)


class CheckpointProfileTest(unittest.TestCase):
    def test_default_profile_is_h90(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.checkpoint_profile, "h90")
        self.assertEqual(args.checkpoint_profile, DEFAULT_CHECKPOINT_PROFILE)
        self.assertIsNone(args.world_model_ckpt)
        path, label = resolve_world_model_checkpoint(
            args.checkpoint_profile, args.world_model_ckpt
        )
        self.assertEqual(path, WORLD_MODEL_CHECKPOINT_PROFILES["h90"].resolve())
        self.assertEqual(label, "h90")

    def test_short_flag_selects_h30(self):
        args = build_parser().parse_args(["--ckpt", "h30"])
        path, label = resolve_world_model_checkpoint(
            args.checkpoint_profile, args.world_model_ckpt
        )
        self.assertEqual(path, WORLD_MODEL_CHECKPOINT_PROFILES["h30"].resolve())
        self.assertEqual(path.name, "best_multisample_finetuned.pt")
        self.assertEqual(label, "h30")

    def test_explicit_checkpoint_path_overrides_profile(self):
        args = build_parser().parse_args(
            [
                "--ckpt",
                "h30",
                "--world-model-ckpt",
                "/tmp/custom-interactive.pt",
            ]
        )
        path, label = resolve_world_model_checkpoint(
            args.checkpoint_profile, args.world_model_ckpt
        )
        self.assertEqual(path, Path("/tmp/custom-interactive.pt"))
        self.assertEqual(label, "custom")


class ContextRolloutProtocolTest(unittest.TestCase):
    def test_2d_parser_defaults_to_eleven_recorded_and_eighty_generated_frames(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.renderer, "2d")
        self.assertEqual(args.start_frame, 0)
        self.assertEqual(args.context_frames, DEFAULT_CONTEXT_FRAMES)
        self.assertEqual(args.context_frames, 11)
        self.assertEqual(args.max_steps, DEFAULT_ROLLOUT_STEPS)
        self.assertEqual(args.max_steps, DEFAULT_2D_ROLLOUT_STEPS)
        self.assertEqual(args.max_steps, 80)

    def test_puffer_parser_defaults_to_one_hundred_fifty_generated_frames(self):
        args = build_parser().parse_args(["--renderer", "puffer"])
        self.assertEqual(args.renderer, "puffer")
        self.assertEqual(args.context_frames, 11)
        self.assertEqual(args.max_steps, DEFAULT_3D_ROLLOUT_STEPS)
        self.assertEqual(args.max_steps, 150)

    def test_explicit_max_steps_overrides_renderer_default_in_either_order(self):
        for argv in (
            ["--renderer", "puffer", "--max-steps", "37"],
            ["--max-steps", "37", "--renderer", "puffer"],
            ["--renderer", "2d", "--max-steps", "37"],
        ):
            with self.subTest(argv=argv):
                self.assertEqual(build_parser().parse_args(argv).max_steps, 37)

    def test_explicit_unroll_steps_overrides_renderer_default(self):
        for renderer, steps in (("2d", 80), ("puffer", 150)):
            with self.subTest(renderer=renderer, steps=steps):
                args = build_parser().parse_args(
                    ["--renderer", renderer, "--unroll-steps", str(steps)]
                )
                self.assertEqual(args.max_steps, steps)

    def test_default_frame_ranges_are_replay_1_to_11_and_model_2_to_11(self):
        self.assertEqual(context_frame_bounds(0, 91, 11), (0, 11))
        # Half-open [1, 11) is human-readable Waymo frames 2 through 11.
        self.assertEqual(prediction_context_bounds(0, 11, 11), (1, 11))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            context_frame_bounds(81, 91, 11)

    def test_puffer_requires_a_valid_chase_focus_during_all_replay_frames(self):
        agents = torch.zeros(11, 2, 8)
        agents[:, 0, 5] = 1.0
        item = {
            "agents": agents,
            "agent_mask": torch.ones(2, dtype=torch.bool),
            "lights": torch.zeros(11, 1, 1),
        }
        server = WaymoInteractiveServer.__new__(WaymoInteractiveServer)
        server.args = SimpleNamespace(start_frame=0)
        server.context_frames = 11
        server.dataset = [item]
        self.assertTrue(server._has_continuous_puffer_focus_context(0))

        agents[4, 0, 5] = 0.0
        self.assertFalse(server._has_continuous_puffer_focus_context(0))

    def test_sampler_first_prediction_receives_frames_2_through_11(self):
        class RecordingDynamics:
            def __init__(self):
                self.packed = None
                self.actions = None
                self.action_mask = None

            def __call__(
                self,
                actions,
                _step_idxs,
                _signal_idxs,
                packed,
                *,
                act_mask,
                **_kwargs,
            ):
                self.packed = packed.detach().clone()
                self.actions = actions.detach().clone()
                self.action_mask = act_mask.detach().clone()
                return packed, None

        dynamics = RecordingDynamics()
        past = torch.arange(1, 12, dtype=torch.float32).reshape(1, 11, 1, 1)
        actions = torch.zeros(1, 12, 16)
        actions[0, :, 0] = torch.arange(1, 13, dtype=torch.float32)
        action_mask = torch.ones_like(actions)
        result = wm.sample_one_timestep_packed(
            dynamics,
            past_packed=past,
            actions_seq=actions,
            act_mask_seq=action_mask,
            map_tokens=None,
            map_mask=None,
            k_max=64,
            sched={
                "K": 1,
                "e": 0,
                "tau": [0.0, 1.0],
                "tau_idx": [0, 64],
                "dt": 1.0,
            },
            max_rollout_window=11,
        )

        self.assertEqual(tuple(result.shape), (1, 1, 1))
        self.assertEqual(tuple(dynamics.packed.shape), (1, 11, 1, 1))
        torch.testing.assert_close(
            dynamics.packed[0, :10, 0, 0],
            torch.arange(2, 12, dtype=torch.float32),
        )
        torch.testing.assert_close(
            dynamics.actions[0, :, 0],
            torch.arange(2, 13, dtype=torch.float32),
        )
        self.assertEqual(tuple(dynamics.action_mask.shape), (1, 11, 16))

    @staticmethod
    def make_replay_state(*, paused=False):
        focus = [
            FocusState(x=float(i), y=float(-i), speed=float(i + 1), yaw=0.01 * i)
            for i in range(11)
        ]
        world = np.stack(
            [np.asarray([[i, -i], [i + 100, -i]], dtype=np.float32) for i in range(11)]
        )
        valid = np.ones((11, 2), dtype=bool)
        yaw = np.zeros((11, 2), dtype=np.float32)
        velocity = np.ones((11, 2, 2), dtype=np.float32)
        return SimpleNamespace(
            context_start_frame=0,
            context_focus=focus,
            context_world=world,
            context_valid=valid,
            context_yaw=yaw,
            context_velocity=velocity,
            focus=focus[0],
            world_history=[world[0].copy()],
            valid_history=[valid[0].copy()],
            yaw_history=[yaw[0].copy()],
            velocity_history=[velocity[0].copy()],
            keys_down=set(),
            replay_index=0,
            step=0,
            paused=paused,
            step_once=False,
            reset_requested=False,
            new_scene_requested=False,
            cached_jpeg=None,
            cached_frame_id=-1,
            last_inference_ms=0.0,
            renderer_name="2d",
            renderer_error=None,
            scene_index=0,
            scenario_id="scenario-test",
            scene_path="/data/scenario-test_focus_123_src0.npz",
            agent_ids=np.asarray([123, 456], dtype=np.int64),
        )

    @staticmethod
    def make_timeline_server():
        server = WaymoInteractiveServer.__new__(WaymoInteractiveServer)
        server.args = SimpleNamespace(max_steps=80)
        server.model_rollout_window = 11
        server.checkpoint_profile = "h90"
        server._render = lambda state: bytes([server._display_frame_id(state)])
        server._status = lambda state: {
            "replay_index": state.replay_index,
            "step": state.step,
        }
        return server

    def test_autoplay_replays_all_context_before_first_model_step(self):
        server = self.make_timeline_server()
        state = self.make_replay_state(paused=False)
        generated = []

        def fake_advance(current, analog_control=None):
            generated.append(analog_control)
            current.step += 1

        server._advance = fake_advance

        jpeg, _status = server._tick_sync(state)
        self.assertEqual(jpeg, bytes([0]))
        for expected_replay_index in range(1, 11):
            jpeg, _status = server._tick_sync(state)
            self.assertEqual(state.replay_index, expected_replay_index)
            self.assertEqual(state.step, 0)
            self.assertEqual(jpeg, bytes([expected_replay_index]))
        self.assertEqual(generated, [])
        self.assertEqual(state.focus, state.context_focus[10])
        self.assertEqual(len(state.world_history), 11)

        jpeg, _status = server._tick_sync(state)
        self.assertEqual(state.step, 1)
        self.assertEqual(len(generated), 1)
        self.assertEqual(jpeg, bytes([11]))

    def test_paused_single_step_moves_one_context_frame_without_inference(self):
        server = self.make_timeline_server()
        state = self.make_replay_state(paused=True)
        state.cached_jpeg = bytes([0])
        state.cached_frame_id = 0
        state.step_once = True
        server._advance = lambda *_args, **_kwargs: self.fail(
            "model inference must not run during recorded replay"
        )

        jpeg, _status = server._tick_sync(state)
        self.assertEqual(jpeg, bytes([1]))
        self.assertEqual(state.replay_index, 1)
        self.assertEqual(state.step, 0)
        self.assertFalse(state.step_once)

    def test_reset_restarts_at_recorded_frame_one_without_skipping(self):
        server = self.make_timeline_server()
        state = self.make_replay_state(paused=False)
        state.replay_index = 10
        state.step = 7
        state.cached_jpeg = b"old scene"
        state.cached_frame_id = 17
        state.reset_requested = True

        def fake_reset(current, *, new_scene):
            self.assertFalse(new_scene)
            replacement = self.make_replay_state(paused=False)
            current.__dict__.clear()
            current.__dict__.update(replacement.__dict__)

        server._reset_session = fake_reset
        server._advance = lambda *_args, **_kwargs: self.fail(
            "reset tick must render frame one before advancing"
        )
        jpeg, _status = server._tick_sync(state)
        self.assertEqual(jpeg, bytes([0]))
        self.assertEqual(state.replay_index, 0)
        self.assertEqual(state.step, 0)
        self.assertEqual(state.focus, state.context_focus[0])

    def test_rollout_stops_exactly_at_eighty_generated_steps(self):
        server = self.make_timeline_server()
        state = self.make_replay_state(paused=False)
        state.replay_index = 10
        state.focus = state.context_focus[-1]
        state.step = 79
        state.cached_jpeg = b"previous"
        state.cached_frame_id = 89

        def fake_advance(current, analog_control=None):
            current.step += 1

        server._advance = fake_advance
        jpeg, _status = server._tick_sync(state)
        self.assertEqual(state.step, 80)
        self.assertTrue(state.paused)
        self.assertEqual(server._display_frame_id(state), 90)
        self.assertEqual(jpeg, bytes([90]))

    def test_3d_rollout_stops_exactly_at_one_hundred_fifty_generated_steps(self):
        server = self.make_timeline_server()
        server.args.max_steps = 150
        state = self.make_replay_state(paused=False)
        state.replay_index = 10
        state.focus = state.context_focus[-1]
        state.step = 149
        state.cached_jpeg = b"previous"
        state.cached_frame_id = 159

        def fake_advance(current, analog_control=None):
            current.step += 1

        server._advance = fake_advance
        jpeg, _status = server._tick_sync(state)
        self.assertEqual(state.step, 150)
        self.assertTrue(state.paused)
        self.assertEqual(server._display_frame_id(state), 160)
        self.assertEqual(jpeg, bytes([160]))

    def test_status_exposes_context_and_rollout_protocol(self):
        server = WaymoInteractiveServer.__new__(WaymoInteractiveServer)
        server.args = SimpleNamespace(max_steps=80)
        server.model_rollout_window = 11
        server.checkpoint_profile = "h90"
        state = self.make_replay_state(paused=True)

        status = server._status(state)
        self.assertEqual(status["phase"], "context_replay")
        self.assertEqual(status["replay_step"], 1)
        self.assertEqual(status["replay_steps"], 11)
        self.assertEqual(status["model_context_start"], 2)
        self.assertEqual(status["model_context_end"], 11)
        self.assertEqual(status["timeline_frame"], 1)
        self.assertEqual(status["scene_index"], 0)
        self.assertEqual(status["scenario_id"], "scenario-test")
        self.assertEqual(status["focus_track_id"], 123)
        self.assertEqual(status["scene_file"], "scenario-test_focus_123_src0.npz")
        self.assertEqual(
            status["scene_label"],
            "scene #0 | scenario scenario-test | focus 123",
        )
        self.assertIn(status["scene_label"], status["text"])

        state.replay_index = 10
        state.step = 1
        state.focus = state.context_focus[-1]
        status = server._status(state)
        self.assertEqual(status["phase"], "rollout")
        self.assertEqual(status["step"], 1)
        self.assertEqual(status["timeline_frame"], 12)

        server.args.max_steps = 150
        state.step = 150
        status = server._status(state)
        self.assertEqual(status["max_steps"], 150)
        self.assertEqual(status["step"], 150)
        self.assertEqual(status["timeline_frame"], 161)

    def test_long_rollout_decoder_metadata_clamps_to_context_handoff(self):
        server = WaymoInteractiveServer.__new__(WaymoInteractiveServer)
        agents = torch.zeros(1, 11, 2, 8)
        lights = torch.arange(11, dtype=torch.float32).reshape(1, 11, 1, 1)
        light_mask = torch.ones(1, 11, 1, dtype=torch.bool)
        state = SimpleNamespace(
            context_start_frame=0,
            context_focus=[object() for _ in range(11)],
            base_batch={
                "agents": agents,
                "agent_mask": torch.ones(1, 2, dtype=torch.bool),
                "lights": lights,
                "light_mask": light_mask,
            },
        )

        decoded_batch = server._decode_batch(
            state,
            timeline_start=129,
            time_steps=32,
        )
        self.assertEqual(tuple(decoded_batch["agents"].shape[:2]), (1, 32))
        self.assertEqual(tuple(decoded_batch["lights"].shape[:2]), (1, 32))
        self.assertEqual(tuple(decoded_batch["light_mask"].shape[:2]), (1, 32))
        torch.testing.assert_close(
            decoded_batch["lights"],
            lights[:, 10:11].expand(-1, 32, -1, -1),
        )

    def test_browser_ui_consumes_phase_specific_progress(self):
        html_path = Path(__file__).resolve().parents[1] / "interactive_world_model_game.html"
        html = html_path.read_text(encoding="utf-8")
        self.assertIn('message.phase === "context_replay"', html)
        self.assertIn("message.replay_step", html)
        self.assertIn("message.replay_steps", html)
        self.assertIn("message.max_steps", html)
        self.assertIn('id="sceneLabel"', html)
        self.assertIn("message.scene_label", html)
        self.assertIn("message.scene_file", html)

    def test_scene_identity_distinguishes_focus_views_of_the_same_scenario(self):
        first = self.make_replay_state()
        second = self.make_replay_state()
        second.agent_ids[0] = 999
        second.scene_path = "/data/scenario-test_focus_999_src1.npz"
        self.assertEqual(scene_identity(first)["scenario_id"], "scenario-test")
        self.assertEqual(scene_identity(second)["scenario_id"], "scenario-test")
        self.assertNotEqual(
            scene_identity(first)["scene_label"],
            scene_identity(second)["scene_label"],
        )


if __name__ == "__main__":
    unittest.main()
