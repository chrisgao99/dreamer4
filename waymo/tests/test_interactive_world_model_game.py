import math
import unittest
from types import SimpleNamespace

import torch

from waymo.interactive_world_model_game import (
    ControlConfig,
    FocusState,
    initial_focus_action,
    integrate_focus_control,
    wm,
)


class InteractiveControlTest(unittest.TestCase):
    def setUp(self):
        self.config = ControlConfig(
            dt=0.1,
            acceleration_mps2=5.0,
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


if __name__ == "__main__":
    unittest.main()
