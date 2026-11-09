from __future__ import annotations

import unittest

import numpy as np

from interaction_contrastive_learning.legacy.soft_pair_samples import (
    SoftPairConfig,
    extract_soft_pair_sample,
    find_constrained_closest_points,
)
from interaction_contrastive_learning.legacy.build_soft_pair_dataset import fit_robust_scaler


def _agent(x: np.ndarray, y: np.ndarray, yaw: float = 0.0) -> np.ndarray:
    t = len(x)
    result = np.zeros((t, 8), dtype=np.float32)
    result[:, 0] = x
    result[:, 1] = y
    result[:, 3] = np.gradient(x, 0.1)
    result[:, 4] = np.gradient(y, 0.1)
    result[:, 2] = np.sqrt(result[:, 3] ** 2 + result[:, 4] ** 2)
    result[:, 5] = 1.0
    result[:, 6] = yaw
    result[:, 7] = 1.0
    return result


class SoftPairSampleTests(unittest.TestCase):
    def test_global_scaler_does_not_amplify_degenerate_heading_channels(self) -> None:
        sequence = np.zeros((3, 60, 12), dtype=np.float32)
        sequence[:, :, 5] = 1.0
        sequence[:, :, 11] = 1.0
        median, scale = fit_robust_scaler(sequence)
        np.testing.assert_allclose(median[[4, 5, 10, 11]], 0.0)
        np.testing.assert_allclose(scale[[4, 5, 10, 11]], 1.0)
        self.assertTrue((scale > 0.0).all())

    def test_pet_constraint_is_applied_before_spatial_minimum(self) -> None:
        t = 91
        steps = np.arange(t, dtype=np.float32)
        a = _agent(steps, np.zeros(t, dtype=np.float32))
        b = _agent(steps - 40.0, np.ones(t, dtype=np.float32))
        cfg = SoftPairConfig(max_pet_steps=30, max_spatial_distance_m=10.0)
        closest = find_constrained_closest_points(a, b, cfg)
        self.assertIsNotNone(closest)
        assert closest is not None
        step_a, step_b, distance = closest
        self.assertLessEqual(abs(step_a - step_b), 30)
        self.assertAlmostEqual(distance, np.sqrt(101.0), places=4)

    def test_first_arrival_is_at_index_19_in_sixty_step_sequence(self) -> None:
        t = 91
        steps = np.arange(t, dtype=np.float32)
        a = _agent(steps - 30.0, np.zeros(t, dtype=np.float32))
        b = _agent(np.zeros(t, dtype=np.float32), steps - 40.0, yaw=np.pi / 2.0)
        cfg = SoftPairConfig()
        sample = extract_soft_pair_sample(a, b, index_a=0, index_b=1, cfg=cfg)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.sequence.shape, (60, 12))
        self.assertEqual(sample.first_arrival_step, 30)
        self.assertEqual(sample.second_arrival_step, 40)
        self.assertEqual(sample.pet_steps, 10)
        # Conflict center is the origin; both arrival points are exactly there.
        np.testing.assert_allclose(sample.sequence[19, 0:2], 0.0, atol=1e-5)
        np.testing.assert_allclose(sample.sequence[29, 6:8], 0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
