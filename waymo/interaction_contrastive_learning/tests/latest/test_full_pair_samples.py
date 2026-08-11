import unittest

import numpy as np

from interaction_contrastive_learning.latest.full_pair_samples import (
    FullPairConfig,
    contact_components,
    continuous_path_intersections,
    extract_full_pair_sample,
)


def _agent() -> np.ndarray:
    result = np.zeros((91, 8), dtype=np.float32)
    result[:, 5] = 1.0
    result[:, 7] = 1.0
    return result


def _size(length: float = 4.8, width: float = 2.0) -> np.ndarray:
    return np.tile(np.asarray([[length, width]], dtype=np.float32), (91, 1))


class FullPairSampleTests(unittest.TestCase):
    def test_continuous_crossing_and_full_trajectory(self):
        a = _agent()
        b = _agent()
        steps = np.arange(91, dtype=np.float32)
        a[:, 0] = steps - 45.25
        a[:, 3] = 10.0
        b[:, 1] = steps - 65.75
        b[:, 4] = 10.0
        b[:, 6] = np.pi / 2.0
        intersections = continuous_path_intersections(a, b)
        self.assertTrue(intersections)
        primary = min(intersections, key=lambda item: abs(item.step_a - item.step_b))
        self.assertAlmostEqual(primary.step_a, 45.25, places=4)
        self.assertAlmostEqual(primary.step_b, 65.75, places=4)

        sample = extract_full_pair_sample(
            a, b, _size(), _size(),
            index_a=0, index_b=1, agent_id_a=10, agent_id_b=20,
            is_original_ooi_pair=False, cfg=FullPairConfig(),
        )
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.event_mode, "path_intersection")
        self.assertEqual(sample.trajectory.shape, (2, 91, 6))
        self.assertEqual(sample.valid_mask.shape, (2, 91))
        self.assertTrue(sample.valid_mask.all())
        self.assertGreater(sample.interval_end_first, sample.interval_start_first)
        self.assertGreater(sample.interval_end_second, sample.interval_start_second)

    def test_following_creates_long_contact_interval(self):
        a = _agent()
        b = _agent()
        steps = np.arange(91, dtype=np.float32)
        a[:, 0] = 0.5 * steps
        b[:, 0] = 0.5 * steps + 8.0
        a[:, 3] = b[:, 3] = 5.0
        components = contact_components(a, b, _size(), _size(), FullPairConfig())
        self.assertTrue(components)
        longest = max(components, key=lambda component: component.num_cells)
        self.assertGreater(longest.end_a - longest.start_a, 60.0)
        self.assertGreater(longest.end_b - longest.start_b, 60.0)
        self.assertGreater(longest.zone_pet_steps, 0.0)

    def test_ooi_without_contact_is_retained_with_mask(self):
        a = _agent()
        b = _agent()
        steps = np.arange(91, dtype=np.float32)
        a[:, 0] = b[:, 0] = steps
        b[:, 1] = 30.0
        a[80:, 5] = 0.0
        non_ooi = extract_full_pair_sample(
            a, b, _size(), _size(),
            index_a=0, index_b=1, agent_id_a=10, agent_id_b=20,
            is_original_ooi_pair=False, cfg=FullPairConfig(),
        )
        self.assertIsNone(non_ooi)
        ooi = extract_full_pair_sample(
            a, b, _size(), _size(),
            index_a=0, index_b=1, agent_id_a=10, agent_id_b=20,
            is_original_ooi_pair=True, cfg=FullPairConfig(),
        )
        self.assertIsNotNone(ooi)
        assert ooi is not None
        self.assertEqual(ooi.event_mode, "ooi_closest_fallback")
        self.assertEqual(int(ooi.valid_mask[0].sum()), 80)
        self.assertTrue(np.all(ooi.trajectory[0, 80:] == 0.0))


if __name__ == "__main__":
    unittest.main()
