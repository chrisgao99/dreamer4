import unittest

import numpy as np

from interaction_contrastive_learning.latest.build_full_pair_rms_neighbors import (
    RmsConfig,
    align_trajectory_batch,
    build_exact_neighbours,
    exact_masked_rms,
    fit_robust_scaler,
    retrieval_descriptor,
)


class FullPairRmsNeighbourTest(unittest.TestCase):
    def test_fractional_event_alignment_removes_absolute_time_shift(self):
        trajectory = np.zeros((2, 2, 91, 6), dtype=np.float32)
        valid = np.ones((2, 2, 91), dtype=bool)
        primary = np.asarray([20.5, 40.5], dtype=np.float32)
        for sample in range(2):
            relative = np.arange(91, dtype=np.float32) - primary[sample]
            for agent in range(2):
                trajectory[sample, agent, :, 0] = relative + agent * 2.0
                trajectory[sample, agent, :, 2] = 1.0
                trajectory[sample, agent, :, 5] = 1.0
        aligned, mask = align_trajectory_batch(
            trajectory,
            valid,
            primary,
            np.arange(-3, 4, dtype=np.float32),
        )
        self.assertTrue(mask.all())
        np.testing.assert_allclose(aligned[0], aligned[1], atol=1e-6)

    def test_scaler_ignores_invalid_padding(self):
        aligned = np.zeros((2, 3, 2, 6), dtype=np.float32)
        mask = np.ones((2, 3, 2), dtype=bool)
        aligned[..., 5] = 1.0
        aligned[0, :, :, 0] = 1.0
        aligned[1, :, :, 0] = 3.0
        aligned[1, 2, :, 0] = 1e6
        mask[1, 2] = False
        median, iqr = fit_robust_scaler(aligned, mask)
        np.testing.assert_allclose(median[:, 0], 1.0)
        self.assertTrue((iqr[:, 0] >= 1.0).all())
        np.testing.assert_array_equal(median[:, 4:6], 0.0)
        np.testing.assert_array_equal(iqr[:, 4:6], 1.0)

    def test_exact_rms_uses_only_common_valid_values(self):
        normalized = np.zeros((3, 4, 2, 6), dtype=np.float32)
        mask = np.ones((3, 4, 2), dtype=bool)
        normalized[1, :, :, 0] = 1.0
        normalized[2, :, :, 0] = 100.0
        mask[2, 2:] = False
        distance, overlap = exact_masked_rms(
            normalized,
            mask,
            0,
            np.asarray([1, 2]),
            min_pair_overlap=0.75,
        )
        self.assertAlmostEqual(float(distance[0]), np.sqrt(2.0 / 12.0), places=6)
        self.assertTrue(np.isinf(distance[1]))
        np.testing.assert_allclose(overlap, [1.0, 0.5])

    def test_coarse_search_is_reranked_by_exact_rms_and_excludes_same_scene(self):
        normalized = np.zeros((6, 5, 2, 6), dtype=np.float32)
        for index, value in enumerate((0.0, 0.05, 0.10, 1.0, 2.0, 3.0)):
            normalized[index, :, :, 0] = value
            normalized[index, :, :, 5] = 1.0
        mask = np.ones((6, 5, 2), dtype=bool)
        descriptor = retrieval_descriptor(normalized, mask, coefficients=2, mask_weight=0.25)
        scenarios = np.asarray(["same", "same", "b", "c", "d", "e"])
        strata = np.full((6,), 110, dtype=np.int16)
        eligible = np.ones((6,), dtype=bool)
        cfg = RmsConfig(
            history_steps=2,
            future_steps=2,
            min_valid_fraction=1.0,
            min_pair_overlap=1.0,
            dct_coefficients=2,
            retrieval_candidates=5,
            retrieval_buffer=1,
            num_neighbours=3,
            query_chunk_size=6,
        )
        neighbours, _, _ = build_exact_neighbours(
            normalized,
            mask,
            descriptor,
            scenarios,
            strata,
            eligible,
            cfg,
        )
        # Index 1 is geometrically closest to anchor 0 but shares its scenario.
        self.assertEqual(int(neighbours["neighbor_indices"][0, 0]), 2)
        self.assertGreater(float(neighbours["rms_distances"][0, 1]), float(neighbours["rms_distances"][0, 0]))


if __name__ == "__main__":
    unittest.main()
