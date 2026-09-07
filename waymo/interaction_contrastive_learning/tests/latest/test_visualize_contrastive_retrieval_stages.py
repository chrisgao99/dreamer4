import unittest

import numpy as np

from interaction_contrastive_learning.latest.visualize_contrastive_retrieval_stages import (
    choose_rms_examples,
    choose_stage_examples,
    decode_reason_bits,
    pairwise_ranking_accuracy,
    select_anchor_rows,
    trajectory_svg,
)


class ContrastiveRetrievalGalleryTest(unittest.TestCase):
    def test_stage_examples_show_positive_and_negative_extremes(self):
        similarities = {1: 0.8, 2: 0.2, 3: 0.7, 4: -0.4}
        selected = choose_stage_examples(
            np.asarray([1, 2]), np.asarray([3, 4]), similarities
        )
        self.assertEqual(
            selected,
            [
                ("most similar GT positive", 1),
                ("least similar GT positive", 2),
                ("hardest GT negative", 3),
                ("easiest GT negative", 4),
            ],
        )

    def test_rms_reference_uses_distance_extremes(self):
        selected = choose_rms_examples(
            np.asarray([10, 11, 12]),
            np.asarray([0.3, 0.1, 0.2]),
            np.asarray([20, 21]),
            np.asarray([1.4, 0.9]),
        )
        self.assertEqual(selected[0], ("nearest RMS positive", 11))
        self.assertEqual(selected[1], ("furthest stored RMS positive", 10))
        self.assertEqual(selected[2], ("closest relation negative", 21))
        self.assertEqual(selected[3], ("furthest stored relation negative", 20))

    def test_pairwise_ranking_accuracy_uses_physical_distance_order(self):
        self.assertEqual(
            pairwise_ranking_accuracy(
                np.asarray([0.9, 0.5, 0.1]), np.asarray([0.1, 0.2, 0.3])
            ),
            1.0,
        )
        self.assertEqual(
            pairwise_ranking_accuracy(
                np.asarray([0.1, 0.5, 0.9]), np.asarray([0.1, 0.2, 0.3])
            ),
            0.0,
        )

    def test_ten_anchor_selection_covers_rare_strata(self):
        manifest = np.arange(100, dtype=np.int64)
        strata = np.asarray([110] * 80 + [111] * 2 + [120] * 5 + [130] * 4 + [210] * 4 + [220] * 3 + [310] * 2)
        selected = select_anchor_rows(manifest, strata, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(set(strata[selected].tolist()), {110, 111, 120, 130, 210, 220, 310})

    def test_svg_marks_future_as_dashed_and_decodes_reasons(self):
        positions = np.zeros((60, 2, 2), dtype=np.float32)
        positions[:, 0, 0] = np.arange(60)
        positions[:, 1, 1] = np.arange(60)
        mask = np.ones((60, 2), dtype=bool)
        rendered = trajectory_svg(
            positions, mask, radius=80.0, title="example", subtitle="sample"
        )
        self.assertIn('stroke-dasharray="5 4"', rendered)
        self.assertIn("solid: history", rendered)
        self.assertEqual(
            decode_reason_bits(1 | 4), "order swap, future order opposite"
        )


if __name__ == "__main__":
    unittest.main()
