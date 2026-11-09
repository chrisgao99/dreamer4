import unittest

import numpy as np

from interaction_contrastive_learning.legacy.visualize_matched_samples import (
    related_samples,
    select_stratified_anchors,
)


class VisualAuditSelectionTest(unittest.TestCase):
    def test_stratified_selection_round_robins_groups(self):
        candidates = np.arange(8, dtype=np.int64)
        relation = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        response = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
        lead = np.asarray([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int16)
        selected = select_stratified_anchors(
            candidates,
            relation_index=relation,
            response_index=response,
            lead_steps=lead,
            count=4,
            seed=0,
        )
        groups = {(int(relation[i]), int(response[i]), int(lead[i])) for i in selected}
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(groups), 4)

    def test_related_samples_skips_padding_and_preserves_roles(self):
        matches = {
            "positive_indices": np.asarray([[2, -1]], dtype=np.int64),
            "positive_distances": np.asarray([[1.5, np.inf]], dtype=np.float32),
            "hard_negative_indices": np.asarray([[3, 4, -1]], dtype=np.int64),
            "hard_negative_distances": np.asarray([[2.5, 3.5, np.inf]], dtype=np.float32),
            "negative_indices": np.asarray([[5, -1]], dtype=np.int64),
            "negative_distances": np.asarray([[9.0, np.inf]], dtype=np.float32),
        }
        related = related_samples(0, matches)
        self.assertEqual(
            [(role, rank, index) for role, rank, index, _ in related],
            [
                ("anchor", 0, 0),
                ("positive", 1, 2),
                ("hard_negative", 1, 3),
                ("hard_negative", 2, 4),
                ("easy_negative", 1, 5),
            ],
        )


if __name__ == "__main__":
    unittest.main()
