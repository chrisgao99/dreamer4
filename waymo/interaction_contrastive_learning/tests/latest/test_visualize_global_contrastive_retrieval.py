import unittest

import numpy as np

from interaction_contrastive_learning.latest.visualize_global_contrastive_retrieval import (
    STAGE_ORDER,
    parse_anchor_indices,
    rank_unique_scenarios,
    select_candidate_rows,
)


class GlobalContrastiveRetrievalTest(unittest.TestCase):
    def test_stage_order_tracks_training_depth(self):
        self.assertEqual(
            STAGE_ORDER, ("raw_z", "reader_z", "hard", "hybrid")
        )

    def test_candidate_scope_uses_every_causally_encodable_history(self):
        cache = {
            "query_step": np.asarray([30, 31, 40, 50]),
            "causal_eligible_mask": np.asarray([True, True, False, True]),
            "training_eligible_mask": np.asarray([True, False, False, True]),
        }
        np.testing.assert_array_equal(
            select_candidate_rows(cache, history_steps=32, scope="history"),
            np.asarray([1, 2, 3]),
        )
        np.testing.assert_array_equal(
            select_candidate_rows(cache, history_steps=32, scope="causal"),
            np.asarray([1, 3]),
        )
        np.testing.assert_array_equal(
            select_candidate_rows(cache, history_steps=32, scope="training"),
            np.asarray([3]),
        )

    def test_ranking_excludes_anchor_scene_and_collapses_candidate_scenes(self):
        scenario_ids = np.asarray(["anchor", "anchor", "a", "a", "b", "c"])
        candidates = np.asarray([0, 1, 2, 3, 4, 5])
        scores = np.asarray([1.0, 0.99, 0.80, 0.95, 0.90, 0.70])
        selected = rank_unique_scenarios(
            anchor=0,
            candidate_rows=candidates,
            similarities=scores,
            scenario_ids=scenario_ids,
            top_k=3,
        )
        self.assertEqual([row["row"] for row in selected], [3, 4, 5])
        self.assertEqual([row["scenario_id"] for row in selected], ["a", "b", "c"])
        self.assertEqual([row["rank"] for row in selected], [1, 2, 3])

    def test_anchor_parser_preserves_explicit_order(self):
        np.testing.assert_array_equal(
            parse_anchor_indices("5, 60,97"), np.asarray([5, 60, 97])
        )
        with self.assertRaises(ValueError):
            parse_anchor_indices("5,5")


if __name__ == "__main__":
    unittest.main()
