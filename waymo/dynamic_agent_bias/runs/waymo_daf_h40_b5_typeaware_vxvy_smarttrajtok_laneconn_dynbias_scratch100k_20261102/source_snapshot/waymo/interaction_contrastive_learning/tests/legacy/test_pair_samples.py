from __future__ import annotations

import unittest

import numpy as np

from interaction_contrastive_learning.legacy.build_matched_samples import build_matches
from interaction_contrastive_learning.legacy.pair_samples import (
    RELATION_NAMES,
    RESPONSE_NAMES,
    SampleConfig,
    build_scene_samples,
    detect_pair_event,
    extract_query_history,
)


def crossing_scene() -> tuple[np.ndarray, np.ndarray]:
    t = 70
    agents = np.zeros((2, t, 8), dtype=np.float32)
    agents[:, :, 5] = 1.0
    agents[:, :, 7] = 1.0
    steps = np.arange(t, dtype=np.float32)
    # Focus reaches the conflict point at step 35; other reaches it at step 40.
    agents[0, :, 0] = steps - 35.0
    agents[0, :, 2] = 10.0
    agents[0, :, 3] = 10.0
    agents[0, :, 6] = 0.0
    agents[1, :, 1] = steps - 40.0
    agents[1, :, 2] = 10.0
    agents[1, :, 4] = 10.0
    agents[1, :, 6] = np.pi / 2.0
    return agents, np.ones((2,), dtype=bool)


class PairSampleTests(unittest.TestCase):
    def test_crossing_event_and_event_relative_query(self) -> None:
        agents, mask = crossing_scene()
        cfg = SampleConfig(history_steps=20, lead_steps=(10,), event_search_start=10)
        event = detect_pair_event(agents[0], agents[1], candidate_index=1, cfg=cfg)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(RELATION_NAMES[event.relation_index], "crossing_or_oncoming_conflict")
        self.assertEqual(event.event_step, 35)
        self.assertAlmostEqual(event.pet_s, 0.5, places=5)

        samples = build_scene_samples(agents, mask, cfg)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].query_step, 25)
        self.assertEqual(RESPONSE_NAMES[samples[0].response_index], "goes_first")
        self.assertTrue(samples[0].eligible)

    def test_query_frame_history_is_causal_and_fixed(self) -> None:
        agents, _ = crossing_scene()
        history = extract_query_history(agents[0], agents[1], query_step=25, history_steps=20, dt=0.1)
        self.assertIsNotNone(history)
        assert history is not None
        self.assertEqual(history.shape, (20, 11))
        # At query=25, focus=(-10,0), other=(0,-15), in focus heading 0.
        np.testing.assert_allclose(history[-1, :2], np.asarray([10.0, -15.0]), atol=1e-5)

    def test_matching_finds_same_response_positive_and_different_response_hard_negative(self) -> None:
        vectors = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [5.0, 5.0],
                [5.1, 5.0],
            ],
            dtype=np.float32,
        )
        result, stats = build_matches(
            vectors,
            eligible=np.ones(6, dtype=bool),
            scene_ids=np.asarray(["a", "b", "c", "d", "e", "f"]),
            lead_steps=np.full(6, 10),
            relation=np.asarray([2, 2, 2, 2, 3, 3]),
            response=np.asarray([0, 0, 1, 1, 0, 0]),
            focus_type=np.ones(6),
            candidate_type=np.ones(6),
            max_positives=1,
            max_hard_negatives=1,
            max_negatives=1,
            search_k=5,
            caliper_quantile=1.0,
            caliper_multiplier=4.0,
            seed=0,
        )
        self.assertGreaterEqual(int(result["positive_indices"][0, 0]), 0)
        self.assertGreaterEqual(int(result["hard_negative_indices"][0, 0]), 0)
        self.assertGreater(int(stats["trainable_anchors"]), 0)


if __name__ == "__main__":
    unittest.main()
