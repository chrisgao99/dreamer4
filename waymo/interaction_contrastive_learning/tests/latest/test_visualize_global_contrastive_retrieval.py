import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from interaction_contrastive_learning.latest.visualize_global_contrastive_retrieval import (
    STAGE_ORDER,
    build,
    build_argparser,
    exact_rms_reference,
    parse_anchor_indices,
    rank_unique_scenarios,
    select_candidate_rows,
    retrieval_metrics,
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

    @staticmethod
    def reference_fixture():
        # Rows 1/2/3/4 would look close but are respectively same scene,
        # wrong stratum, a near-duplicate, and insufficient overlap.
        values = np.asarray([0, .03, .04, .01, .05, .30, .20, .25, .50], dtype=np.float32)
        n = len(values)
        cache = {
            "sample_index": np.arange(n),
            "scenario_id": np.asarray(["anchor", "anchor", "wrong", "dup", "short", "a", "a", "b", "c"]),
            "source_path": np.asarray([f"scene_{row}.npz" for row in range(n)]),
            "first_agent_id": np.ones(n, dtype=int),
            "second_agent_id": np.full(n, 2),
            "stratum_key": np.asarray([110, 110, 210, 110, 110, 110, 110, 110, 110]),
            "query_step": np.full(n, 31),
            "positive_indices": np.full((n, 1), -1),
            "negative_indices": np.full((n, 1), -1),
        }
        features = {key: cache[key].copy() for key in (
            "sample_index", "scenario_id", "source_path", "first_agent_id", "second_agent_id", "stratum_key"
        )}
        mask = np.ones((n, 60, 2), dtype=bool)
        mask[4, 20:] = False
        features.update({
            "normalized_sequence": np.broadcast_to(values[:, None, None, None], (n, 60, 2, 6)).copy(),
            "aligned_mask": mask,
            "normalization_median": np.zeros((2, 6)),
            "normalization_iqr": np.ones((2, 6)),
            "time_offsets": np.arange(-19, 41),
        })
        return cache, features

    def test_rms_reference_exhaustive_filtering_and_best_pair_per_scene(self):
        cache, features = self.reference_fixture()
        reference = exact_rms_reference(
            anchor=0, candidate_rows=np.asarray([8, 5, 1, 2, 4, 3, 6, 7, 0]),
            cache=cache, features=features, top_k=5,
        )
        self.assertEqual([r["row"] for r in reference["results"]], [6, 7, 8])
        np.testing.assert_allclose([r["exact_rms"] for r in reference["results"]], [.2, .25, .5])
        self.assertEqual(reference["eligible_pair_rows"], 4)
        self.assertEqual(reference["eligible_unique_scenarios"], 3)
        self.assertTrue(all("cosine" not in r for r in reference["results"]))
        # The reference must not reach outside the saved model candidate corpus.
        limited = exact_rms_reference(
            anchor=0, candidate_rows=np.asarray([5, 7]), cache=cache, features=features,
        )
        self.assertEqual([r["row"] for r in limited["results"]], [7, 5])

    def test_rms_reference_handles_no_comparable_candidates(self):
        cache, features = self.reference_fixture()
        reference = exact_rms_reference(
            anchor=0, candidate_rows=np.arange(5), cache=cache, features=features,
        )
        self.assertEqual(reference["results"], [])
        self.assertEqual(reference["eligible_pair_rows"], 0)

    def test_rms_reference_promotes_compressed_features_before_squaring(self):
        cache, features = self.reference_fixture()
        features["normalized_sequence"][8] = 300.0
        features["normalized_sequence"] = features["normalized_sequence"].astype(np.float16)
        reference = exact_rms_reference(
            anchor=0, candidate_rows=np.asarray([8]), cache=cache, features=features,
        )
        self.assertEqual(len(reference["results"]), 1)
        self.assertAlmostEqual(reference["results"][0]["exact_rms"], 300.0)

    def test_cpu_refresh_preserves_model_retrievals_and_scores(self):
        cache, features = self.reference_fixture()
        reference = exact_rms_reference(
            anchor=0, candidate_rows=np.arange(9), cache=cache, features=features,
        )
        result = {**reference["results"][0], "cosine": .9}
        metrics = retrieval_metrics(anchor=0, results=[result], annotations={6: result}, cache=cache)
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            np.savez(directory / "cache.npz", **cache)
            np.savez(directory / "features.npz", **features)
            np.savez(directory / "retrieval_scores.npz", anchor_rows=[0], candidate_rows=np.arange(9),
                     **{stage: np.zeros((1, 9), dtype=np.float32) for stage in STAGE_ORDER})
            scores_before = (directory / "retrieval_scores.npz").read_bytes()
            manifest = {
                "contrastive_cache": str(directory / "cache.npz"),
                "rms_features": str(directory / "features.npz"),
                "candidate_pair_rows": 9, "candidate_unique_scenarios": 7,
                "anchors": [{
                    "anchor_row": 0, "scenario_id": "anchor", "stratum": "vehicle → vehicle · contact",
                    "query_step": 31, "page": "anchor_01.html",
                    "retrievals": {stage: [result] for stage in STAGE_ORDER},
                    "metrics": {stage: metrics for stage in STAGE_ORDER},
                }],
            }
            manifest_path = directory / "gallery_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            args = build_argparser().parse_args(["--refresh_rms_reference", "--output_dir", str(directory)])
            with patch(
                "interaction_contrastive_learning.latest.visualize_global_contrastive_retrieval.score_encoder_group",
                side_effect=AssertionError("Refreshing an existing report must not run model inference"),
            ):
                build(args)
            updated = json.loads(manifest_path.read_text())
            self.assertEqual(updated["anchors"][0]["retrievals"], manifest["anchors"][0]["retrievals"])
            self.assertEqual(updated["anchors"][0]["metrics"], manifest["anchors"][0]["metrics"])
            self.assertEqual((directory / "retrieval_scores.npz").read_bytes(), scores_before)
            page = (directory / "anchor_01.html").read_text()
            self.assertLess(page.index('id="rms-reference"'), page.index("Raw z: base tokenizer</h2>"))
            self.assertEqual(page.count('class="candidate reference"'), 3)
            self.assertIn("Exact RMS", (directory / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
