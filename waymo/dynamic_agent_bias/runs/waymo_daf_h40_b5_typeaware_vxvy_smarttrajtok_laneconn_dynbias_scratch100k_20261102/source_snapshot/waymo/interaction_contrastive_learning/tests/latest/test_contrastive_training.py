import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from interaction_contrastive_learning.latest.build_contrastive_training_cache import (
    REASON_FUTURE_ORDER_OPPOSITE,
    REASON_SWAP_DIFF,
    _relation_masks,
    select_duplicate_safe_positives,
)
from interaction_contrastive_learning.latest.train_interaction_contrastive import (
    ArbitraryPairContrastiveHead,
    InteractionContrastiveDataset,
    hard_contrastive_loss,
    hybrid_soft_contrastive_loss,
    select_stratified_validation_anchors,
    soft_contrastive_loss,
)


class ContrastiveTrainingTest(unittest.TestCase):
    def test_hard_dataset_requires_negatives_on_both_endpoints_and_keeps_nearest_positive(self):
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.npz"
            np.savez_compressed(
                cache_path,
                training_eligible_mask=np.ones((4,), dtype=bool),
                positive_indices=np.asarray(
                    [[1, -1], [2, 3], [1, -1], [1, -1]], dtype=np.int32
                ),
                negative_indices=np.asarray(
                    [[3, -1], [2, 3], [1, 3], [1, 2]], dtype=np.int32
                ),
            )
            dataset = InteractionContrastiveDataset(
                cache_path,
                mode="hard",
                history_steps=32,
                hard_negatives_per_side=2,
                soft_positives=1,
                soft_negatives=1,
                soft_beta=0.5,
                training=True,
            )
            self.assertNotIn(0, dataset.anchor_indices.tolist())
            dataset._load_scene = lambda sample_index: {"index": int(sample_index)}
            anchor_one_item = int(np.flatnonzero(dataset.anchor_indices == 1)[0])
            for _ in range(8):
                item = dataset[anchor_one_item]
                self.assertEqual(int(item["sample_indices"][1]), 2)

    def test_duplicate_safe_positive_skips_near_identical_cross_scene_slice(self):
        indices = np.asarray([[1, 2, -1], [0, 2, -1], [0, 1, -1]], dtype=np.int32)
        distances = np.asarray(
            [[1e-7, 0.12, np.inf], [1e-7, 0.15, np.inf], [0.12, 0.15, np.inf]],
            dtype=np.float32,
        )
        causal = np.ones((3,), dtype=bool)
        paths = np.asarray(["a.npz", "b.npz", "c.npz"])
        positives, positive_distances, original_ranks, duplicate_counts = (
            select_duplicate_safe_positives(
                indices,
                distances,
                causal,
                paths,
                duplicate_rms_threshold=0.02,
            )
        )
        self.assertEqual(int(positives[0, 0]), 2)
        self.assertAlmostEqual(float(positive_distances[0, 0]), 0.12, places=6)
        self.assertEqual(int(original_ranks[0, 0]), 2)
        self.assertEqual(int(duplicate_counts[0]), 1)

    def test_relation_mask_marks_swap_and_opposite_future_order(self):
        anchor = {
            "order_swap": np.asarray([False]),
            "order_outcome": np.asarray([1]),
            "gap_trend": np.asarray([1]),
            "distance_trend": np.asarray([1]),
        }
        candidates = {
            "order_swap": np.asarray([True, False]),
            "order_outcome": np.asarray([-1, 1]),
            "gap_trend": np.asarray([-1, 1]),
            "distance_trend": np.asarray([1, -1]),
        }
        tier1, tier2, reasons = _relation_masks(anchor, candidates)
        self.assertTrue(bool(tier1[0]))
        self.assertTrue(bool(tier2[0]))
        self.assertEqual(
            int(reasons[0]) & (REASON_SWAP_DIFF | REASON_FUTURE_ORDER_OPPOSITE),
            REASON_SWAP_DIFF | REASON_FUTURE_ORDER_OPPOSITE,
        )
        self.assertFalse(bool(tier1[1]))
        self.assertTrue(bool(tier2[1]))

    def test_arbitrary_pair_head_accepts_nonzero_first_slot(self):
        head = ArbitraryPairContrastiveHead(
            d_bottleneck=8,
            d_model=16,
            n_heads=4,
            n_latents=4,
            n_agents=6,
            embedding_dim=12,
            dropout=0.0,
            scale_pos_embeds=True,
        )
        z = torch.randn(3, 4, 8)
        embedding, pair_token = head(
            z,
            torch.tensor([2, 3, 4]),
            torch.tensor([5, 1, 0]),
        )
        self.assertEqual(tuple(embedding.shape), (3, 12))
        self.assertEqual(tuple(pair_token.shape), (3, 16))
        torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(3))
        self.assertEqual(embedding.dtype, torch.float32)

    def test_stratified_validation_subset_is_fixed_and_covers_strata(self):
        anchors = np.arange(20, dtype=np.int64)
        strata = np.asarray([110] * 12 + [120] * 6 + [310] * 2)
        first = select_stratified_validation_anchors(anchors, strata, 10)
        second = select_stratified_validation_anchors(anchors, strata, 10)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(set(strata[first].tolist()), {110, 120, 310})

    def test_hard_and_soft_losses_reward_correct_similarity_order(self):
        good = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        )
        bad = good.clone()
        bad[1] = torch.tensor([-1.0, 0.0])
        offsets = torch.tensor([0])
        good_hard, _ = hard_contrastive_loss(
            good, offsets, negatives_per_side=2, temperature=0.1
        )
        bad_hard, _ = hard_contrastive_loss(
            bad, offsets, negatives_per_side=2, temperature=0.1
        )
        self.assertLess(float(good_hard), float(bad_hard))

        soft_embeddings = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]]
        )
        target = torch.tensor([[0.8, 0.2, 0.0]])
        soft_loss, _ = soft_contrastive_loss(
            soft_embeddings,
            offsets,
            target,
            positives_per_anchor=2,
            negatives_per_anchor=1,
            temperature=0.1,
        )
        self.assertTrue(torch.isfinite(soft_loss))

    def test_hybrid_loss_rewards_separation_and_positive_rms_order(self):
        def unit_vector(cosine):
            return [cosine, float(np.sqrt(1.0 - cosine**2))]

        good_similarities = [0.98, 0.94, 0.90, 0.84, 0.78, 0.70, 0.62, 0.52]
        negative_similarities = [0.25, 0.10, -0.10, -0.30]
        good = torch.tensor(
            [[1.0, 0.0]]
            + [unit_vector(value) for value in good_similarities]
            + [unit_vector(value) for value in negative_similarities]
        )
        bad = torch.tensor(
            [[1.0, 0.0]]
            + [unit_vector(value) for value in reversed(good_similarities)]
            + [unit_vector(value) for value in negative_similarities]
        )
        weights = np.exp(-np.arange(8) / 3.0)
        target = torch.from_numpy(
            np.asarray(
                [np.concatenate((weights / weights.sum(), np.zeros(4)))],
                dtype=np.float32,
            )
        )
        kwargs = dict(
            positives_per_anchor=8,
            negatives_per_anchor=4,
            separation_temperature=0.07,
            rank_temperature=0.10,
            rank_relative_weight=0.2,
        )
        good_loss, good_metrics = hybrid_soft_contrastive_loss(
            good, torch.tensor([0]), target, **kwargs
        )
        bad_loss, bad_metrics = hybrid_soft_contrastive_loss(
            bad, torch.tensor([0]), target, **kwargs
        )
        self.assertLess(float(good_loss), float(bad_loss))
        self.assertEqual(float(good_metrics["positive_rank_accuracy"]), 1.0)
        self.assertLess(float(bad_metrics["positive_rank_accuracy"]), 0.1)
        self.assertEqual(float(good_metrics["separation_accuracy"]), 1.0)


if __name__ == "__main__":
    unittest.main()
