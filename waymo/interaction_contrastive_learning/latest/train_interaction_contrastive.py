"""Two-stage hard or soft interaction-contrastive tokenizer fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

WAYMO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT / "core", WAYMO_ROOT / "training" / "tokenizer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer4.model import add_sinusoidal_positions
from waymo.core.vector_tokenizer_decoder import vector_tokenizer_reconstruction_loss
from waymo.training.tokenizer.train_waymo_vector_tokenizer import build_argparser as tokenizer_argparser
from waymo.training.tokenizer.train_waymo_vector_tokenizer import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed_everything(int(info.seed))


class ArbitraryPairContrastiveHead(nn.Module):
    """Read an arbitrary ordered agent pair from query-time Z only."""

    def __init__(
        self,
        *,
        d_bottleneck: int,
        d_model: int,
        n_heads: int,
        n_latents: int,
        n_agents: int,
        embedding_dim: int,
        dropout: float,
        scale_pos_embeds: bool,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_latents = int(n_latents)
        self.n_agents = int(n_agents)
        self.scale_pos_embeds = bool(scale_pos_embeds)
        self.z_proj = nn.Linear(d_bottleneck, d_model)
        self.slot_queries = nn.Parameter(torch.empty(n_agents, d_model))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.pair_query_mlp = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )

    @torch.no_grad()
    def init_from_decoder_queries(self, decoder_queries: torch.Tensor) -> None:
        count = min(len(self.slot_queries), len(decoder_queries))
        self.slot_queries[:count].copy_(
            decoder_queries[:count].detach().to(
                device=self.slot_queries.device, dtype=self.slot_queries.dtype
            )
        )

    def forward(
        self,
        z_current: torch.Tensor,
        first_slots: torch.Tensor,
        second_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_current.dim() != 3:
            raise ValueError(
                f"Expected query-time z (B,N_latents,D), got {tuple(z_current.shape)}"
            )
        if z_current.shape[1] != self.n_latents:
            raise ValueError(
                f"Expected {self.n_latents} latent tokens, got {z_current.shape[1]}"
            )
        if bool((first_slots < 0).any()) or bool((second_slots < 0).any()):
            raise ValueError("Pair slots must be non-negative")
        first = self.slot_queries[first_slots]
        second = self.slot_queries[second_slots]
        pair_query = self.pair_query_mlp(
            torch.cat([first, second, first * second, (first - second).abs()], dim=-1)
        )
        memory = torch.tanh(self.z_proj(z_current))
        memory = add_sinusoidal_positions(
            memory[:, None, :, :], self.scale_pos_embeds
        )[:, 0]
        attended, _ = self.attn(
            query=pair_query[:, None, :], key=memory, value=memory, need_weights=False
        )
        pair_token = self.norm1(pair_query + attended[:, 0])
        pair_token = self.norm2(pair_token + self.ff(pair_token))
        # Keep normalization and all downstream cosine/log-softmax operations
        # in FP32.  The previous BF16 soft run became non-finite late in Stage
        # B; a larger epsilon also protects against a near-zero projector norm.
        embedding = F.normalize(
            self.projector(pair_token).float(), dim=-1, eps=1e-6
        )
        return embedding, pair_token


def hard_contrastive_loss(
    embeddings: torch.Tensor,
    group_offsets: torch.Tensor,
    *,
    negatives_per_side: int,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = []
    positive_cosines = []
    negative_cosines = []
    for offset_value in group_offsets.tolist():
        offset = int(offset_value)
        anchor = embeddings[offset]
        positive = embeddings[offset + 1]
        anchor_negatives = embeddings[offset + 2 : offset + 2 + negatives_per_side]
        positive_negatives = embeddings[
            offset + 2 + negatives_per_side : offset + 2 + 2 * negatives_per_side
        ]
        anchor_logits = torch.cat(
            [(anchor * positive).sum()[None], anchor_negatives @ anchor], dim=0
        ) / float(temperature)
        positive_logits = torch.cat(
            [(positive * anchor).sum()[None], positive_negatives @ positive], dim=0
        ) / float(temperature)
        target = torch.zeros((1,), device=embeddings.device, dtype=torch.long)
        losses.extend(
            [
                F.cross_entropy(anchor_logits[None], target),
                F.cross_entropy(positive_logits[None], target),
            ]
        )
        positive_cosines.append((anchor * positive).sum())
        negative_cosines.extend((anchor_negatives @ anchor).unbind())
        negative_cosines.extend((positive_negatives @ positive).unbind())
    loss = torch.stack(losses).mean()
    pos = torch.stack(positive_cosines).mean()
    neg = torch.stack(negative_cosines).mean()
    return loss, {
        "positive_cosine": pos.detach(),
        "negative_cosine": neg.detach(),
        "cosine_margin": (pos - neg).detach(),
    }


def soft_contrastive_loss(
    embeddings: torch.Tensor,
    group_offsets: torch.Tensor,
    target_probabilities: torch.Tensor,
    *,
    positives_per_anchor: int,
    negatives_per_anchor: int,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = []
    positive_cosines = []
    negative_cosines = []
    candidate_count = positives_per_anchor + negatives_per_anchor
    for group, offset_value in enumerate(group_offsets.tolist()):
        offset = int(offset_value)
        anchor = embeddings[offset]
        candidates = embeddings[offset + 1 : offset + 1 + candidate_count]
        logits = candidates @ anchor / float(temperature)
        q = target_probabilities[group, :candidate_count].to(logits.dtype)
        losses.append(-(q * F.log_softmax(logits, dim=0)).sum())
        positive_cosines.extend((candidates[:positives_per_anchor] @ anchor).unbind())
        negative_cosines.extend((candidates[positives_per_anchor:] @ anchor).unbind())
    loss = torch.stack(losses).mean()
    pos = torch.stack(positive_cosines).mean()
    neg = torch.stack(negative_cosines).mean()
    return loss, {
        "positive_cosine": pos.detach(),
        "negative_cosine": neg.detach(),
        "cosine_margin": (pos - neg).detach(),
    }


def hybrid_soft_contrastive_loss(
    embeddings: torch.Tensor,
    group_offsets: torch.Tensor,
    target_probabilities: torch.Tensor,
    *,
    positives_per_anchor: int,
    negatives_per_anchor: int,
    separation_temperature: float,
    rank_temperature: float,
    rank_relative_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Hard outcome separation plus RMS ordering within positive neighbours."""

    separation_losses = []
    rank_kls = []
    rank_cross_entropies = []
    target_entropies = []
    nearest_cosines = []
    furthest_cosines = []
    positive_cosines = []
    negative_cosines = []
    separation_accuracies = []
    ranking_accuracies = []
    for group, offset_value in enumerate(group_offsets.tolist()):
        offset = int(offset_value)
        anchor = embeddings[offset].float()
        positives = embeddings[
            offset + 1 : offset + 1 + positives_per_anchor
        ].float()
        negatives = embeddings[
            offset
            + 1
            + positives_per_anchor : offset
            + 1
            + positives_per_anchor
            + negatives_per_anchor
        ].float()
        positive_similarity = positives @ anchor
        negative_similarity = negatives @ anchor

        separation_logits = torch.cat(
            (positive_similarity[:1], negative_similarity), dim=0
        ) / float(separation_temperature)
        separation_losses.append(
            F.cross_entropy(
                separation_logits[None],
                torch.zeros((1,), device=embeddings.device, dtype=torch.long),
            )
        )

        q = target_probabilities[group, :positives_per_anchor].float()
        q = q / q.sum().clamp_min(1e-8)
        rank_log_probability = F.log_softmax(
            positive_similarity / float(rank_temperature), dim=0
        )
        rank_cross_entropy = -(q * rank_log_probability).sum()
        target_entropy = -(q * q.clamp_min(1e-8).log()).sum()
        rank_kl = (rank_cross_entropy - target_entropy).clamp_min(0.0)
        rank_cross_entropies.append(rank_cross_entropy)
        target_entropies.append(target_entropy)
        rank_kls.append(rank_kl)

        nearest_cosines.append(positive_similarity[0])
        furthest_cosines.append(positive_similarity[-1])
        positive_cosines.extend(positive_similarity.unbind())
        negative_cosines.extend(negative_similarity.unbind())
        separation_accuracies.append(
            (positive_similarity[0] > negative_similarity.max()).float()
        )
        if positives_per_anchor > 1:
            comparisons = positive_similarity[:-1, None] > positive_similarity[None, 1:]
            upper_triangle = torch.triu(
                torch.ones_like(comparisons, dtype=torch.bool), diagonal=0
            )
            ranking_accuracies.append(comparisons[upper_triangle].float().mean())

    separation = torch.stack(separation_losses).mean()
    rank_kl = torch.stack(rank_kls).mean()
    total = separation + float(rank_relative_weight) * rank_kl
    nearest = torch.stack(nearest_cosines).mean()
    furthest = torch.stack(furthest_cosines).mean()
    positive = torch.stack(positive_cosines).mean()
    negative = torch.stack(negative_cosines).mean()
    metrics = {
        "loss_separation": separation.detach(),
        "loss_rank_kl": rank_kl.detach(),
        "loss_rank_cross_entropy": torch.stack(rank_cross_entropies).mean().detach(),
        "target_rank_entropy": torch.stack(target_entropies).mean().detach(),
        "nearest_positive_cosine": nearest.detach(),
        "furthest_positive_cosine": furthest.detach(),
        "positive_cosine": positive.detach(),
        "negative_cosine": negative.detach(),
        "cosine_margin": (nearest - negative).detach(),
        "positive_cosine_drop": (nearest - furthest).detach(),
        "separation_accuracy": torch.stack(separation_accuracies).mean().detach(),
    }
    if ranking_accuracies:
        metrics["positive_rank_accuracy"] = (
            torch.stack(ranking_accuracies).mean().detach()
        )
    return total, metrics


def _tensor_scene_from_npz(
    path: str,
    *,
    query_step: int,
    history_steps: int,
    first_agent_id: int,
    second_agent_id: int,
) -> dict[str, torch.Tensor]:
    start = int(query_step) - int(history_steps) + 1
    if start < 0:
        raise ValueError(f"query_step={query_step} lacks {history_steps} causal steps")
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["agent_ids"], dtype=np.int64)
        first_matches = np.flatnonzero(ids == int(first_agent_id))
        second_matches = np.flatnonzero(ids == int(second_agent_id))
        if len(first_matches) != 1 or len(second_matches) != 1:
            raise ValueError(
                f"Cannot map pair ids ({first_agent_id}, {second_agent_id}) exactly once in {path}"
            )
        end = int(query_step) + 1
        return {
            "agents": torch.from_numpy(np.asarray(data["agents"][:, start:end])).float(),
            "agent_mask": torch.from_numpy(np.asarray(data["agent_mask"])).bool(),
            "map_polylines": torch.from_numpy(np.asarray(data["map_polylines"])).float(),
            "map_mask": torch.from_numpy(np.asarray(data["map_mask"])).bool(),
            "lights": torch.from_numpy(np.asarray(data["lights"][start:end])).float(),
            "light_mask": torch.from_numpy(np.asarray(data["light_mask"][start:end])).bool(),
            "first_slot": torch.tensor(int(first_matches[0]), dtype=torch.long),
            "second_slot": torch.tensor(int(second_matches[0]), dtype=torch.long),
        }


class InteractionContrastiveDataset(Dataset):
    def __init__(
        self,
        cache_path: Path,
        *,
        mode: str,
        history_steps: int,
        hard_negatives_per_side: int,
        soft_positives: int,
        soft_negatives: int,
        soft_beta: float,
        training: bool,
        soft_sigma_by_stratum: dict[int, float] | None = None,
        soft_sigma_floor: float = 0.02,
    ):
        with np.load(cache_path, allow_pickle=False) as cache:
            self.arrays = {key: np.asarray(cache[key]) for key in cache.files}
        self.mode = str(mode)
        self.history_steps = int(history_steps)
        self.hard_negatives_per_side = int(hard_negatives_per_side)
        self.soft_positives = int(soft_positives)
        self.soft_negatives = int(soft_negatives)
        self.soft_beta = float(soft_beta)
        self.training = bool(training)
        self.soft_sigma_floor = float(soft_sigma_floor)
        eligible = self.arrays["training_eligible_mask"].astype(bool)
        positives = self.arrays["positive_indices"]
        negatives = self.arrays["negative_indices"]
        self.endpoint_ok = None
        if mode == "hard":
            # Symmetric InfoNCE requires certified negatives for both endpoints.
            endpoint_ok = eligible & ((negatives >= 0).sum(axis=1) >= hard_negatives_per_side)
            self.endpoint_ok = endpoint_ok
            has_endpoint_positive = np.asarray(
                [
                    any(endpoint_ok[int(candidate)] for candidate in row if int(candidate) >= 0)
                    for row in positives
                ],
                dtype=bool,
            )
            # Both sides of the symmetric hard loss need their own certified
            # negative set.  Do not retain an anchor merely because its
            # positive endpoint is valid.
            eligible = endpoint_ok & has_endpoint_positive
        elif mode in ("soft", "hybrid"):
            eligible &= (positives >= 0).sum(axis=1) >= soft_positives
            eligible &= (negatives >= 0).sum(axis=1) >= soft_negatives
        else:
            raise ValueError(f"Unknown contrastive mode: {mode}")
        self.anchor_indices = np.flatnonzero(eligible).astype(np.int64)
        if not len(self.anchor_indices):
            raise RuntimeError(f"No eligible anchors in {cache_path} for mode={mode}")
        if mode == "hybrid":
            if soft_sigma_by_stratum is None:
                self.soft_sigma_by_stratum = self._estimate_soft_sigmas()
            else:
                self.soft_sigma_by_stratum = {
                    int(key): float(value)
                    for key, value in soft_sigma_by_stratum.items()
                }
        else:
            self.soft_sigma_by_stratum = {}

    def __len__(self) -> int:
        return len(self.anchor_indices)

    def _valid_row(self, key: str, anchor: int) -> np.ndarray:
        row = self.arrays[key][anchor]
        return row[row >= 0].astype(np.int64)

    def _sample(self, values: np.ndarray, count: int) -> np.ndarray:
        if len(values) < count:
            raise ValueError(f"Cannot sample {count} values from pool of {len(values)}")
        if not self.training:
            return values[:count]
        return np.random.choice(values, size=count, replace=False)

    def _soft_positive_offsets(self, count: int) -> np.ndarray:
        # Evenly cover the sorted positive neighbourhood. This is equivalent to
        # stratified sampling across the top-32 after duplicate removal.
        if count < self.soft_positives:
            raise ValueError(
                f"Need {self.soft_positives} soft positives, only {count} available"
            )
        boundaries = np.linspace(0, count, self.soft_positives + 1).astype(np.int64)
        selected = []
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            high = max(high, low + 1)
            if self.training:
                selected.append(int(np.random.randint(low, min(high, count))))
            else:
                selected.append(int(min(low, count - 1)))
        return np.asarray(selected, dtype=np.int64)

    def _hybrid_positive_offsets(self, count: int) -> np.ndarray:
        """Always retain the nearest positive, then cover the remaining range."""

        if count < self.soft_positives:
            raise ValueError(
                f"Need {self.soft_positives} hybrid positives, only {count} available"
            )
        if self.soft_positives == 1:
            return np.zeros((1,), dtype=np.int64)
        boundaries = np.linspace(
            1, count, self.soft_positives, dtype=np.int64
        )
        selected = [0]
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            high = max(high, low + 1)
            if self.training:
                selected.append(int(np.random.randint(low, min(high, count))))
            else:
                selected.append(int(min(low, count - 1)))
        return np.asarray(selected, dtype=np.int64)

    def _estimate_soft_sigmas(self) -> dict[int, float]:
        """Estimate train-only stratum scales from the top-neighbour RMS spread."""

        distances = self.arrays["positive_rms_distances"]
        spreads = []
        strata = []
        for anchor in self.anchor_indices.tolist():
            row = distances[int(anchor)]
            row = row[np.isfinite(row)]
            if len(row) < self.soft_positives:
                continue
            spreads.append(max(float(row[-1] - row[0]), self.soft_sigma_floor))
            strata.append(int(self.arrays["stratum_key"][int(anchor)]))
        spread_array = np.asarray(spreads, dtype=np.float64)
        stratum_array = np.asarray(strata, dtype=np.int64)
        global_sigma = max(float(np.median(spread_array)), self.soft_sigma_floor)
        result = {-1: global_sigma}
        for stratum in np.unique(stratum_array):
            values = spread_array[stratum_array == stratum]
            result[int(stratum)] = max(
                float(np.median(values)), self.soft_sigma_floor
            )
        return result

    def _load_scene(self, sample_index: int) -> dict[str, torch.Tensor]:
        return _tensor_scene_from_npz(
            str(self.arrays["source_path"][sample_index]),
            query_step=int(self.arrays["query_step"][sample_index]),
            history_steps=self.history_steps,
            first_agent_id=int(self.arrays["first_agent_id"][sample_index]),
            second_agent_id=int(self.arrays["second_agent_id"][sample_index]),
        )

    def __getitem__(self, item: int) -> dict[str, Any]:
        anchor = int(self.anchor_indices[item])
        positive_pool = self._valid_row("positive_indices", anchor)
        if self.mode == "hard":
            positive_pool = positive_pool[self.endpoint_ok[positive_pool]]
            # Hard InfoNCE uses the closest non-duplicate exact-RMS neighbour,
            # not a random member of the top-32 neighbourhood.
            positive = int(positive_pool[0])
            anchor_negatives = self._sample(
                self._valid_row("negative_indices", anchor), self.hard_negatives_per_side
            )
            positive_negatives = self._sample(
                self._valid_row("negative_indices", positive), self.hard_negatives_per_side
            )
            sample_indices = np.concatenate(
                ([anchor, positive], anchor_negatives, positive_negatives)
            ).astype(np.int64)
            target_probabilities = np.empty((0,), dtype=np.float32)
        else:
            valid_count = len(positive_pool)
            offsets = (
                self._hybrid_positive_offsets(valid_count)
                if self.mode == "hybrid"
                else self._soft_positive_offsets(valid_count)
            )
            positives = positive_pool[offsets]
            positive_distance_row = self.arrays["positive_rms_distances"][anchor]
            positive_distance_pool = positive_distance_row[np.isfinite(positive_distance_row)]
            sampled_distances = positive_distance_pool[offsets].astype(np.float32)
            negatives = self._sample(
                self._valid_row("negative_indices", anchor), self.soft_negatives
            )
            d_first = float(positive_distance_pool[0])
            if self.mode == "hybrid":
                stratum = int(self.arrays["stratum_key"][anchor])
                sigma = self.soft_sigma_by_stratum.get(
                    stratum, self.soft_sigma_by_stratum[-1]
                )
                scaled_distance = (sampled_distances - d_first) / max(sigma, 1e-6)
            else:
                d_edge = float(positive_distance_pool[-1])
                scaled_distance = (sampled_distances - d_first) / max(
                    (d_edge - d_first) * self.soft_beta, 1e-6
                )
            weights = np.exp(-scaled_distance).astype(np.float32)
            target_probabilities = np.concatenate(
                (weights / weights.sum(), np.zeros((self.soft_negatives,), dtype=np.float32))
            )
            sample_indices = np.concatenate(([anchor], positives, negatives)).astype(np.int64)
        return {
            "scenes": [self._load_scene(int(index)) for index in sample_indices],
            "sample_indices": torch.from_numpy(sample_indices),
            "target_probabilities": torch.from_numpy(target_probabilities),
        }


def select_stratified_validation_anchors(
    anchor_indices: np.ndarray,
    stratum_key: np.ndarray,
    count: int,
) -> np.ndarray:
    """Return a fixed proportional, within-stratum evenly spaced subset."""

    anchors = np.asarray(anchor_indices, dtype=np.int64)
    if count <= 0 or count >= len(anchors):
        return anchors.copy()
    anchor_strata = np.asarray(stratum_key)[anchors]
    unique, counts = np.unique(anchor_strata, return_counts=True)
    ideal = count * counts.astype(np.float64) / counts.sum()
    allocation = np.floor(ideal).astype(np.int64)
    if count >= len(unique):
        allocation[(counts > 0) & (allocation == 0)] = 1
    allocation = np.minimum(allocation, counts)
    while allocation.sum() > count:
        candidates = np.flatnonzero(allocation > (1 if count >= len(unique) else 0))
        if not len(candidates):
            break
        index = int(candidates[np.argmax(allocation[candidates] - ideal[candidates])])
        allocation[index] -= 1
    while allocation.sum() < count:
        candidates = np.flatnonzero(allocation < counts)
        if not len(candidates):
            break
        index = int(candidates[np.argmax(ideal[candidates] - allocation[candidates])])
        allocation[index] += 1
    selected = []
    for stratum, number in zip(unique.tolist(), allocation.tolist()):
        group = anchors[anchor_strata == stratum]
        offsets = np.linspace(0, len(group) - 1, int(number), dtype=np.int64)
        selected.extend(group[offsets].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def contrastive_collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    flat_scenes = [scene for item in items for scene in item["scenes"]]
    tensor_keys = (
        "agents",
        "agent_mask",
        "map_polylines",
        "map_mask",
        "lights",
        "light_mask",
        "first_slot",
        "second_slot",
    )
    batch = {
        key: torch.stack([scene[key] for scene in flat_scenes], dim=0)
        for key in tensor_keys
    }
    group_sizes = [len(item["scenes"]) for item in items]
    offsets = np.cumsum([0] + group_sizes[:-1]).astype(np.int64)
    batch["group_offsets"] = torch.from_numpy(offsets)
    batch["sample_indices"] = torch.cat([item["sample_indices"] for item in items])
    if items[0]["target_probabilities"].numel():
        batch["target_probabilities"] = torch.stack(
            [item["target_probabilities"] for item in items], dim=0
        )
    else:
        batch["target_probabilities"] = torch.empty((len(items), 0), dtype=torch.float32)
    return batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, argparse.Namespace):
        return vars(value)
    return dict(value) if isinstance(value, dict) else {}


def load_baseline_tokenizer(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_args = _as_dict(checkpoint.get("args", {}))
    defaults = vars(tokenizer_argparser().parse_args([]))
    defaults.update(saved_args)
    model_args = argparse.Namespace(**defaults)
    state = checkpoint["model"]
    n_agents = int(state["decoder.agent_queries"].shape[0])
    n_lights = int(state["decoder.light_queries"].shape[0])
    tokenizer = build_model(model_args, n_agents=n_agents, n_lights=n_lights)
    tokenizer.load_state_dict(state, strict=True)
    tokenizer.to(device)
    return tokenizer, model_args, checkpoint


def configure_trainable_parameters(
    tokenizer: nn.Module,
    *,
    encoder_unfreeze_blocks: int,
) -> list[nn.Parameter]:
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    encoder = tokenizer.encoder
    if encoder_unfreeze_blocks <= 0:
        return []
    for layer in encoder.layers[-encoder_unfreeze_blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for module_name in ("bottleneck_proj", "bottleneck_norm"):
        module = getattr(encoder, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return [parameter for parameter in encoder.parameters() if parameter.requires_grad]


def reconstruction_loss(tokenizer, encoder_output, batch, anchor_indices, model_args):
    decoder_kwargs = {}
    if getattr(tokenizer.decoder, "use_agent_tokens", False):
        decoder_kwargs["encoder_agent_tokens"] = encoder_output.agent_tokens[anchor_indices]
        agents_btkf = batch["agents"][anchor_indices].transpose(1, 2).contiguous()
        decoder_kwargs["encoder_agent_mask"] = (
            (agents_btkf[..., 5] > 0.5)
            & batch["agent_mask"][anchor_indices, None, :]
        )
    if getattr(tokenizer.decoder, "attend_map", False):
        decoder_kwargs["encoder_map_tokens"] = encoder_output.map_tokens[anchor_indices]
        decoder_kwargs["encoder_map_mask"] = encoder_output.map_token_mask[anchor_indices]
    decoded = tokenizer.decoder(
        encoder_output.z[anchor_indices],
        agent_mask=batch["agent_mask"][anchor_indices],
        light_mask=batch["light_mask"][anchor_indices],
        **decoder_kwargs,
    )
    loss, metrics = vector_tokenizer_reconstruction_loss(
        decoded,
        agents=batch["agents"][anchor_indices],
        agent_mask=batch["agent_mask"][anchor_indices],
        lights=batch["lights"][anchor_indices],
        light_mask=batch["light_mask"][anchor_indices],
        agent_xy_weight=model_args.agent_xy_weight,
        agent_vel_weight=model_args.agent_vel_weight,
        agent_yaw_weight=model_args.agent_yaw_weight,
        agent_valid_weight=model_args.agent_valid_weight,
        light_state_weight=model_args.light_state_weight,
        light_valid_weight=model_args.light_valid_weight,
        agent_delta_xy_weight=model_args.agent_delta_xy_weight,
        agent_fde_xy_weight=model_args.agent_fde_xy_weight,
        agent_kinematic_xy_weight=model_args.agent_kinematic_xy_weight,
        agent_speed_yaw_kinematic_weight=model_args.agent_speed_yaw_kinematic_weight,
        kinematic_dt=model_args.kinematic_dt,
        focus_agent_weight=model_args.focus_agent_weight,
        agent_xy_loss=model_args.agent_xy_loss,
        agent_xy_parameterization=model_args.agent_xy_parameterization,
    )
    return loss, metrics


def compute_contrastive_loss(head, encoder_output, batch, args):
    embeddings, _ = head(
        encoder_output.z[:, -1], batch["first_slot"], batch["second_slot"]
    )
    if args.mode == "hard":
        return hard_contrastive_loss(
            embeddings,
            batch["group_offsets"],
            negatives_per_side=args.hard_negatives_per_side,
            temperature=args.temperature,
        )
    if args.mode == "soft":
        return soft_contrastive_loss(
            embeddings,
            batch["group_offsets"],
            batch["target_probabilities"],
            positives_per_anchor=args.soft_positives,
            negatives_per_anchor=args.soft_negatives,
            temperature=args.temperature,
        )
    return hybrid_soft_contrastive_loss(
        embeddings,
        batch["group_offsets"],
        batch["target_probabilities"],
        positives_per_anchor=args.soft_positives,
        negatives_per_anchor=args.soft_negatives,
        separation_temperature=args.temperature,
        rank_temperature=args.rank_temperature,
        rank_relative_weight=(
            args.hybrid_rank_weight / max(args.hybrid_separation_weight, 1e-12)
        ),
    )


@torch.no_grad()
def evaluate(tokenizer, head, loader, device, args, model_args, stage_b: bool):
    tokenizer.eval()
    head.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if args.eval_batches > 0 and batch_index >= args.eval_batches:
            break
        batch = move_batch(batch, device)
        encoder_output = tokenizer.encoder(
            agents=batch["agents"],
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
            lights=batch["lights"],
            light_mask=batch["light_mask"],
        )
        contrastive, metrics = compute_contrastive_loss(head, encoder_output, batch, args)
        values = {"loss_contrastive": float(contrastive.item())}
        values.update({key: float(value.item()) for key, value in metrics.items()})
        if stage_b:
            recon, recon_metrics = reconstruction_loss(
                tokenizer,
                encoder_output,
                batch,
                batch["group_offsets"],
                model_args,
            )
            values["loss_reconstruction"] = float(recon.item())
            for key in ("agent_xy_mae_m", "agent_speed_mae_mps", "agent_yaw_mae_deg"):
                if key in recon_metrics:
                    values[key] = float(recon_metrics[key].item())
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(path, tokenizer, head, optimizer, scaler, args, model_args, step, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    saved_args = vars(model_args).copy()
    saved_args.update(
        contrastive_refined=True,
        contrastive_mode=args.mode,
        contrastive_history_steps=args.history_steps,
    )
    saved_contrastive_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "format": "waymo_interaction_contrastive_tokenizer_v1",
            "model": tokenizer.state_dict(),
            "contrastive_head": head.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": saved_args,
            # Keep the checkpoint compatible with PyTorch's safe
            # weights_only=True loader (Path objects are not allow-listed).
            "contrastive_args": saved_contrastive_args,
            "step": int(step),
            "best_val_contrastive": float(best_val),
        },
        temporary,
    )
    temporary.replace(path)


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.5f}" for key, value in sorted(metrics.items()))


def append_metrics(path: Path, *, kind: str, step: int, stage: str, metrics) -> None:
    payload = {
        "kind": str(kind),
        "step": int(step),
        "stage": str(stage),
        **{key: float(value) for key, value in metrics.items()},
    }
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_nonfinite_report(
    path: Path,
    *,
    step: int,
    stage: str,
    reason: str,
    batch: dict[str, torch.Tensor],
) -> None:
    payload = {
        "step": int(step),
        "stage": str(stage),
        "reason": str(reason),
        "sample_indices": batch["sample_indices"].detach().cpu().tolist(),
        "group_offsets": batch["group_offsets"].detach().cpu().tolist(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def tensor_gradient_norm(
    loss: torch.Tensor, parameters: list[nn.Parameter]
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    if not squared:
        return loss.new_zeros((), dtype=torch.float32)
    return torch.stack(squared).sum().sqrt()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    seed_everything(args.seed)
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    tokenizer, model_args, baseline_checkpoint = load_baseline_tokenizer(
        args.tokenizer_ckpt, device
    )
    encoder_parameters = configure_trainable_parameters(
        tokenizer, encoder_unfreeze_blocks=args.encoder_unfreeze_blocks
    )
    for parameter in tokenizer.decoder.parameters():
        parameter.requires_grad_(False)
    tokenizer.decoder.eval()
    head = ArbitraryPairContrastiveHead(
        d_bottleneck=model_args.d_bottleneck,
        d_model=args.head_dim,
        n_heads=args.head_heads,
        n_latents=model_args.n_latents,
        n_agents=tokenizer.decoder.n_agents,
        embedding_dim=args.embedding_dim,
        dropout=args.head_dropout,
        scale_pos_embeds=model_args.scale_pos_embeds,
    ).to(device)
    if args.head_init_ckpt is not None:
        head_checkpoint = torch.load(args.head_init_ckpt, map_location="cpu")
        head.load_state_dict(head_checkpoint["contrastive_head"], strict=True)
        print(
            f"initialized contrastive head from {args.head_init_ckpt} "
            f"step={head_checkpoint.get('step')}",
            flush=True,
        )
    elif args.init_decoder_queries and args.head_dim == tokenizer.decoder.d_model:
        head.init_from_decoder_queries(tokenizer.decoder.agent_queries)

    train_dataset = InteractionContrastiveDataset(
        args.cache_dir / "train_contrastive_training.npz",
        mode=args.mode,
        history_steps=args.history_steps,
        hard_negatives_per_side=args.hard_negatives_per_side,
        soft_positives=args.soft_positives,
        soft_negatives=args.soft_negatives,
        soft_beta=args.soft_beta,
        training=True,
        soft_sigma_floor=args.soft_sigma_floor,
    )
    val_dataset = InteractionContrastiveDataset(
        args.cache_dir / "val_contrastive_training.npz",
        mode=args.mode,
        history_steps=args.history_steps,
        hard_negatives_per_side=args.hard_negatives_per_side,
        soft_positives=args.soft_positives,
        soft_negatives=args.soft_negatives,
        soft_beta=args.soft_beta,
        training=False,
        soft_sigma_by_stratum=train_dataset.soft_sigma_by_stratum,
        soft_sigma_floor=args.soft_sigma_floor,
    )
    full_val_anchor_count = len(val_dataset)
    if args.val_anchors > 0:
        val_dataset.anchor_indices = select_stratified_validation_anchors(
            val_dataset.anchor_indices,
            val_dataset.arrays["stratum_key"],
            args.val_anchors,
        )
    np.savez_compressed(
        args.output_dir / "validation_manifest.npz",
        cache_row=val_dataset.anchor_indices,
        sample_index=val_dataset.arrays["sample_index"][val_dataset.anchor_indices],
        scenario_id=val_dataset.arrays["scenario_id"][val_dataset.anchor_indices],
        stratum_key=val_dataset.arrays["stratum_key"][val_dataset.anchor_indices],
    )
    if args.mode == "hybrid":
        (args.output_dir / "soft_scales.json").write_text(
            json.dumps(
                {
                    str(key): value
                    for key, value in sorted(
                        train_dataset.soft_sigma_by_stratum.items()
                    )
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=contrastive_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=contrastive_collate,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(head.parameters()), "lr": args.head_lr},
            {"params": encoder_parameters, "lr": 0.0},
        ],
        weight_decay=args.weight_decay,
    )
    # BF16 has FP32-like exponent range and does not need loss scaling.  Keep a
    # scaler only for the optional FP16 path.
    scaler = GradScaler(
        device="cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    total_steps = args.stage_a_steps + args.stage_b_steps
    step = 0
    best_val_by_stage = {"A": math.inf, "B": math.inf}
    checkpoint_dir = args.output_dir
    metrics_path = checkpoint_dir / "metrics.jsonl"
    metrics_path.write_text("")
    iterator = iter(train_loader)
    started = time.time()
    print(
        f"mode={args.mode} device={device} amp={use_amp} amp_dtype={args.amp_dtype} "
        f"train_anchors={len(train_dataset)} "
        f"val_anchors={len(val_dataset)}/{full_val_anchor_count} "
        f"baseline_step={baseline_checkpoint.get('step')} "
        f"stage_a={args.stage_a_steps} stage_b={args.stage_b_steps}"
    )
    print(
        f"duplicate_filter={json.loads((args.cache_dir / 'summary.json').read_text())['config']['duplicate_rms_threshold']} "
        f"history={args.history_steps} encoder_blocks={args.encoder_unfreeze_blocks}"
    )
    if args.mode == "hybrid":
        print(
            f"hybrid_separation_weight={args.hybrid_separation_weight} "
            f"hybrid_rank_weight={args.hybrid_rank_weight} "
            f"rank_temperature={args.rank_temperature} "
            f"soft_sigma_floor={args.soft_sigma_floor}",
            flush=True,
        )

    if args.eval_at_start:
        initial_metrics = evaluate(
            tokenizer, head, val_loader, device, args, model_args, True
        )
        if not all(math.isfinite(value) for value in initial_metrics.values()):
            raise FloatingPointError(f"Non-finite initial validation: {initial_metrics}")
        print(f"eval step=0 stage=init {format_metrics(initial_metrics)}", flush=True)
        append_metrics(
            metrics_path, kind="eval", step=0, stage="init", metrics=initial_metrics
        )
        best_val_by_stage["A"] = initial_metrics["loss_contrastive"]
        save_checkpoint(
            checkpoint_dir / "initial.pt",
            tokenizer,
            head,
            optimizer,
            scaler,
            args,
            model_args,
            0,
            best_val_by_stage["A"],
        )
        save_checkpoint(
            checkpoint_dir / "best_stage_a.pt",
            tokenizer,
            head,
            optimizer,
            scaler,
            args,
            model_args,
            0,
            best_val_by_stage["A"],
        )

    while step < total_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        step += 1
        stage_b = step > args.stage_a_steps
        if stage_b:
            optimizer.param_groups[1]["lr"] = args.encoder_lr
            tokenizer.encoder.eval()
        else:
            optimizer.param_groups[1]["lr"] = 0.0
            tokenizer.encoder.eval()
        head.train()
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if stage_b:
                encoder_output = tokenizer.encoder(
                    agents=batch["agents"],
                    agent_mask=batch["agent_mask"],
                    map_polylines=batch["map_polylines"],
                    map_mask=batch["map_mask"],
                    lights=batch["lights"],
                    light_mask=batch["light_mask"],
                )
            else:
                with torch.no_grad():
                    encoder_output = tokenizer.encoder(
                        agents=batch["agents"],
                        agent_mask=batch["agent_mask"],
                        map_polylines=batch["map_polylines"],
                        map_mask=batch["map_mask"],
                        lights=batch["lights"],
                        light_mask=batch["light_mask"],
                    )
            contrastive, contrastive_metrics = compute_contrastive_loss(
                head, encoder_output, batch, args
            )
            reconstruction = contrastive.new_zeros(())
            reconstruction_metrics = {}
            if stage_b:
                reconstruction, reconstruction_metrics = reconstruction_loss(
                    tokenizer,
                    encoder_output,
                    batch,
                    batch["group_offsets"],
                    model_args,
                )
                progress = min(
                    1.0,
                    max(0.0, (step - args.stage_a_steps) / max(args.contrastive_ramp_steps, 1)),
                )
                maximum_contrastive_weight = (
                    args.hybrid_separation_weight
                    if args.mode == "hybrid"
                    else args.contrastive_weight
                )
                contrastive_weight = maximum_contrastive_weight * progress
                loss = reconstruction + contrastive_weight * contrastive
            else:
                contrastive_weight = 1.0
                loss = contrastive
        stage_name = "B" if stage_b else "A"
        if not bool(torch.isfinite(loss)):
            write_nonfinite_report(
                checkpoint_dir / "nonfinite.json",
                step=step,
                stage=stage_name,
                reason="non-finite forward loss",
                batch=batch,
            )
            raise FloatingPointError(f"Non-finite loss at step={step} stage={stage_name}")
        gradient_probe_metrics = {}
        if stage_b and args.gradient_probe_every > 0 and (
            step == args.stage_a_steps + 1 or step % args.gradient_probe_every == 0
        ):
            reconstruction_gradient_norm = tensor_gradient_norm(
                reconstruction, encoder_parameters
            )
            contrastive_gradient_norm = tensor_gradient_norm(
                contrastive, encoder_parameters
            )
            weighted_ratio = (
                float(contrastive_weight)
                * contrastive_gradient_norm
                / reconstruction_gradient_norm.clamp_min(1e-12)
            )
            gradient_probe_metrics = {
                "encoder_reconstruction_grad_norm": float(
                    reconstruction_gradient_norm.item()
                ),
                "encoder_contrastive_grad_norm": float(
                    contrastive_gradient_norm.item()
                ),
                "encoder_weighted_contrastive_grad_ratio": float(
                    weighted_ratio.item()
                ),
            }
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ],
                args.grad_clip,
                error_if_nonfinite=True,
            )
        except RuntimeError as error:
            write_nonfinite_report(
                checkpoint_dir / "nonfinite.json",
                step=step,
                stage=stage_name,
                reason=f"non-finite gradient: {error}",
                batch=batch,
            )
            raise
        scaler.step(optimizer)
        scaler.update()

        metrics = {
            "loss_total": float(loss.detach().item()),
            "loss_contrastive": float(contrastive.detach().item()),
            "loss_reconstruction": float(reconstruction.detach().item()),
            "contrastive_weight": float(contrastive_weight),
            "grad_norm": float(grad_norm),
            **gradient_probe_metrics,
            **{key: float(value.item()) for key, value in contrastive_metrics.items()},
        }
        for key in ("agent_xy_mae_m", "agent_speed_mae_mps", "agent_yaw_mae_deg"):
            if key in reconstruction_metrics:
                metrics[key] = float(reconstruction_metrics[key].detach().item())
        if step == 1 or step % args.log_every == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"step={step} stage={'B' if stage_b else 'A'} {format_metrics(metrics)} "
                f"steps_per_sec={step / elapsed:.3f}",
                flush=True,
            )
            append_metrics(
                metrics_path,
                kind="train",
                step=step,
                stage=stage_name,
                metrics=metrics,
            )
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.stage_a_steps):
            val_metrics = evaluate(
                tokenizer, head, val_loader, device, args, model_args, stage_b
            )
            if not all(math.isfinite(value) for value in val_metrics.values()):
                raise FloatingPointError(
                    f"Non-finite validation at step={step}: {val_metrics}"
                )
            print(f"eval step={step} stage={'B' if stage_b else 'A'} {format_metrics(val_metrics)}", flush=True)
            append_metrics(
                metrics_path,
                kind="eval",
                step=step,
                stage=stage_name,
                metrics=val_metrics,
            )
            if val_metrics["loss_contrastive"] < best_val_by_stage[stage_name]:
                best_val_by_stage[stage_name] = val_metrics["loss_contrastive"]
                save_checkpoint(
                    checkpoint_dir / ("best.pt" if stage_b else "best_stage_a.pt"),
                    tokenizer,
                    head,
                    optimizer,
                    scaler,
                    args,
                    model_args,
                    step,
                    best_val_by_stage[stage_name],
                )
        if args.save_every > 0 and step % args.save_every == 0:
            current_best_val = best_val_by_stage["B" if stage_b else "A"]
            save_checkpoint(
                checkpoint_dir / f"step_{step:08d}.pt",
                tokenizer,
                head,
                optimizer,
                scaler,
                args,
                model_args,
                step,
                current_best_val,
            )
            save_checkpoint(
                checkpoint_dir / "latest.pt",
                tokenizer,
                head,
                optimizer,
                scaler,
                args,
                model_args,
                step,
                current_best_val,
            )

    save_checkpoint(
        checkpoint_dir / f"final_step_{step:08d}.pt",
        tokenizer,
        head,
        optimizer,
        scaler,
        args,
        model_args,
        step,
        best_val_by_stage["B" if args.stage_b_steps > 0 else "A"],
    )
    save_checkpoint(
        checkpoint_dir / "latest.pt",
        tokenizer,
        head,
        optimizer,
        scaler,
        args,
        model_args,
        step,
        best_val_by_stage["B" if args.stage_b_steps > 0 else "A"],
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["hard", "soft", "hybrid"], required=True)
    parser.add_argument("--tokenizer_ckpt", type=Path, required=True)
    parser.add_argument("--head_init_ckpt", type=Path)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--stage_a_steps", type=int, default=5000)
    parser.add_argument("--stage_b_steps", type=int, default=20000)
    parser.add_argument("--encoder_unfreeze_blocks", type=int, default=1)
    parser.add_argument("--head_dim", type=int, default=256)
    parser.add_argument("--head_heads", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--head_dropout", type=float, default=0.05)
    parser.add_argument("--init_decoder_queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard_negatives_per_side", type=int, default=2)
    parser.add_argument("--soft_positives", type=int, default=8)
    parser.add_argument("--soft_negatives", type=int, default=4)
    parser.add_argument("--soft_beta", type=float, default=0.5)
    parser.add_argument("--soft_sigma_floor", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--rank_temperature", type=float, default=0.10)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--hybrid_separation_weight", type=float, default=0.005)
    parser.add_argument("--hybrid_rank_weight", type=float, default=0.001)
    parser.add_argument("--contrastive_ramp_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gradient_probe_every", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--amp_dtype", choices=["bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--val_anchors", type=int, default=0)
    parser.add_argument(
        "--eval_at_start", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--save_every", type=int, default=1000)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2, sort_keys=True) + "\n"
    )
    train(args)


if __name__ == "__main__":
    main()
