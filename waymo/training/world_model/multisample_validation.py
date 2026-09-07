"""Distribution-level validation metrics for stochastic joint Waymo rollouts."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import torch


# Waymo object types used by Flow-ERD: vehicle, pedestrian, cyclist.  Keep the
# tensor order stable because CPD component sidecars use the same convention.
FLOW_ERD_CPD_TYPE_IDS = (1, 2, 3)
FLOW_ERD_CPD_TYPE_NAMES = ("vehicle", "pedestrian", "cyclist")


def _masked_scene_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reduce trailing dimensions while preserving scene/candidate prefixes."""
    weight = weight.to(device=value.device, dtype=value.dtype)
    while weight.dim() < value.dim():
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(value)
    reduce_dims = tuple(range(2, value.dim()))
    return (value * weight).sum(dim=reduce_dims) / weight.sum(dim=reduce_dims).clamp_min(1.0)


def _trajectory_weights(
    agents_btkf: torch.Tensor,
    *,
    future_start: int,
    exclude_focus: bool,
    agent_weight_multiplier: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    target = agents_btkf[:, int(future_start) :].float()
    valid = target[..., 5] > 0.5
    weight = valid.to(dtype=target.dtype)
    if agent_weight_multiplier is not None:
        multiplier = agent_weight_multiplier[:, int(future_start) :].to(
            device=target.device,
            dtype=target.dtype,
        )
        if multiplier.shape != weight.shape:
            raise ValueError(
                "agent_weight_multiplier must match future (B,H,K); got "
                f"{tuple(multiplier.shape)} vs {tuple(weight.shape)}"
            )
        weight = weight * multiplier

    if exclude_focus and int(weight.shape[-1]) > 0:
        nonfocus = weight.clone()
        nonfocus[..., 0] = 0.0
        # Scenes containing only the focus slot fall back to all-agent metrics.
        has_nonfocus = nonfocus.sum(dim=(1, 2)) > 0.0
        weight = torch.where(has_nonfocus[:, None, None], nonfocus, weight)
    return target, weight


def _scope_metrics(
    predicted_xy: torch.Tensor,
    agents_btkf: torch.Tensor,
    *,
    future_start: int,
    exclude_focus: bool,
    agent_weight_multiplier: Optional[torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    """Metrics for (B,N,T,K,2) joint candidates under one agent scope."""
    target, weight = _trajectory_weights(
        agents_btkf,
        future_start=future_start,
        exclude_focus=exclude_focus,
        agent_weight_multiplier=agent_weight_multiplier,
    )
    pred = predicted_xy[:, :, int(future_start) :].float()
    if pred.shape[0] != target.shape[0] or pred.shape[2:4] != target.shape[1:3]:
        raise ValueError(
            "Predicted and target future shapes disagree: "
            f"{tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    target_xy = target[..., 0:2]
    distance = (pred - target_xy[:, None]).norm(dim=-1)
    candidate_ade = _masked_scene_mean(distance, weight[:, None])

    valid = weight > 0.0
    any_valid = valid.any(dim=1)
    time_index = torch.arange(valid.shape[1], device=valid.device).view(1, -1, 1)
    last_index = torch.where(valid, time_index, torch.zeros_like(time_index)).amax(dim=1)
    pred_final_index = last_index[:, None, :, None].expand(
        -1,
        int(pred.shape[1]),
        -1,
        2,
    )
    pred_final = pred.gather(dim=2, index=pred_final_index.unsqueeze(2)).squeeze(2)
    target_final_index = last_index[:, :, None].expand(-1, -1, 2)
    target_final = target_xy.gather(dim=1, index=target_final_index.unsqueeze(1)).squeeze(1)
    final_distance = (pred_final - target_final[:, None]).norm(dim=-1)
    final_weight = any_valid.to(dtype=pred.dtype)
    if agent_weight_multiplier is not None:
        multiplier_future = agent_weight_multiplier[:, int(future_start) :].to(
            device=pred.device,
            dtype=pred.dtype,
        )
        final_weight = final_weight * multiplier_future.gather(
            dim=1,
            index=last_index[:, None],
        ).squeeze(1)
    if exclude_focus and int(final_weight.shape[-1]) > 0:
        nonfocus_final = final_weight.clone()
        nonfocus_final[..., 0] = 0.0
        has_nonfocus_final = nonfocus_final.sum(dim=1) > 0.0
        final_weight = torch.where(has_nonfocus_final[:, None], nonfocus_final, final_weight)
    candidate_fde = (final_distance * final_weight[:, None]).sum(dim=-1) / final_weight.sum(
        dim=-1
    ).clamp_min(1.0)[:, None]

    ade_winner = candidate_ade.argmin(dim=1)
    winner_fde = candidate_fde.gather(dim=1, index=ade_winner[:, None]).squeeze(1)

    num_candidates = int(pred.shape[1])
    if num_candidates > 1:
        pair_distance = (pred[:, :, None] - pred[:, None, :]).norm(dim=-1)
        pair_traj = (
            pair_distance * weight[:, None, None]
        ).sum(dim=(3, 4)) / weight.sum(dim=(1, 2)).clamp_min(1.0)[:, None, None]
        pair_final_distance = (pred_final[:, :, None] - pred_final[:, None, :]).norm(dim=-1)
        pair_endpoint = (
            pair_final_distance * final_weight[:, None, None]
        ).sum(dim=-1) / final_weight.sum(dim=-1).clamp_min(1.0)[:, None, None]
        upper = torch.triu(
            torch.ones((num_candidates, num_candidates), device=pred.device, dtype=torch.bool),
            diagonal=1,
        )
        pairwise_trajectory = pair_traj[:, upper].mean()
        pairwise_endpoint = pair_endpoint[:, upper].mean()
    else:
        pairwise_trajectory = pred.sum() * 0.0
        pairwise_endpoint = pred.sum() * 0.0

    scene_minade = candidate_ade.min(dim=1).values
    scene_minfde = candidate_fde.min(dim=1).values
    scene_meanade = candidate_ade.mean(dim=1)
    scene_meanfde = candidate_fde.mean(dim=1)
    scene_worstade = candidate_ade.max(dim=1).values
    metrics = {
        f"{prefix}_minade_m": scene_minade.mean(),
        f"{prefix}_minfde_m": scene_minfde.mean(),
        f"{prefix}_ade_winner_fde_m": winner_fde.mean(),
        f"{prefix}_mean_ade_m": scene_meanade.mean(),
        f"{prefix}_mean_fde_m": scene_meanfde.mean(),
        f"{prefix}_worst_ade_m": scene_worstade.mean(),
        f"{prefix}_first_ade_m": candidate_ade[:, 0].mean(),
        f"{prefix}_oracle_ade_gain_m": (candidate_ade[:, 0] - scene_minade).mean(),
        f"{prefix}_pairwise_trajectory_distance_m": pairwise_trajectory,
        f"{prefix}_pairwise_endpoint_distance_m": pairwise_endpoint,
    }
    # Waymo is sampled at 10 Hz. Future index 79 is therefore the position at
    # t+8.0 s (future steps are numbered 1..H). Spatial spread is the radial
    # population standard deviation sqrt(var_x + var_y) across all N joint
    # candidates, averaged over valid/relevance-weighted agents and scenes.
    if int(pred.shape[2]) >= 80:
        endpoint_8s = pred[:, :, 79]
        endpoint_8s_mean = endpoint_8s.mean(dim=1, keepdim=True)
        endpoint_8s_spatial_std = (
            (endpoint_8s - endpoint_8s_mean).pow(2).mean(dim=1).sum(dim=-1).sqrt()
        )
        endpoint_8s_weight = weight[:, 79]
        metrics[f"{prefix}_8s_endpoint_mean_spatial_std_m"] = (
            endpoint_8s_spatial_std * endpoint_8s_weight
        ).sum() / endpoint_8s_weight.sum().clamp_min(1.0)
    return metrics


def flow_erd_cpd_metrics(
    predicted_xy: torch.Tensor,
    agents_btkf: torch.Tensor,
    *,
    future_start: int,
    type_scales: Sequence[float],
    exclude_focus: bool = True,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute Flow-ERD Cross-Pair Diversity (Eq. 20--21).

    ``predicted_xy`` contains full joint closed-loop rollouts with shape
    ``(B, K, T, A, 2)``.  Agent membership and type are frozen at the final
    context frame, so this metric does not inspect the logged future.  The
    returned component tensors retain each scene/pair/type mean squared
    displacement, allowing CPD to be re-normalized later with different
    training-set scales without rerunning the model.
    """
    if predicted_xy.dim() != 5 or int(predicted_xy.shape[-1]) != 2:
        raise ValueError(f"Expected predicted_xy=(B,K,T,A,2), got {tuple(predicted_xy.shape)}")
    if agents_btkf.dim() != 4 or int(agents_btkf.shape[-1]) < 8:
        raise ValueError(f"Expected agents_btkf=(B,T,A,F>=8), got {tuple(agents_btkf.shape)}")
    if len(type_scales) != len(FLOW_ERD_CPD_TYPE_IDS):
        raise ValueError(
            f"Expected {len(FLOW_ERD_CPD_TYPE_IDS)} CPD type scales, got {len(type_scales)}"
        )
    if any((not math.isfinite(float(scale))) or float(scale) <= 0.0 for scale in type_scales):
        raise ValueError(f"CPD type scales must be finite and positive, got {tuple(type_scales)}")

    future_start = int(future_start)
    future = predicted_xy[:, :, future_start:].float()
    num_rollouts = int(future.shape[1])
    horizon = int(future.shape[2])
    num_agents = int(future.shape[3])
    if num_rollouts < 2:
        raise ValueError("Flow-ERD CPD requires at least two rollouts")
    if horizon < 1:
        raise ValueError("Flow-ERD CPD requires at least one future step")
    if tuple(agents_btkf.shape[:1]) != tuple(future.shape[:1]) or int(agents_btkf.shape[2]) != num_agents:
        raise ValueError(
            "Predicted rollouts and agent metadata disagree: "
            f"{tuple(predicted_xy.shape)} vs {tuple(agents_btkf.shape)}"
        )

    # p_0 in the paper is the final context state.  Freeze the roster here so
    # neither future GT validity nor future GT positions enter this log-free metric.
    context_index = max(0, future_start - 1)
    if context_index >= int(agents_btkf.shape[1]):
        raise ValueError(
            f"CPD context index {context_index} is outside metadata length {agents_btkf.shape[1]}"
        )
    context = agents_btkf[:, context_index]
    roster_valid = context[..., 5] > 0.5
    agent_type = context[..., 7].round().long()
    if exclude_focus and num_agents > 0:
        roster_valid = roster_valid.clone()
        roster_valid[..., 0] = False

    pair_squared_distance = (future[:, :, None] - future[:, None, :]).pow(2).sum(dim=-1)
    type_mse_parts = []
    type_count_parts = []
    type_present_parts = []
    for type_id in FLOW_ERD_CPD_TYPE_IDS:
        selected = roster_valid & (agent_type == int(type_id))
        count = selected.sum(dim=-1)
        denominator = (count * horizon).clamp_min(1).to(dtype=pair_squared_distance.dtype)
        type_mse = (
            pair_squared_distance * selected[:, None, None, None].to(pair_squared_distance.dtype)
        ).sum(dim=(3, 4)) / denominator[:, None, None]
        present = count > 0
        type_mse = torch.where(present[:, None, None], type_mse, torch.zeros_like(type_mse))
        type_mse_parts.append(type_mse)
        type_count_parts.append(count)
        type_present_parts.append(present)

    # (B,K,K,C), with absent types represented by a zero contribution exactly
    # as Eq. 20's "types with N_c=0 are omitted" prescription.
    type_mse = torch.stack(type_mse_parts, dim=-1)
    type_count = torch.stack(type_count_parts, dim=-1)
    type_present = torch.stack(type_present_parts, dim=-1)
    upper = torch.triu(
        torch.ones((num_rollouts, num_rollouts), device=future.device, dtype=torch.bool),
        diagonal=1,
    )
    pair_type_mse = type_mse[:, upper]
    pair_indices = upper.nonzero(as_tuple=False)
    scale = torch.as_tensor(type_scales, device=future.device, dtype=pair_type_mse.dtype)
    pair_cpd = (pair_type_mse / scale.square()).sum(dim=-1).clamp_min(0.0).sqrt()
    pair_cpd_unscaled = pair_type_mse.sum(dim=-1).clamp_min(0.0).sqrt()
    scene_cpd = pair_cpd.mean(dim=-1)
    scene_cpd_unscaled = pair_cpd_unscaled.mean(dim=-1)
    scene_valid = type_present.any(dim=-1)
    valid_weight = scene_valid.to(dtype=scene_cpd.dtype)
    valid_denominator = valid_weight.sum().clamp_min(1.0)

    metrics = {
        "flow_erd_cpd": (scene_cpd * valid_weight).sum() / valid_denominator,
        "flow_erd_cpd_unscaled": (scene_cpd_unscaled * valid_weight).sum() / valid_denominator,
        "flow_erd_cpd_valid_scene_fraction": valid_weight.mean(),
    }
    components = {
        "type_mse": pair_type_mse,
        "type_count": type_count,
        "type_present": type_present,
        "pair_indices": pair_indices,
        "scene_cpd": scene_cpd,
        "scene_cpd_unscaled": scene_cpd_unscaled,
        "scene_valid": scene_valid,
    }
    return metrics, components


def multisample_trajectory_metrics(
    agent_continuous: torch.Tensor,
    agents_btkf: torch.Tensor,
    *,
    future_start: int,
    agent_weight_multiplier: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate N full joint rollouts without per-agent oracle selection.

    Args:
        agent_continuous: decoded states with shape (B,N,T,K,7).
        agents_btkf: raw targets with shape (B,T,K,F>=7).
    """
    if agent_continuous.dim() != 5 or agent_continuous.shape[-1] != 7:
        raise ValueError(f"Expected agent_continuous=(B,N,T,K,7), got {tuple(agent_continuous.shape)}")
    if agents_btkf.dim() != 4 or agents_btkf.shape[-1] < 7:
        raise ValueError(f"Expected agents_btkf=(B,T,K,F>=7), got {tuple(agents_btkf.shape)}")
    predicted_xy = agent_continuous[..., 0:2]
    metrics = _scope_metrics(
        predicted_xy,
        agents_btkf,
        future_start=future_start,
        exclude_focus=False,
        agent_weight_multiplier=None,
        prefix="multisample_all",
    )
    metrics.update(
        _scope_metrics(
            predicted_xy,
            agents_btkf,
            future_start=future_start,
            exclude_focus=True,
            agent_weight_multiplier=agent_weight_multiplier,
            prefix="multisample_nonfocus",
        )
    )
    metrics["multisample_num_rollouts"] = torch.tensor(
        float(agent_continuous.shape[1]),
        device=agent_continuous.device,
    )
    return metrics


_SELECTION_COMPONENTS = (
    ("multisample_nonfocus_minade_m", 0.30, 0.25),
    ("multisample_nonfocus_ade_winner_fde_m", 0.15, 0.5),
    ("multisample_nonfocus_mean_ade_m", 0.20, 0.25),
    ("multisample_nonfocus_mean_fde_m", 0.10, 0.5),
    ("multisample_nonfocus_worst_ade_m", 0.05, 0.25),
    ("collision_overlap_rate_proxy", 0.075, 0.01),
    ("offroad_rate_proxy", 0.075, 0.01),
    ("kinematic_xy_mae_m", 0.05, 0.01),
)


def multisample_selection_score(
    metrics: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    diversity_floor_ratio: float = 0.5,
) -> Dict[str, float]:
    """Return a Stage-1-normalized balanced score; lower is better.

    Accuracy coverage and average candidate quality carry 80% of the base
    score. All-candidate safety and kinematics carry 20%. A checkpoint that
    retains less than ``diversity_floor_ratio`` of Stage-1 trajectory diversity
    is marked ineligible and receives a large penalty. Large but useless
    diversity cannot win because mean/worst candidate accuracy remains scored.
    """
    components: Dict[str, float] = {}
    score = 0.0
    finite = True
    for name, weight, absolute_floor in _SELECTION_COMPONENTS:
        value = float(metrics.get(name, math.nan))
        ref_value = float(reference.get(name, math.nan))
        if not math.isfinite(value) or not math.isfinite(ref_value):
            finite = False
            normalized = math.inf
        else:
            # The additive floor prevents tiny proxy rates from exploding while
            # keeping the Stage-1 reference score exactly 1.0.
            normalized = (value + float(absolute_floor)) / (
                abs(ref_value) + float(absolute_floor)
            )
        components[f"selection_normalized_{name}"] = normalized
        score += float(weight) * normalized

    diversity_name = "multisample_nonfocus_pairwise_trajectory_distance_m"
    diversity = float(metrics.get(diversity_name, math.nan))
    reference_diversity = float(reference.get(diversity_name, math.nan))
    if not math.isfinite(diversity) or not math.isfinite(reference_diversity):
        finite = False
        diversity_ratio = math.nan
    elif reference_diversity <= 1e-6:
        diversity_ratio = 1.0
    else:
        diversity_ratio = diversity / reference_diversity

    floor = max(0.0, float(diversity_floor_ratio))
    diversity_eligible = math.isfinite(diversity_ratio) and diversity_ratio >= floor
    collapse_fraction = 0.0 if floor <= 0.0 else max(0.0, floor - diversity_ratio) / floor
    diversity_penalty = 10.0 * collapse_fraction if math.isfinite(collapse_fraction) else math.inf
    score += diversity_penalty
    eligible = finite and diversity_eligible and math.isfinite(score)
    return {
        "checkpoint_selection_score": float(score),
        "checkpoint_selection_eligible": float(eligible),
        "checkpoint_diversity_ratio_to_reference": float(diversity_ratio),
        "checkpoint_diversity_penalty": float(diversity_penalty),
        **components,
    }
