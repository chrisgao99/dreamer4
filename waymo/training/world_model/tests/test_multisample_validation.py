from __future__ import annotations

import torch

from waymo.training.world_model.multisample_validation import (
    flow_erd_cpd_metrics,
    multisample_selection_score,
    multisample_trajectory_metrics,
)


def _targets() -> torch.Tensor:
    target = torch.zeros((2, 4, 3, 7), dtype=torch.float32)
    target[..., 5] = 1.0
    target[:, :, :, 0] = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
    return target


def _decoded(target: torch.Tensor) -> torch.Tensor:
    decoded = torch.zeros((2, 3, 4, 3, 7), dtype=torch.float32)
    decoded[..., 0:2] = target[:, None, ..., 0:2]
    decoded[:, 1, :, :, 0] += 1.0
    decoded[:, 2, :, :, 0] += 2.0
    decoded[..., 6] = 1.0
    return decoded


def test_joint_minade_uses_one_full_candidate_and_reports_diversity() -> None:
    target = _targets()
    decoded = _decoded(target)
    metrics = multisample_trajectory_metrics(decoded, target, future_start=1)

    torch.testing.assert_close(metrics["multisample_all_minade_m"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["multisample_all_mean_ade_m"], torch.tensor(1.0))
    torch.testing.assert_close(metrics["multisample_all_worst_ade_m"], torch.tensor(2.0))
    assert metrics["multisample_all_pairwise_trajectory_distance_m"].item() > 0.0
    assert metrics["multisample_nonfocus_pairwise_endpoint_distance_m"].item() > 0.0


def test_nonfocus_metrics_ignore_focus_only_error() -> None:
    target = _targets()
    decoded = _decoded(target)
    decoded[:, 0, 1:, 0, 0] += 20.0
    metrics = multisample_trajectory_metrics(decoded, target, future_start=1)

    assert metrics["multisample_all_minade_m"].item() > 0.0
    torch.testing.assert_close(metrics["multisample_nonfocus_minade_m"], torch.tensor(0.0))


def test_selection_rejects_collapsed_checkpoint() -> None:
    reference = {
        "multisample_nonfocus_minade_m": 2.0,
        "multisample_nonfocus_ade_winner_fde_m": 4.0,
        "multisample_nonfocus_mean_ade_m": 3.0,
        "multisample_nonfocus_mean_fde_m": 5.0,
        "multisample_nonfocus_worst_ade_m": 4.0,
        "collision_overlap_rate_proxy": 0.1,
        "offroad_rate_proxy": 0.2,
        "kinematic_xy_mae_m": 0.1,
        "multisample_nonfocus_pairwise_trajectory_distance_m": 2.0,
    }
    good = dict(reference)
    good["multisample_nonfocus_minade_m"] = 1.0
    good_score = multisample_selection_score(good, reference, diversity_floor_ratio=0.5)
    assert good_score["checkpoint_selection_eligible"] == 1.0

    collapsed = dict(good)
    collapsed["multisample_nonfocus_pairwise_trajectory_distance_m"] = 0.2
    collapsed_score = multisample_selection_score(collapsed, reference, diversity_floor_ratio=0.5)
    assert collapsed_score["checkpoint_selection_eligible"] == 0.0
    assert collapsed_score["checkpoint_selection_score"] > good_score["checkpoint_selection_score"]


def test_8s_endpoint_spatial_std_uses_future_step_80() -> None:
    target = torch.zeros((1, 81, 2, 7), dtype=torch.float32)
    target[..., 5] = 1.0
    decoded = torch.zeros((1, 3, 81, 2, 7), dtype=torch.float32)
    decoded[..., 6] = 1.0
    decoded[:, 1, 80, :, 0] = 1.0
    decoded[:, 2, 80, :, 0] = 2.0

    metrics = multisample_trajectory_metrics(decoded, target, future_start=1)

    expected = torch.tensor((2.0 / 3.0) ** 0.5)
    torch.testing.assert_close(
        metrics["multisample_all_8s_endpoint_mean_spatial_std_m"], expected
    )
    torch.testing.assert_close(
        metrics["multisample_nonfocus_8s_endpoint_mean_spatial_std_m"], expected
    )


def test_flow_erd_cpd_is_type_normalized_rms_and_log_free() -> None:
    # One final context frame followed by two rollout steps.  Future GT is
    # deliberately invalid: CPD must freeze the roster at the context frame.
    target = torch.zeros((1, 3, 3, 8), dtype=torch.float32)
    target[:, 0, :, 5] = 1.0
    target[:, 0, 0, 7] = 1.0  # controlled focus vehicle
    target[:, 0, 1, 7] = 1.0  # simulated vehicle
    target[:, 0, 2, 7] = 2.0  # simulated pedestrian

    predicted = torch.zeros((1, 2, 3, 3, 2), dtype=torch.float32)
    predicted[:, 1, 1:, 0, 0] = 100.0  # ignored controlled focus
    predicted[:, 1, 1:, 1, 0] = 3.0
    predicted[:, 1, 1:, 2, 0] = 4.0

    metrics, components = flow_erd_cpd_metrics(
        predicted,
        target,
        future_start=1,
        type_scales=(3.0, 4.0, 2.0),
        exclude_focus=True,
    )

    torch.testing.assert_close(metrics["flow_erd_cpd"], torch.sqrt(torch.tensor(2.0)))
    torch.testing.assert_close(metrics["flow_erd_cpd_unscaled"], torch.tensor(5.0))
    torch.testing.assert_close(components["type_mse"][0, 0], torch.tensor([9.0, 16.0, 0.0]))
    torch.testing.assert_close(components["type_count"][0], torch.tensor([1, 1, 0]))
    assert components["pair_indices"].tolist() == [[0, 1]]
