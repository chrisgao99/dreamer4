from argparse import Namespace

import numpy as np
import pytest

from waymo.evaluation.eval_waymo_direct_action_flow_multisample import (
    resolve_focus_conditioning,
    summarize_agent_scopes,
)


def test_focus_mode_follows_checkpoint_and_rejects_accidental_conditioning():
    assert resolve_focus_conditioning(Namespace(), "checkpoint") is True
    args = Namespace(condition_focus_actions=False)
    assert resolve_focus_conditioning(args, "checkpoint") is False
    assert resolve_focus_conditioning(args, "generate_all") is False
    with pytest.raises(ValueError, match="conflicts"):
        resolve_focus_conditioning(args, "conditioned")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_focus_conditioning(Namespace(), "generate_all")


def test_scope_ade_weights_points_selects_joint_rollouts_and_omits_empty_scenes():
    # Scene 0: focus has 1 valid step, nonfocus has 3. The best joint
    # candidate differs from independently choosing the best for each agent.
    # Scene 1: only focus is valid, so it must not dilute nonfocus ADE.
    ade = np.array([[[2., 10.], [8., 4.]], [[6., np.nan], [4., np.nan]]])
    steps = np.array([[1, 3], [2, 0]])
    result = summarize_agent_scopes(ade, steps)
    assert result["all"]["mean_ade_m"] == pytest.approx(5.75)
    assert result["all"]["minade_m"] == pytest.approx(4.5)
    assert result["nonfocus"]["scene_count"] == 1
    assert result["nonfocus"]["mean_ade_m"] == pytest.approx(7.)
    assert result["nonfocus"]["minade_m"] == pytest.approx(4.)
    assert result["focus"]["mean_ade_m"] == pytest.approx(5.)
    assert result["focus"]["minade_m"] == pytest.approx(3.)
    assert result["all"]["valid_agent_time_points_per_rollout"] == 6


def test_empty_scope_is_not_reported_as_perfect_prediction():
    result = summarize_agent_scopes(np.ones((1, 2, 1)), np.ones((1, 1)))
    assert result["nonfocus"] == {"scene_count": 0, "mean_ade_m": None, "minade_m": None}
