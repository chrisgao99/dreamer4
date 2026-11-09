from types import SimpleNamespace

import torch
from waymo.training.world_model.mon_violations import (SceneReference, candidate_violations, segments_cross,
                                                       select_winners, violation_counts)


def make_scene(gt_y=(5., 10.), T=6):
    """Two vehicles driving east at y=gt_y; solid white line at y=0; road edges at y=-20 / y=+20."""
    N = 2
    gt = torch.zeros(1, N, T, 3)
    gt[..., 0] = torch.arange(1, T + 1) * 1.0
    for n, y in enumerate(gt_y):
        gt[:, n, :, 1] = y
    gt[:, 1, :, 0] += 30.
    current = gt[:, :, 0].clone(); current[..., 0] -= 1
    hist = torch.zeros(1, N, 11, 8)
    hist[..., 0] = current[..., 0, None] - torch.arange(10, -1, -1) * 1.0
    hist[..., 1] = current[..., 1, None]
    hist[..., 5] = 1; hist[..., 7] = 1
    targets = SimpleNamespace(future_pose=gt, current_pose=current, valid=torch.ones(1, N, T, dtype=torch.bool),
                              agent_type=torch.ones(1, N, dtype=torch.long))
    xs = torch.linspace(-60, 80, 6)
    def line(y, typ, direction):
        p = torch.zeros(6, 6); p[:, 0] = xs if direction > 0 else xs.flip(0); p[:, 1] = y
        p[:, 2] = direction; p[:, 4] = typ; p[:, 5] = 1
        return p
    poly = torch.stack((line(0., 7, 1), line(-20., 15, 1), line(20., 15, -1), line(5., 2, 1), line(10., 2, 1)))[None]
    batch = dict(map_polylines=poly, map_mask=torch.ones(1, 5, 6, dtype=torch.bool), map_ids=torch.tensor([[7, 15, 16, 1, 2]]),
                 map_is_lane=torch.tensor([[False, False, False, True, True]]))
    return targets, hist, batch


def test_offroad_near_a_lane_not_counted():
    """An excursion past a (mis-signed / overpass) edge while on a lane centreline is not offroad."""
    targets, hist, batch = make_scene()
    batch['map_polylines'][0, 4, :, 1] = 25.             # the second lane now runs at y=25, beyond the y=20 edge
    p = targets.future_pose[0].clone(); p[1, 3:, 1] = 25.
    v, c = run(targets, hist, batch, [p])
    assert c[0, 1].item() == 0 and v['off_excess_raw'][0, 1] > 4


def run(targets, hist, batch, cands):
    ref = SceneReference(targets, hist, batch, agent_lengths=torch.full((1, 2), 4.5))
    v = candidate_violations(torch.stack(cands), ref, len(cands), batch['map_polylines'], batch['map_mask'])
    return v, violation_counts(v)


def test_segments_cross():
    p0, p1 = torch.tensor([[0., -1.], [0., 1.]]), torch.tensor([[0., 1.], [0., 2.]])
    q0, q1 = torch.tensor([[-1., 0.]]), torch.tensor([[1., 0.]])
    assert segments_cross(p0, p1, q0, q1)[:, 0].tolist() == [True, False]


def test_clean_candidate_is_clean():
    targets, hist, batch = make_scene()
    v, c = run(targets, hist, batch, [targets.future_pose[0]])
    assert c.tolist() == [[0, 0, 0]]
    assert torch.allclose(v['ade'], torch.zeros(1))


def test_collision_counts_agents_once_and_logged_overlap_excluded():
    targets, hist, batch = make_scene()
    p = targets.future_pose[0].clone()
    p[1, 2:, 0] = p[0, 2:, 0] + 1.0; p[1, 2:, 1] = 5.    # agent 1 drives into agent 0 for 4 steps
    _, c = run(targets, hist, batch, [p])
    assert c[0, 0].item() == 2                            # two agents, however many steps
    targets2, hist2, batch2 = make_scene()
    targets2.future_pose[0, 1, -1, :2] = targets2.future_pose[0, 0, -1, :2]  # the LOG also overlaps once
    _, c2 = run(targets2, hist2, batch2, [p])
    assert c2[0, 0].item() == 0


def test_solid_line_new_crossing_counted_once_and_logged_crossing_excluded():
    targets, hist, batch = make_scene()
    p = targets.future_pose[0].clone()
    p[0, :, 1] = torch.tensor([5., -3., 3., -3., 3., 5.])  # crosses back and forth
    v, c = run(targets, hist, batch, [p])
    assert c[0, 2].item() == 1 and v['line_events'][0, 0].item() == 4
    targets2, hist2, batch2 = make_scene()
    targets2.future_pose[0, 0, 3:, 1] = -5.                # the logged agent crosses the same line
    _, c2 = run(targets2, hist2, batch2, [p])
    assert c2[0, 2].item() == 0


def test_offroad_counted_only_beyond_edge():
    targets, hist, batch = make_scene()
    p = targets.future_pose[0].clone()
    p[1, 3:, 1] = 25.                                     # beyond the upper edge at y=20
    v, c = run(targets, hist, batch, [p])
    assert c[0, 1].item() == 1 and v['off_excess'][0, 1] > 4
    assert v['off_excess'][0, 0] < 0


def test_select_winners_guard_then_lexicographic():
    d = torch.tensor([[1.0, 1.1, 1.2, 1.4]])
    counts = torch.tensor([[[2, 0, 0], [1, 1, 0], [1, 0, 3], [0, 0, 0]]])
    w, pool = select_winners(d, counts, ratio=1.3)
    assert w.item() == 2 and pool.tolist() == [[True, True, True, False]]
    w, _ = select_winners(d, counts, ratio=1.0)
    assert w.item() == 0
    w, _ = select_winners(d, torch.zeros_like(counts), ratio=2.0)
    assert w.item() == 0
