"""Per-agent rule violations of closed-loop candidates, for violation-aware MoN selection.

Raw per-agent quantities (thresholds are applied by ``violation_counts`` so they can be tuned offline):
  coll_depth : deepest NEW box overlap (m, >0 = overlap) of the agent with any other agent over the rollout.
               Boxes: measured length, type-proxy width (vehicle 2.0 m, ped/cyclist 0.8 m). A pair counts only
               when both agents are logged-valid at that step. Excluded pairs: ped-ped, and pairs whose LOGGED
               boxes overlap at t=0 or at any logged future step (proxy-box / tracking artifacts).
  off_excess : largest footprint excursion (m) beyond the road edge (map types 15/16) for vehicles/cyclists, only
               at steps where the logged footprint is reliably on road (``reliable_road_mask``, 0.3 m margin) AND
               the agent centre is farther than ``lane_threshold`` from every lane centreline (map_is_lane). The
               nearest-edge-point sign is unreliable near free edge ends (median tips, crop borders) and at
               overpass edges; a car near a lane centreline is on the road. off_excess_raw: without the lane test.
  line_events: number of steps at which a vehicle's centre path crosses a solid line (7 solid white, 8 solid
               double white, 11 solid yellow, 12 solid double yellow) that the same agent's logged path
               (history + future) never crosses; lines are identified by map_ids.
An agent counts once per category however many times or steps it violates.
"""
import torch
from waymo.training.world_model.direct_action_flow_mon_losses import footprint, road_geometry, reliable_road_mask

SOLID_LINES = (7, 8, 11, 12)


def dimensions(types, lengths=None):
    length, width = footprint(types)
    if lengths is not None:
        length = torch.where(lengths > 0.1, lengths.to(length.dtype), length)
    return length, width


def box_gap(poses, length, width):
    """poses (B,N,T,3), length/width (B,N) -> (B,T,N,N) separating-axis gap (negative = overlap depth)."""
    xy = poses[..., :2].transpose(1, 2).float()
    yaw = poses[..., 2].transpose(1, 2).float()
    forward = torch.stack((yaw.cos(), yaw.sin()), -1)
    lateral = torch.stack((-yaw.sin(), yaw.cos()), -1)
    hl = length[:, None].float() / 2; hw = width[:, None].float() / 2
    delta = xy.unsqueeze(-3) - xy.unsqueeze(-2)
    gaps = []
    for axes in (forward.unsqueeze(-2), lateral.unsqueeze(-2), forward.unsqueeze(-3), lateral.unsqueeze(-3)):
        center = (delta * axes).sum(-1).abs()
        ri = hl.unsqueeze(-1) * (forward.unsqueeze(-2) * axes).sum(-1).abs() + hw.unsqueeze(-1) * (lateral.unsqueeze(-2) * axes).sum(-1).abs()
        rj = hl.unsqueeze(-2) * (forward.unsqueeze(-3) * axes).sum(-1).abs() + hw.unsqueeze(-2) * (lateral.unsqueeze(-3) * axes).sum(-1).abs()
        gaps.append(center - ri - rj)
    return torch.stack(gaps, -1).max(-1).values


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def segments_cross(p0, p1, q0, q1):
    """Path segments p0->p1 (Q,2) vs line segments q0->q1 (S,2) -> (Q,S) proper crossing."""
    r, s = p1 - p0, q1 - q0
    d = _cross(r[:, None], s[None])
    qp = q0[None] - p0[:, None]
    ok = d.abs() > 1e-9
    d = torch.where(ok, d, torch.ones_like(d))
    t = _cross(qp, s[None]) / d
    u = _cross(qp, r[:, None]) / d
    return ok & (t >= 0) & (t < 1) & (u >= 0) & (u <= 1)


def lane_segments(map_polylines, map_mask, map_is_lane):
    m = map_mask.bool() & map_is_lane.bool()[:, None]
    seg = m[:, :-1] & m[:, 1:]
    return map_polylines[:, :-1, :2][seg].float(), map_polylines[:, 1:, :2][seg].float()


def point_segment_distance(q, a, e, chunk=2048):
    """(Q,2) points to the nearest of segments a->e (S,2) -> (Q,) distance."""
    if not len(a):
        return torch.full((len(q),), float('inf'), device=q.device)
    ab = e - a; l2 = (ab * ab).sum(-1).clamp_min(1e-9)
    out = []
    for s in range(0, len(q), chunk):
        x = q[s:s + chunk, None] - a[None]
        t = ((x * ab[None]).sum(-1) / l2).clamp(0, 1)
        out.append((x - t[..., None] * ab[None]).norm(dim=-1).amin(1))
    return torch.cat(out) if out else torch.zeros(0, device=q.device)


def solid_segments(map_polylines, map_mask, map_ids):
    """One scene: (M,P,F) polylines -> segment starts, ends, line ids of the solid lines."""
    typ = map_polylines[..., 4].round().long()
    solid = map_mask.bool() & torch.isin(typ, torch.tensor(SOLID_LINES, device=typ.device))
    seg = solid[:, :-1] & solid[:, 1:]
    ids = map_ids[:, None].expand(-1, seg.shape[1])
    return map_polylines[:, :-1, :2][seg].float(), map_polylines[:, 1:, :2][seg].float(), ids[seg]


class SceneReference:
    """Log-derived quantities of the B original scenes, shared by their K candidates."""

    def __init__(self, targets, history, batch, agent_lengths=None):
        self.gt, self.current, self.valid, self.types = (targets.future_pose.float(), targets.current_pose.float(),
                                                          targets.valid.bool(), targets.agent_type.long())
        self.length, self.width = dimensions(self.types, agent_lengths)
        B, N = self.types.shape
        alive0 = history[:, :, -1, 5] > 0.5
        vt = self.valid.transpose(1, 2)
        gap_gt = box_gap(self.gt, self.length, self.width)
        gap0 = box_gap(self.current[:, :, None], self.length, self.width)[:, 0]
        logged = ((gap_gt < 0) & vt[..., :, None] & vt[..., None, :]).any(1) | ((gap0 < 0) & alive0[:, :, None] & alive0[:, None, :])
        ped = self.types == 2
        eye = torch.eye(N, dtype=torch.bool, device=ped.device)
        self.pair_ok = ~eye & ~(ped[:, :, None] & ped[:, None, :]) & ~logged
        self.logged_overlap_pairs = (logged & ~eye).sum((1, 2)) / 2
        self.gt_min_gap = torch.where(vt[..., :, None] & vt[..., None, :] & ~eye[None, None], gap_gt, torch.full_like(gap_gt, 1e3)).amin((1, 3))
        self.road_support = reliable_road_mask(self.gt, self.current, self.valid, self.types,
                                               batch['map_polylines'], batch['map_mask'])
        self.gt_off, _ = road_geometry(self.gt, self.types, batch['map_polylines'], batch['map_mask'])
        # Logged path: history + future positions, consecutive valid frames.
        hist_xy, hist_valid = history[..., :2].float(), history[..., 5] > 0.5
        self.log_xy = torch.cat((hist_xy, self.gt[..., :2]), 2)
        self.log_valid = torch.cat((hist_valid, self.valid), 2)
        self.lanes = [lane_segments(batch['map_polylines'][b], batch['map_mask'][b], batch['map_is_lane'][b]) for b in range(B)]
        self.lines = [solid_segments(batch['map_polylines'][b], batch['map_mask'][b], batch['map_ids'][b]) for b in range(B)]
        self.gt_crossed = []          # per scene: (N, U) logged agent crossed line u ; and the id -> u index
        for b in range(B):
            a, e, ids = self.lines[b]
            uniq, inv = torch.unique(ids, return_inverse=True)
            crossed = torch.zeros(N, len(uniq), dtype=torch.bool, device=ids.device)
            if len(ids):
                p0, p1 = self.log_xy[b, :, :-1], self.log_xy[b, :, 1:]
                m = self.log_valid[b, :, :-1] & self.log_valid[b, :, 1:]
                hit = segments_cross(p0.reshape(-1, 2), p1.reshape(-1, 2), a, e).view(N, -1, len(ids)) & m[..., None]
                onehot = torch.nn.functional.one_hot(inv, len(uniq)).float()
                crossed = (hit.any(1).float() @ onehot) > 0
            self.gt_crossed.append((crossed, inv))


def offroad_excess(signed, support, centers, ref, K, lane_threshold):
    """signed (BK,N,T) corner excursion, support (BK,N,T), centers (BK,N,T,2) -> (BK,N) lane-filtered max excursion."""
    cand = support & (signed > 0)
    far = torch.zeros_like(cand)
    for r in torch.nonzero(cand.flatten(1).any(1)).flatten().tolist():
        a, e = ref.lanes[r // K]
        idx = cand[r].nonzero(as_tuple=True)
        far[r][idx] = point_segment_distance(centers[r][idx], a, e) > lane_threshold
    return torch.where(support & far, signed, torch.full_like(signed, -1e3)).amax(-1)


def candidate_violations(poses, ref, K, map_polylines, map_mask, lane_threshold=1.5):
    """poses (B*K,N,T,3) -> dict of (B*K,N) raw per-agent quantities (see module doc) + path lengths."""
    rep = lambda x: x.repeat_interleave(K, 0)  # noqa: E731
    poses = poses.float()
    BK, N, T, _ = poses.shape
    valid, types = rep(ref.valid), rep(ref.types)
    vt = valid.transpose(1, 2)
    gap = box_gap(poses, rep(ref.length), rep(ref.width))
    allowed = vt[..., :, None] & vt[..., None, :] & rep(ref.pair_ok)[:, None]
    coll_depth = -torch.where(allowed, gap, torch.full_like(gap, 1e3)).amin((1, 3))
    signed, _ = road_geometry(poses, types, rep(map_polylines), rep(map_mask))
    support = rep(ref.road_support)
    off_excess_raw = torch.where(support, signed, torch.full_like(signed, -1e3)).amax(-1)
    off_excess = offroad_excess(signed, support, poses[..., :2], ref, K, lane_threshold)
    current = rep(ref.current)
    path = torch.cat((current[:, :, None, :2], poses[..., :2]), 2)
    line_events = torch.zeros(BK, N, device=poses.device)
    vehicle = types == 1
    for r in range(BK):
        b = r // K
        a, e, ids = ref.lines[b]
        if not len(ids):
            continue
        crossed, inv = ref.gt_crossed[b]
        p0, p1 = path[r, :, :-1], path[r, :, 1:]
        hit = segments_cross(p0.reshape(-1, 2), p1.reshape(-1, 2), a, e).view(N, T, len(ids))
        new = hit & ~crossed[:, None, inv] & (valid[r] & vehicle[r][:, None])[..., None]
        line_events[r] = new.any(-1).sum(-1).float()
    step = (path[:, :, 1:] - path[:, :, :-1]).norm(dim=-1) * valid
    gt_path = torch.cat((current[:, :, None, :2], rep(ref.gt)[..., :2]), 2)
    gt_step = (gt_path[:, :, 1:] - gt_path[:, :, :-1]).norm(dim=-1) * valid
    dist = (poses[..., :2] - rep(ref.gt)[..., :2]).norm(dim=-1)
    ade = (dist * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    return dict(coll_depth=coll_depth, off_excess=off_excess, off_excess_raw=off_excess_raw, line_events=line_events,
                pred_path=step.sum(-1), gt_path=gt_step.sum(-1), ade=ade)


def violation_counts(v, depth=0.0, margin=0.0):
    """Raw per-agent quantities -> (...,3) agent counts: collision, offroad, solid line."""
    return torch.stack(((v['coll_depth'] > depth).sum(-1), (v['off_excess'] > margin).sum(-1),
                        (v['line_events'] > 0).sum(-1)), -1)


def select_winners(d, counts, ratio, slack=0.0):
    """d (B,K) motion distances, counts (B,K,3) -> winners (B,), pool mask (B,K).

    Pool = candidates within max(ratio*min d, min d + slack). In the pool: fewest collision agents, then offroad,
    then solid-line, then smallest d. ratio=1, slack=0 reduces to plain MoN (argmin d) up to exact ties."""
    dmin = d.min(1, keepdim=True).values
    pool = d <= torch.maximum(dmin * ratio, dmin + slack)
    c = counts.long().clamp(max=63)
    key = c[..., 0] * 4096 + c[..., 1] * 64 + c[..., 2]
    key = torch.where(pool, key, torch.full_like(key, 1 << 40))
    best = pool & (key == key.min(1, keepdim=True).values)
    winners = torch.where(best, d, torch.full_like(d, float('inf'))).argmin(1)
    return winners, pool
