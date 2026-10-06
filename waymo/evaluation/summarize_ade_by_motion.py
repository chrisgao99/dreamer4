"""Group saved joint rollouts by GT motion, averaging agents and samples equally."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    rows = []
    scene_indices = []
    for path in sorted((root / "rollout_batches").glob("batch_*.npz")):
        with np.load(path) as data:
            valid = data["valid"].astype(bool)
            assert not np.any(valid[..., 1:] & ~valid[..., :-1]), "Expected contiguous valid prefixes"
            count = valid.sum(axis=-1)
            gt = data["gt_pose"][..., :2]
            initial = data["initial_pose"][..., :2]
            last_index = np.maximum(count - 1, 0)
            last = np.take_along_axis(gt, last_index[..., None, None], axis=2).squeeze(2)
            net = np.linalg.norm(last - initial, axis=-1)
            previous = np.concatenate((initial[:, :, None], gt[:, :, :-1]), axis=2)
            length = (np.linalg.norm(gt - previous, axis=-1) * valid).sum(axis=-1)
            ade = data["candidate_agent_ade"]
            scene_indices.extend(data["dataset_index"].tolist())
            for b, n in zip(*np.where(count > 0)):
                samples = ade[b, :, n]
                assert np.isfinite(samples).all()
                rows.append((count[b, n], net[b, n], length[b, n], samples.mean(), samples.min()))
    assert rows, "No saved trajectory batches found"
    assert len(scene_indices) == len(set(scene_indices)), "Duplicate scenes"
    values = np.asarray(rows, dtype=np.float64)
    result = {
        "source": str(root.resolve()),
        "scene_count": len(scene_indices),
        "valid_agent_tracks": len(values),
        "aggregation": "Equal weight per scene-agent track and per rollout. minADE is per-agent best sample, not joint scene minADE.",
        "groups": {},
    }
    for coverage in ("full80", "any_valid_prefix"):
        eligible = values[:, 0] == 80 if coverage == "full80" else values[:, 0] > 0
        for distance_name, column in (("net_displacement", 1), ("path_length", 2)):
            distance = values[:, column]
            output = []
            for label, mask in (("<2m", distance < 2), ("2-20m", (distance >= 2) & (distance <= 20)), (">20m", distance > 20)):
                selected = values[eligible & mask]
                output.append({
                    "group": label,
                    "agent_tracks": len(selected),
                    "fraction": float(len(selected) / max(1, eligible.sum())),
                    "mean_ade_m": float(selected[:, 3].mean()) if len(selected) else None,
                    "per_agent_minade_m": float(selected[:, 4].mean()) if len(selected) else None,
                    "mean_valid_steps": float(selected[:, 0].mean()) if len(selected) else None,
                })
            result["groups"][f"{coverage}/{distance_name}"] = output
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    output = args.root / "ade_by_gt_motion.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
