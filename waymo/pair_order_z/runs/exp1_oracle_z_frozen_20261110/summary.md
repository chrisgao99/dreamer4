# 实验 1 结果：oracle z，冻结 85k WM，只训练 g

512 场景 × 32 rollout，同一组种子。有标签车对的场景 115 个。

| 指标 | unknown | oracle | reversed |
|---|---|---|---|
| ADE | 2.4686 | 2.4679 | 2.4683 |
| minADE | 1.4281 | 1.4323 | 1.4411 |
| CPD | 0.2201 | 0.2200 | 0.2200 |
| 碰撞 场景-rollout（全部） | 0.4130 | 0.4134 | 0.4133 |
| 碰撞 场景-rollout（有标签场景） | 0.3924 | 0.3943 | 0.3937 |
| 碰撞 场景-rollout（无标签场景） | 0.4189 | 0.4189 | 0.4189 |
| 碰撞 agent-time（全部） | 0.0248 | 0.0248 | 0.0248 |
| 碰撞 agent-time（有标签场景） | 0.0196 | 0.0195 | 0.0195 |
| 标签车对之间的碰撞 | 0.0732 | 0.0687 | 0.0703 |
| offroad 场景-rollout | 0.3560 | 0.3557 | 0.3561 |
| 一直不起步 | 0.3764 | 0.3752 | 0.3759 |
| 后 4 s 卡住 | 0.1044 | 0.1041 | 0.1042 |
| 通行顺序：same_as_log | 0.378 | 0.384 | 0.384 |
| 通行顺序：reversed | 0.070 | 0.070 | 0.070 |
| 通行顺序：no_conflict | 0.545 | 0.539 | 0.539 |
| 通行顺序：tie | 0.007 | 0.007 | 0.007 |

## 场景配对 bootstrap（相对 unknown，负数 = 碰撞下降）

- oracle scene_collision_all: +0.0004，95% CI [-0.0007, +0.0016]，n=512，P(Δ≥0)≈0.777
- oracle scene_collision_labelled: +0.0019，95% CI [-0.0033, +0.0071]，n=115，P(Δ≥0)≈0.782
- oracle pair_collision_by_scene: -0.0043，95% CI [-0.0104, +0.0010]，n=115，P(Δ≥0)≈0.068
- reversed scene_collision_all: +0.0003，95% CI [-0.0008, +0.0015]，n=512，P(Δ≥0)≈0.713
- reversed scene_collision_labelled: +0.0014，95% CI [-0.0035, +0.0065]，n=115，P(Δ≥0)≈0.717
- reversed pair_collision_by_scene: -0.0020，95% CI [-0.0076, +0.0036]，n=115，P(Δ≥0)≈0.242

## 训练（验证集 flow loss：真值 z 相对 未知 的下降 %）

best step 1500：{"val_all_gain_pct": 0.08483746320429064, "val_active_gain_pct": 0.5658490485354094, "val_yielder_gain_pct": 0.8598494422744335, "val_priority_gain_pct": 0.09760771602952414, "val_active_in_window_gain_pct": 0.7911383039362561, "val_active_beyond_window_gain_pct": 0.3836351721922642}
- oracle bias：{"yield": {"edges": 35170, "per_head_mean": [-1.6223, -0.7316, -2.8944, 2.3594, -2.2591, -0.8223, 0.0189, -2.5466], "mean_abs": 1.9598313706942097, "max_head_mean": 2.3878826751588997, "frac_any_head_abs_gt_2.5": 0.9425362524879158}, "priority": {"edges": 35170, "per_head_mean": [-0.2077, 0.1979, -0.973, 0.8527, -0.1785, -0.0717, 0.1919, -0.3992], "mean_abs": 1.034452231044894, "max_head_mean": 1.63800763971288, "frac_any_head_abs_gt_2.5": 0.214046061984646}}
- oracle 重规划时车对状态比例：{"pre_entry_log_order": 0.261, "first_in_overlap": 0.093, "second_in_overlap": 0.002, "both_in_unknown": 0.001, "released": 0.345, "deviated_unknown": 0.277, "tie_unknown": 0.0, "not_alive": 0.021}
- reversed bias：{"yield": {"edges": 35107, "per_head_mean": [-1.1957, -0.4989, -2.5974, 2.0645, -1.5587, -0.6366, 0.5434, -2.0343], "mean_abs": 1.666829580816384, "max_head_mean": 2.1958116062078235, "frac_any_head_abs_gt_2.5": 0.7770245250234996}, "priority": {"edges": 35107, "per_head_mean": [-0.5555, -0.0016, -1.7886, 1.2449, -0.979, -0.155, -0.2999, -1.1792], "mean_abs": 1.2670422708172906, "max_head_mean": 1.6577444744466046, "frac_any_head_abs_gt_2.5": 0.42376164297718405}}
- reversed 重规划时车对状态比例：{"pre_entry_log_order": 0.261, "first_in_overlap": 0.092, "second_in_overlap": 0.002, "both_in_unknown": 0.001, "released": 0.344, "deviated_unknown": 0.279, "tie_unknown": 0.0, "not_alive": 0.021}
