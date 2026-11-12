# 实验 1b-v2 结果：z v2（加同向车对），解冻去噪器，loss 集中在有 z 的车上

512 场景；1b-v2 三种模式每场景 16 个 rollout、同一组种子；85k 基线是实验 1 的 32 个 rollout（种子不同，只比总体）。标签 v2 车对 5189 对，开始时 z 开着的 1300 对。

| 指标 | 85k frozen, unknown (32 rollouts) | 1b-v2 unknown | 1b-v2 oracle | 1b-v2 reversed |
|---|---|---|---|---|
| ADE | 2.4686 | 2.1399 | 2.1302 | 2.1360 |
| minADE | 1.4281 | 1.4055 | 1.3961 | 1.4026 |
| CPD | 0.2201 | 0.2296 | 0.2293 | 0.2297 |
| 碰撞 场景-rollout | 0.4130 | 0.5137 | 0.5009 | 0.5044 |
| 碰撞 agent-time | 0.0248 | 0.0300 | 0.0288 | 0.0291 |
| offroad 场景-rollout | 0.3560 | 0.3931 | 0.3917 | 0.3936 |
| all pairs：车对碰撞 | 0.014 | 0.020 | 0.018 | 0.018 |
| all pairs：冲突中按日志顺序 | 0.992 | 0.992 | 0.992 | 0.992 |
| all pairs：冲突中反序 | 0.008 | 0.008 | 0.008 | 0.008 |
| all pairs：没相遇 | 0.196 | 0.143 | 0.140 | 0.141 |
| all pairs：车对数 | 5189 | 5189 | 5189 | 5189 |
| z set, crossing：车对碰撞 | 0.091 | 0.092 | 0.079 | 0.083 |
| z set, crossing：冲突中按日志顺序 | 0.826 | 0.885 | 0.892 | 0.883 |
| z set, crossing：冲突中反序 | 0.174 | 0.115 | 0.108 | 0.117 |
| z set, crossing：没相遇 | 0.574 | 0.452 | 0.440 | 0.439 |
| z set, crossing：车对数 | 156 | 156 | 156 | 156 |
| z set, same direction：车对碰撞 | 0.041 | 0.069 | 0.064 | 0.065 |
| z set, same direction：冲突中按日志顺序 | 0.922 | 0.913 | 0.909 | 0.905 |
| z set, same direction：冲突中反序 | 0.078 | 0.087 | 0.091 | 0.095 |
| z set, same direction：没相遇 | 0.588 | 0.485 | 0.479 | 0.483 |
| z set, same direction：车对数 | 183 | 183 | 183 | 183 |
| z set, all：车对碰撞 | 0.064 | 0.079 | 0.071 | 0.073 |
| z set, all：冲突中按日志顺序 | 0.876 | 0.899 | 0.900 | 0.895 |
| z set, all：冲突中反序 | 0.124 | 0.101 | 0.100 | 0.105 |
| z set, all：没相遇 | 0.581 | 0.470 | 0.461 | 0.463 |
| z set, all：车对数 | 339 | 339 | 339 | 339 |

## 场景配对 bootstrap（相对 1b-v2 unknown，同一组种子）

- 1b-v2 oracle scene_collision_all: -0.0128，95% CI [-0.0175, -0.0084]，n=512，P(Δ≥0)≈0.000
- 1b-v2 oracle pair_collision_by_scene [z set, all]: -0.0090，95% CI [-0.0170, -0.0023]，n=205，P(Δ≥0)≈0.002
- 1b-v2 oracle reversed_share_by_scene [z set, all]: -0.0003，95% CI [-0.0094, +0.0084]，n=173，P(Δ≥0)≈0.467
- 1b-v2 oracle pair_collision_by_scene [z set, crossing]: -0.0151，95% CI [-0.0297, -0.0033]，n=104，P(Δ≥0)≈0.004
- 1b-v2 oracle reversed_share_by_scene [z set, crossing]: -0.0070，95% CI [-0.0192, +0.0030]，n=81，P(Δ≥0)≈0.101
- 1b-v2 oracle pair_collision_by_scene [z set, same direction]: -0.0025，95% CI [-0.0082, +0.0028]，n=111，P(Δ≥0)≈0.180
- 1b-v2 oracle reversed_share_by_scene [z set, same direction]: +0.0045，95% CI [-0.0077, +0.0172]，n=98，P(Δ≥0)≈0.770
- 1b-v2 reversed scene_collision_all: -0.0093，95% CI [-0.0135, -0.0051]，n=512，P(Δ≥0)≈0.000
- 1b-v2 reversed pair_collision_by_scene [z set, all]: -0.0081，95% CI [-0.0151, -0.0026]，n=205，P(Δ≥0)≈0.001
- 1b-v2 reversed reversed_share_by_scene [z set, all]: +0.0009，95% CI [-0.0065, +0.0075]，n=173，P(Δ≥0)≈0.613
- 1b-v2 reversed pair_collision_by_scene [z set, crossing]: -0.0112，95% CI [-0.0235, -0.0020]，n=104，P(Δ≥0)≈0.005
- 1b-v2 reversed reversed_share_by_scene [z set, crossing]: -0.0007，95% CI [-0.0108, +0.0081]，n=81，P(Δ≥0)≈0.457
- 1b-v2 reversed pair_collision_by_scene [z set, same direction]: -0.0041，95% CI [-0.0099, +0.0011]，n=111，P(Δ≥0)≈0.066
- 1b-v2 reversed reversed_share_by_scene [z set, same direction]: +0.0024，95% CI [-0.0074, +0.0110]，n=98，P(Δ≥0)≈0.713

## 训练（EMA 权重，验证集 flow loss；gain = 真值 z 相对未知的下降 %）

| step | all_loss_unknown | all_gain_pct | z_gain_pct | crossing_gain_pct | same_dir_gain_pct | yielder_gain_pct | priority_gain_pct | other_gain_pct |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0556 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1000 | 0.0554 | 0.0149 | 0.0475 | 0.0600 | 0.0306 | 0.0398 | 0.0341 | 0.0035 |
| 2000 | 0.0553 | 0.0262 | 0.0841 | 0.0690 | 0.0750 | 0.0545 | 0.0864 | 0.0059 |
| 3000 | 0.0553 | 0.0329 | 0.1257 | -0.0153 | 0.1578 | 0.1184 | 0.1494 | 0.0004 |
| 4000 | 0.0553 | 0.0527 | 0.1770 | 0.3880 | 0.1424 | 0.2148 | 0.1257 | 0.0090 |
| 5000 | 0.0553 | 0.0590 | 0.2107 | 0.2783 | 0.2026 | 0.2298 | 0.2031 | 0.0058 |
| 6000 | 0.0552 | 0.0586 | 0.2022 | 0.4316 | 0.1666 | 0.1996 | 0.1921 | 0.0084 |
| 7000 | 0.0552 | 0.0642 | 0.2250 | 0.5455 | 0.1737 | 0.2537 | 0.1819 | 0.0079 |
| 8000 | 0.0552 | 0.0687 | 0.2456 | 0.5293 | 0.1944 | 0.2681 | 0.1969 | 0.0068 |
| 9000 | 0.0552 | 0.0668 | 0.2394 | 0.5158 | 0.1982 | 0.2534 | 0.2206 | 0.0064 |
| 10000 | 0.0552 | 0.0670 | 0.2358 | 0.4700 | 0.2002 | 0.2491 | 0.2192 | 0.0080 |
- 1b-v2 oracle 重规划时车对状态比例：{"pre_entry_log_order": 0.026, "first_in_overlap": 0.053, "second_in_overlap": 0.0, "both_in_unknown": 0.001, "released": 0.265, "deviated_unknown": 0.282, "tie_unknown": 0.002, "not_alive": 0.038, "same_dir_far": 0.276, "same_dir_both_in_front_first": 0.057}
- 1b-v2 oracle bias：{"yield": {"edges": 180736, "per_head_mean": [-0.4362, -0.0814, 0.6232, 0.3132, 0.566, -0.0001, 0.4947, 0.1665], "mean_abs": 0.5244664320786201, "max_head_mean": 1.1867087956020943, "frac_any_head_abs_gt_2.5": 0.01385999468838527}, "priority": {"edges": 180736, "per_head_mean": [0.0285, -0.2763, 0.4652, -0.7813, -0.0113, 1.4305, 0.5076, -0.5475], "mean_abs": 0.6959118282921394, "max_head_mean": 1.5821940271330934, "frac_any_head_abs_gt_2.5": 0.009516643059490085}}
- 1b-v2 reversed 重规划时车对状态比例：{"pre_entry_log_order": 0.026, "first_in_overlap": 0.053, "second_in_overlap": 0.0, "both_in_unknown": 0.001, "released": 0.265, "deviated_unknown": 0.282, "tie_unknown": 0.002, "not_alive": 0.038, "same_dir_far": 0.276, "same_dir_both_in_front_first": 0.057}
- 1b-v2 reversed bias：{"yield": {"edges": 180831, "per_head_mean": [-0.519, -0.1658, 0.563, 0.2283, 0.8183, -0.1453, 1.0633, 0.3105], "mean_abs": 0.6304933939519564, "max_head_mean": 1.6027094343491477, "frac_any_head_abs_gt_2.5": 0.013084039794061858}, "priority": {"edges": 180831, "per_head_mean": [0.0897, -0.1673, 0.4734, -0.7572, -0.285, 1.6162, -0.1109, -0.7141], "mean_abs": 0.7173711781120742, "max_head_mean": 1.7052029791458339, "frac_any_head_abs_gt_2.5": 0.00882592033445593}}
