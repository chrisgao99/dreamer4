# Step-40k 极端随机 focus-action 配对测试

## 结论

这个 checkpoint 明显使用了 focus action：极端随机 action 对 focus agent 的影响很强，并且会传播到其他 agent，但后者显著更弱。

- H90 时，focus agent 相对正常-action rollout 的响应为 ADE 48.32 m、FDE 75.20 m。
- H90 时，其他 agent 的平均响应为 ADE 4.51 m、FDE 9.23 m，约为 focus ADE 响应的 9.3%。
- 传播不是均匀的小扰动：H90 有 52.0% 的有效 other-agent/timestep 响应超过 1 m，20.7% 超过 5 m；每个场景中最受影响 other agent/timestep 的响应平均为 47.08 m。
- 对 GT 误差的恶化小于输出本身的响应，尤其是其他 agent：H90 focus ADE 对 GT 增加 47.59 m，other ADE 对 GT 增加 1.55 m。这说明 other-agent 位移有一部分与原有预测误差方向相消或近似正交，不能只看 GT metric 判断 action sensitivity。

## 主要结果

所有单位均为米。`response` 是相同 scene、相同初始 latent、相同 rollout noise 下，极端随机 action 与 recorded action 两条模型输出之间的距离。95% CI 是 128 个场景均值的 normal-approximation CI。

| Horizon | Focus response ADE | Focus response FDE | Other response ADE | Other response FDE | Other/Focus ADE | Other >1 m | Other >5 m |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 4.23 [2.66, 5.79] | 6.30 [4.66, 7.94] | 0.80 [0.62, 0.99] | 1.30 [0.96, 1.64] | 19.0% | 7.0% | 2.9% |
| 30 | 12.68 [10.83, 14.53] | 29.01 [25.54, 32.49] | 1.38 [1.15, 1.62] | 2.10 [1.80, 2.41] | 10.9% | 22.2% | 5.2% |
| 50 | 29.92 [27.60, 32.25] | 64.86 [60.08, 69.63] | 2.10 [1.82, 2.38] | 3.87 [3.32, 4.42] | 7.0% | 36.5% | 10.1% |
| 80 | 44.87 [42.48, 47.26] | 75.25 [70.57, 79.94] | 3.79 [3.23, 4.35] | 7.70 [6.37, 9.03] | 8.4% | 49.1% | 18.4% |
| 90 | 48.32 [45.87, 50.76] | 75.20 [69.98, 80.42] | 4.51 [3.81, 5.21] | 9.23 [7.53, 10.93] | 9.3% | 52.0% | 20.7% |

## 相对 GT 的误差变化

| Horizon | Focus baseline ADE | Focus random ADE | Focus ADE increase | Other baseline ADE | Other random ADE | Other ADE increase |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.33 | 4.19 | +3.86 | 2.82 | 3.01 | +0.19 |
| 30 | 0.38 | 12.71 | +12.34 | 3.82 | 4.06 | +0.25 |
| 50 | 0.52 | 29.96 | +29.43 | 4.55 | 4.99 | +0.44 |
| 80 | 0.70 | 44.92 | +44.21 | 5.72 | 6.89 | +1.17 |
| 90 | 0.78 | 48.37 | +47.59 | 6.13 | 7.67 | +1.55 |

H90 的 focus/other FDE 对 GT 分别增加 73.69 m 和 3.82 m。

## 协议

- checkpoint: `step_00040000.pt`
- validation: 固定 `val_random128_seed0_manifest.json`，128 scenes
- rollout: ctx=1，共享 H90，H10/H30/H50/H80/H90 都是同一 rollout 的 prefix
- solver: shortcut-4, d=0.25
- control: dataset recorded focus action
- treatment: 每个未来 timestep 独立随机化全部 focus action 字段
  - delta-x、delta-y：各自 Uniform[-50, 50] m / 0.1 s
  - delta-yaw：Uniform[-pi, pi] rad / 0.1 s
  - signed speed：Uniform[-200, 200] m/s
  - vx、vy：各自 Uniform[-200, 200] m/s
  - valid=1，action mask 的字段 0..6 全开
- random seed: 20260807
- control/treatment 在每个场景恢复同一 PyTorch RNG state，因此 rollout noise 完全配对。

H90 实际抽样均值：delta-XY norm 38.40 m/tick，abs(delta-yaw) 90.43 deg/tick，abs(speed) 99.96 m/s，velocity norm 152.97 m/s。这个测试是有意构造的强 OOD stress test，不代表现实可执行 control 的效果。

## 完整性检查

- 128/128 scenes 完成；全部 420 个 aggregate scalar 有限，无 NaN/Inf。
- control rollout 的所有已有指标与此前同 checkpoint、同 manifest、同 protocol 的 baseline JSON 完全一致，最大绝对差为 0。
