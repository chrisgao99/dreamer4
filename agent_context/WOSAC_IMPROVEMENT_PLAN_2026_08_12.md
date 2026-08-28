# WOSAC 提升方案：保留随机性并改善交互与地图约束

日期：2026-08-12

项目根目录：`/p/yufeng/tri30/dreamer4`

状态：研究结论和第一轮训练方案已确定；MoN/物理约束/diversity-retention 训练尚未实现或启动。

## 1. 目标

下一轮 world-model rollout fine-tuning 要同时做到：

1. 保留 Stage 1 对输入噪声的响应，不再让不同噪声收敛到几乎相同的未来；
2. 保持 rollout fine-tuning 带来的 ADE 改善；
3. 降低多车联合 rollout 的 collision 和 offroad；
4. 让不同样本对应不同但联合一致、物理可行的交互模式，而不是无意义的轨迹扰动。

核心原则：

> 用 joint Minimum-over-N 避免把所有噪声强制拟合到唯一 GT；对所有候选施加 collision/offroad/kinematic 约束；用 Stage 1 teacher 保留 noise-dependent diversity。

## 2. 已确认的问题与证据

### 2.1 Rollout fine-tuning 导致 mode collapse

已由当前 Stage 1 与 rollout-finetuned 模型的生成对比确认：

- Stage 1 训练结束时，不同随机噪声能生成明显不同的 joint futures；
- rollout fine-tuning 后 ADE 降低；
- 但不同噪声生成的轨迹几乎相同，说明模型在 fine-tuning 中学会忽略噪声。

最可能的机制是：每个噪声样本都被同一个单一 GT future 做回归监督，因此优化目标会把所有样本拉向同一个条件均值/单一 mode。

### 2.2 当前 WOSAC-style 结果

当前评估使用的 rollout 模型：

```text
waymo/checkpoints/
  waymo_wm_original_stmlayer_3_stage/
  waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/
  step_00040000.pt
```

1000 focus views（500个场景，每个场景两个 OOI）的结果：

```text
realism meta-metric:           0.4211
kinematic bucket:              0.4679
interactive bucket:            0.3608
map-based bucket:              0.4719
ADE:                           2.6302 m
MinADE over 32 joint samples:  2.6067 m
collision rate:                28.11%
offroad rate:                  42.42%
traffic-light violation rate:   1.47%
```

结果位置：

```text
waymo/eval_results/wosac/
  wosac_oracle_focus_step40k_val1000_k4_rb8/wosac_summary.json
```

解释：

- interactive bucket 最低，collision 和 nearest-agent relation 是主要短板；
- offroad 很高，道路边界/drivable-area 约束不足；
- ADE 与 best-of-32 MinADE 只差约 `0.0235 m`（约0.9%），32次采样几乎没有带来 best-of-N 收益，是 mode collapse 的强证据。

### 2.3 当前评估不是官方 WOSAC

以上数值是内部 WOSAC-style diagnostic，不可与 leaderboard 直接比较：

- 最多保留 focus 附近32个 agent，官方可要求最多128个；
- focus agent 使用 logged oracle future；
- 未选中的 agent 不补齐；
- 每个 OOI 建立独立的 focus-centered view；
- collision/interaction 只在保留的 agent 子集中计算；
- 模型预测 planar `x/y/heading`，`z` 使用 logged value。

这些限制通常使任务比官方设定更容易。因此即使在当前协议中 collision/offroad 仍高，也应视为明确问题。

### 2.4 Tokenizer 分块诊断

固定 val128 的纯 tokenizer GT -> encode -> decode 测试表明：

```text
                         stride30 keep-first   stride16 center   stride16 keep-first
agent ADE                      0.4375 m             0.4304 m          0.4186 m
delta-XY error                 0.0361 m             0.0350 m          0.0297 m
kinematic-XY error             0.0348 m             0.0339 m          0.0295 m
maximum delta-XY spike         0.4045 m             0.2124 m          0.1376 m
```

`stride16 keep-first` 是三者中最好的 tokenizer 使用方式。`center` 在第24帧提前切换 chunk，会引入更多中间边界，不作为首选。

但是第一轮 MoN loss ablation 应先固定现有 `chunk32/stride30/keep-first`，避免同时改变 tokenizer 分布和训练目标而无法归因。选出有效 loss 组合后，再做独立的 `stride16 keep-first` matched fine-tuning 实验。

## 3. 第一轮训练：不先改网络结构

### 3.1 初始化与固定项

Stage 1 teacher/初始化候选（启动前必须再次核对这是产生强随机性的具体 checkpoint）：

```text
waymo/checkpoints/
  waymo_wm_original_stmlayer_3_stage/
  waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m/
  best.pt
```

第一轮固定：

- tokenizer checkpoint不变并冻结；
- world-model architecture不变；
- map-conditioning architecture不变；
- context、action conditioning、shortcut sampler和数据顺序不变；
- tokenizer先保持 `window=32, stride=30, keep-first`；
- 从同一个有随机性的 Stage 1 checkpoint 初始化所有实验；
- paired evaluation 使用同一组 scenes、同一组 noise seeds。

冻结 tokenizer 参数，但 decoded loss 必须允许梯度经过 frozen decoder 回传到 world-model latent。不能在 decoder 前 `detach` world-model output。

### 3.2 Joint Minimum-over-N（MoN）

同一个 context 在训练时采样 `N=4` 个独立噪声，生成4个完整 joint rollouts：

```text
same context + eps_1 -> Y_1
same context + eps_2 -> Y_2
same context + eps_3 -> Y_3
same context + eps_4 -> Y_4
```

对每个候选计算与 GT 的整场误差：

```text
E_n = joint trajectory/latent reconstruction error(Y_n, GT)
winner = argmin_n E_n
L_MoN = E_winner
```

必须遵守：

- winner 按完整 scene-level joint rollout 选择；
- 所有有效 future agents 共享同一个 winner；
- 不允许每个 agent 独立挑选不同 winner，否则得到的不是一个真实存在的联合未来；
- `argmin` 只做离散选择，不对 winner index 反传；
- GT reconstruction/regression loss只回传 winner；
- 先用 `N=4` 控制显存，验证有效后再测试 `N=8`。

MoN解决的问题：不同噪声不再全部被要求拟合同一个 GT；只需至少一个候选覆盖 observed future，其余候选可以表达其他合理模式。

MoN不能单独保证非 winner 合理，因此必须配合下一节对所有候选的物理约束。

### 3.3 所有候选都施加物理约束

```text
L_physical = mean_n(
    lambda_collision * L_collision(Y_n)
  + lambda_offroad   * L_offroad(Y_n)
  + lambda_kinematic * L_kinematic(Y_n)
)
```

#### Interaction / collision

第一版目标：降低碰撞，同时避免用“驶出道路”来绕开其他车辆。

- 使用车辆 oriented boxes，而不是只比较 agent center distance；
- 对 box overlap/penetration 使用连续的 soft penalty；
- 增加较弱的 near-distance/TTC risk term，使“刚好没重叠但极危险”也有梯度；
- 对所有有效 agent pairs计算；
- 对 OOI pair、低TTC pair、未来路径有冲突的pair提高权重；
- loss作用于所有 `N` 个候选，而不仅是 MoN winner。

计算量控制：先用 OOI pair + 每个 agent 的 top-k nearby/conflict pairs，而不是无条件计算所有 `32 x 32` pairs。

#### Map / offroad

第一版目标：让 vehicle footprint 而不只是车辆中心保持在 drivable region 内。

推荐：

- 从 drivable-area polygon/road edge建立 signed distance field（SDF）或可微距离查询；
- 检查 oriented vehicle box 的四角和必要的边缘采样点；
- 接近道路边界时即产生连续梯度，越过边界后增加惩罚；
- 加入较弱的 lane-heading alignment loss，防止车辆虽然未出界但横穿车道；
- 与 collision loss联合训练，避免模型通过offroad来规避碰撞。

实现前必须确认现有 vector dataset 是否保存了完整 drivable-area/road-edge geometry。若只有稀疏 polyline，第一版可先做 road-edge/lane-distance proxy，但需要在日志中明确其不是严格 offroad loss。

#### Kinematic consistency

保留并作用于所有候选：

- delta-XY continuity；
- predicted velocity与位置差分一致；
- speed/yaw与位置更新一致；
- acceleration和jerk正则；
- 单独记录 tokenizer chunk seams 附近的误差。

### 3.4 Stage 1 diversity retention

MoN减少 collapse 的压力，但不能保证 student一定继续使用噪声。使用冻结 Stage 1 teacher显式保留相同 noise 的相对输出差异。

对相同 context 和相同 `eps_1 ... eps_N`：

```text
Stage 1 teacher -> Y^T_1 ... Y^T_N
student         -> Y^S_1 ... Y^S_N
```

计算 teacher/student 的 pairwise trajectory distance：

```text
D_T(i,j) = distance(Y^T_i, Y^T_j)
D_S(i,j) = distance(Y^S_i, Y^S_j)
```

第一版建议用 diversity floor，而不是要求 student逐条复制 teacher：

```text
L_div = mean_{i<j} max(0, rho * stopgrad(D_T(i,j)) - D_S(i,j))
```

建议从 `rho=0.5~0.7` 的小范围开始。含义是 student 至少保留 Stage 1 的一部分噪声响应，但允许为了 ADE/物理合理性修改具体轨迹。

距离定义应：

- 对所有有效非focus agent的全时域或终点 `x/y` 计算；
- 使用同一个 scene-level mask；
- 归一化 agent数和有效帧数；
- 同时报告 joint-scene distance与agent-level分布；
- 可对不合理的 teacher轨迹设置距离上限，避免复刻 Stage 1 的碰撞/offroad diversity。

physical losses作用于所有候选，因此 diversity不能靠碰撞、offroad或高频抖动来满足。

后续版本可将 `D` 替换或补充为 interaction-embedding distance，使差异表达“谁先走/谁让行”，而不仅是XY分散。

### 3.5 总训练目标

```text
L_total =
    L_shortcut
  + lambda_MoN       * L_MoN
  + lambda_collision * L_collision_all_samples
  + lambda_offroad   * L_offroad_all_samples
  + lambda_kinematic * L_kinematic_all_samples
  + lambda_div       * L_diversity_retention
```

注意：

- 现有 rollout reconstruction若对每个noise都回归同一个GT，必须由 joint MoN替代，不能在保留原强回归的同时仅追加MoN；
- `lambda_div` 在 fine-tuning 初期较强，随后可衰减；
- collision/offroad 应逐步warm up，避免训练初期大梯度破坏已有动力学；
- 所有loss分别记录，不能只记录总loss。

## 4. 第一轮最小实验矩阵

所有实验必须从同一个 Stage 1 checkpoint初始化，并使用相同数据、步数和随机种子。

| Run | Joint MoN | All-sample physical loss | Stage 1 diversity retention | 目的 |
|---|---:|---:|---:|---|
| Control | No | No | No | 复现当前 rollout FT 的 ADE下降与collapse |
| A | N=4 | No | No | 判断MoN本身能否保留随机性 |
| B | N=4 | collision + offroad + kinematic | No | 判断物理loss能否降低WOSAC违规 |
| C | N=4 | collision + offroad + kinematic | Yes | 完整推荐方案，判断teacher retention是否必要 |

推荐先用短训练筛选，而不是直接跑完整50k：

1. 每组先跑相同的2k~5k optimizer steps；
2. 固定 val128，每500或1000 steps检查 ADE、collision/offroad proxy和diversity；
3. 只有保持多样性且不显著损害ADE的组继续到10k；
4. 候选checkpoint先跑固定 WOSAC-style 100 views；
5. 最终候选再跑1000 views。

## 5. Horizon 与优化稳定性

推荐使用 curriculum：

```text
H20 -> H40 -> H80/H90
```

同时混合 short- and long-horizon batches，防止只优化长时终点而破坏短时运动质量。

训练建议：

- 从 Stage 1 使用较小 learning rate开始；
- 保留 Stage 1 shortcut objective batch，避免只训练 autoregressive rollout；
- decoded rollout loss从短horizon warm up；
- gradient clipping保持开启；
- 第一次实验不要同时增加pair-token新结构、替换map encoder或更换tokenizer；
- checkpoint选择采用多目标指标，而不是只看 validation loss/ADE。

## 6. Interaction contrastive learning 的接入顺序

当前 pairwise contrastive learning 可帮助模型区分：

- 谁先走、谁让行；
- following、crossing、converging等关系；
- 相似几何下不同的交互响应。

第一轮不改 architecture，先把已有 pair encoder作为辅助评估/监督：

1. 对 GT future与每个生成future提取 pair interaction embedding；
2. 将 embedding distance作为 MoN winner score的一个较弱分量；
3. 记录不同noise是否覆盖不同interaction modes；
4. collision/TTC仍提供物理约束，不能只依赖contrastive embedding。

后续版本再考虑：

- 将pair tokens通过cross-attention输入world model；
- 从future pair embeddings聚类 interaction prototypes；
- 采样 scene-level interaction mode，并在完整 rollout 中保持不变；
- 增加 generated trajectory -> mode prediction / mutual-information loss，防止模型忽略mode。

不要让每个agent独立采样“先走/让行”mode；scene-level mode必须约束整组agent，否则会增加互相冲突的决策和碰撞。

## 7. Map 使用诊断与后续方案

在修改map architecture之前先做三组严格配对测试：

```text
normal map tokens
zeroed/masked map tokens
map tokens shuffled across scenes
```

只移除 world-model map conditioning，保持 tokenizer map conditioning不变，避免混淆 tokenizer reconstruction与world-model map usage。

记录：ADE、offroad、road-edge likelihood、map bucket、collision和diversity。

后续增强方案：

- agent-to-polyline相对位置与相对heading encoding；
- stronger/gated map cross-attention；
- nearby lane/road-edge token selection；
- drivable-area、road-edge和lane-heading auxiliary prediction；
- map dropout，防止模型只记特定map token pattern；
- 检查坐标变换、padding mask、地图覆盖范围和traffic-light对齐。

## 8. Tokenizer oracle 与 Stage 对照诊断

训练前后都保留以下归因测试：

### Tokenizer oracle WOSAC-style test

```text
GT trajectory -> tokenizer encode -> tokenizer decode -> WOSAC metrics
```

比较 raw GT 与 reconstructed GT 的 collision/offroad增量。GT本身可能有少量事件，因此看增量而不是假设GT为零。

### Stage 1 vs rollout-finetuned model

严格保持相同 tokenizer、scene、context、sampler、noise seeds和rollout设置，比较：

- ADE/FDE与MinADE；
- collision/offroad；
- kinematic/interactive/map WOSAC buckets；
- sample pairwise distance、endpoint std和interaction-mode coverage。

## 9. 必须监控的指标

### Accuracy

- mean ADE/FDE across all samples；
- best-of-N MinADE/MinFDE；
- focus与nonfocus分别报告；
- ADE-MinADE gap只能作为多样性提示，不能单独作为质量指标。

### Stochasticity

- full-horizon mean pairwise trajectory distance；
- 8 s endpoint spatial standard deviation；
- agent-level endpoint std P50/P90/max；
- latent variance与decoded trajectory variance；
- noise-to-output sensitivity；
- interaction-mode coverage；
- 每种mode的有效样本数，防止只有极少数outlier制造方差。

### Physical realism

- oriented-box collision rate和penetration depth；
- nearest-agent distance与TTC；
- offroad rate与road-edge distance；
- traffic-light violation；
- delta-XY、kinematic consistency、acceleration和jerk；
- WOSAC kinematic/interactive/map buckets与meta-metric。

### Candidate-level safety

不能只报告 best sample。需要同时报告：

- 所有 `N` 个候选的平均collision/offroad；
- 至少一个安全候选的scene比例；
- 所有候选都安全的scene比例；
- quality+diversity Top-5后的安全率。

## 10. 第一轮验收标准

相对于相同步数的 current rollout FT control：

1. ADE继续改善，或退化控制在很小范围内；
2. student保留至少约50%~70%的 Stage 1 pairwise trajectory diversity；
3. 不同noise产生稳定可重复的不同joint futures，而不是单agent随机抖动；
4. collision与offroad在固定 WOSAC100 上有明确下降；
5. interactive和map bucket不下降；
6. candidate diversity增加不能由碰撞、offroad、异常速度或chunk seam跳变造成；
7. 使用多个固定seed重复趋势，不能只依据一个case或一个seed。

这些是第一轮筛选标准，不应在实现前写成不可调整的硬阈值。先记录 Control/A/B/C 的paired confidence intervals，再决定正式门槛。

## 11. 建议的实现顺序

1. 固定并记录准确的 Stage 1 teacher checkpoint与当前 collapsed checkpoint；
2. 建立同一context、`N=4` noise的训练batch与paired evaluation protocol；
3. 先实现 joint MoN winner selection并复现 Run A；
4. 实现 differentiable collision、offroad proxy和kinematic losses，完成 Run B；
5. 加入冻结 Stage 1 teacher与diversity floor，完成 Run C；
6. 固定 val128做短间隔评估；
7. 用 WOSAC100筛选，WOSAC1000确认；
8. 在最佳loss配置上独立测试 matched `stride16 keep-first` fine-tuning；
9. 再接入pair interaction embedding/mode conditioning；
10. 最后考虑map encoder/cross-attention architecture改动。

## 12. 需要保存的实验产物

每个run必须保存：

- 完整CLI和resolved config；
- init checkpoint、tokenizer checkpoint和teacher checkpoint绝对路径；
- 训练seed与evaluation noise seed列表；
- 每项loss曲线；
- val128 paired per-sample metrics；
- 代表性场景的所有32条可视化；
- WOSAC100/1000 summary与逐view JSONL；
- Stage 1、Control、A、B、C 的统一对比表。

## 13. 一句话版本

> Fine-tune with joint Minimum-over-N so only one sampled joint future must match the observed GT, constrain every candidate with collision/offroad/kinematic losses, and distill Stage 1's noise-dependent diversity so long-horizon accuracy improves without mode collapse.
