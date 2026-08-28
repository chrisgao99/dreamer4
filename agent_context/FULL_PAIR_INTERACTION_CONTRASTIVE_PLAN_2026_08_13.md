# Full-Pair Interaction Filtering, RMS Matching, and Contrastive Training Plan

日期：2026-08-13

项目根目录：`/p/yufeng/tri30/dreamer4`

当前代码：`waymo/interaction_contrastive_learning/latest/`

状态：

- full-pair v2 筛选已经实现，并已完成 50k focus-file 数据生成；
- event-aligned、mask-aware RMS neighbour v0 已经实现并完成计算；
- hard/soft contrastive tokenizer fine-tuning 尚未实现或启动；
- `waymo/interaction_contrastive_learning/legacy/` 中的 soft-pair v1 和
  discrete-label matched-pair 实验不是当前主线。

本文档是当前 interaction contrastive 工作的最新上下文。它取代旧文档中
关于 pair 筛选、matching target 和下一步训练方式的过时描述。

## 1. 研究目标

最终目标不是只训练一个能区分 pair 的临时 head，而是重新组织 tokenizer
encoder 的 latent `Z`：

> 保留场景重建所需信息，同时让相似的双车联合运动和协调模式更容易从
> `Z` 中读取。

期望 contrastively refined `Z` 为后续 world model 提供更明确的 interaction
state，例如冲突关系、到达次序、减速和让行模式。

但必须保持以下边界：

- contrastive learning 首先是 representation-learning 实验；
- 它可能改善 downstream interaction 表现，但不保证直接改善 WOSAC；
- WOSAC interaction bucket 主要衡量 collision、nearest-object distance 和
  TTC 分布，不等同于丰富的 action-conditioned reaction；
- 若希望其他车辆随 focus action/plan 改变而反应，world model 仍需要明确的
  action conditioning、interaction-aware training loss 和 action-swap 评估。

## 2. 最新 full-pair v2 筛选

### 2.1 数据来源与枚举

输入是 50k OOI-centered focus files：

```text
data/waymo_vector_dataset_ooi_centered_50k
```

对每个 focus file：

1. focus agent 固定为输入中的 slot 0；
2. 枚举每一个有效 non-focus agent；
3. 对两条完整 91-step trajectory 的所有有效异步 timestep combinations
   计算物理接近关系，而不是只比较相同绝对 timestep；
4. 使用 raw Waymo length/width 和 heading 计算 oriented bounding-box (OBB)
   edge clearance；
5. 若任意 timestep pair 的 OBB edge clearance `<= 1.0 m`，则形成 physical
   path-contact region；
6. 额外检查连续 line segments 的轨迹交点，避免漏掉发生在两个 10 Hz
   samples 之间的 crossing。

代码：

```text
waymo/interaction_contrastive_learning/latest/full_pair_samples.py
```

关键函数：

```text
continuous_path_intersections()
obb_separation_matrix()
contact_components()
select_primary_component()
extract_full_pair_sample()
```

### 2.2 Pair inclusion rule

原始 two-OOI pair 与 mined non-OOI pair 使用不同 inclusion rule：

- original two-OOI pair 无条件保留；
- mined non-OOI pair 必须至少有一个 physical path-contact component；
- 如果 original OOI pair 没有 contact component，使用 continuous closest
  points 建立 `ooi_closest_fallback` event，而不是删除样本；
- track 缺帧不会删除 original OOI pair，缺帧信息保存在 validity mask 中。

因此，`1.0 m` 是 non-OOI physical-contact 筛选的 OBB edge-clearance threshold，
不是简单的 agent-center distance threshold。

### 2.3 Event selection and relevance

contact matrix 使用 8-connected components 形成 contact intervals。primary
component 按以下顺序确定：

1. 最小 zone PET；
2. 优先真实 continuous path intersection；
3. 最小 center PET；
4. 更大的 contact component。

pair 按 primary event 的到达次序排列为 `first_agent` 和 `second_agent`。
soft relevance 只用于 deterministic ranking：

```text
pet_s = zone_pet_steps * 0.1
relevance = exp(-0.5 * (pet_s / 3.0)^2)
```

当前正式数据使用：

```text
non_ooi_top_k_per_focus = 0
```

即不做 per-focus top-k 截断，所有满足 physical-contact 条件的 non-OOI pairs
均被保留。

### 2.4 Stored representation and deduplication

每个样本保存：

```text
trajectory:  (2, 91, 6)
valid_mask:  (2, 91)
features:    x, y, vx, vy, heading_sin, heading_cos
```

坐标以 primary conflict point 为原点，并使用 first-arrival agent 在 primary
step 的 heading 定义旋转坐标系。

同一 Waymo scenario 中，同一 unordered agent pair 可能因为两个 focus views
重复出现。使用以下 key 去重：

```text
(split, scenario_id, min(agent_id_a, agent_id_b), max(agent_id_a, agent_id_b))
```

这就是此前所说的 duplicate pair/window；它不是删除两个外观相似但来自
不同 driving records 的窗口。

### 2.5 Completed dataset

正式结果：

```text
waymo/cache/interaction_full_pairs_50k_v2_no_topk
```

版本和统计：

```text
version:                         full_91_physical_contact_intervals_v2
focus files:                     50,000
train samples:                   117,540
val samples:                      13,261
expected original OOI pairs:      24,985
retained original OOI pairs:      24,985
missing original OOI pairs:            0
non-OOI top-k removals:                 0
```

入口：

```bash
bash waymo/interaction_contrastive_learning/latest/run_full_pair_50k_v2_tmux.sh
```

## 3. 最新两阶段 similarity search

这里统一使用以下术语：

- **query pair**：当前需要寻找相似 pair 的 interaction sample；
- **candidate pair**：DCT 粗召回的候选 pair；
- **nearest neighbour / best match**：候选中 exact RMS 最小的 pair；
- 旧 slide 中的 **anchor** 与 query pair 含义相同，但后续优先使用 query pair，
  避免和 event-alignment anchor 混淆。

### 3.1 Alignment and normalization

每个 pair 以 `primary_step_first` 为 event anchor，线性插值到 60-step window：

```text
offsets: -19 ... +40 steps
time:    -1.9 ... +4.0 seconds
shape:   (60, 2, 6)
```

heading sin/cos 插值后重新投影到 unit circle。

normalization 仅使用 train split 的有效状态拟合：

- 按 first/second agent role 和 channel 分别计算 median/IQR；
- invalid padding 不参与统计；
- position/velocity IQR floor 为 1.0；
- heading sin/cos 保持原 unit-circle scale；
- train scaler 原样应用于 val。

进入 neighbour search 的 query pair 要求 joint valid fraction `>= 0.80`。
两个 pair 计算 exact RMS 时要求 common joint overlap `>= 0.70`。

### 3.2 Stage 1: DCT candidate retrieval

不能对 117k train samples 直接执行完整 all-to-all exact RMS。每个 normalized
trajectory 被压缩成 low-frequency DCT descriptor：

```text
3 DCT coefficients x 12 trajectory channels = 36
3 DCT coefficients x 2 validity-mask channels = 6
descriptor dimension                         = 42
```

query 和 candidate 只在相同 stratum 中比较：

```text
(ordered first-agent type,
 ordered second-agent type,
 contact event vs OOI fallback event)
```

这里的 contact event 合并 `path_intersection` 与 `obb_contact_interval`；fallback
单独成组。

使用 DCT descriptor 的 KD-tree distance 为每个 query pair 检索最多 1,024 个
candidate pairs，并排除：

- query 本身；
- 来自同一 scenario 的 samples。

DCT 只用于快速 candidate retrieval，DCT distance 永远不作为最终 similarity。

### 3.3 Stage 2: exact mask-aware RMS reranking

对 1,024 个 retrieved candidates 使用完整 normalized `(60,2,6)` states 计算：

```text
RMS(i,j) = sqrt(
  sum_common_valid ||X_i - X_j||^2
  / number_of_common_valid_scalar_values
)
```

只使用两个 pair 共同有效的 agent/timestep/channel。RMS 越小表示越相似。
按 exact RMS rerank 后保存 top 32：

```text
neighbor_indices
rms_distances
common_valid_fraction
```

当前方法不使用 cosine distance 或 DTW。

### 3.4 Important approximation boundary

存储的结果是：

> the exact RMS ranking within the 1,024 DCT-retrieved candidates

而不是 117k 数据集的 exhaustive global exact-RMS ranking。96-query sampled
audit 中 exact-top-32 recall 为：

```text
train mean recall: 0.8646
val mean recall:   0.9915
```

因此在报告中应说 “best exact-RMS match among the DCT candidates”，除非未来
增加 candidate count 或完成 exhaustive top-1 validation。训练前应继续记录
DCT recall，尤其关注 train split 中的低-recall query。

### 3.5 Completed similarity cache

结果：

```text
waymo/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0
```

统计：

```text
version:                 full_pair_event_aligned_masked_rms_v0
train total/eligible:    117,540 / 74,892
val total/eligible:       13,261 /  8,413
retrieval candidates:      1,024
stored neighbours:            32
```

入口：

```bash
bash waymo/interaction_contrastive_learning/latest/run_full_pair_rms_neighbors_tmux.sh
```

## 4. Contrastive representation design

当前 tokenizer latent 默认形状为：

```text
Z: (B, T, N_latents=8, D_bottleneck=32)
```

计划中的 pair representation 必须只从 `Z` 读取，不允许 raw pair geometry
进入 contrastive head：

```text
Z
  -> learned query for pair (i,j)
  -> cross-attention over Z
  -> pair token h_ij
  -> projection head
  -> L2-normalized contrastive embedding g_ij
```

当前 `TokenizerInteractionAuxHead` 已实现 focus-slot-0 到 candidate-slot-j 的
`Z`-only cross-attention readout，并输出：

```text
pair_tokens: (B, T, K, 128)
```

代码：

```text
waymo/core/vector_tokenizer_decoder.py
```

新的 RMS contrastive projector、temporal/query-step selection、paired sampler
和 hard/soft contrastive loss 尚未实现。

### 4.1 Which embedding receives the loss?

contrastive loss 直接计算在临时 embedding `g_ij` 上，但最终希望永久改变的是
encoder 输出 `Z`：

```text
contrastive loss
  -> g_ij
  -> projector
  -> h_ij
  -> pair-query cross-attention
  -> Z
  -> unfrozen tokenizer encoder blocks
```

projector/head 可以在 fine-tuning 后丢弃；保留的是 refined tokenizer encoder
及其 `Z`。必须用 fresh probe/readout 验证信息确实进入 `Z`，不能只报告训练
head 的准确率。

### 4.2 Causal readout requirement

RMS target 使用完整 interaction trajectory，可以离线读取 future；但若目标是
帮助 causal world model，contrastive embedding 不应通过读取 future `Z` 来
轻易复现 target。

第一版建议：

- 使用 event-aligned causal history；
- 从 `primary_step_first` 对应的 query-time `Z` 读取一个 pair embedding；
- query-time `Z` 只能看到该时刻及之前的 states；
- future trajectory 仅用于离线定义 RMS neighbours/targets。

开始实现前必须固定 query step 和 tokenizer history window，并写入训练配置。
如果改成对多个 `h_ij(t)` 做 temporal pooling，必须保证 pooling 不跨越 query
time，避免 future leakage。

### 4.3 Pair-slot mapping requirement

full-pair 数据按 first arrival/agent ID 排列，而 source OOI-centered scene 的
slot 0 是原 focus，两者不一定相同。训练 loader 必须：

1. 根据 stored `first_agent_id`、`second_agent_id` 找回 tokenizer input 中的
   两个 agent slots；
2. 保证两个 agent 都被 tokenizer selection 保留；
3. 支持 arbitrary `(slot_i, slot_j)` pair query，或明确重排输入使 first agent
   成为 slot 0；
4. 不可默认 full-pair 的 first agent 永远等于原 focus slot 0。

## 5. Hard contrastive plan

先实现 hard contrastive，验证完整 sampler、pair readout 和 gradient path：

- query：当前 eligible interaction pair；
- positive：来自另一 scenario 的最低 exact-RMS neighbour；
- hard negatives：相同 matching stratum 中、RMS 明显较远且不在 positive
  exclusion region 的 pairs；
- easy negatives：不同 agent-type strata 的 in-batch pairs，可以保留但不能是
  唯一 negatives；
- objective：symmetric InfoNCE with cosine similarity。

只使用 non-same-type negatives 会让任务过于简单：模型只需识别
vehicle-vehicle、vehicle-pedestrian 等类别，而不必学习 interaction dynamics。
因此优先使用 **behaviorally dissimilar pairs with the same ordered agent types**
作为 hard negatives。

需要避免 false negatives：不能简单地把所有不在 stored top-K 中的 same-type
pairs 都视为 negative；应设置 RMS distance/rank margin 或 positive exclusion
范围。

初始 temperature 建议：

```text
tau = 0.07
```

## 6. Soft contrastive plan

hard pipeline 稳定后，再使用 top-K exact-RMS neighbours 建立 soft targets：

- top-K neighbours 都是 weighted positives；
- smaller RMS -> higher target probability/similarity；
- farther RMS -> lower target probability/similarity；
- unrelated same-stratum pairs 作为 hard negatives；
- non-same-type pairs 可作为 easy in-batch negatives。

目标可写为：

```text
q_ij = normalized RMS-derived target weights
p_ij = softmax(cos(g_i, g_j) / tau)
L_soft = -sum_j q_ij log p_ij
```

RMS-to-weight kernel、local scale 和 K 必须通过 val ablation 选择，不能在没有
验证的情况下把 raw RMS 直接解释成概率。第一版可从 stored top 32 开始。

## 7. Two-stage tokenizer fine-tuning

### Stage A: train the readout/head only

目的：验证固定的 baseline `Z` 是否已经包含足够信息，让小 head 恢复 RMS
similarity structure。

冻结：

- tokenizer encoder；
- tokenizer decoder。

训练：

- pair-query cross-attention/readout；
- projection head。

loss：

```text
L_stage_A = L_contrastive
```

此阶段 reconstruction loss 可以监控，但不需要加入 head optimizer，因为 head
不影响 reconstruction path。Stage A 不会 reshape `Z`。

### Stage B: reshape Z while preserving reconstruction

冻结：

- tokenizer decoder parameters。

解冻并训练：

- pair-query readout；
- projector；
- tokenizer encoder 最后 1--2 个 blocks。

loss：

```text
L_total = L_reconstruction + lambda_contrastive * L_contrastive
```

含义：

- `L_contrastive` 让 RMS 相似的 joint interactions 在 embedding space 中接近；
- `L_reconstruction` 约束变化后的 `Z` 仍能被原 frozen decoder 解码，避免
  representation drift 和 catastrophic loss of scene information。

decoder 权重虽然冻结，但 forward/backward 必须保留：

```text
reconstruction loss
  -> frozen decoder operations
  -> Z
  -> unfrozen encoder blocks
```

禁止：

- `Z.detach()`；
- 用 `torch.no_grad()` 包裹 decoder forward；
- 从第一步起冻结随机初始化的 pair head/projector。

建议初始 learning rates：

```text
pair readout/projector LR: approximately 1e-4
encoder LR:                approximately 1e-5
```

逐步 ramp `lambda_contrastive`。初期让 contrastive gradient norm 约为
reconstruction gradient norm 的 10%--30%，并同时监控 reconstruction validation
metrics。

如果 Stage B 不稳定：

1. 只解冻最后一个 encoder block；
2. 降低 encoder LR；
3. 降低或更慢 ramp `lambda_contrastive`；
4. 检查是否存在 future leakage、false negatives 或 DCT retrieval failure。

## 8. Training data and sampling requirements

实现 paired loader 时必须保留：

- source scenario ID；
- source path / pair IDs；
- event anchor 和 valid mask；
- query-positive/candidate indices；
- exact RMS and common-valid fraction；
- ordered agent types and fallback/contact stratum。

每个 contrastive edge 必须：

- 跨 scenario；
- agent ID 到 tokenizer slots 的映射有效；
- 满足 query 和 pair-overlap eligibility；
- 不使用 DCT distance 作为 training similarity target；
- train/val neighbour graph 分开，normalizer 只由 train 拟合。

推荐实现顺序：

1. 建立 sample index 到 raw/tokenizer scene 和 agent slots 的可靠 join；
2. 实现 query-positive paired batch 和 hard-negative mask；
3. 实现 causal pair readout 和 projector；
4. 仅训练 head，验证 hard InfoNCE；
5. 解冻 encoder 最后 1--2 blocks，加入 reconstruction anchor；
6. hard loss 稳定后实现 soft top-K targets。

## 9. Evaluation and ablations

### 9.1 Representation evaluation

使用 held-out val scenes 检查：

- exact-RMS neighbour retrieval Recall@K；
- embedding cosine similarity 与 exact RMS 的 rank correlation；
- same-type hard-negative discrimination；
- fresh `Z`-only probe/readout，不复用训练 projector；
- reconstruction quality before/after fine-tuning。

### 9.2 Downstream world-model evaluation

用相同 world-model architecture、data、optimization budget 比较：

```text
baseline tokenizer Z
vs
contrastively refined tokenizer Z
```

至少报告：

- standard latent/trajectory prediction metrics；
- conflict-relevant agent subsets；
- WOSAC interaction metrics，作为 secondary ablation outcome；
- action-swap test：固定 history，改变 focus action/plan，比较 relevant other
  agents 的预测是否以合理方向发生变化；
- unrelated agents 在 action swap 下应保持相对稳定。

不能只根据 collision 降低断言模型理解了 interaction；过度停车或过度拉开
距离也能降低 collision。reactive behavior 必须用受控 action-swap 和
interaction-specific qualitative/quantitative evaluation 单独验证。

## 10. Current implementation and result locations

Current filtering and matching code：

```text
waymo/interaction_contrastive_learning/latest/
  common.py
  full_pair_samples.py
  build_full_pair_dataset.py
  build_full_pair_rms_neighbors.py
  run_full_pair_50k_v2_tmux.sh
  run_full_pair_rms_neighbors_tmux.sh
  build_contrastive_training_cache.py
  train_interaction_contrastive.py
  run_hard_soft_contrastive_cuda3_tmux.sh
  run_hybrid_soft_v2_cuda3_tmux.sh
  summarize_hybrid_soft_metrics.py
  visualize_full_pair_audit.py
  visualize_full_pair_rms_neighbors.py
  visualize_full_pair_hard_negatives.py
  visualize_full_pair_relation_negatives.py
  animate_full_pair_audit.py
```

Tests：

```text
waymo/interaction_contrastive_learning/tests/latest/
```

Dataset：

```text
waymo/cache/interaction_full_pairs_50k_v2_no_topk/
```

Similarity cache：

```text
waymo/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0/
```

Contrastive training cache：

```text
waymo/cache/interaction_full_pairs_50k_v2_contrastive_v1/
```

Key summaries：

```text
waymo/cache/interaction_full_pairs_50k_v2_no_topk/summary.json
waymo/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0/rms_summary.json
waymo/cache/interaction_full_pairs_50k_v2_contrastive_v1/summary.json
```

### 10.1 Implemented hard/soft experiments (2026-08-13)

Positive safeguards：

- positive candidates must be cross-scenario and cross-source-file；
- any edge with exact mask-aware RMS `<= 0.02` is rejected as a duplicate
  continuous-drive slice, even if its Waymo scenario ID differs；
- hard mode uses the closest remaining exact-RMS positive whose two endpoints
  each have enough certified negatives；
- the final cache contains zero same-scenario/same-source positive edges and its
  minimum retained RMS is `0.020005` on train and `0.026208` on val；
- duplicate filtering removed 7,538 train and 101 val edges.

Negative and loss definitions：

- search all 1,024 DCT candidates in the same ordered-type/contact stratum；
- require substantially different full-trajectory RMS and a different relative
  outcome (order swap/future order, gap trend, or distance trend)；
- hard experiment uses symmetric InfoNCE with 2 certified negatives per endpoint；
- soft experiment samples 8 points across the duplicate-filtered top-32 RMS
  neighbourhood, uses locally normalized exponential RMS weights, and includes
  4 certified relation negatives with zero target mass.

Both experiments use 5,000 head-only steps followed by 20,000 steps that unfreeze
the final encoder block, keep the decoder frozen, and optimize reconstruction plus
a contrastive weight ramping to 0.1. They were submitted sequentially to physical
CUDA 3 so the two full models do not contend for one GPU：

```text
tmux: interaction_contrastive_hard_soft_cuda3

hard output:
waymo/checkpoints/interaction_contrastive_hard_relneg_dupfiltered_v1/

soft output (automatically starts after hard):
waymo/checkpoints/interaction_contrastive_soft_relneg_dupfiltered_v1/

logs:
waymo/logs/interaction_contrastive_learning/hard_relneg_dupfiltered_v1_cuda3.log
waymo/logs/interaction_contrastive_learning/soft_relneg_dupfiltered_v1_cuda3.log
```

Hard completed successfully. Its best Stage-B checkpoint is step 18,500 with
validation contrastive loss 0.73324 and nearest-positive/negative cosine margin
0.13496. The original standalone soft run became non-finite between steps 13,900
and 13,920; only its pre-failure `best.pt` at step 13,500 is finite.

### 10.2 Hybrid-soft-v2 from hard best (2026-08-14)

The follow-up experiment treats soft learning as local refinement instead of an
independent replacement for hard contrastive learning：

- initialize both tokenizer and pair head from hard `best.pt` step 18,500；
- each group contains the nearest non-duplicate RMS positive, 7 additional
  positives spread through the top-32 neighbourhood, and 4 relation negatives；
- `L_separation` is hard InfoNCE between the nearest positive and 4 negatives；
- `L_rank` is KL divergence between train-stratum RMS weights and cosine ranking
  over the 8 positives only；
- Stage B uses `L_reconstruction + 0.005 L_separation + 0.001 L_rank` after a
  4,000-step ramp. A CUDA gradient probe measured a 0.299 weighted
  contrastive/reconstruction encoder-gradient ratio at full weight；
- cosine normalization and log-softmax run in FP32, and any non-finite forward
  loss or gradient immediately writes sample IDs and terminates；
- train 5,000 head-only plus 40,000 final-encoder-block refinement steps with
  batch size 2；
- evaluate on a fixed proportional-stratified manifest of 512 validation anchors
  and report separation accuracy, positive RMS rank accuracy, cosine margins,
  rank KL, reconstruction, and component gradient norms.

Initial fixed-manifest metrics before hybrid training：

```text
separation accuracy:          0.50977
positive rank accuracy:       0.55978
nearest-vs-negative margin:   0.12076
nearest-vs-farthest drop:     0.04678
rank KL:                      0.51674
reconstruction loss:          0.00921
```

The first fixed-manifest Stage-A evaluation at step 2,000 was finite and moved in
the intended aggregate direction：composite contrastive loss `1.41652 -> 1.30910`,
separation loss `1.31317 -> 1.21098`, separation accuracy `0.50977 -> 0.55469`,
and rank KL `0.51674 -> 0.49057`. Positive pairwise rank accuracy was slightly
lower (`0.55978 -> 0.55197`), so it remains a tracked trade-off rather than a
claimed improvement at this early point.

Run locations：

```text
tmux: interaction_hybrid_soft_v2_cuda3

output:
waymo/checkpoints/interaction_contrastive_hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3/

log:
waymo/logs/interaction_contrastive_learning/hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3.log

live machine-readable metrics:
.../metrics.jsonl

final self-contained report:
.../training_report.html
```

## 11. Concise presentation wording

Filtering：

> For each focus--candidate pair, search all 91 x 91 valid timestep
> combinations and retain the pair if the oriented-box edge clearance is at
> most 1.0 m or the continuous paths intersect. Original OOI pairs are always
> retained.

Similarity search：

> DCT retrieves 1,024 candidate pairs for each query pair. Exact mask-aware RMS
> then reranks those candidates and selects the nearest neighbours.

Contrastive purpose：

> Contrastive learning is used as an auxiliary objective to make latent `Z`
> better capture similar joint-motion and coordination patterns. Its downstream
> interaction and WOSAC effects must be measured through controlled ablations.
