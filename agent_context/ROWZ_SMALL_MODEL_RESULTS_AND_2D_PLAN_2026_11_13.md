# 通行顺序小模型（rowz）：历次实验结果与 2D 实验计划（2026-11-13）

> **本文件汇总 rowz 小模型 pilot → v2 → v3 → v3noz 的结果，并记录 v4（2D，模型控制 heading）的实验计划和重跑方法。**
>
> - 小模型的最初设计见 `dreamer4/agent_context/ROW_PAIR_ORDER_Z_WM_FINETUNE_PLAN_2026_11_09.md`（下称"原方案"）。大 WM 这边的讨论见 `dreamer4/agent_context/CLOSED_LOOP_PREFERENCE_AND_EXOGENOUS_AGENTS_2026_11_12.md`。
> - 状态：v4 在 kuorobot01 上训练到 epoch 16/80 时按用户要求停止，改到另一台 server 重跑（§6）。
> - 路径约定：文中路径都相对于 `tri30/`。

## 0. 一句话

- **小模型能不能对 ego 做出反应，取决于训练里有没有"ego 干预 + 按规则改写其他车的目标"，不取决于 z，也不取决于安全层。**
  - pilot 没有干预：训练时不带安全层的版本，stop-go 跟车碰撞 37–38%。
  - v2 起加了干预：v3、v3noz 的 stop-go 碰撞降到 11–17%。
  - 去掉 z（v3noz）不变差，对 ego 抢行的反应反而更大。
- **学到的反应正好是改写目标写进去的那部分：** "路线前方有车身就减速"学会了，交叉口里提前让行没学会。
- **v4 要回答的问题：** 让模型自己控制 heading（真正的 2D）之后，刹车反应还在不在；模型是用刹车还是用打方向来躲。

## 1. 小模型的共同设定

- **数据：** 由 `waymo_interaction_screening/code/rowz_views.py` 从 candidate bank 构建。
  - 每个 view 是在路口相遇的一对 focus 车，加最多 3 辆与它们共享空间的车，共 2–5 辆。
  - 训练：`waymo_interaction_screening/results/rowz_views_pilot100_20261109` 的 train split，11,811 对，即 23,622 个 item（两辆 focus 车各当一次 ego）。
  - 验证：1,308 对，即 2,616 个 item，取自 pilot100 和 `rowz_views_extra100_off5_20261109` 的 validation split，固定在第一个 anchor。
  - v2 起还要用路线缓存 `rowz_routes5_{pilot100,extra100_off5}_20261109`，由 `code/rowz_routes.py v2` 构建。四个缓存一共约 224 MB。
- **运动（pilot 到 v3noz）：** 每辆车沿自己的日志路线运动，状态是路程 s 和速度 v，模型只输出纵向加速度。2D 位置按 s 在日志路线上插值得到。
  - ego（0 号车）回放日志，或被干预；其余车都由模型控制。
  - 历史 1.1 s，预测 6 s。
- **训练：** 闭环 rollout，梯度穿过整条 rollout。
  - 闭环 horizon 按课程增加：10 → 20 → 40 → 60 步；horizon 60 时学习率 × 0.3。
  - 80 epoch，Adam 6e-4，hard 和 undecided 车对过采样 3 倍。
- **两种执行层：**
  - policy：每步重新算一次的 Markov 策略，用关系注意力，带 null slot。
  - MATS-lite：每轮重规划由 GRU 生成一组系数，加速度对当前的自身特征和车对特征是仿射的：a = a_last + b(t) + S(t)·ψ + Σ_j A_ij(t)·φ_ij。
- **关键评估（验证集，6 s）：**
  - `collision`：按日志回放 ego 的事实 rollout 里的碰撞率。
  - `cf_assert`：ego 从真实状态起以 +2 m/s² 加速抢行。看日志里对方先过的那一半：对方进入冲突区的时间推迟了多少（`d_entry_s`），以及 focus 车对的碰撞率。
  - `cf_brake`：ego 以 3 m/s² 刹停。看日志里 ego 先过的那一半：对方是否提前通过（`earlier_frac`）。
  - `stopgo`：ego 以 4 m/s² 刹停，等到 6 s，再以 2 m/s² 起步。统计 ego 身后 0–40 m 的跟车：碰撞、6 s 时已停下、4 s 内重新起步、一直没起步。

## 2. 版本时间线

| 版本 | 日期 | 运行目录（`MATS/waymo_runs/`） | 相对上一版的主要改动 |
|---|---|---|---|
| pilot | 11-09 上午 | `rowz_pilot_20261109` | z 头读日志路线和 ROW 标签，z 在起点定一次；执行层以 z 为条件。4 个 run：policy / MATS × 安全层开 / 关。**没有 ego 干预** |
| z 信息测试 | 11-09 | `rowz_zhead_present_20261109` | 只训练 z 头，比较不同的路线信息来源 |
| v2 | 11-09 下午 | `rowz_v2_20261109` | z 头改为 C2（只读当前信息），每 0.5 s 重估，带滞回；安全层只在紧急时介入，外加模仿 loss；**50% 样本做 ego 干预，其他车用包络目标**；同向车对的 z 在两车离远后释放。4 个 run：policy / MATS × 重规划 0.5 s / 5 s |
| v3 | 11-10 | `rowz_v3_20261110` | 去掉安全层；所有车互相可见；同向车对的 z 也由 z 头给；MATS 的线性车对项加入障碍 gap 和接近速度 |
| v3noz | 11-10 | `rowz_v3noz_20261110` | v3 去掉 z（任何地方 z = 0），其余不变 |
| v4（2D） | 11-13 | `rowz_v4_2d_20261113` | 模型同时控制 heading，30 个路线点作输入，用横向 loss 约束（§5）。停在 epoch 16 |

**ego 干预**（v2 起，代码 `MATS/waymo_mats/rowz/v2.py` 的 `ego_interventions`）：50% 的训练样本里，ego 沿自己的路线改变速度，三种方式等概率：
- 从随机时刻起速度乘 0.5–1.4；
- 刹停，停 1–3 s，再恢复日志速度；
- 以 1–2 m/s² 加速 1–3 s。

这些样本里，其他车不再用日志作目标。速度目标是 `min(日志速度, 路线前方有车身时按 2.5 m/s² 算的舒适包络)`（`comfort_speed`），超过目标的部分权重 1.0、低于目标的部分权重 0.3，另加软碰撞 loss。

## 3. 结果

### 3.1 z 头

- **pilot：** 起点处准确率 0.98。hard 子集 0.94–0.96（按匀速推算的顺序只有 0.57），undecided 子集 0.95–0.96（匀速 0.89）。
- **z 信息测试**（`rowz_zhead_present_20261109/README.md`）：

  | 输入 | 准确率 |
  |---|---|
  | 日志路线 | 0.976 |
  | 与日志匹配的地图分支 | 0.956 |
  | 只用当前信息：所有地图分支 + 路线头（C2） | 0.950 |
  | 没有路线 | 0.915 |

  - ROW 标签在有路线特征后不再提供信息。
  - 路线头在有歧义的车上只有 49% 准确，但选错分支多半不影响通行顺序的判断。
- **v2 / v3 部署时（C2，滞回）：** 准确率 0.93–0.95，hard 子集 0.84–0.90。v3 在起点处：交叉车对 0.95，同向车对 0.97。
- **z 是很强、但会被盲目服从的控制量**（pilot `analysis/z_effect.md`）：把 focus 车对的 z 翻过来，对方的进入时间平均移动约 1 s，碰撞也升到 10–19%。

### 3.2 反应：各版本对比（验证集 2,616 个 item，6 s）

| 版本 / run | 推理时的安全层 | ADE s (m) | 事实碰撞 | stop-go 碰撞 | stop-go 6 s 已停 | 抢行：对方推迟 (s) | 抢行：focus 碰撞 | 刹车：对方提前比例 |
|---|---|---|---|---|---|---|---|---|
| pilot policy_nosafety（z = head） | 无 | 1.73 | 1.5% | 36.9% | 72% | −0.03 | 35.0% | 0.00 |
| pilot mats_nosafety（z = head） | 无 | 1.61 | 1.6% | 37.5% | 49% | +0.01 | 35.1% | 0.03 |
| pilot policy / mats（训练时带安全层，推理时关掉） | 无 | – | – | 52.4% / 36.3% | 46% / 52% | – | – | – |
| v2 policy_r5 | 紧急层 | 1.70 | 1.6% | 1.8%* | 92% | +0.25 | 27.4% | 0.17 |
| v2 policy_r50 / mats_r50 | 紧急层 | 1.73 / 1.63 | 1.9% / 1.6% | 3.0% / 3.0%* | 93% / 94% | +0.14 / +0.20 | 28.2% / 29.0% | 0.06 / 0.08 |
| v3 policy_r5 | 无 | 1.69 | 1.6% | 15.5% | 48% | +0.15 | 30.5% | 0.17 |
| v3 mats_r5 | 无 | 1.54 | 1.0% | 11.3% | 47% | +0.16 | 29.5% | 0.17 |
| v3noz policy_r5 | 无 | 1.64 | 1.0% | 16.7% | 47% | +0.32 | 25.1% | 0.25 |
| v3noz mats_r5 | 无 | 1.48 | 0.5% | 15.5% | 53% | +0.40 | 23.2% | 0.21 |

- **表的口径：**
  - "抢行"两列取日志里对方先过的那一半；"刹车"一列取日志里 ego 先过的那一半，ego 刹停后对方提前超过 0.2 s 通过的比例。
  - \* v2 的 stop-go 是带紧急安全层测的，和其他行不能直接比。
  - v2 `mats_r5` 在 epoch 40 因为 GPU 被 ollama 占用而 OOM，只有 best.pt（epoch 36），没有最终指标。
- **数据来源：** pilot 的数字来自 `rowz_pilot_20261109/analysis/{summary,counterfactual_split,stop_go}.md`；其余来自各 run 的 `final_val_metrics.json`。

### 3.3 其他发现

- **pilot：**
  - 唯一的反应来自固定安全层：ego 刹停时 focus 碰撞 0%，没有安全层时 3–11%。
  - 抢行时 focus 碰撞 28–35%，而在 ego 车身刚上对方路线的时刻，以 ≤ 6 m/s² 还能刹住的只占全部样本的 1–4%，其余要靠提前预判。
  - policy 有 11% 的跟车停下后一直不起步：一部分卡在停止线前，另一部分是同向车对的 z = "前车先"一直不释放。
- **v2，刹车是否要等 z 翻转**（`rowz_v2_20261109/analysis/brake_before_flip.md`）：
  - 把 z 冻住，执行层仍有约 25% 的样本会刹车，所以刹车不依赖 z 翻转。
  - 这些刹车几乎都是被动反应：大多在 ego 车身已经上了对方路线之后才开始（ego 车身上对方路线的中位时刻，是 ego 进入冲突区前 0.2 s）；提前预判只占 0–13%。
  - z 在 r5 下翻转能让刹车开始得早一点，但碰撞仍是 27–30%。
  - mats_r50 有 98% 的样本完全不刹车。
- **v2 修好了跟车不起步：** 一直不起步从 11% 降到 0–2%。
- **v3，汇入类车对过早被判为"已解决"**（`rowz_v3_20261110/analysis/resolved_follow*.py`）：
  - 两条路径汇合的交叉车对（一车转进另一车的车道），在第一辆车驶出冲突区后 z 就置 0，之后不会再按同向车对处理。验证集里这样的车对占交叉车对的 34%。
  - 由此导致的碰撞很少，优先级低。
- **v3noz 比 v3 好的可能原因：**
  - v3 训练时给执行层喂日志 z，部署时用 z 头的 z，而 z 头恰好在 hard / undecided 车对上最弱（0.82–0.88）；
  - 日志 z 让执行层可以不去读状态。
- **反应仍然偏弱：** v3 / v3noz 的 stop-go 里，只有约一半跟车在 6 s 时停下，碰撞 11–17%。

### 3.4 交互游戏（键盘控制 ego 的速度）

在 `MATS/` 下运行：

| 游戏 | 命令 | 加载的 checkpoint |
|---|---|---|
| pilot | `.venv-waymo/bin/python -m waymo_mats.rowz.game --port 7911` | 4 个 pilot best.pt |
| v2 | `... -m waymo_mats.rowz.game_v2 --port 7912` | 4 个 v2 best.pt |
| v3 | `... -m waymo_mats.rowz.game_v3 --port 7913` | v3 和 v3noz 的 policy、MATS 共 4 个 |

- **用户试玩的感受（v3 / v3noz）：** 有车挡路时大部分会减速，停车后会重新起步，ego 抢行后会让；有 z 和没 z 体验差不多。
- **注意：** 游戏代码里 run 目录是写死的绝对路径（`/p/yufeng/tri30/...`），换 server 要改。

## 4. 结论与对大 WM 的启示

1. **反应来自干预和改写的目标。**
   - pilot 训练时有没有安全层，学习层都躲不开：MATS 36% 对 38%，policy 52% 对 37%。所以安全层是次要因素（它会替学习层兜底，policy 上看得到一点影响）。
   - 加了干预之后 stop-go 碰撞降到 11–17%。
   - 不足：pilot → v3 同时改了好几处，不是单变量消融。干净的检验是 v3noz 加 `--perturb 0` 再训一次，看 stop-go 是否回到约 37%。这个实验还没做。
2. **学到的反应 = 目标里写进去的反应。** 包络目标只说"路线前方有车身就减速"，所以交叉口里的提前让行学不会（抢行时对方只推迟 +0.3–0.4 s）。要学提前让行，需要带交叉项的目标，v2 分析里提过，还没做。
3. **z 不是反应的来源。** 训练时给"特权"的顺序信号，反而让执行层少读状态。
4. **对大 WM（讨论记录，未定）：**
   - WM 不反应的根因相同：日志里别人的未来可以从场景直接推出来，读 ego 的偏离没有梯度；MoN 闭环也没有碰撞项和干预。
   - generate-all 里联合采样出的"对方未来"相当于 v3 的日志 z，是一条协调捷径；游戏里人类的未来并不是那份计划。
   - 可借鉴的做法：
     - 闭环 fine-tune 时对随机一辆车做干预，执行时用扰动动作覆盖它（`CLOSED_LOOP_PREFERENCE_AND_EXOGENOUS_AGENTS_2026_11_12.md` §3.3）；
     - 受影响的车用"日志路径 + 重新定时的速度"作目标，并打开碰撞 loss；
     - 外部车的未来 token 不让其他车看到；
     - pair 特征加相对速度和接近速度；
     - 评估照搬 stop-go 和 `cf_assert` / `cf_brake`。

## 5. v4：2D 实验计划

### 5.1 问题

前面所有版本里，车按构造钉在日志路线上，只控制速度，横向偏差永远是 0。v4 放开 heading，回答两个问题：
- 刹车反应还在不在；
- 模型是刹车还是打方向躲。

路线仍然用日志路线（以路程参数化），所以"路线未知"和"路口多模态"这两个问题还不涉及（§5.6）。

### 5.2 设计（代码 `MATS/waymo_mats/rowz/v4.py`，以 v3noz 为基础）

- **运动：** 模型车用运动学 unicycle，状态是 x、y、heading、速度。每步输出加速度和曲率，曲率限制在 |κ| ≤ min(0.25 1/m, 6 m/s² / v²)。
  - **没有沿路线的前馈**：初始化时车直着开，跟随路线要自己学。
  - ego 仍沿日志路线回放，只控制速度；干预方式不变。
- **路线是输入，不是约束。** 路线是本车的日志路径，以路程参数化，不带时间（不给速度剖面）。输入包括：
  - 从本车在路线上的投影位置起，前方每 2 m 一个点，共 30 个（2–60 m），在本车坐标系下；
  - 横向偏移 d、heading 误差、前方 5 m 的路线曲率；
  - 到前方 3 / 6 / 10 / 15 / 25 m 路线点的 pure-pursuit 曲率。
- **其余特征：** v3 的所有特征（停止线、冲突区、同向 gap、灯、编码器历史）里的路程 s，都改用 2D 位置在路线上的投影。车身相关的计算（障碍特征、碰撞、软碰撞 loss）用真实的 2D 位姿。
- **横向 loss：** 不做硬限幅，只用 loss。
  - 每个模型车的每一步（真实样本和干预样本都算）：`huber(d) + 0.5·huber(5·heading 误差)`，权重 1（`--lat-weight`）；
  - 日志状态上的单步 heading 变化 loss，权重 0.5，和单步加速度 loss 并列；
  - 软碰撞 loss 作用在 2D 位姿上，梯度也会传到转向。模型可以打方向躲，只是横向 loss 让这样做有代价。
- **执行层：**
  - MATS：转向对 8 个横向特征是仿射的（节点系数，每步都算），再加两个车对项：j 在我坐标系下的横向、纵向位置，乘以门控 exp(−距离 / 10 m)。30 个路线点通过每轮重规划时的 context 进入。
  - policy：加一个转向输出头，每步都能看到 30 个路线点。
- **其他：**
  - 选 checkpoint 的标准改为 2D ADE + 5 × 碰撞。
  - 跳过大梯度的规则改为相对阈值：某步梯度范数超过最近 200 步中位数的 20 倍才跳过。v3 用的是固定阈值 100，而 MATS 初始化时转向梯度有 500–700，按老规则每一步都会被跳过；v3 自己的梯度最大才 17。
  - 遇到显存不足只跳过当前 batch（记为 `skipped_oom`），评估会重试一次。
- **超参数：** 与 v3noz 相同：重规划 0.5 s，80 epoch，batch 128，Adam 6e-4，perturb 0.5，seed 17，没有安全层，没有 z。

### 5.3 显存与速度

在 kuorobot01 的 4090 上实测，horizon 60 时一步训练的峰值显存（不含每个进程约 0.47 GB 的 CUDA context）：

| micro-batch | policy | MATS |
|---|---|---|
| 32 | 0.62 GiB | 0.95 GiB |
| 64 | 1.23 GiB | 1.83 GiB |
| 128（整个 batch） | 2.42 GiB | OOM（当时只剩约 3.7 GB） |

- **为什么用了 micro-batch：** kuorobot01 上另一个用户的 ollama `gemma3:27b` 常驻占了 19.7 GB，所以用 `--micro-batch 32` 做梯度累积，每次更新仍等于 batch 128。
- **速度：** micro-batch 32、两个 run 并行时，horizon 10 约 0.35 s 一次更新，horizon 20 约 0.7 s，按此估计整个 run 要 6–7 h。
- **换到空闲的 GPU：** 用 `MB=0`（整个 batch）应该快很多，但没有实测；v3 整个 batch 一个 run 是 1.4–1.5 h。
- **`MB=0` 和 `MB=32` 的数值差别：** micro-batch 下每块内部各自归一化 loss，所以两者不会逐位相同，但期望一致。

### 5.4 停止前的部分结果（kuorobot01，2026-11-13 12:52–13:20）

停在 epoch 16（horizon 20），没有跳过的步（大梯度和 OOM 都没有）。下面是 epoch 12 的验证结果（5 s，全部 2,616 个 item）：

| | 2D ADE | ADE s | 速度 MAE | 碰撞 | 平均 \|d\| | 最大 \|d\| 超过 1 m 的车 | heading 误差 |
|---|---|---|---|---|---|---|---|
| v4 policy | 1.52 | 1.86 | 0.89 | 2.6% | 0.08 m | 1.6% | 1.1° |
| v4 MATS | 1.48 | 1.86 | 0.91 | 1.5% | 0.06 m | 0.7% | 1.0° |
| v3noz policy（1D，同 epoch） | – | 1.57 | 1.01 | 1.9% | 0 | 0 | 0 |
| v3noz MATS（1D，同 epoch） | – | 1.47 | 0.91 | 1.8% | 0 | 0 | 0 |

- 初始化时平均 |d| 约 1.4 m，epoch 4 就降到 0.08–0.09 m。车道保持学得很快，碰撞和 v3noz 同期接近。
- v4 的 ADE s 是用投影位置算的，和 v3noz 不完全同口径。
- stop-go 和反事实指标只在最终评估里算，还没有。
- 部分结果在 `MATS/waymo_runs/rowz_v4_2d_20261113/stopped_kuorobot01_20261113/`（logs、metrics.jsonl、epoch 12 的 best.pt / last.pt）。

### 5.5 在另一台 server 上重跑

1. **需要的文件：**
   - 运行目录 `MATS/waymo_runs/rowz_v4_2d_20261113/`：`run_all.sh`、`README.md`，以及 `source_snapshot/`（冻结的代码，与 `MATS/waymo_mats/rowz/` 当前版本相同）。
   - 4 个数据缓存（约 224 MB）放在 `waymo_interaction_screening/results/` 下：
     - `rowz_views_pilot100_20261109`
     - `rowz_views_extra100_off5_20261109`
     - `rowz_routes5_pilot100_20261109`
     - `rowz_routes5_extra100_off5_20261109`
   - Python 环境：默认 `MATS/.venv-waymo/bin/python`（torch 2.8.0+cu128，numpy 1.24.4）；也可以用 `PY=...` 指定。
2. **数据路径：** `v2.py` 的数据根目录已改为可用环境变量 `ROWZ_SCREEN` 覆盖，默认值仍是本机的 `/p/yufeng/...`，所以旧 run 能原样复现。`run_all.sh` 会自动设成 `<tri30>/waymo_interaction_screening/results`，启动前检查 4 个缓存都在，也会检查 python 是否存在。
3. **启动：**
   ```bash
   cd <tri30>/MATS/waymo_runs/rowz_v4_2d_20261113
   GPU=0 MB=0 bash run_all.sh    # GPU 显存够时（每个 run ≥ ~4.5 GB）；不够就 MB=32
   ```
   会在 tmux 会话 `rowz_v4` 里开两个窗口：`policy_r5` 和 `mats_r5`。如果 `policy_r5/`、`mats_r5/` 已经存在，脚本会拒绝覆盖；kuorobot01 上那两个停掉的 run 已经移到 `stopped_kuorobot01_20261113/`。
4. **查看进度：** 日志在 `logs/<run>.log`，进度在 `<run>/status.json` 和 `<run>/metrics.jsonl`。`<run>/completed.json` 出现就是跑完了，结果在 `final_val_metrics.json`。
5. **已经验证过：** 在本机用 launcher 同样的命令从 `source_snapshot/` 跑 3 步，exit 0（`smoke_recheck/launcher_check_mats.log`）；`ROWZ_SCREEN` 的覆盖确实生效。

### 5.6 跑完后看什么

和 v3noz（§3.2）比：

| 问题 | 指标 | v3noz 参考值（policy / MATS） |
|---|---|---|
| 车道保持 | `lat_abs_m`、`car_max_lat_gt1m`、`heading_err_deg` | 1D，恒为 0 |
| 刹车反应还在不在 | `stopgo/collision`、`stopgo/stopped_6s` | 16.7% / 15.5%；47% / 53% |
| 抢行时会不会让 | `cf_assert/other_first_log/d_entry_s`、`.../focus_collision` | +0.32 / +0.40 s；25% / 23% |
| 是刹车还是打方向 | `stopgo/max_lat_gt1m`、`cf_assert/*/other_max_lat_gt1m` | 1D，恒为 0 |
| 整体 | `collision`、`ade_xy` | 碰撞 1.0% / 0.5% |

**读法：**
- 碰撞和 v3noz 相近，横向偏移又小：放开 heading 没有破坏刹车反应。
- 碰撞下降，但横向偏移大：模型学成了打方向躲。
- 碰撞上升，横向偏移小：2D 本身破坏了反应。

### 5.7 后续（讨论过，未定）

- **去掉"路线已知"这个特权：** 训练时用日志匹配的地图分支中心线，部署时用路线头（C2）选的分支。这更接近 WM 的处境：反应依赖的特征（路线上有没有车身、冲突区）都建立在路线已知的前提上。
- 甚至可以保持 1D，只把路径换成预测的分支，单独测"反应在多大程度上依赖知道路线"。
- 横向交互（绕开路边停着的车、找空档换道）需要单独设计目标，不会从"沿路线走 + 刹车"里自然出现。
- 带交叉项的干预目标，用来学提前让行。
- v3noz 加 `--perturb 0` 的单变量消融。
- 适配 v4 的游戏（game_v4）。

## 6. 文件索引

- **代码** `MATS/waymo_mats/rowz/`：
  - `data.py`：数据集；
  - `models.py`：pilot 模型、特征、碰撞；
  - `train.py`：pilot 训练；
  - `zexp.py`：z 信息测试；
  - `v2.py`、`v3.py`（`--no-z` 即 v3noz）、`v4.py`；
  - `game*.py` 和对应的 `.html`。
- **运行目录** `MATS/waymo_runs/`：`rowz_pilot_20261109`、`rowz_zhead_present_20261109`、`rowz_v2_20261109`、`rowz_v3_20261110`、`rowz_v3noz_20261110`、`rowz_v4_2d_20261113`。每个目录都有 README、`run_all.sh` 和 `source_snapshot/`。
- **分析：**
  - `rowz_pilot_20261109/analysis/`：summary、counterfactual_split、z_effect、stop_go、route_ambiguity；
  - `rowz_v2_20261109/analysis/brake_before_flip.md`；
  - `rowz_v3_20261110/analysis/`：resolved_follow、game_check。
- **数据构建：** `waymo_interaction_screening/code/rowz_views.py`（需要 candidate bank `waymo_interaction_screening/candidate_bank_v1_20261022`）、`code/rowz_routes.py`。
