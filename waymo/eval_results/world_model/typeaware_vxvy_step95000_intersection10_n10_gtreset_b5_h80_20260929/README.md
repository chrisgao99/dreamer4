# GT 历史重置对照：H40 / execute 5 / H80

同一批 10 个场景；EMA step95000；每场景 10 次随机采样；8 步 Euler；seed=20260907+sample_order。

每隔 5 步，从当前时刻的真实 11 帧历史重新开始，重置所有 agent 的位置、朝向、vx/vy 和有效性。每轮预测 40 步，只保存前 5 步。地图保持不变，使用规划时刻的灯态。没有使用未来动作作为条件。

图中的线连接这些短预测段；连接处可能含重置跳变，不代表可连续执行的 8 秒轨迹。ADE 表示反复重置条件下的短期预测误差，不是闭环 ADE；改善不能单独证明模型已经理解地图。

[汇总图](gtreset_b5_contact_sheet.png)

| 场景 | 图片 |
|---|---|
| 00 · 341ded3ad87d6f02 | [right_turn](gtreset_b5_scene_00_341ded3ad87d6f02.png) |
| 01 · 4b2dced3ebba31b2 | [left_turn](gtreset_b5_scene_01_4b2dced3ebba31b2.png) |
| 02 · df8ce087d18f6c83 | [right_turn](gtreset_b5_scene_02_df8ce087d18f6c83.png) |
| 03 · 94e7762cbdee8792 | [left_turn](gtreset_b5_scene_03_94e7762cbdee8792.png) |
| 04 · f781bc3966a2eec3 | [left_turn](gtreset_b5_scene_04_f781bc3966a2eec3.png) |
| 05 · d3156c727eba44e6 | [left_turn](gtreset_b5_scene_05_d3156c727eba44e6.png) |
| 06 · a9a7480e5c110232 | [left_turn](gtreset_b5_scene_06_a9a7480e5c110232.png) |
| 07 · dfa6cf95daff60a5 | [left_turn](gtreset_b5_scene_07_dfa6cf95daff60a5.png) |
| 08 · ea4e999edf5e2196 | [right_turn](gtreset_b5_scene_08_ea4e999edf5e2196.png) |
| 09 · 74627b54baec865 | [right_turn](gtreset_b5_scene_09_74627b54baec865.png) |

原始预测：`scene_XX_trajectories.npz`，包含 `[10,32,80,3]` 的 predicted_poses、prediction_valid、GT 状态和重置时刻。

复现脚本：`generate.py`；完整配置：`run_config.json`；每场景数值：`metrics.json`。
