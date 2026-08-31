# RH20T cfg4 UR5 Joint Stage 2 iter10000 验收报告

日期：2026-08-31
结论：**训练与执行链路完成；离线 gate 经显式后处理通过；RoboLab 控制 gate 通过、任务行为 gate 失败。iter10000 仅接受为诊断基线，不记为 Stage 2 任务策略通过。**

## 1. 训练产物与冻结合同

- 数据：`rh20t_cfg4_ur5_joint_ext2_15hz_v1`，1776 episodes，1,255,431 frames。
- 初始化：Cosmos3-Edge DCP；64 GPUs，micro-batch 1/rank，gradient accumulation 8，global batch 512。
- 优化：10,000 iterations，base LR `5e-5`，action modules LR multiplier `5`，每 500 iter 保存。
- 时序：固定 15 Hz，train/predict/execute=`32/32/8`。
- 状态：当前 6D UR5 joint + gripper close fraction。
- 目标：未来 32 个 7D absolute joint targets `[q1..q6, gripper]`。
- 视觉：两路真实外部相机，各 `360x640`，沿高度拼接为 `720x640`；video subsample 2。
- 推理质量 oracle：UniPC denoise 30，guidance 1，seed 0，action normalization depth 2。

主产物：

```text
/mnt/pfs/swy/cosmos3/runs/cosmos3_action_rh20t_ur5_joint/cosmos3_action_rh20t_ur5_joint/stage2_c32/rh20t_cfg4_ur5_joint_edge_64gpu_10k
```

最终 checkpoint 的 model/optim/scheduler/trainer 各 64 shards 完整；W&B 未出现 NaN，action loss 尾段仍缓慢下降。

## 2. G0：合同与静态链路

通过：

- checkpoint、resolved config、manifest 与 quantile stats sidecar 可加载；
- wire action 严格为 `32x7` absolute joint，关节顺序使用 URDF 名称；
- gripper 在训练、server 与 client 中均为 `close_fraction`；
- client 真实构造 vertical-pair 画布，不补训练中不存在的 wrist view；
- server-client handshake、current-state conditioning、execute-8、Isaac controller 和多视角录制均可运行。

Cosmos server commit：`36ca431`。RoboLab commit：`484f7f1`。

## 3. G1：256-window 离线 gate

`triangular_3tap` 只处理六个 arm joints，保留 gripper 原值；server 同时归档 raw action。该功能由通用 `JointChunkPostprocessorConfig` 控制，不含 RH20T 专用分支。

| 指标 | Raw | Postprocessed | 结论 |
|---|---:|---:|---|
| arm MAE | 0.04974 rad | 0.04960 rad | 接近 |
| max velocity | 1.452 rad/s | 1.168 rad/s | 通过 |
| max acceleration | 40.981 rad/s² | 8.896 rad/s² | Raw 失败；处理后通过 |
| P99 step scale ratio | 1.165 | 1.132 | 通过 |
| gripper accuracy | 0.9348 | 0.9348 | 通过 |
| hard gripper violations | 0 | 0 | 通过 |
| joint-limit violations | 0 | 0 | 通过 |

gripper raw range为 `[-0.0054, 1.0030]`，属于已批准的 ±0.05 数值容差，二值语义不变。

结论：**部署合同包含 postprocessor 时 G1 通过；raw action 本身仍有加速度尖峰，不能宣称原始模型输出已完全满足执行质量。**

报告：

```text
<run>/evaluation/iter_000010000/offline_gate_postprocessed_256.json
<run>/evaluation/iter_000010000/offline_gate_postprocessed_256_samples.npz
```

## 4. G2：RoboLab Banana in Bowl canary

配置：Isaac Sim 6.0.1，`rh20t_vertical_pair`，`rh20t_ur5` reset，denoise/guidance=`30/1`，execute 8，单 episode 750 frames。

验收前修复了两个确定性问题：

1. closed reset 原先只初始化 Robotiq master joint，五个 mimic joints 保持 0，导致首帧物理冲击；现六个 gripper DOF 均初始化为 `0.82474`。
2. Isaac 偶发阻塞超过 WebSocket 默认 20 秒 ping timeout；server 改为 20 秒 interval、120 秒 timeout。完整 rollout 多次跨过 20 秒级阻塞且未断线。

最终轨迹：

| 指标 | 结果 |
|---|---:|
| 首帧 arm 相对 reset 最大偏差 | 0.0164 rad |
| 前 5 帧最大偏差 | 0.0342 rad |
| joint tracking MAE | 0.00653 rad |
| tracking P99 / max | 0.03694 / 0.04675 rad |
| EEF path length | 0.664 m |
| EEF speed mean / max | 0.0243 / 0.258 m/s |
| gripper | 全程 closed，无切换 |
| EEF 到 banana 最小距离 | 0.281 m |
| EEF 到 bowl 最小距离 | 0.0774 m |
| banana 最大位移 | 0.014 mm |
| bowl 最大位移 | 14.48 mm |
| task success | 0/1 |

目检结果：轨迹连续且受控，机械臂稳定接近并触碰 bowl，没有接近、抓取或搬运 banana。它不是此前 EEF 模型的无序大幅运动，但没有形成正确任务进展。

产物：

```text
/home/lenovo/swy/RoboLab_div/output/rh20t_cfg4_ur5_joint_iter10000_rh20t_mimic_fixed_keepalive120/
```

## 5. Gate 判定

| Gate | 判定 |
|---|---|
| G0 合同与工具链 | Pass |
| G1 离线安全分布 | Conditional pass：必须启用已登记 postprocessor |
| G2 低层执行与控制 | Pass |
| G2 任务语义进展 | Fail |
| Stage 2 总体 | Diagnostic baseline only |

可以确认的是：图像与状态能进入模型、action 可按正确关节顺序传输、后处理与 controller 能产生连续可跟踪的运动。不能据此确认的是：模型已经学会根据视觉和语言选择正确对象、生成正确 gripper phase，以及 raw action 已完全达到部署质量。

## 6. 下一步

1. 先选择 RH20T 训练分布内、RoboLab 可复现的任务，冻结真实训练 prompt、双相机画布和初始状态，运行同一闭环 gate。
2. 对固定 validation windows 增加对象/任务分层、gripper transition 与方向性指标，确认正确 GT 节点是否能被模型复现。
3. 若训练分布内 gate 有正确方向但完成度不足，再从 iter10000 resume 或做目标域微调；优先增加任务平衡、目标物 grounding 和有效 gripper transition。
4. 若训练分布内仍持续选择错误目标，不直接堆 steps；先审计 prompt-image-action 配对、camera domain gap 和 task sampling。

因此后续需要继续做视觉—语义—动作联合对齐，但不是把本次结果简单归因于“动作控制错误”，也不能把平滑运动误记为完整 action policy 已通过。
