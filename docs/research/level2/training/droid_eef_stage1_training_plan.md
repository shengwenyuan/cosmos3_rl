# Cosmos3-Edge 第一轮 DROID EEF 训练计划

日期：2026-08-21

执行状态（2026-08-24）：训练与部署链路已跑通，但策略行为门禁未通过。实测记录见 `droid_eef_stage1_validation_report.md`。本文件保留为 v1 历史计划；2026-08-25 后的重训决策以 `stage1_stage2_reframing_after_droid_eef_v1.md` 为准。

## 1. 目标与边界

目标是在 Cosmos3-Edge MT 基座上学习跨机器人 EEF 动作先验，为 RH20T cfg4 和 UR12e 迁移提供 Level 2 起点。Nano Policy DROID 是数据口径参考，不直接继承其 joint-space action head。

首轮固定：DROID success、官方 `keep_ranges_1_0_1`、`easy_v1` 二次筛选、15 Hz、三相机、无 proprio state、EEF delta action，以及 `train chunk=32 / predict=32 / execute=8 / denoise=4`。

## 2. 标签语义先审计

本轮实际 DROID loader 的 `ee_pose_delta` 使用 window 起点 `T0` 与未来帧 `Tk` 计算 anchored SE(3) 增量：

```text
delta[k] = inverse(T0) @ Tk,  k = 1..32
```

再编码为：

```text
[dx, dy, dz, rot6d(6), gripper_open_fraction]  # 10D
```

这本质上是“未来实测 EEF 轨迹代理”，不是数据里的 Cartesian command；也是相对同一 `T0` 的累计目标，不是 `inverse(T[k-1]) @ T[k]` 的逐帧增量。若改用 command 或逐帧增量，必须建立新数据版本、统计量和实验，不能静默切换。

## 3. 第一轮配置

| 项 | 建议值 | 说明 |
|---|---|---|
| base checkpoint | 官方 Cosmos3-Edge MT | 不加载 Nano joint action head |
| dataset profile | `cosmos3_droid_success_640x360_v1` | 显式指定，禁用 legacy 自动探测 |
| action space | `ee_pose_delta` | backward-anchored、relative-to-window-start |
| action dim | 10 | xyz + rot6d + gripper |
| state | `use_state=false` | 保持跨机器人接口 |
| FPS | 15 | 首轮不重采样 |
| action chunk | 32 | 建立官方 Edge DROID 时间窗口对齐基线 |
| exact duration | 33 | 1 observation + 32 future actions |
| image | 三视角、480 训练分辨率 | 沿用 Edge/Nano augmentation 与 prompt 组织 |
| normalization | `quantile_rot` | 仅从 `easy_v1_train` 的 10D delta 重算 |
| token budget | 74,000 起测 | 先沿用 Edge LIBERO，按 dry-run 调整 |
| optimizer | FusedAdamW，base LR `5e-5` | fresh action head multiplier 5 起测 |
| action loss | scale/weight 10 | 与 Edge action 配置保持一致 |
| activation checkpoint | selective AC | 8 GPU 首选 |
| inference | 32-step 预测、滚动执行 8-step | 不复制官方 open-loop 32；每 8 步重规划 |
| denoise | 4 steps | 对齐官方 Edge DROID；8 steps 只作为质量消融 |

Stage 1 对齐官方 Edge DROID 的 15 Hz、chunk 32、predict 32 和 4-step denoise，以建立最少变量的基线；不对齐其 8D joint action、proprio state 和 open-loop 32 执行方式。Nano DROID 的全局 batch=8192、LR=`2e-4` 也不原样复制。

chunk 16 只作为后续 RH20T/UR12e 备选：若 3 mm 重采样后 episode 明显变短、chunk 32 显存不足，或 32-step 长期预测误差明显增大，再做 16-step 消融，不作为 DROID Stage 1 首跑配置。

## 4. 必要代码工作

1. 新建独立 Edge-DROID-EEF experiment config，禁止直接修改 Nano 配置。
2. 把 filter path、manifest path、dataset revision、profile、窗口上限做成显式配置。
3. 让 dataset 只暴露 manifest 中的 episode/range；记录每层过滤计数。
4. 从筛选后的 train split 计算 10D EEF delta 统计量；禁止复用 absolute pose 或 joint action stats。
5. 增加数据 smoke test：action shape/finite、rot6d 可还原、当前帧基准、夹爪范围、三相机顺序、窗口不跨 range。
6. checkpoint 名称必须编码数据版本、action 语义、chunk 和迭代数。

## 5. 分阶段训练

### P0：无 GPU 数据门禁

- metadata-only 生成 `official_keep_v1`、`easy_v1`、split 和统计。
- 随机可视化 200 条窗口；审计 measured-vs-command。
- 通过后冻结 manifest SHA256 和统计文件 SHA256。

### P1：加载与显存 smoke

- 1/10/100 episode 跑通单卡，再跑 8 卡 20 iterations。
- 自动搜索每卡 micro-batch；优先目标 global batch 256–512，必要时使用 gradient accumulation。
- 验证 resume、checkpoint 写入和 dataloader 吞吐。

### P2：小集过拟合

- 100–500 episodes，200–500 iterations。
- 要求 action loss 明显下降；固定样本的预测方向、旋转和夹爪趋势可解释。
- 若失败，先查标签/归一化/时间对齐，不直接扩大训练。

### P3：`easy_v1` 首轮

- 先跑 2,000 iterations；每 500 iterations 保存并做固定离线评测。
- 门禁通过后延长到 5,000 iterations。
- 同一 run 最终只保留最大 iteration 权重；中间评测表和日志保留。

### P4：规模与窗口消融

至少比较：

- `easy_v1 + cap64`；
- `easy_v1` 不限每 episode 窗口；
- `official_keep_v1`。
- 在主线 chunk 32 基线完成后，仅按既定触发条件增加 chunk 16 消融。

比较时固定总 optimizer steps、global batch 和 seed，区分“数据更干净”与“训练看过更多样本”的收益。

## 6. 验收指标

- 数据：有效窗口数、任务族/场景覆盖、过滤原因、视频解码失败率。
- 优化：action total/translation/rotation/gripper loss，梯度范数，吞吐和峰值显存。
- 离线：按任务族报告 delta L1、旋转 geodesic error、gripper F1、32-step rollout drift，并单独报告前 8 步误差。
- 部署前：目标 4070 Ti 级主机测 4/8 denoise steps 的 P50/P95 推理时延；验证 32 预测/8 执行没有明显停顿。

本轮 P3 离线与 RoboLab 行为门禁未通过，因此 Stage 1 不记为成功基线。RH20T cfg4 只允许进入受控 Stage 2 pilot，并必须同时保留 Edge MT 直接初始化对照；只有坐标、归一化、action 解码及小集过拟合门禁通过，才允许扩大训练，更不能直接进入 Level 3 真机阶段。
