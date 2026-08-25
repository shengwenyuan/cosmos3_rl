# Stage 1 / Stage 2 路线重构

日期：2026-08-25
背景：DROID EEF Stage 1 v1 的训练链路完成，但离线分布与 RoboLab 行为 gate 失败；历史 Cosmos3-Nano MT→EEF 实验也长期不理想。

## 1. BEHAVIOR1K Action SFT 参考记录

参考：`cosmos3-behavior1k-sft-v0-dejie-20260815.md`。

关键事实：

- 20,000 episodes、100 tasks、210,916,774 raw frames；30 FPS 数据按 5 FPS 采样。
- 每个 sample 为 65 video frames + 64-step、23D native action，最终得到 539,206 samples。
- 先训练 Vision SFT `iter 500`，再 fresh-init action heads 做 Action SFT。
- 48 GPUs、global batch 48、一个 logical epoch，共 11,234 optimizer steps。
- model、optimizer、scheduler 与 trainer checkpoint 正常落盘。
- 没有 validation action metric、rollout 或 task success；action 未 normalization，代码 dirty，数据无不可变 hash。

客观边界：该记录只证明 Vision SFT→Action SFT、episode-aware sampling、小 global batch 和完整 logical epoch 的工程链可运行。即便外部观察认为模型效果差，文档本身也不能证明其动作是否稳定，更不能证明它不会产生混乱运动。它同样不能作为 23D unnormalized action 或 64-step horizon 的质量依据。

对本项目最有启发的不是最终 checkpoint，而是两点：

1. 不直接要求原始 MT 同时完成目标视觉域适配和 policy learning；先做目标域 Vision SFT。
2. 不用大 GBS 替代 optimizer updates。BEHAVIOR1K 的 GBS 约 48；Stage 1 v1 的 GBS 512，每个数据 epoch 只有约 3,231 次更新。

## 2. 总体判断

不再把“DROID EEF Stage 1 成功”设为 Stage 2 前置条件。Stage 1 v1 保留为负结果和初始化消融；Stage 2 改为目标域重新起跑。

```text
旧主链：MT → DROID EEF → UR5 EEF → UR5 joint → UR12e joint

新主链：MT → UR5/RH20T Vision SFT → stateful UR5 joint → UR12e joint
                    ├─ DROID Stage 1 backbone initialization control
                    └─ UR5 EEF representation ablation
```

Stage 1 不再消耗全量算力追求“必须通过”；Stage 2 的成功也不能反推 Stage 1 成功。

## 3. Stage 1 新思路：低成本机制验证

### 3.1 目标

回答两个有限问题：

1. DROID 目标域 Vision SFT 是否能让后续 action learning 不再出现明显尺度失控？
2. GBS 512 是否因 optimizer updates 过少而加重欠优化？

不再以 full DROID task success 为必达结果。

### 3.2 实验顺序

#### S1-R0：v1 收口

- 对 iter 6500 复跑与 iter 5000 完全相同的 128-window / 49-shard 离线分布 gate。
- 只作为结果归档；若仍越界，不再做无保护 RoboLab rollout。

#### S1-R1：Vision-first 小规模诊断

- 使用 Fx4 DROID 的 RGB、语言和未来视频做短程 Vision SFT。
- 从该 Vision SFT checkpoint fresh-init `action2llm`、`llm2action`、action modality/position modules。
- 首个诊断保持 v1 anchored target、manifest 和 stats 不变，只改变初始化与优化 schedule，避免无法归因。
- 只使用冻结的小 subset；先验证 fixed-sample overfit，不直接启动全量 DROID。

#### S1-R2：表示诊断，可选

只有 R1 仍不能过拟合时，才在同一小 subset 比较：

- anchored cumulative + horizon-aware、hold-centered normalization；
- stateful frame-wise EEF delta。

它们必须使用独立 action spec、stats 和 checkpoint 名称，不能覆盖 v1。

### 3.3 优化配置

- 首选 GBS 64；GBS 128 只作吞吐/质量对照，不再使用 GBS 512 作为首轮基线。
- 多卡用于缩短 wall time；64 GPUs 时 micro-batch 1、gradient accumulation 1 即 GBS 64。
- `steps_per_epoch = ceil(train_windows / GBS)`，报告 optimizer updates 与 window exposures。
- 不按卡数线性放大 LR；base/action-head LR 由小集 overfit 冻结。
- action modules fresh-init；v1 action head 不继承。

### 3.4 Stop/Go

必须依次通过：fixed subset overfit → 反归一化分布 gate → 有保护 RoboLab progress。任一步失败即停止扩数据。只有 R1/R2 小集明显通过，才讨论是否值得重跑全量 DROID；该决定不阻塞 Stage 2。

## 4. Stage 2 新思路：目标域重新起跑

### 4.1 Stage 2-J：主线

目标是在 RH20T cfg4 / UR5 上建立可学习、可闭环的 stateful joint policy，并直接服务后续 UR12e joint 迁移。

```text
initialization: RH20T/UR5 Vision SFT
state: current UR5 joints + gripper state
target: future UR5 joint targets + gripper command
domain/stats: UR5 独立
chunk: 16
predict/execute: 16 / 8
denoise: 4
GBS: 64 primary, 128 ablation
```

原因：它更接近已有官方 action-policy 的 state/action 范式，也避免继续把无 state EEF 作为 UR12e 主线的必经依赖。UR5 与 UR12e 即使同为 6 DoF，也必须使用独立 head、normalizer、joint limits 与 TCP 合同。

### 4.2 初始化对照

在完全相同的 RH20T subset、action spec、stats、steps 和 seed 下比较：

1. RH20T/UR5 Vision SFT → fresh UR5 joint modules（主候选）。
2. 已验证 action-policy checkpoint 的 shared backbone → fresh UR5 joint modules。
3. DROID EEF iter 6500 shared backbone → fresh UR5 joint modules（迁移对照）。
4. Edge MT → fresh UR5 joint modules（资源允许时的下界）。

禁止直接加载 iter 6500 的 action head 和 v1 stats。若 iter 6500 初始化不改善 overfit 速度、离线指标或闭环 progress，则结论为 DROID EEF Stage 1 没有可测迁移收益。

### 4.3 Stage 2-E：降级为消融

UR5 EEF 不再是主链前置，只保留 matched representation experiment：

- 使用与 Stage 2-J 相同 episodes、canvas、split 和 window starts。
- 显式输入 current EEF pose + gripper state。
- chunk 16；新建 hold-aware stats 与输出安全 guard。
- 与 joint policy 比较数据效率、离线 action validity 和 RoboLab progress。

EEF 成功只能说明 target-domain EEF 可学；不能自动证明 DROID EEF transfer 成立。

### 4.4 Stage 2 执行门禁

```text
50–100 episode conversion/overlay
→ fixed subset overfit, 200–500 optimizer steps 起测
→ offline prediction distribution gate
→ RoboLab protected rollout
→ medium subset
→ full accepted RH20T/UR5 data
```

扩大训练前必须满足：

- fixed samples 可过拟合，translation/rotation 或 joint/gripper 各项均改善；
- 输出无 NaN、无系统性越界，预测/目标尺度比进入预注册范围；
- safety guard 不持续接管大部分动作；
- RoboLab 至少出现方向正确的 approach/grasp/transport progress；
- validation 按 episode/task/scene 切分，无 source leakage。

## 5. 预期管理

- Stage 1 v2 的合理成功标准是“可学习且不尺度失控”，不是恢复一个已被否定的必经 stage。
- Stage 2 初始不期待零样本任务成功；先期待 target-domain overfit 和安全、方向正确的闭环 progress。
- 若 Stage 2-J 成功而 Stage 2-E 失败，后续取消 EEF→joint 的串行依赖，直接推进 UR5 joint→UR12e joint。
- 若 Vision-first + 小 GBS 仍不能通过小集 overfit，应优先怀疑 action token/flow objective、label timing 或模型架构适配，而不是继续堆数据和 GPU。
