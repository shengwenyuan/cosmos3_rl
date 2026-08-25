# Cosmos3-Edge：DROID-EEF → UR5 → UR12e 分阶段后训练计划

> 状态：Frozen mainline v1.0
> 日期：2026-08-21
> 范围：第二学期 Level 2 主线与 Level 3 衔接
> 原则：先冻结主链；消融实验单独登记，按资源优先级补充。

> 2026-08-25 amendment：Stage 1 v1 失败后，Stage 1 降级为机制验证，Stage 2 改为 RH20T/UR5 Vision SFT→stateful joint 主线；详见 `../level2/training/stage1_stage2_reframing_after_droid_eef_v1.md`。与本文冻结主链冲突处以 amendment 为准。

## 0. 结论与冻结决策

主训练链冻结为：

```text
Stage 1：Cosmos3-Edge MT → DROID-EEF policy
Stage 2：DROID-EEF → UR5-EEF
Stage 3：UR5-EEF checkpoint → 独立 UR5-joint checkpoint
Stage 4：UR5-joint → 独立 UR12e-joint domain
```

本计划采用以下确定性决策：

1. **Stage 1 从原始 Cosmos3-Edge MT checkpoint 开始。**
   - Edge-LIBERO 已证明 MT → EEF policy 的实现路径可行；
   - 它只作为代码、数据合同、导出和闭环测试的已验证参考；
   - 主训练不加载 Edge-LIBERO 权重，避免增加早期 checkpoint 构成和归因复杂度。
2. **DROID 的 7-DoF Franka 数据只在 EEF action domain 中使用。**
   - 不把 Franka raw joint state 输入共享 EEF domain；
   - EEF 阶段默认不使用 proprioception，以复用现有无 state 的 EEF policy 路径；
   - current EEF pose + gripper state 作为后续可选消融，不进入首版主链。
3. **joint-position 数据的跨机器人使用范围严格受限。**
   - joint PT 主线只接受中大型 6-DoF 机械臂；
   - 不把 Franka/WidowX 等不同 DoF 或不同 joint 语义的数据直接用于 UR joint head；
   - UR5 与 UR12e 虽同为 UR 系列 6-DoF，仍使用独立 domain、normalizer 和 action spec。
4. **Stage 3 不要求同时在线的“双头模型”。**
   - 当前已发布/已验证的 Cosmos3 action-policy 路径按 experiment 选择单一 domain；它不构成“同时双头训练已受支持”的证据；
   - 现有 domain/action adapter 配置方式只用于建立两个独立 experiment；
   - `ur5_eef` 与 `ur5_joint` 分别训练、导出和评测；
   - Stage 3 从 UR5-EEF checkpoint 加载共享权重，重新初始化/加载独立 joint adapter；
   - 不修改主干结构，不要求一次 forward 同时输出 EEF 与 joint。
5. **FK 首先是离线一致性指标，不是主训练损失。**
   - 首版 joint policy 使用现有 action flow loss；
   - FK consistency loss 需要新增训练逻辑，且若 EEF target 由同一 joint label 派生，监督并不独立；
   - 只有主链跑通且离线/闭环结果显示必要时，才作为可选消融加入。
6. **完整清洗与等 EEF 增量间隔采样是所有 EEF 数据的统一前处理。**
   - 删除坏帧、未对齐帧、冻结/异常信号和不完整 action horizon；
   - DROID、UR5 real、UR5 sim 使用同一采样语义；
   - 由于该采样会减少可用 windows，训练规模按最终 complete windows 计算，不按 raw frames 估计。

## 1. 研究问题与可支持的结论

### 1.1 主问题

在不同时改变模型角色、embodiment 和 action representation 的前提下，分阶段训练是否能够：

1. 将 Cosmos3-Edge MT 转换成可靠的单臂 EEF policy；
2. 将大规模 7-DoF Franka EEF 经验迁移到 6-DoF UR5；
3. 在固定 UR5 embodiment 后，将 EEF policy 转换为 joint-position policy；
4. 提高未在前序训练中出现的 UR12e 在 10/50/100 demonstrations 下的数据效率。

### 1.2 主链可以支持的结论

- Stage 1 成功：Edge MT 可以通过干净的 DROID-EEF policy PT 获得单臂 EEF 控制能力。
- Stage 2 成功：跨 DoF/embodiment 的 EEF adaptation 在 UR5 上成立。
- Stage 3 成功：固定 embodiment 后，joint representation 可以从已适配的 EEF checkpoint 获益。
- Stage 4 成功：UR5 joint initialization 可以降低 UR12e 的目标数据需求。

### 1.3 主链不能自动支持的结论

- DROID-EEF 成功不等于 UR5/UR12e 零样本成功。
- RH20T measured-joint proxy 不等于真实 commanded joint target。
- UR5-joint 成功不证明 EEF representation 一定劣于 joint。
- UR5 与 UR12e action tensor 同形，不代表它们共享相同数值 action space。
- 离线 loss、FK error 或 future-video quality 不能替代闭环任务成功率。

## 2. 统一数据与动作合同

### 2.1 EEF action contract

所有 Stage 1–2 EEF 数据统一为单臂 10D frame-wise relative action：

```text
action[k] = [translation_delta(3), rotation_rot6d(6), gripper(1)]
delta_T[k] = inverse(T[k-1]) @ T[k]
raw_action_dim = 10
arms = 1
```

必须冻结并写入 dataset card：

- transform 的乘法方向；
- translation 所在坐标系与单位；
- rotation-6D 定义和 quaternion/matrix 转换顺序；
- TCP/tool frame；
- gripper open/close sign、范围和 command/state 语义；
- chunk length、采样语义和真实时间跨度统计；
- normalizer 版本、train split checksum 和转换器 commit。

首版 EEF observation：

```text
RGB canvas + language
use_state = false
```

首版不混入任何 raw joint proprioception。若后续加入 current EEF pose + gripper，DROID 与 UR5 必须同时使用相同 schema，并建立新的 matched experiment。

### 2.2 Joint action contract

UR5 和 UR12e 分别建立独立的 7D absolute joint domain：

```text
action[k] = [q_target_1..6, gripper_command]
observation.state = [q_current_1..6, gripper_state_or_previous_command]
representation = absolute
```

其中：

- UR5：允许使用明确标注的 future measured-joint proxy；
- UR12e：采集时同时保留 commanded joints 与 measured joints，主 label 优先使用已验证的 command；
- 每个 domain 分别保存 joint order、rad/degree、zero/sign、limits、TCP、controller、normalizer；
- UR5 normalizer 和统计量不得直接用于 UR12e。

### 2.3 相机合同

不同 stage 可以拥有不同 canvas 布局，但每个 domain 内必须固定：

- 相机角色与拼接顺序；
- 分辨率、resize/pad；
- missing-view policy；
- prompt 中的 view description；
- train/server/evaluator pixel-level parity。

缺失 wrist 时不得复制 exterior 并伪装成 wrist。若使用外视角-only 数据，必须注册独立、真实的 view description。

### 2.4 采样与 window 统计

等 EEF 增量间隔采样后，每个正式数据版本必须报告：

```text
source episodes
accepted episodes
accepted segments
retained keyframes
complete action windows
windows per episode: P5 / median / P95
windows per task: min / median / max
real-time span per chunk: P5 / median / P95
rejection reason counts
```

训练 step 不从 raw frames 或旧 dense anchors 推导。每个 run 必须记录：

```text
unique windows
window draws
nominal window exposures
episode/task sampling weights
```

## 3. Stage 0：合同冻结与实现准备

### 3.1 目标

在任何大规模训练前，冻结跨 stage 可比较的数据、action、checkpoint 和评测接口。

### 3.2 必做工作

1. 从已完成的 Edge-LIBERO 项目复用且只复用：
   - Edge action-policy experiment/注册方式；
   - EEF 10D adapter 和 normalizer 接口；
   - export → server → evaluator round trip；
   - action/future-video shape tests。
2. 不加载 Edge-LIBERO checkpoint 权重。
3. 建立独立 domain：

```text
droid_eef
ur5_eef
ur5_joint
ur12e_joint
```

4. 每个 domain 具有独立配置、normalizer、checkpoint metadata 和 server action spec。
5. 对 EEF 与 joint 分别写线性 ramp、恒定位姿、旋转、gripper step 和 episode-tail 单元测试。

### 3.3 Gate 0

以下条件全部通过后才进入 Stage 1：

- Edge MT checkpoint load report 与预期一致；
- EEF action modules 明确 fresh-init；
- 17/33 或实际配置的 vision/action shape 完整 round trip；
- row 0 condition 与 future action rows 无 off-by-one；
- train/server action 反归一化一致；
- 不存在 future observation leakage；
- 固定样本导出前后输出数值一致。

## 4. Stage 1：Cosmos3-Edge MT → DROID-EEF policy

### 4.1 目标

保持 Cosmos MT 的单臂 frame-wise relative EEF 表示，将通用 MT checkpoint 专门化为 action-generating policy。

Stage 1 实际同时完成：

- task mode 转为 policy；
- condition/output token 角色冻结；
- clean EEF action supervision；
- 单臂 domain adapter 学习；
- future RGB 辅助监督；
- EEF policy server 合同建立。

因此它不被描述为“只改模型角色”，但不引入 embodiment/joint representation shift。

### 4.2 数据

主数据：质量审计后的 DROID real-robot 数据，转换为 `droid_eef`。

EEF label 来源必须在转换前判定：

1. 优先：语义和坐标系已确认的 Cartesian command；
2. 次选：future measured EEF proxy，必须显式标记 derived；
3. command 与 measured proxy 不得静默混用。

过滤至少包括：

- success/quality gate；
- causal RGB/action alignment；
- video decode、黑帧、冻结帧、timestamp gap；
- non-finite pose、rotation discontinuity、TCP jump；
- gripper frozen/invalid/sign ambiguity；
- action horizon incomplete；
- 等 EEF 增量采样后的 complete-window gate。

### 4.3 训练顺序

```text
1/10 episodes data smoke
→ fixed small-set overfit
→ medium subset stability run
→ full accepted DROID-EEF run
```

每次扩大数据前都要求：

- action/vision loss finite；
- 固定样本能够过拟合；
- translation/rotation/gripper 方向正确；
- checkpoint export/server round trip 通过；
- 至少一个闭环或可执行 controller probe 出现非 noop、方向正确行为。

### 4.4 Stage 1 输出

```text
checkpoint: edge_droid_eef
domain: droid_eef
action: 10D frame-wise relative EEF
state: none in v1
data manifest + stats + filter report
closed-loop/action probe report
```

### 4.5 2026-08-24 实施状态

Stage 1 v1 已完成数据、训练、checkpoint、server-client 与 RoboLab 链路，但行为 gate 未通过。实际训练 action 为：

```text
delta[k] = inverse(T0) @ Tk,  k = 1..32
```

即 anchored cumulative delta，而不是本计划 2.1 节定义的 frame-wise delta。该差异与失败结果均不得在 Stage 2 中静默继承。Stage 1 v1 checkpoint 只可作为受控初始化消融；Stage 2 主实验必须保留 `Edge MT → UR5-EEF` 直接初始化对照。详见 `../level2/training/droid_eef_stage1_validation_report.md`。

## 5. Stage 2：DROID-EEF → UR5-EEF

### 5.1 目标

保持 EEF representation 和模型角色不变，只适配：

- Franka 7-DoF → UR5 6-DoF；
- robot/camera appearance；
- workspace、TCP/tool、gripper；
- real/sim scene and controller distribution。

### 5.2 数据优先级

#### 主数据：RH20T cfg4

- 与 UR5/Robotiq 路线最接近；
- 有 camera calibration、TCP 和 joint；
- 先审计 TCP command 是否能作为可靠 EEF supervision；
- 若只能使用 measured TCP，则标记为 future measured EEF proxy；
- `gripper_info` 常量通道禁止进入模型；gripper command 需独立确认。

保留两个 manifest：

```text
cfg4_ext3       # 较高数据量，主候选
cfg4_wrist_ext2 # 接近 wrist + exterior prior 的严格对照
```

#### 条件数据

- Berkeley AUTOLab UR5：只有 raw joint/TCP/action 重建和完整 QA 后才能进入；
- UR5 simulation：必须输出与真实 UR5 完全相同的 EEF schema、TCP 和 controller interface；
- 不因数据量大而混入双臂、移动底盘或不同 action 语义的数据。

### 5.3 训练策略

首版保持标准模型结构：

- 从 `edge_droid_eef` 加载共享 checkpoint；
- 注册独立 `ur5_eef` domain 和 normalizer；
- 新 domain action modules 按实际兼容性选择 fresh-init 或显式加载 DROID-EEF EEF adapter；
- 先训练 action modules，再开放少量 LoRA/末端 blocks；
- 小数据下 validation/closed-loop early stopping；
- 不执行无评测的长 schedule。

### 5.4 Gate 2

Stage 2 完成至少满足：

- 50-episode conversion/overlay gate；
- 100-episode overfit gate；
- train/val 无 source episode leakage；
- 反归一化 translation/rotation/gripper error 可解释；
- UR5 controller 中无系统性坐标反向、TCP scale error 或 IK runaway；
- 冻结的 RoboLab UR5 tasks 上产生方向正确的 stage progress；
- action validity 达到预注册阈值。

### 5.5 Stage 2 输出

```text
checkpoint: edge_ur5_eef
domain: ur5_eef
action: 10D frame-wise relative EEF
state: none in v1
UR5 real/sim manifests and per-source stats
RoboLab UR5 closed-loop report
```

## 6. Stage 3：UR5-EEF → 独立 UR5-joint checkpoint

### 6.1 目标

固定 UR5 embodiment、episode、图像、语言和采样 anchor，只改变 action representation：

```text
10D relative EEF
→
6 absolute joints + gripper
```

### 6.2 不修改模型结构的实现

主链不实现同时活动的 EEF/joint 双头。

当前可依赖的是“每个 experiment 选择一个 domain/action adapter”的既有路径；不假设框架已经验证一次训练同时维护 EEF 与 joint 两个 head。

采用两个独立 experiment：

```text
action_policy_ur5_eef_edge
action_policy_ur5_joint_edge
```

Stage 3：

1. 从 `edge_ur5_eef` checkpoint 加载共享 backbone；
2. 注册/启用现有机制下的 `ur5_joint` action domain；
3. 使用 7D joint state/action adapter；
4. joint action modules fresh-init；
5. 保存为独立 `edge_ur5_joint` checkpoint；
6. EEF checkpoint 保持不变，用于离线和闭环对照。

这只增加 dataset/config/domain adapter，不要求一次 forward 同时输出两种 action，也不要求改变 transformer 主体。

### 6.3 Matched dataset

Stage 3 必须使用 Stage 2 的同一批：

- source episode IDs；
- train/val/test split；
- 相机 canvas；
- instruction；
- 等 EEF 增量采样 anchor；
- action chunk 起点与终点。

只替换 label/state：

```text
observation.state = current UR5 measured joints + previous gripper command
action = future measured-joint proxy + future gripper command
```

metadata 必须标记：

```text
action_is_derived = true
action_arm_source = future_measured_joint_position_proxy
```

### 6.4 FK 的首版用途

首版不增加 FK training loss。离线报告计算：

- `FK(q_pred)` 对 target TCP 的 translation error；
- SO(3) geodesic rotation error；
- predicted joint limits/velocity/acceleration；
- FK/TCP calibration residual；
- EEF 与 joint checkpoints 在相同 rollout seeds 下的 stage progress。

若 EEF target 由同一 measured joint 通过 FK 派生，FK loss 不是独立标签。只有在标准 joint loss 已跑通且显示任务空间误差需要重加权时，再加入低权重 FK regularization 消融。

### 6.5 Gate 3

- joint order、zero/sign 和 row0/row1 单元测试通过；
- joint head 小数据 overfit 成功；
- action validity、joint limit 和速度约束通过；
- 在相同 UR5 tasks/seeds 下完成 EEF 与 joint checkpoint 对照；
- 不因 joint train loss 更低而跳过闭环比较。

### 6.6 Stage 3 输出

```text
checkpoint A: edge_ur5_eef   # Stage 2 冻结产物
checkpoint B: edge_ur5_joint # Stage 3 新产物
matched EEF-vs-joint offline report
matched EEF-vs-joint RoboLab report
```

## 7. Stage 4：UR5-joint → UR12e-joint

### 7.1 目标

测试 UR5 joint policy 是否能够降低未在前序训练中出现的中大型 6-DoF UR12e 的 demonstration demand。

### 7.2 UR12e 独立合同

UR12e 必须注册独立：

```text
domain = ur12e_joint
action spec
state spec
normalizer
joint limits and zero/sign convention
TCP/Hand-E transform
controller and safety limits
camera canvas and prompt description
```

不得直接复用：

- UR5 train stats；
- cfg4 Robotiq 2F-85 数值范围；
- UR5 joint limits、workspace 或 TCP；
- UR5 server-side clipping 常量。

### 7.3 数据

按照已有实验室计划采集至少两个任务：

```text
D10  ⊂ D50 ⊂ D100
```

每条数据必须保留：

- commanded 与 measured joints；
- Hand-E command 与 actual state；
- 三路 RGB 和独立时间戳；
- operator、teleop mode、success/failure；
- latency、安全停止和人工干预。

### 7.4 训练顺序

首版：

1. 建立新的 UR12e state/action adapter；
2. 从 UR5 joint checkpoint 加载共享 backbone；
3. 在 action dimension/模块结构兼容时，显式 warm-start UR5 joint adapter/head；
4. 使用 UR12e 独立 normalizer；
5. 先训练 action adapter/head；
6. 再开放 LoRA/末端 blocks；
7. 小数据不立即 full SFT。

### 7.5 评测

每个 D10/D50/D100 checkpoint 报告：

- task success 与 Wilson interval；
- approach/align/grasp/transport/place stage progress；
- joint/action validity；
- safety stop、intervention、timeout；
- inference/control latency；
- failure taxonomy；
- success-vs-demonstrations curve。

## 8. 消融与对照实验清单

消融不阻塞主链，按资源优先级执行。

### P0：主链必要验证，不视为可删减消融

1. 每个 stage 的 small-set overfit。
2. 每个 checkpoint 的 export → server round trip。
3. Stage 2/3 的 UR5 闭环评测。
4. UR12e D10/D50/D100 严格嵌套和安全评测。
5. 数据清洗前后最终 episode/window/task-density 统计。

### P1：最高价值、资源允许时优先

#### A1. DROID-EEF 阶段是否有价值

```text
Edge MT → UR5-EEF
vs.
Edge MT → DROID-EEF → UR5-EEF
```

资源节约版本：在相同 100-episode UR5 subset、相同步数、单 seed 下先比较；只有差异明显再扩 seeds/full data。

#### A2. UR5 EEF vs joint representation

直接使用 Stage 2/3 已产生的两个 checkpoints，在相同 RoboLab tasks、seeds、相机和 controller budget 下比较。原则上不需要额外主训练，只增加评测成本。

#### A3. UR5 joint-head warm start 是否帮助 UR12e

```text
fresh UR12e joint adapter/head
vs.
UR5 joint adapter/head warm start
```

先在 D10 上筛选；只有 warm start 显示收益，再扩展 D50/D100 对照。

### P2：有研究价值，但不阻塞主线

1. `cfg4_ext3` vs `cfg4_wrist_ext2`。
2. RH20T-only vs UR5-sim → RH20T staged adaptation。
3. DROID/UR5 EEF `use_state=false` vs current EEF pose + gripper state。
4. FK consistency regularization vs standard joint flow loss。
5. UR5-EEF → UR12e-EEF 与 UR5-joint → UR12e-joint。
6. Berkeley raw-reconstructed UR5 数据增益。
7. LoRA/末端 blocks vs full SFT，仅在 D100 且小数据方案已稳定后执行。

### P3：资源富余时

1. Edge-LIBERO checkpoint 初始化对照。该实验不进入主计划，以避免早期 checkpoint 构成复杂化。
2. 多 seeds 的完整复现。
3. chunk length、执行 horizon、相机数量、denoise steps 等系统消融。
4. EEF commanded label vs measured EEF proxy 的 matched comparison。
5. UR12e EEF/joint 混合或同时输出的多头架构研究。

## 9. 资源控制规则

1. 先跑 shape/data/overfit，再扩大数据和步数。
2. 小数据 stage 使用 validation/closed-loop early stopping，不照搬官方 DROID no-eval reproduction。
3. 32 GPU 资源优先用于并行必要对照，不用于放大小数据单 run。
4. 每个 stage 只保留预注册的少量 checkpoints；smoke artifacts 与正式结果分目录。
5. 一次实验只改变一个主要变量：初始化、数据源、action representation、相机或训练策略。
6. 任何 stage 若 small-set overfit 不能产生方向正确的闭环动作，停止扩算力，先查数据/坐标/server/controller。

## 10. Stop/Go 决策

### Stage 1 No-Go

- DROID Cartesian action 语义无法确认；
- EEF window 数在清洗后不足以支撑预定任务范围；
- fixed subset 可拟合 loss 但 server action 持续 noop/反向；
- future-video/action 时序无法统一。

### Stage 2 No-Go

- RH20T TCP/gripper/相机无法通过因果和语义 gate；
- UR5-EEF 反复离开 IK/workspace 且 controller parity 已确认；
- 100-episode overfit 仍无方向正确 progress。

### Stage 3 No-Go

- measured-joint proxy 与视频/TCP 严重不一致；
- joint target 存在系统性 off-by-one、冻结或错误 joint order；
- joint checkpoint 无法通过 action validity gate。

### Stage 4 No-Go

- UR12e command/measured joint、Hand-E 或相机时间戳不完整；
- D10 不能稳定 overfit；
- warm-start 导致越界/不安全行为且 fresh head 不存在同样问题；
- 真机安全链、watchdog 或急停未验证。

## 11. 最小完成标准

若资源有限，论文/项目的最小完整版本为：

1. Edge MT → DROID-EEF 数据与训练链跑通；
2. DROID-EEF → cleaned RH20T cfg4 UR5-EEF；
3. 同一 UR5 episodes 上生成独立 EEF/joint checkpoints；
4. 在冻结 RoboLab UR5 tasks 上完成 matched EEF-vs-joint 评测；
5. UR12e 至少两个任务完成 D10/D50/D100 数据与一种 joint warm-start 主方案；
6. 若只能完成一个关键对照，优先执行 P1-A3：fresh UR12e joint head vs UR5 joint-head warm start。

## 12. 产物与命名

```text
datasets/
  droid_eef_<revision>/
  rh20t_cfg4_ur5_eef_<revision>/
  rh20t_cfg4_ur5_joint_<revision>/
  ur12e_joint_<revision>/

checkpoints/
  edge_droid_eef/
  edge_ur5_eef/
  edge_ur5_joint/
  edge_ur12e_joint_d10/
  edge_ur12e_joint_d50/
  edge_ur12e_joint_d100/

reports/
  stage1_droid_eef/
  stage2_ur5_eef/
  stage3_ur5_eef_vs_joint/
  stage4_ur12e_data_efficiency/
```

每个正式产物必须保存：

- source/dataset/split manifest checksum；
- converter、code、environment commit；
- action/state/camera spec；
- normalizer；
- checkpoint load report；
- training config、seed、GPU、step 和 effective window exposure；
- export/server/eval config；
- closed-loop logs、videos 和 failure labels。

## 13. 与现有文档的关系

- `summer_milestone/final_report/eece7945_final_report.tex`：Level 1 证据、action contract 与 Level 2/3 claim boundary。
- `cosmos3_edge_rh20t_posttrain_reassessment.md`：DROID joint contract、RH20T 规模与训练重评估。
- `rh20t_cfg3_cfg4_lerobot_conversion_audit.md`：RH20T 因果转换、相机、joint/gripper 和 filter gates。
- `rh20t_ur5_to_cosmos3_spec.md`：原 UR5 joint 主线规格；本计划将 EEF adaptation 前置，但保留其 joint/safety约束。
- `ur12e_level3_12x7_lab_plan.md`：UR12e D10/D50/D100 采集和实验室执行计划。

本文件是当前跨 stage 训练顺序、初始化和消融优先级的主计划；若与旧计划的训练顺序冲突，以本文件为准，但数据安全、因果对齐和真机安全要求继续沿用旧文档中的更严格版本。
