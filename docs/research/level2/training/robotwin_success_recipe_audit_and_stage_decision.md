# RoboTwin 成功配方审计与 Stage 决策

日期：2026-08-27  
审计基线：作者 `0bc3de6`，其 NVIDIA 父提交 `326b399`，最新 NVIDIA `de925b6`；本项目 `d02af68`。

## 结论

1. 高分最可信的解释是 **stateful absolute joint + 同构数据/推理合同 + 约 12 次有效 window 遍历**，不是 Nano、大 GBS 或单一超参的魔法。
2. 它强烈支持 Stage 2 改走 RH20T/UR5 stateful joint；不支持继续把无 state、anchored EEF 当作必经桥梁。
3. **不立即重做 Stage 1**：先用现有 iter 6500 补 `30-step / guidance=1` 质量上界；只有小集可学习且离线分布通过，才讨论 DROID EEF v2。
4. **立即推进 Stage 2**：第一版直接 Action SFT；Vision SFT 从前置条件降为消融。部署仍用 Edge，Nano 只作云端上界，不改变 4070 Ti 目标。

## 1. 作者改了什么

作者分支仅比其父提交多 1 个提交：34 个文件、`+3721/-4`，其中 26 个新增、8 个修改；没有改模型主体、flow loss 或 trainer 核心。

| 层 | 新增/修改 | 判断 |
|---|---|---|
| 数据 | RoboTwin LeRobot loader、14D stats、factory、domain/view 描述 | 核心新增；绝对关节、state row、三相机 canvas、视频降采样在此闭环 |
| 训练 | Nano experiment、clean/all TOML 与短 launcher | 核心配方；沿用官方 DROID 的 WAM、FusedAdam、JSON prompt、fresh action head、rank/worker episode shuffle |
| 推理 | RoboTwin server、`vla-eval-cosmos3` bridge/config | 价值很高；把 state、stats、canvas、fps、chunk/vsub 固化到服务端 |
| 验收 | Docker、50-task sharding/汇总、3 个 RoboTwin task 修复、官方 per-task step limit | 修复 policy eval 的未初始化状态与欠配 step；未改 success 定义 |
| 主线兼容 | 注册入口、domain id、viewpoint 文本 | 小改；最新 NVIDIA 已自带 `robotwin=17` 和通用 multiview，应重做小范围移植而非整提交 cherry-pick |

关键代码锚点：动作/状态合同见 `DelinQu@0bc3de6:cosmos_framework/data/generator/action/datasets/robotwin_lerobot_dataset.py:195`、`:331`；训练配置见 `.../action_policy_robotwin_nano.py:183`、`:268`、`:282`；推理合同见 `DelinQu@0bc3de6:packages/vla-eval-cosmos3/configs/robotwin.yaml:24`。

作者分支相对最新 NVIDIA 为 8 commits behind / 1 ahead；本项目只包含到 `5eee9ed`。实现前先吸收后续官方修复，再按现有 manifest/action-spec 抽象移植 loader、recipe 和 parity gate；不要覆盖我们的 EEF server-client 合同。

## 2. 真正独特的训练配方

| 维度 | RoboTwin 成功配方 | DROID EEF Stage 1 v1 | 含义 |
|---|---|---|---|
| target | 14D absolute joints，当前 14D state 作为第 0 row，共用逐通道 q01/q99 | 10D anchored cumulative EEF，无 numeric state，跨 horizon `quantile_rot` | 最大差异；前者可学“复制当前状态/hold”，后者 model-zero 不是 hold |
| 数据 | 27,500 eps / 50 tasks；约 91% randomized；不删静止段 | 28,358 eps；motion-biased starts | RoboTwin 抽样约 35% 相邻动作近似 hold；目标多样性和静止先验都被保留 |
| 时间 | action 30 Hz、chunk 64；video 每 4 步取样，17 frames；2.13 s | action/video 15 Hz、chunk 32，33 frames；2.13 s | 成功点是保持物理 horizon，同时把视觉 token 与动作频率解耦 |
| 数据预算 | GBS 2048 ×25k = 51.2M exposures，约 11.9 window passes；clean run 约 13.1 passes | GBS 512 ×6500 = 3.33M，约 2.0 passes | 大 GBS 并非原罪；Stage 1 更像样本遍历和 optimizer updates 均不足，不能只怪 GBS |
| 优化 | LR `5e-5`，action heads ×5，warmup 500，LR floor 0.1，clip 1 | LR `1e-5`，heads ×5，warmup 500，衰减到 0 | RoboTwin 给新 head 更强且不归零的学习信号 |
| 模型 | 原始 Cosmos3-Nano，WAM，仅训练 generation/action 相关模块 | Edge-MT，WAM，同类模块 | 不是全参数 SFT；也证明独立 Vision SFT 并非所有目标域的必需条件 |
| 推理 | UniPC 30 steps，guidance 1；同步执行完整 64-action chunk | 4 steps，既有服务默认 guidance 3，execute 8 | 高分没有验证 4-step、异步或 receding horizon |
| 视觉 | 640×480 三相机；训练/推理相同 canvas、view 文本、17-frame duration | 已修复三相机合同，但 action 表示失败 | 作者报告 320×240 使抓放约 90%→0%，parity 是硬门禁，不是实现细节 |

这里的 WAM 不是 17 帧观测历史：服务端把当前 canvas 重复成模型所需形状，再联合去噪未来 video/action；`vsub=4` 节省的是未来视觉 token，action 仍为 30 Hz dense target（`DelinQu@0bc3de6:cosmos_framework/scripts/action_policy_server_robotwin.py:225`、`:305`）。action 从 `t0` 开始且 state 也是 `t0`，天然提供“当前位置→当前位置”锚点。

配方原文：数据与成绩见 `DelinQu@0bc3de6:examples/robotwin/README.md:11`、`:80`；超参见 `:120`；state/quantile/vsub 见 `DelinQu@0bc3de6:cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_robotwin_nano.py:210`；优化器与 scheduler 见 `:86`、`:109`。

### 证据边界

- 95.24% / 94.24% / 53.44% 是作者报告，仓库没有原始 rollout 包、checkpoint hash 或训练曲线可独立复核；但 50 tasks ×100 episodes、无 error/truncate/reject 的协议比单任务视频强得多（`DelinQu@0bc3de6:examples/robotwin/README.md:11`）。
- whole-scene 模型训练时已见 randomized episodes，94.24% 不是 clean→random OOD；真正 OOD 是 clean-only→random 的 53.44%。这反而说明目标场景多样性不可省。
- clean-only 仍读取由全量 parquet 计算的 stats，存在轻微分布信息泄漏；不影响 whole-scene 主结果，但本项目 stats 必须只用 train split（`DelinQu@0bc3de6:examples/robotwin/tools/compute_robotwin_action_stats.py:39`）。
- `val_ratio=0`，质量主要由闭环 benchmark 证明；我们的真实数据仍必须按 episode/task/scene 做 held-out split。
- eval 使用更高分辨率相机和官方任务时长，并修复 3 个 policy-path 状态 bug；这些是合同/基准修正，不是降低成功阈值（`DelinQu@0bc3de6:examples/robotwin/eval/run_task.sh:62`、`:104`）。
- serving 的 state-conditioned batch path 不安全，作者强制 batch 1；完整 chunk 被同步缓存执行，不具备 RTC/异步证据（`DelinQu@0bc3de6:packages/vla-eval-cosmos3/vla_eval_cosmos3/server.py:130`）。
- 输出反归一化后没有 clamp，但策略仍成功；因此“缺 clamp”不能单独解释我们的混乱运动。真机 joint/workspace/rate guard 仍是安全要求，不是模型质量修复。
- 新 loader/server 没有单元测试，三份 RoboTwin task 是整文件 fork；Docker preflight 和大规模 rollout 提供运行证据，但移植时应补最小 contract test，避免继承维护债务。

## 3. Stage 1 / Stage 2 决策

### Stage 1：不重做主线，只做两级诊断

**S1-D0，先做，无训练：** 对 iter 6500 的同一固定离线集比较 `(steps,guidance) = (30,1), (8,1), (4,1), (4,3)`。若仅 4-step/高 guidance 越界，问题属于采样/部署；只在 `30/1` 通过后做一次受保护 RoboLab rollout。

**S1-D1，可选小集：** 若所有组合都失败，只允许冻结 subset 比较 stateful frame-wise EEF 与官方 absolute-joint control；保留低运动窗口，fresh head，目标 10–12 window passes。fixed-set overfit 或反归一化 gate 失败即停止，不启动全量 DROID EEF v2。

理由：v1 已确认 SE(3) 方向正确，却在单步平移 P90 上放大 18.3×，且只有约 2 次 window 遍历（`docs/research/level2/training/droid_eef_stage1_validation_report.md:44`、`:86`）。RoboTwin 支持修正表示/预算，不支持反向 delta 或继续盲加 iteration。

### Stage 2：现在启动，joint 为唯一主线

按以下顺序冻结配置：

1. **Schema gate**：确认 RH20T `joint.npy` 的关节顺序、单位、限位、gripper、时间戳，以及 command/measured lag；FK 必须能重建 `tcp_base.npy`。若不是可执行关节目标，才降级 EEF。
2. **小集 baseline**：Edge base → fresh UR5 domain/head；输入当前 joints+gripper，预测 absolute future joints+gripper；quantile stats 只由 train split 计算。
3. **时间基线**：15 Hz action、chunk 32、video subsample 2，仍为 17 visual frames / 2.13 s；predict 32、execute 8。7.5 Hz/chunk 16 保持同物理 horizon，作为短 episode/显存回退。
4. **数据**：第一版保留全部合法 starts；若必须压窗，使用分层采样而不是删除静止段，监控 20–35% low-motion/hold windows。保留堆叠、整理和场景多样性。
5. **优化**：小集 bracket base LR `1e-5` 与 `5e-5`，action heads ×5；warmup 500、LR floor 0.1。GBS 64/128 由吞吐决定，但 full run 以 `sample_exposures / valid_windows` 和 optimizer updates 共同收口，不能只固定 iter。
6. **推理 gate**：先用 cloud `30-step/guidance=1` 作为质量 oracle，再降到 8/4-step；若 30 成功而 4 失败，只优化 sampler/部署。真机仍 execute 8 + safety guard，不照搬 RoboTwin 的 open-loop 64。
7. **消融顺序**：Edge base vs 官方 Edge-DROID shared backbone（都 fresh UR5 head）→ Vision SFT 初始化 → EEF target。最新上游 `mode=policy`、diffusion cache、FP8 只在 WAM 正确后做 4070 Ti 优化，不混入首轮质量基线。

Stage 2 的 Go 条件仍是：schema/FK → fixed-set overfit → held-out offline validity/hold → `30/1` RoboLab progress → 8/4-step → medium/full data。任何扩量前，camera identity、canvas pixels、prompt、fps、chunk/vsub、state layout 和 stats hash 必须成为可机器校验的单一合同。

## 最终路线

```text
现有 Stage1-6500 ── 30/1 sampler 复核 ──┬─ 通过：只修推理链，不重训
                                       └─ 失败：归档；最多做小集机制实验

Edge base / official Edge-DROID backbone
        → RH20T stateful absolute-joint WAM
        → RoboLab 30-step quality oracle
        → Edge 8/4-step + execute-8
        → UR5/UR12e 真机
```

这份结论更新 `stage1_stage2_reframing_after_droid_eef_v1.md` 的两点：Vision SFT 从主前置降为消融；Stage 2 主时序从 chunk 16 改为 `15 Hz / chunk 32 / vsub 2`，以保持已被 DROID 与 RoboTwin 共同支持的约 2.13 s horizon。
