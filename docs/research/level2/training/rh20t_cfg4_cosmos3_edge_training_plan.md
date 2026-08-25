# RH20T CFG4 Cosmos3-Edge 后训练与真机部署计划

状态：Draft v0.1
日期：2026-08-21
训练资源：8 GPU 云端开发机
目标部署：约 RTX 4070 Ti 级别单卡主机

修订提示（2026-08-25）：本文件保留 RH20T 数据、部署和安全细节；初始化、GBS、action 主线和 Stage 2 定位已由 `stage1_stage2_reframing_after_droid_eef_v1.md` 重构。冲突处以后者为准。

## 1. 目标

基于清洗并转换完成的 RH20T CFG4 LeRobot 数据，对 Cosmos3-Edge 进行 WAM/action-policy 后训练，输出可在单卡真机端运行的 EEF 相对动作策略。

首版优先验证：

1. 数据和动作坐标链正确。
2. 16步chunk在 RH20T 空间重采样轨迹上可学习。
3. 4-step UniPC 推理质量可接受。
4. 16步预测、执行8步后异步预取下一chunk，在4070 Ti级设备上无明显停顿和边界跳变。

首版不承诺完整RTC。先建立同步基线和异步prefetch基线，再决定是否修改denoising实现RTC式重叠约束。

## 2. 已冻结的首版建议

```text
model                    = Cosmos3-Edge
mode                     = wam
action space             = frame_wise_relative
rotation                 = rot6d
pose convention          = backward_framewise
raw action dim           = 10
action normalization     = quantile_rot
chunk_length             = 16
vision frames            = 17
nominal conditioning fps = 6
format_prompt_as_json    = true
inference sampler        = UniPC
inference denoise steps  = 4
guidance                 = 3.0
shift                    = 5.0
predicted horizon        = 16
executed horizon         = 8
```

说明：denoise step是推理参数，不是SFT训练步数。训练仍使用flow-matching的噪声时间采样。

## 3. 为什么选择chunk 16

官方 Cosmos3 DROID policy 使用15 Hz、chunk 32，对应约2.13秒。RH20T 清洗后的有效频率和各 chunk 真实跨度由正式 dry-run 重新统计。

因此：

- 32步可能在清洗数据上形成过长 open-loop。
- chunk 16 暂作首选，正式跨度统计完成后冻结。
- 当前 Edge-LIBERO 模型配置已包含 `encode_exact_durations=[17,61,73]`，16步对应17帧，无需新增tokenizer duration。
- 执行 8 步的推理预算以正式跨度统计和 4070 Ti 实测为准。

## 4. 计划代码改动

当前代码仓：

```text
/root/workspace/cosmos3_rl
```

建议新增/修改：

```text
cosmos_framework/data/generator/action/datasets/rh20t_lerobot_dataset.py
cosmos_framework/data/generator/action/datasets/action_sft_dataset.py
cosmos_framework/data/generator/action/utils/domain_utils.py
cosmos_framework/data/generator/action/normalizer_stats/rh20t_cfg4_*.json
cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_rh20t_cfg4_edge.py
examples/toml/sft_config/action_policy_rh20t_cfg4_edge_8gpu.toml
examples/launch_sft_action_policy_rh20t_cfg4_edge_8gpu.sh
docs/action_policy_rh20t_cfg4_edge_posttrain.md
```

真机服务端优先扩展：

```text
cosmos_framework/scripts/action_policy_server_robolab.py
```

该服务器已经支持 `action_space=midtrain` 的 EEF/rot6d路径，但需要增加 RH20T/UR5 观测适配、动作统计、相机拼接和客户端协议测试。

## 5. 数据与切分

首轮至少保留两个数据配置：

- A：`ext2`，两路外部视角，episode覆盖最大。
- B：`ext2_wrist`，两路外部视角+wrist，数据较少但具备手眼信息。

切分原则：

- Train/val/test 不按随机frame切分。
- 至少按 `task + scene` 或 `task + scene + user` 分组，避免同场景同任务重复轨迹泄漏。
- 固定split seed并落盘split manifest。
- Action stats只用train split计算。
- 首版目标不是宣称任务泛化，测试报告分别给出 seen-task/seen-scene 和 held-out-scene 指标。

## 6. 训练配置起点

从当前 `action_policy_libero_all_edge` 复制并最小改动：

```text
model precision                  = bfloat16
activation checkpointing        = selective
max_num_tokens_after_packing     = 74000
optimizer                       = FusedAdamW
base lr                         = 5e-5
action head lr multiplier       = 5.0
weight decay                    = 0.05
grad clip                       = 1.0
cfg dropout                     = 0.1
chunk_length                    = 16
fps                             = 6
format_prompt_as_json           = true
iterable episode shuffle        = true
```

首轮不直接照搬LIBERO的全部batch设置。先用真实RH20T样本测量单step显存、token数和吞吐后，再决定：

```text
max_samples_per_batch in {32, 64, 128}
grad_accum_iter in {1, 2, 4}
```

目标是在8 GPU上保持稳定的有效global batch，同时避免少数长episode/多视角样本触发74k token cap或OOM。

## 7. 分阶段训练

### Phase 0：数据和shape smoke

- 1、10、100条episode逐级加载。
- 检查17帧vision、16×10 action、prompt、domain id、normalizer和sequence plan。
- 8 rank × 多worker连续运行至少200 batch。
- 记录GPU显存、CPU RSS、PFS吞吐和video decoder cache。

通过条件：无NaN、无旋转尖峰、无episode跨界window、无持续文件句柄增长。

### Phase 1：小数据过拟合

- 选取10个任务，每任务5–10条高质量episode。
- 训练200–500 iterations。
- 验证action loss明显下降，固定样本预测可接近GT。
- 解码少量未来vision检查时序和动作方向，不以视觉质量作为唯一指标。

通过条件：绝对/相对pose回放方向正确，夹爪语义正确，模型能够过拟合。

### Phase 2：全量baseline

- 使用冻结的train split和stats。
- 首轮 `max_iter=5000`，`save_iter=500`。
- 只保留最大iter的长期checkpoint，smoke checkpoint按既定规则清理。
- 固定100–200条validation window，每500 iter运行离线action指标。

主要监控：

- action flow loss及各动作维分项误差。
- translation/rotation/gripper反归一化误差。
- 预测chunk首部、尾部和边界速度/加速度。
- 每任务和每视角配置的样本覆盖。
- 梯度范数、NaN、OOM、dataloader等待占比。

### Phase 3：必要消融

按优先级执行：

1. `ext2` vs `ext2_wrist`。
2. nominal fps 6、真实时间窗anchor方案。
3. chunk 12 vs 16。
4. denoise steps 4/6/8，仅在同一checkpoint上做推理消融。

不要同时改变数据清洗、相机、chunk和denoise steps，否则无法定位收益来源。

## 8. 离线评测

离线指标全部在反归一化动作上计算：

- translation MAE/RMSE，单位mm。
- rotation geodesic error，单位degree。
- gripper accuracy、开合事件F1和事件延迟。
- chunk积分后的末端pose drift。
- action finite difference、速度和加速度尖峰。
- 相邻chunk overlap区域的首动作差、平均差和最大差。
- 接触前后窗口的动作误差单独统计。

至少保留以下固定对照：

```text
teacher forcing single-window
synchronous execute-16
synchronous replan-8
asynchronous predict-16/execute-8
```

## 9. 4070 Ti级真机部署验证

### 9.1 模型导出

- 从DCP导出单机 safetensors/HF格式。
- 推理默认 `decode_video=false`，真机只返回action，避免无必要的VAE decode延迟。
- batch size=1，固定分辨率和相机拼接布局。
- 先测bf16；显存不够再评估FP8/INT8或低分辨率，量化前保留bf16质量基线。

### 9.2 性能目标

执行8步后的剩余时间预算：中位1.18秒，P5约0.84秒。因此首版目标：

```text
4-step UniPC端到端 latency P50 <= 500 ms
4-step UniPC端到端 latency P95 <= 700 ms
稳定显存 <= 11 GB（兼容12 GB级显卡为目标）
连续运行30分钟无显存增长
```

若P95超过0.84秒：

1. 关闭vision decode和所有debug dump。
2. 降低输入分辨率/视角数量。
3. 使用CUDA graph/compile或合适量化。
4. 将预取点从第8步提前到第6步。
5. 最后才考虑减少denoise steps到2；2-step必须单独验证质量。

### 9.3 执行状态机

首版异步prefetch：

1. 获取当前观测，生成16步chunk A。
2. 开始逐步执行A。
3. A执行到第8步时，用最新观测异步生成chunk B。
4. B生成期间继续执行A剩余动作。
5. B到达后进行安全检查和2–4步边界融合，再切换到B。
6. B未按时到达时不得无限重复最后动作；进入减速保持或安全停止。

必须记录：观测时间、请求开始/结束、每个动作实际下发时间、chunk id、动作来自哪个chunk、融合权重和丢弃原因。

### 9.4 连贯性指标

- Chunk边界的position/rotation/gripper跳变。
- 速度、加速度、jerk峰值及P95/P99。
- 新旧chunk重叠8步的动作偏差。
- 推理延迟超预算率。
- stale observation年龄。
- 安全停止次数和任务成功率。

首版只做插值/限速属于异步prefetch，不称为RTC。若边界不连续仍明显，再实现RTC式“新chunk denoising受已承诺旧动作约束”，参考现有flow action tensor mask/conditioning路径设计。

## 10. 安全要求

- 真机第一阶段限制工作空间、EEF速度、旋转速度、夹爪力和单步位移。
- 软件急停、硬件急停和watchdog均需有效。
- 推理超时、NaN、越界、相机丢帧、网络断连立即进入安全状态。
- 首次运行从空载、低速、无障碍工作区开始。
- 所有模型输出经过限幅、碰撞/工作空间检查后才发送机器人。

## 11. 交付物

- RH20T dataset adapter及测试。
- Edge 8-GPU训练experiment、TOML和launcher。
- 固定split和normalizer stats。
- 5k baseline checkpoint及HF导出。
- 离线评测报告。
- 4070 Ti级设备延迟/显存benchmark。
- 同步与异步execute-8连贯性对比。
- 真机部署runbook和安全checklist。

## 12. Go/No-Go门槛

进入真机前必须同时满足：

- 数据相对动作积分回放验证通过。
- 小数据过拟合通过。
- 离线action无系统性坐标/夹爪方向错误。
- 4-step推理P95落入剩余执行预算。
- 异步chunk边界指标优于同步execute-16，且没有新的大幅尖峰。
- Watchdog、限幅和急停演练通过。
