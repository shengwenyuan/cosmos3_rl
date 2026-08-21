# RH20T CFG4 转 LeRobot v3 开发计划

状态：Draft v0.1
日期：2026-08-21
前置依赖：`RH20T_CFG4_DATA_CLEANING_PLAN.md` 的 v1 manifest 已冻结

## 1. 目标

把 RH20T CFG4 清洗索引转换为本地 LeRobot v3 数据集，并为 Cosmos3-Edge WAM/action-policy 后训练提供稳定的数据契约。

转换器不得修改：

```text
/mnt/pfs/swy/dataset/RH20T/RH20T_cfg4
```

目标输出建议：

```text
/mnt/pfs/swy/dataset/RH20T/lerobot/rh20t_cfg4_3mm_se3_v1
```

输出先写入独立 staging 目录，校验通过后原子重命名；不在最终目录中边转换边暴露半成品。

## 2. 数据设计决策

### 2.1 Episode与任务

- 一个 RH20T 机器人目录映射为一个 LeRobot episode。
- `task_xxxx` 不能直接作为语言指令；必须建立 `task_catalog.csv`：

```text
task_id,instruction_en,instruction_zh,review_status,source
```

- 训练前只接受 `review_status=approved` 的英文指令。
- 同一 task 可保留多个经过审核的 paraphrase，但 train/val 必须按 task/scene 分组，防止相邻重复episode泄漏。

### 2.2 时间语义

清洗后是可变时间间隔的空间序列。LeRobot/Cosmos 当前接口仍需要单一 `fps`，首版采用：

```text
nominal_fps = 6
timestamp = cleaned_index / 6.0
```

同时额外保存：

```text
source.timestamp_s
source.delta_t_s
source.timestamp_ms
```

名义时间只用于 LeRobot 索引和 Cosmos prompt；真机执行与离线分析必须优先使用 source timestamp。禁止丢弃原时间戳后再把数据解释为严格6 Hz轨迹。

### 2.3 Action表示

首版采用与当前 Cosmos midtrain/LIBERO 路径兼容的 10D frame-wise relative action：

```text
action[0:3]  = delta xyz
action[3:9]  = relative rotation 6D
action[9]    = gripper target
```

位姿差分统一调用 Cosmos 现有实现：

```text
cosmos_framework/data/generator/action/utils/pose_utils.py
pose_convention = "backward_framewise"
rotation_format = "rot6d"
```

不得在转换脚本里重新实现另一套旋转/坐标约定。转换前后都保存少量 absolute pose，用相对动作积分回放验证误差。

建议辅助状态：

```text
observation.state.eef_abs = xyz + rot6d
observation.state.gripper = 1D
observation.state.wrench  = force_xyz + torque_xyz
```

力数据先作为观察/审计字段保存，首版模型输入是否使用另行做消融，不直接拼入 action。

### 2.4 图像变体

由于 wrist 覆盖不足，首版同时产出两个数据变体或两个 episode filter manifest：

1. `ext2`：两路高覆盖外部视角，最大化episode数量。
2. `ext2_wrist`：两路外部视角+wrist，仅保留有效 wrist episode。

候选外部相机：

```text
104122062295
104122062823
```

Wrist：

```text
045322071843
```

正式冻结前随机审计100帧，确认外部双视角具有互补覆盖。相机组合不得仅依据 serial 名称决定。

LeRobot key建议：

```text
observation.images.exterior_1
observation.images.exterior_2
observation.images.wrist
```

缺失 wrist 不用黑帧伪造；使用独立 variant/filter，避免模型把黑帧学成合法观测。

### 2.5 分辨率和编码

- 原始有效视频分辨率为640×360。
- LeRobot落盘保留640×360 H.264，避免提前损失信息。
- Cosmos dataloader阶段再执行训练分辨率缩放。
- 视频时间基必须与名义6 Hz一致；source timestamp写入parquet辅助字段。
- 转码必须记录 ffmpeg 命令、版本、codec、pixel format、GOP和帧数。

## 3. 目标LeRobot结构

```text
rh20t_cfg4_3mm_se3_v1/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   ├── episodes/
│   │   └── chunk-*/file-*.parquet
│   └── stats.json
├── data/
│   └── chunk-*/file-*.parquet
├── videos/
│   └── chunk-*/<video_key>/file-*.mp4
├── conversion_config.json
├── conversion_summary.json
└── rejected_episodes.jsonl
```

核心frame字段：

```text
episode_index, frame_index, index, timestamp, task_index
action[10]
observation.state.eef_abs[9]
observation.state.gripper[1]
observation.state.wrench[6]
source.timestamp_s
source.delta_t_s
source.keep_reason
```

`info.json` 中 fps、features、video_path、data_path 必须由 LeRobot API 生成或按当前安装版本验证，不手写未经校验的旧版schema。

## 4. 计划脚本

建议扩展同一工具包：

```text
/root/workspace/cosmos3_rl/tools/rh20t/
├── rh20t_tools/
│   ├── lerobot_schema.py
│   ├── action_conversion.py
│   ├── video_writer.py
│   ├── task_catalog.py
│   └── conversion.py
├── scripts/
│   ├── convert_cfg4_to_lerobot.py
│   ├── compute_action_stats.py
│   └── validate_lerobot_dataset.py
└── tests/
```

建议 CLI：

```bash
python -m rh20t_tools.scripts.convert_cfg4_to_lerobot \
  --source-root /mnt/pfs/swy/dataset/RH20T/RH20T_cfg4 \
  --manifest-root /mnt/pfs/swy/dataset/RH20T/derived/cfg4_clean_3mm_se3_v1 \
  --task-catalog /root/workspace/cosmos3_rl/docs/research/level2/data/rh20t_cfg4_task_catalog.csv \
  --output-root /mnt/pfs/swy/dataset/RH20T/lerobot/rh20t_cfg4_3mm_se3_v1.staging \
  --variant ext2_wrist \
  --nominal-fps 6 \
  --action-space frame_wise_relative \
  --rotation-space 6d \
  --pose-convention backward_framewise \
  --resume
```

功能要求：

- `--episode-limit`：用于1/10/100条逐级smoke。
- `--workers`：限制并发解码，默认保守值，避免PFS随机读打满。
- `--resume`：episode级幂等恢复。
- `--overwrite` 默认关闭。
- 每个episode完成后写校验记录，失败时不污染其他episode。

## 5. Cosmos数据集适配

不强行伪装成 DROID 目录命名。建议新增：

```text
cosmos_framework/data/generator/action/datasets/rh20t_lerobot_dataset.py
```

并在：

```text
cosmos_framework/data/generator/action/datasets/action_sft_dataset.py
```

新增：

```text
get_action_rh20t_cfg4_sft_dataset(...)
```

优先继承/复用：

```text
BaseActionLeRobotDataset
ActionSFTDataset
ActionIterableShuffleDataset
ActionTransformPipeline
```

同时更新：

```text
cosmos_framework/data/generator/action/utils/domain_utils.py
```

注册 RH20T/UR5 embodiment与 raw action width=10。若现有 `midtrain` domain 已能完整表达，则复用其 action spec，但仍保留独立 dataset/domain name，避免统计和提示词混淆。

## 6. 归一化统计

训练前对训练split单独计算：

- raw action min/max、mean/std。
- q01/q99 quantile。
- translation、rotation6D、gripper分维统计。
- task/scene级离群分布。

首选 `quantile_rot`，沿用当前 Edge-LIBERO 的旋转处理；同时保留 raw stats 供真机反归一化。

禁止把 validation episode 纳入统计。统计文件包含 dataset manifest hash 和 split hash，数据变化后旧统计必须失效。

## 7. 验证与验收

### 7.1 结构验证

- LeRobot metadata可由当前安装版本成功打开。
- Episode、frame、video数量与 manifest 一致。
- 所有episode `frame_index` 从0连续增长。
- 任务文本非空且来自 approved catalog。
- 视频帧数与parquet帧数一致。

### 7.2 数值验证

- Relative action积分回 absolute pose，随机100个window位置误差 < 1 mm、旋转误差 < 0.2°，或给出浮点/表示允许的明确上限。
- Quaternion符号翻转不产生动作尖峰。
- Gripper方向和真机开合语义一致。
- 反归一化后与原始action逐元素一致。
- `source.delta_t_s` 分布与清洗报告一致。

### 7.3 Cosmos smoke

使用新 dataset factory 读取：

- 单样本：shape、dtype、prompt、sequence_plan正确。
- 一个16-step window产生17帧vision序列和16步10D action。
- 100 batch 连续读取无文件句柄持续增长、无随机OOM。
- 8 rank × 多worker时episode sharding互斥且可复现。

## 8. 分阶段实施

1. M1：确认LeRobot版本和schema，转换1条episode。
2. M2：完成task catalog首批审核，转换10条并做action积分回放。
3. M3：转换100条，跑Cosmos dataloader smoke和视频吞吐测试。
4. M4：生成 `ext2` 与 `ext2_wrist` 全量索引/数据变体。
5. M5：冻结stats、split和manifest hash，交给训练计划。

## 9. 未决策项

- `ext2` 与 `ext2_wrist` 哪个作为主训练集，需要用100条smoke比较有效window数量、视觉覆盖和显存。
- 是否把wrench作为模型输入，首版默认仅保存不输入。
- 6 Hz是名义值；若模型对时间提示敏感，需要增加真实 `delta_t` conditioning或改为“空间索引选anchor、window内固定时间重采样”的对照数据集。
