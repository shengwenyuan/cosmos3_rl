# RH20T CFG4 数据清洗开发计划

状态：Draft v0.1
日期：2026-08-21
负责人：swy

## 1. 目标

在不修改原始数据集的前提下，为 RH20T CFG4 生成可复现、可审计的清洗索引。首版规则为“累计 3 mm SE(3) 增量”与“夹爪事件、安全时间间隔”取并集。

原始数据只读路径：

```text
/mnt/pfs/swy/dataset/RH20T/RH20T_cfg4
```

首版不复制或重编码原始视频，只生成 manifest、统计和拒绝列表。后续 LeRobot 转换器根据 manifest 从原始视频解码所需帧。

## 2. 已知基线

- CFG4 共 2,194 条机器人 episode、130 个任务、20 个场景。
- Scene 1–10 共 2,100 条；`rating >= 2 && calib_quality in [1,2,3]` 后约 1,805 条。
- 高质量 episode 抽样的原始相机对齐轨迹平均 474.5 帧、平均 61.5 秒。
- 原始有效频率中位约 7.0 Hz，不能使用 MP4 容器声明的 25 FPS 代替真实时间戳。
- Wrist 相机有效视频覆盖明显低于外部相机，清洗层必须记录逐相机可用性，不能假设八相机齐全。
- 3 mm SE(3) 单独保留约 66.8% 帧；加入夹爪和最大间隔保护后的最终保留率由正式 dry-run 冻结。

## 3. v1 清洗规则

### 3.1 Episode入口过滤

默认训练集合：

```text
cfg == 4
scene in [1, 10]
rating >= 2
calib_quality in [1, 2, 3]
非 *_human 目录
metadata、tcp_base、gripper 至少可读
```

不满足条件的条目写入 `rejected_episodes.jsonl`，保留明确 reason code，不静默丢弃。

### 3.2 主时间线

首版使用：

```text
transformed/tcp_base.npy["104122062295"]
```

原因：该外部相机覆盖率高，且现有探针已验证其时间戳、base-frame TCP 和夹爪数据可对齐。

处理要求：

1. 仅保留 `timestamp <= metadata.finish_time`。
2. 时间戳升序排序并去重。
3. 检查 quaternion 有限且模长有效。
4. 原始毫秒时间戳、相邻 `delta_t` 必须保留在 manifest 中。

若主 serial 缺失，v1 直接拒绝该 episode；fallback serial 留到 v2，避免混入未经验证的外参链。

### 3.3 SE(3) 等增量采样

定义相邻运动标量：

```text
d_i = ||p_i - p_(i-1)||_2 + 0.05 m/rad * geodesic_angle(R_i, R_(i-1))
```

累计 `d_i` 达到 0.003 m 时保留当前帧并清零 residual。始终保留首帧和末帧。

注意：这里是累计弧长，不是“当前帧相对上一个保留帧的净位移”，否则往返运动可能被错误抵消。

### 3.4 强制保留事件

以下索引与 SE(3) 索引取并集：

- 夹爪命令显著变化：沿用数据量纲下 `abs(delta_gripper_command) >= 0.5` 的首版阈值。
- Episode 首帧、末帧。
- 最大时间间隔保护：相邻保留帧的原始时间差不得超过 0.5 秒；超出时从原时间线补点。

### 3.5 多相机对齐

对每个保留时间戳，在相机各自 `timestamps.npy` 中找最近帧：

- 默认最大误差 100 ms。
- 保存 `source_frame_index`、`source_timestamp_ms` 和 `alignment_error_ms`。
- 某一视角缺失不直接拒绝 episode；写入 camera availability mask。
- 训练数据变体可以在后续转换阶段选择“外部双视角全集”或“外部双视角+wrist子集”。

## 4. 计划脚本

建议放入 Cosmos 仓库外的独立数据工具目录，避免数据预处理逻辑和训练核心耦合：

```text
/root/workspace/cosmos3_rl/tools/rh20t/
├── pyproject.toml
├── rh20t_tools/
│   ├── probe.py
│   ├── schema.py
│   ├── alignment.py
│   ├── se3.py
│   └── cleaning.py
├── scripts/
│   ├── build_clean_manifest.py
│   └── validate_clean_manifest.py
└── tests/
```

建议 CLI：

```bash
python -m rh20t_tools.scripts.build_clean_manifest \
  --input-root /mnt/pfs/swy/dataset/RH20T/RH20T_cfg4 \
  --output-root /mnt/pfs/swy/dataset/RH20T/derived/cfg4_clean_3mm_se3_v1 \
  --scene-min 1 --scene-max 10 \
  --min-rating 2 --calib-quality 1 2 3 \
  --se3-step-m 0.003 --rotation-weight-m-per-rad 0.05 \
  --max-gap-s 0.5 \
  --dry-run
```

正式运行前必须先完成 `--dry-run`，输出预测条数和空间占用，不写 episode manifest。

## 5. 输出契约

```text
cfg4_clean_3mm_se3_v1/
├── cleaning_config.json
├── dataset_summary.json
├── episodes.jsonl
├── rejected_episodes.jsonl
├── task_summary.parquet
└── manifests/
    └── <episode_name>.parquet
```

每帧至少包含：

```text
episode_id, cleaned_index
source_timestamp_ms, source_delta_t_s
tcp_xyz, tcp_quat_wxyz
gripper_command
keep_reason bitmask
camera.<serial>.source_frame_index
camera.<serial>.source_timestamp_ms
camera.<serial>.alignment_error_ms
camera.<serial>.available
```

`keep_reason` 至少区分 `FIRST/LAST/SE3/GRIPPER/MAX_GAP`，便于后续统计每类规则贡献。

## 6. 验证与验收

单元测试：

- Quaternion 正负号不影响旋转角。
- 累计弧长往返运动不被抵消。
- 首尾帧必保留。
- 事件并集正确。
- 最大时间间隔补点正确。
- 相机最近邻对齐不跨 episode。

数据级验收：

- 输入原始目录 mtime 和文件哈希抽样不发生变化。
- 所有输出时间戳严格递增。
- 所有保留索引均可映射回原始条目。
- 总体帧保留率由正式 dry-run 冻结；异常偏离必须人工审计。
- 16-step window 数量、每任务episode数、相机覆盖率均写入 summary。
- 随机抽查至少 30 个 episode，并渲染首/中/尾片段叠加 EEF、夹爪和 keep reason。

## 7. 风险与待决策

- 等空间采样产生可变真实时间间隔，不能再把每个动作步严格解释成固定 `1/fps` 秒。
- 任务文本映射不在当前 episode metadata 中，需要在 LeRobot 转换前建立经过人工审核的 task catalog。
- Wrist 缺失率高，首版训练采用哪些视角由转换计划中的数据变体实验决定。

## 8. 里程碑

1. M1：manifest schema、SE(3)与事件单测完成。
2. M2：200条episode dry-run与当前探针结果一致。
3. M3：1,805条候选全量生成manifest和summary。
4. M4：30条可视化审计通过，冻结 `cleaning_config.json` 为 v1。
