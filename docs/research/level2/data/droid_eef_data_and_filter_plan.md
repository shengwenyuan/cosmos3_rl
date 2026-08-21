# DROID EEF 数据获取与二次筛选计划

日期：2026-08-21

## 1. 数据源与固定版本

第一轮只下载 NVIDIA 发布的 LeRobot v3 `success` split，不下载 failure split：

- Hugging Face：`nvidia/Cosmos3-DROID`
- revision：`5c11a20accb11497270a5247a7f1e66ad04c956c`
- 本地路径：`/mnt/pfs/swy/dataset/DROID/Cosmos3-DROID`
- `success`：3,133 个文件，628,036,574,065 bytes
- 57,639 episodes，18,691,281 frames，15 FPS，三路 640×360 AV1 视频

Nano Policy 对齐过滤文件：

- 仓库：`KarlP/droid`
- 文件：`keep_ranges_1_0_1.json`
- revision：`bcb840c3b496533e0adf548a54b51f2f00057837`
- SHA256：`efee695f8228fe19d4f01767b95ab612084c499c8a59421c5f3d2c7ad5f3625e`
- 本地路径：`/mnt/pfs/swy/dataset/DROID/annotations/keep_ranges_1_0_1.json`

该文件针对 DROID 1.0.1，保留不少于 16 帧、且没有被连续 8 个 idle action 打断的运动区间。代码必须把 filter 路径改为显式配置，不能继续依赖源码里的内部绝对路径。

## 2. 下载通道

同一 77,129,770-byte parquet 的单连接探针：Hugging Face 约 11.4 MiB/s；ModelScope 镜像约 0.9 MiB/s。当前机器使用 Hugging Face，ModelScope 仅作故障切换。

下载要求：固定 revision、允许断点续传、日志和 PID 落在 `/mnt/pfs/swy/dataset/DROID/download_logs/`。原始数据只读使用，不做就地转码或改名。

## 3. 官方 filter 后规模

基于官方 `success/meta` 与当前加载器的 `[start, end)` 语义：

| 指标 | 数值 |
|---|---:|
| filter 命中 episodes | 56,715 |
| 保留 ranges | 75,558 |
| 保留 frames | 14,856,095 |
| 可用 16-step windows | 13,735,276（原 success 的 77.05%） |
| 可用 32-step windows | 12,606,835（原 success 的 74.57%） |

## 4. 二次筛选建议

二次筛选分为确定性四层，每层均输出 episode/range manifest 和原因码。

### F1：简单桌面任务语义

- 白名单：单阶段 pick/place/relocate、push/pull/slide、简单 open/close，以及堆叠、归类、整理/摆放。
- 三条语言标注至少两条命中同一白名单族。
- 堆叠、归类、整理作为独立训练命中族；允许这类任务包含必要的多物体和分步操作，后续单独做任务均衡。
- 任一标注命中以下内容即排除：液体/倾倒、擦拭清洁、布料与袋子、工具操作、电器、插接/拧紧，以及不属于堆叠/整理族的复杂多阶段任务。
- 时长先保留 48–450 frames（3.2–30 秒）；边界后续由分布和人工复核调整。

纳入堆叠/整理后，粗关键词探针得到 31,528 episodes、5,221,441 个 16-step windows，约为官方 filter 后窗口的 38.01%；其中保守的两票命中规则识别出 528 个堆叠/整理 episodes。这只是候选集规模，不能直接作为最终训练 manifest。

### F2：EEF 与视频质量

- 时间戳单调，三路视频存在且可解码；排除黑帧、长冻结和严重不同步。
- EEF/夹爪均为 finite；处理 Euler wrap 后再计算相邻 SE(3) 增量。
- 按训练集分布的 P99/P99.9 检查平移、旋转、速度和工作空间异常，不先拍脑袋写固定阈值。
- 每个保留 range 至少形成一个完整 16-step window。
- DROID 首轮不做 3 mm 等 EEF 增量重采样；保持 15 Hz，避免与官方 filter 和 Nano 数据节奏同时发生两项变化。

### F3：资源与分布控制

- 按标准化任务族、采集机构/场景和 episode 分组采样。
- 每 episode 最多取 64 个起始窗口；长 episode 均匀取样，避免连续邻近窗口淹没 batch。
- 对高频任务设上限，对 push/pull 等少数族设最低配额。
- 生成 `easy_v1_train.jsonl`、`easy_v1_val.jsonl` 和汇总表；split 必须在采样窗口前按 episode/场景分组完成。

F1 约 522 万个 16-step windows；若每 episode 封顶 64，理论上限约 202 万，可显著减少一轮训练的重复样本与 I/O。

### F4：人工复核

- 随机抽查至少 200 episodes，并对每个保留/排除任务族各抽样。
- 保存首中尾帧、语言、EEF 轨迹和排除原因。
- 人工误判率超过 5% 时修订规则并生成新版本，禁止覆盖旧 manifest。

## 5. 产物

建议落盘：

```text
/mnt/pfs/swy/dataset/DROID/
├── Cosmos3-DROID/                 # NVIDIA 原始 LeRobot v3，只读
├── annotations/
│   └── keep_ranges_1_0_1.json
├── manifests/
│   ├── official_keep_v1.jsonl
│   ├── easy_v1_train.jsonl
│   ├── easy_v1_val.jsonl
│   └── easy_v1_summary.json
├── stats/
│   └── easy_v1_eef_quantile_rot.json
└── download_logs/
```

首轮优先产出 `easy_v1`；同时保留 `official_keep_v1`，方便在相同代码上做数据规模消融。
