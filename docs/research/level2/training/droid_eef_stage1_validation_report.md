# DROID EEF Stage 1 验证报告

日期：2026-08-24
结论：**训练与动作链路完成；策略行为 gate 失败；不得记为 Stage 1 成功。**

## 1. 实际产物

- 数据：Fx4-filtered DROID success，28,358 episodes，1,653,955 个 chunk-32 training windows。
- 人工语义抽检：38/38 sampled rows 判定 correct；该结果只支持抽样 gate，不代表全量逐条审核。
- 训练：Cosmos3-Edge-MT，64 GPUs，micro-batch 1/rank，gradient accumulation 8，global batch 512，bf16，FusedAdam，LR `1e-5`。
- 时序：固定 15 Hz，SE(3) smoothing 后回投原时间戳，train/predict chunk 32，execute 8，UniPC denoise 4。
- Checkpoints：iter 3500、5000、6500。6500 已归档，但本报告的完整离线分布与 RoboLab 结论基于 iter 5000；不得把 5000 结果直接外推为 6500 结果。
- ModelScope：`sss225/cosmos3-edge-policy-droid-eef-iter5000`、`sss225/cosmos3-edge-policy-droid-eef-iter6500`。

## 2. 冻结的 as-built action contract

```text
observation image: T0 three-camera canvas
numeric model state: none (use_state=false, state_rows=0)
decoder anchor: current panda_link8 pose
target delta[k]: inverse(T0) @ Tk, k=1..32
target layout: xyz(3) + rotation-6D(6) + gripper_open_fraction(1)
wire output: absolute panda_link8 pose, quat_xyzw + gripper_close_fraction
decode: T_pred[k] = T_current @ delta[k]
```

`panda_link8` 表示 Franka flange frame。当前数值 EEF pose 只用于服务端反解锚点，并未作为模型条件输入。

正向重建误差约 `1e-16 m`。若误用反向 delta，horizon 8/32 的中位重建误差约为 8.7/24.1 cm。因此 inverse、quaternion 与 delta 方向不是本轮失败的解释，禁止据此反向重训。

## 3. 数据目标分布抽样

128 个 windows、49 个 data shards 的目标分布：

| Horizon | 平移 P50 / P90 | 旋转 P50 / P90 |
|---|---:|---:|
| 1 | 0.70 / 1.68 cm | 0.80° / 2.48° |
| 8 | 5.04 / 12.09 cm | 6.00° / 16.55° |
| 16 | 9.52 / 19.59 cm | 10.99° / 27.88° |
| 32 | 15.40 / 30.73 cm | 18.16° / 44.51° |

本轮使用 anchored cumulative target，却让所有 horizon 共用一套 `quantile_rot` stats。motion-biased window starts、静止窗口不足和局部坐标平移偏置同时存在。model-space 零向量反归一化后约为 5.21 cm + 4.13°，不是 hold。

## 4. iter 5000 离线预测分布 gate

配置：128 个固定窗口、128 episodes、49 shards、seed 0、与服务端一致的 4-step UniPC。Gate 限制为预测/目标 P90、P99 比值不超过 2，以及落在 train q01–q99 外的总比例不超过 10%。结果：**fail**。

| Horizon | 平移 P90：GT → Pred | 比值 | 旋转 P90：GT → Pred | 比值 |
|---|---:|---:|---:|---:|
| 1 | 1.51 → 27.70 cm | 18.33 | 2.24° → 9.75° | 4.36 |
| 8 | 11.96 → 59.48 cm | 4.97 | 17.39° → 30.95° | 1.78 |
| 16 | 19.86 → 103.19 cm | 5.20 | 29.68° → 62.28° | 2.10 |
| 32 | 29.36 → 162.66 cm | 5.54 | 44.15° → 110.13° | 2.49 |

相邻预测目标：

- translation P90：1.42 → 7.84 cm，5.53×；P99：2.33 → 25.26 cm，10.86×。
- rotation P90：2.12° → 5.46°，2.57×；P99：3.33° → 9.70°，2.92×。
- 29.56% 的 raw action channel values 超出训练 q01–q99；平移三轴平均约 49.1%。
- `apply_forward_clamp=false`；而现有 `apply_forward_clamp` 只约束训练 forward normalization，不构成推理输出保护。

完整产物：

```text
/mnt/pfs/swy/cosmos3/eval/iter_000005000_offline_distribution_gate_128x49_v2/report.json
/mnt/pfs/swy/cosmos3/eval/iter_000005000_offline_distribution_gate_128x49_v2/samples.jsonl
```

## 5. RoboLab gate

Isaac Sim 6.0.1、Panda absolute-IK client、manifest-matched multi-view 输入下运行 `BananaInBowlTaskAbsIK`：

- iter 3500：轨迹无任务导向，未形成稳定 approach/grasp/place progress。
- iter 5000：轨迹仍为大幅无序运动；panda_link8 重命名及 server-client action contract 修复后结果未改善。
- server-client handshake、canvas、absolute pose wire、quaternion order、gripper semantics 与 IK 工具链均可运行；没有证据支持把失败归因于 TCP/flange 混淆或 delta 反向。

视频：

```text
/home/lenovo/swy/RoboLab_div/output/cosmos3_eef_iter3500_banana_isaac601_xyzw/
/home/lenovo/swy/RoboLab_div/output/cosmos3_panda_link8_iter5000_banana_isaac601_multiview/
```

RoboLab G6 的正向 stage progress 与成功线均未达到，结果记为 **fail**，不是 conditional pass。

## 6. 结论与 Stage 2 边界

已通过：数据与 checkpoint 可加载、SE(3) 方向与重建、server-client 编解码、Isaac/RoboLab 执行工具链。

未通过：离线预测尺度、相邻目标连续性、RoboLab task progress。

最可能的共同原因是 anchored cumulative targets、跨 horizon 混合归一化、motion-biased starts、hold 样本不足与无推理范围保护共同放大尚未学稳的输出；不是简单增加训练步数即可判定解决的问题。

Stage 2 可以开始，但仅限 **diagnostic pilot**：

1. 先对 iter 6500 复跑同一离线 gate；若仍越界，不再直接做无保护 RoboLab rollout。
2. 建立新的 Stage 2 action/stats revision，禁止静默复用 v1 stats。优先保留 anchored 方向，使用按 horizon、以 SE(3) hold 为中心的 robust normalization；若改为 frame-wise delta，必须登记为独立 representation ablation。
3. 任何仿真执行增加独立的 post-decode workspace、相邻平移/旋转和 IK validity guard；该 guard 不替代训练修复。
4. 在完全相同的 RH20T/UR5 100-episode subset、stats、步数和 seed 下比较：
   - Edge MT → Stage 2；
   - iter 6500 shared backbone → Stage 2，并重新初始化与新 representation/stats 不兼容的 action modules。
5. 只跑 200–500 iteration small-set overfit。必须通过固定样本反归一化分布 gate，并在 RoboLab 出现方向正确 progress，才扩大数据或 GPU。
6. 若 iter 6500 初始化不优于 Edge MT，Stage 1 只保留为负结果，不再作为后续主链依赖。

此策略允许 Stage 2 继续产生研究信息，同时不把 Stage 1 的失败错误解释为已验证的迁移基础。
