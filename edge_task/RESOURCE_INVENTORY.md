# Cosmos3 Edge Policy LIBERO 资源盘点

盘点时间：2026-08-15 UTC。训练运行时只使用 `/mnt/cfs/data/swy/cosmos3` 和
`/mnt/cfs/data/swy/libero`；BOS 仅为迁移/归档来源，不直接作为训练输入。

## 1. 仓库与能力

| 项目 | 结果 | 判断 |
| --- | --- | --- |
| 工作目录 | `/root/workspace/cosmos-framework` | 正确 |
| `origin` | `https://github.com/NVIDIA/cosmos-framework.git` | NVIDIA 官方源 |
| NVIDIA 基线 | `origin/main` / `c3493f3b5f9a40d381c1d66e548ef249b8f4d64f` | 官方基线 commit |
| 当前分支 / HEAD | `edge-libero` / `8e3b66b1a987e439d5bf81731fb548913919c97c` | 在官方基线上增加 Edge-LIBERO recipes |
| 包版本 | `1.2.2` | `pyproject.toml:8-15` |
| Edge base 配置 | 已有 | `EDGE_MODEL_CONFIG`、Edge inference YAML、DCP converter 均存在 |
| LIBERO train data | 已有 loader | LeRobot v3 parquet + 双路视频 loader、normalizer 已有 |
| LIBERO inference/eval | 已有 | HTTP policy server 和 closed-loop client 已有 |
| Edge-LIBERO recipe | **已新增** | `action_policy_libero_all_edge` + 单节点 8 卡/4 卡 TOML 与 launcher |
| Edge DCP launcher | **已新增并成功执行** | `prepare_edge_dcp.sh`；原子落盘到个人 CFS，单 rank reload 通过 |
| 已训练 Edge-LIBERO 权重 | **不存在** | 不是待下载组件；未来训练产物 |

代码依据：

- `cosmos_framework/configs/base/experiment/sft/models/edge_model_config.py:34-60` 定义 Edge action-capable 模型基线。
- `cosmos_framework/inference/common/checkpoints.py:210-226` 将 Edge backbone 映射到公开 `nvidia/Cosmos3-Edge` 完整快照。
- `cosmos_framework/data/generator/action/datasets/libero_lerobot_dataset.py:4-20` 定义 LIBERO 数据、动作和 20 FPS stats 约束。
- `docs/action_policy_libero_posttrain.md:29-53` 指定 `nvidia/LIBERO_LeRobot_v3`、四 suite 和数据预处理约定。
- `cosmos_framework/scripts/action_policy_server_libero.py:4-33` 支持训练 DCP 或导出的 safetensors checkpoint。
- `docs/action_policy_libero_posttrain.md:105-130` 要求模拟器使用与训练环境隔离的 venv。
- `docs/action_policy_libero_edge_posttrain.md` 定义 Edge 四-suite 的 8 卡/4 卡等效 batch 配方。

Edge recipe 已独立使用 `EDGE_MODEL_CONFIG`，而不是只替换 Nano checkpoint 路径；正式训练前仍须完成本清单的环境、资产和多卡 smoke gate。
其 processor 已改为从 `${EDGE_BASE_PATH}` 读取已固定的完整 Edge 快照，避免训练
启动时重新按 Hub `main` 下载。

## 2. 当前运行时

| 检查项 | 当前结果 | 状态 |
| --- | --- | --- |
| OS / glibc | Ubuntu 24.04.3 / glibc 2.39 | 满足要求 |
| uv | `0.11.25` | 满足 `pyproject.toml:315-316` 的 `>=0.11.3` |
| Python | uv-managed CPython `3.13.12` | 与 `.python-version:1` 一致 |
| CFS 项目 venv | `/mnt/cfs/data/swy/cosmos3/envs/cosmos-framework-cu130-train` | **已就绪**：425 包、关键训练 imports 通过 |
| `cu130-train` 最终 sync | `Checked 425 packages in 1.34s` | `uv.lock` SHA-256 `3af6bd1b80349e0e95ee056770fead8772c7d9495602116420d405c1a5587701` |
| CUDA toolkit | 13.0 / nvcc 13.0.88 | 与 `cu130-train` 相符 |
| NVIDIA driver 文件 | 580.159.04 | 仅表示可读到版本文件 |
| GPU / BF16 | 真实设备视角可见 4 GPU；Torch CUDA BF16 smoke 通过 | 当前开发机通过；8-GPU 正式任务复验 |
| NCCL | 4-rank all-reduce 通过，NCCL `2.28.9+cuda13.0` | `world_size=4`，求和结果 10 |
| RPBZZZ6 任务侧记录 | 8 卡，每卡 97,887 MB | 来自 `/mnt/cfs/data/pmk/outputs/new-ckpt-smoke` 只读日志；正式训练任务仍复验 CUDA/NCCL |
| `/dev/shm` | 约 942 GiB，shared memory create/unlink 成功 | 通过 |
| `ffmpeg`, `git-lfs`, `curl`, `wget`, `tree` | 已安装 | 通过 |
| `libx11-dev`, `libegl1` | 缺失 | 派生镜像补齐 |
| `/root/.cache/uv` | 当前会话不可写 | 必须显式配置 cache |
| Cosmos 持久化根目录 | `/mnt/cfs/data/swy/cosmos3` | 已创建 models/envs/cache/checkpoints/runs 等目录 |
| LIBERO 独立数据根 | `/mnt/cfs/data/swy/libero` | 已创建数据与 manifests；与 Cosmos 环境/输出分离 |
| BOS→CFS 基准 | 1 GiB / 3.016 s / 356 MB/s（含 `fdatasync`） | 历史基准；BOS 副本现在为可选项 |
| `HF_ENDPOINT` | `https://huggingface.co` | 用户已明确批准官方端点；固定 revision 已下载完成 |
| 本地可用空间 | 约 86 GiB | 仅够轻量验证，不够稳定保存完整训练资产/输出 |

项目 uv 环境已完整安装到 CFS，使用持久 CPython 3.13.12、Torch
`2.10.0+cu130` 和 CUDA runtime 13.0。`torch`、`torchvision`、
`transformer_engine`、`natten`、`cosmos_framework`、`transformers`、
`diffusers`、`lerobot`、`pyarrow`、`av` 均可导入。真实设备视角下 4 张 GPU、
CUDA BF16 和 4-rank NCCL all-reduce 均已通过；8 卡正式任务仍须复跑相同 smoke。

`uv pip check` 扫描 425 包后只报告一个非训练阻断项：
`jupyter-compare-view` 声明 Python `<3.13`，而仓库 `.python-version` 固定
3.13。保留仓库 Python 版本，不为该可视化插件降级训练栈。

官方建议首次工作流预留约 150 GiB，并把 `HF_HOME`、`IMAGINAIRE_OUTPUT_ROOT` 放到大容量持久化存储（`docs/setup.md:31-40`、`docs/setup.md:233-243`）。全量训练按至少 1 TiB 规划。

## 3. `/mnt/bos/1011` 可复用资源

| 资源 | 大小 | 对 Edge-LIBERO 的用途 |
| --- | ---: | --- |
| `/mnt/bos/1011/models/Cosmos3-Edge` | 9,198,075,487 B payload | 官方固定 revision 归档副本；不作为训练运行时输入 |
| `/mnt/bos/1011/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | 2,818,839,170 B（2.63 GiB） | 已迁移到个人 CFS；BOS 原件不作为训练运行时输入 |
| `/mnt/bos/1011/models/Wan2.2-TI2V-5B` | 31.85 GiB | 不需复制整仓 |
| `/mnt/bos/1011/models/Cosmos3-Nano` | 34.43 GiB | Nano base，不用于 Edge-LIBERO |
| `/mnt/bos/1011/cosmos3/checkpoints/Cosmos3-Nano-DCP` | 28.26 GiB | Nano DCP，不能代替 Edge DCP |
| `/mnt/bos/1011/models/Qwen3-VL-8B-Instruct` | 16.34 GiB | Edge 使用 Nemotron-2B-Dense-VL，不需要 |
| `/mnt/bos/1011/datasets/*` | 多组机器人/VLM 数据 | 未发现官方 20 FPS `LIBERO_LeRobot_v3`，不能替代 |

当前资产：

```text
READY   /mnt/cfs/data/swy/cosmos3/models/Cosmos3-Edge
READY   /mnt/cfs/data/swy/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
READY   /mnt/cfs/data/swy/cosmos3/checkpoints/Cosmos3-Edge-DCP
READY   /mnt/bos/1011/models/Cosmos3-Edge
READY   /mnt/cfs/data/swy/libero/LIBERO_LeRobot_v3
```

`/mnt/cfs/data/swy` 是用户个人目录；本任务只在其 `cosmos3/` 与 `libero/`
子树写入。其他 `/mnt/cfs/data/*` 用户目录仅可做只读盘点，不作为训练依赖。

## 4. 唯一需要准备的模型/数据组件

| 优先级 | 组件 | 大小 | 来源/位置 | 说明 |
| --- | --- | ---: | --- | --- |
| P0 | Cosmos3 Edge HF base | 9,198,075,487 B | **已就绪两份**：CFS 正本及 `/mnt/bos/1011/models/Cosmos3-Edge` | 官方 revision `6f58f6b4c91288838e60b6bcb2cc45d997e961de`；各 54 文件；size/SHA 一致 |
| P0 | Wan2.2 VAE | 2,818,839,170 B | **已就绪** `/mnt/cfs/data/swy/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | SHA-256 `20eb7896…92b36`；4 卡真实训练 encode 通过 |
| P0 | LIBERO 四标准 suite | 1,867,646,269 B | **已就绪** `/mnt/cfs/data/swy/libero/LIBERO_LeRobot_v3` | revision `ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64`；38 文件；四套均 20 FPS；parquet/双路视频读取通过 |
| P0 | Edge DCP | 6,740,325,663 B | **已就绪** `/mnt/cfs/data/swy/cosmos3/checkpoints/Cosmos3-Edge-DCP` | 4 文件；单 rank reload 为 549 tensors / 3,369,657,024 parameters；完整 SHA manifest 已生成 |
| P0 | LIBERO action stats | 很小 | 已在 repo | `libero_native_frame_wise_relative_rot6d.json` |
| P0 | train venv | 官方 lock 425 包 | `/mnt/cfs/data/swy/cosmos3/envs/cosmos-framework-cu130-train` | `cu130-train`；CFS 持久版，另建议镜像内 `/opt/cosmos-venv` |
| P1 | LIBERO simulator venv | 独立环境 | 镜像内 `/opt/libero-venv` | Python 3.10 + LIBERO + robosuite 1.4.1 + MuJoCo 2.3.7 |
| 不需要 | Edge-DROID policy | 9.17 GB | — | 已排除 |
| 不需要 | DROID 数据集 | — | — | 已排除 |
| 不需要 | `libero_90` | 2.59 GB | — | 不属于当前四-suite训练目标 |
| 不存在 | 发布版 Cosmos3-Edge-Policy-LIBERO | — | — | 未来训练输出，不应列入下载清单 |

数据目录公开大小：`libero_10` 约 631 MB、goal 334 MB、object 537 MB、spatial 366 MB；全仓含 `libero_90` 为 4.46 GB。数据固定使用代码注册 revision `ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64`（`cosmos_framework/inference/common/checkpoints.py:329-334`）。

公开来源：

- <https://huggingface.co/nvidia/Cosmos3-Edge/tree/main>
- <https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3>

下载网络策略已由用户在 2026-08-15 明确变更为允许官方
`https://huggingface.co`。Edge 固定 revision 已分别从官方端点下载到个人 CFS
和唯一授权的 BOS 目标；两份的 size manifest 与 11 项关键 SHA-256 一致。模型清单位于
`/mnt/cfs/data/swy/cosmos3/manifests/Cosmos3-Edge.files.tsv` 与
`/mnt/cfs/data/swy/cosmos3/manifests/Cosmos3-Edge.sha256`。LIBERO 的 38 文件完整
size/SHA-256 清单位于 `/mnt/cfs/data/swy/libero/manifests/`。训练时使用本地路径
和 offline 环境变量，不再按浮动 `main` 拉取权重。
