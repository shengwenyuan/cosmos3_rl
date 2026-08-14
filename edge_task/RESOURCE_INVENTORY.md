# Cosmos3 Edge Policy LIBERO 资源盘点

盘点时间：2026-08-14 UTC。所有 `/mnt` 操作均为只读枚举和元数据统计。

## 1. 仓库与能力

| 项目 | 结果 | 判断 |
| --- | --- | --- |
| 工作目录 | `/root/workspace/cosmos-framework` | 正确 |
| `origin` | `https://github.com/NVIDIA/cosmos-framework.git` | NVIDIA 官方源 |
| 分支 / HEAD | `main` / `c3493f3b5f9a40d381c1d66e548ef249b8f4d64f` | 固定此版本 |
| 包版本 | `1.2.2` | `pyproject.toml:8-15` |
| Edge base 配置 | 已有 | `EDGE_MODEL_CONFIG`、Edge inference YAML、DCP converter 均存在 |
| LIBERO train data | 已有 loader | LeRobot v3 parquet + 双路视频 loader、normalizer 已有 |
| LIBERO inference/eval | 已有 | HTTP policy server 和 closed-loop client 已有 |
| Edge-LIBERO recipe | **已新增** | `action_policy_libero_all_edge` + 单节点 8 卡/4 卡 TOML 与 launcher |
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

## 2. 当前运行时

| 检查项 | 当前结果 | 状态 |
| --- | --- | --- |
| OS / glibc | Ubuntu 24.04.3 / glibc 2.39 | 满足要求 |
| uv | `0.11.25` | 满足 `pyproject.toml:315-316` 的 `>=0.11.3` |
| Python | uv-managed CPython `3.13.12` | 与 `.python-version:1` 一致 |
| 项目 `.venv` | 不存在 | **未就绪** |
| `cu130-train` dry-run | 可解析，预计下载/安装 425 个包 | lock 可用，环境未安装 |
| CUDA toolkit | 13.0 / nvcc 13.0.88 | 与 `cu130-train` 相符 |
| NVIDIA driver 文件 | 580.159.04 | 仅表示可读到版本文件 |
| GPU | 当前诊断会话内 `nvidia-smi` 失败 | **实际 GPU 任务复验** |
| RPBZZZ6 任务侧记录 | 8 卡，每卡 97,887 MB | 来自 `/mnt/cfs/data/pmk/outputs/new-ckpt-smoke` 只读日志；正式训练任务仍复验 CUDA/NCCL |
| `/dev/shm` | 约 942 GiB，shared memory create/unlink 成功 | 通过 |
| `ffmpeg`, `git-lfs`, `curl`, `wget`, `tree` | 已安装 | 通过 |
| `libx11-dev`, `libegl1` | 缺失 | 派生镜像补齐 |
| `/root/.cache/uv` | 当前会话不可写 | 必须显式配置 cache |
| `HF_HOME`, `UV_CACHE_DIR`, `IMAGINAIRE_OUTPUT_ROOT` | 未设置 | **未就绪** |
| 本地可用空间 | 约 86 GiB | 仅够轻量验证，不够稳定保存完整训练资产/输出 |

uv 本体和 Python 是完整的，但“项目 uv 环境”不完整：没有 `.venv`，也没有任何项目依赖。冻结 dry-run 已能解析 lock；`uv lock --check --offline` 仅因 LeRobot、Megatron-LM 两个 Git 源未缓存而无法离线检查，不代表 lock 冲突。

官方建议首次工作流预留约 150 GiB，并把 `HF_HOME`、`IMAGINAIRE_OUTPUT_ROOT` 放到大容量持久化存储（`docs/setup.md:31-40`、`docs/setup.md:233-243`）。全量训练按至少 1 TiB 规划。

## 3. `/mnt/bos/1011` 可复用资源

| 资源 | 大小 | 对 Edge-LIBERO 的用途 |
| --- | ---: | --- |
| `/mnt/bos/1011/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | 2,818,839,170 B（2.63 GiB） | **直接只读复用，设置 `WAN_VAE_PATH`** |
| `/mnt/bos/1011/models/Wan2.2-TI2V-5B` | 31.85 GiB | 不需复制整仓 |
| `/mnt/bos/1011/models/Cosmos3-Nano` | 34.43 GiB | Nano base，不用于 Edge-LIBERO |
| `/mnt/bos/1011/cosmos3/checkpoints/Cosmos3-Nano-DCP` | 28.26 GiB | Nano DCP，不能代替 Edge DCP |
| `/mnt/bos/1011/models/Qwen3-VL-8B-Instruct` | 16.34 GiB | Edge 使用 Nemotron-2B-Dense-VL，不需要 |
| `/mnt/bos/1011/datasets/*` | 多组机器人/VLM 数据 | 未发现官方 20 FPS `LIBERO_LeRobot_v3`，不能替代 |

明确缺失：

```text
MISSING /mnt/bos/1011/models/Cosmos3-Edge
MISSING /mnt/bos/1011/datasets/LIBERO_LeRobot_v3
MISSING writable Cosmos3-Edge-DCP output
```

`/mnt/cfs/data` 含其他用户/项目内容；不视为本任务已授权依赖，也不修改。

## 4. 唯一需要准备的模型/数据组件

| 优先级 | 组件 | 大小 | 来源/位置 | 说明 |
| --- | --- | ---: | --- | --- |
| P0 | Cosmos3 Edge HF base | 约 9.18 GB | `nvidia/Cosmos3-Edge` revision `6f58f6b4c91288838e60b6bcb2cc45d997e961de` | 唯一需要下载的模型；含 Edge 架构、tokenizer、reasoner、vision、generator 权重 |
| P0 | Wan2.2 VAE | 2.63 GiB | 已有只读 `/mnt/.../Wan2.2_VAE.pth` | 不下载、不复制 |
| P0 | LIBERO 四标准 suite | 约 1.87 GB | `nvidia/LIBERO_LeRobot_v3` | `libero_10/goal/object/spatial`，20 FPS |
| P0 | Edge DCP | 由本地转换产生 | 可写 checkpoint 根目录 | 不是下载项；由 Edge HF base 转换 |
| P0 | LIBERO action stats | 很小 | 已在 repo | `libero_native_frame_wise_relative_rot6d.json` |
| P0 | train venv | 官方 lock 约 425 包 | 镜像内 `/opt/cosmos-venv` | `cu130-train`，无需 `policy-server`/OpenPI |
| P1 | LIBERO simulator venv | 独立环境 | 镜像内 `/opt/libero-venv` | Python 3.10 + LIBERO + robosuite 1.4.1 + MuJoCo 2.3.7 |
| 不需要 | Edge-DROID policy | 9.17 GB | — | 已排除 |
| 不需要 | DROID 数据集 | — | — | 已排除 |
| 不需要 | `libero_90` | 2.59 GB | — | 不属于当前四-suite训练目标 |
| 不存在 | 发布版 Cosmos3-Edge-Policy-LIBERO | — | — | 未来训练输出，不应列入下载清单 |

数据目录公开大小：`libero_10` 约 631 MB、goal 334 MB、object 537 MB、spatial 366 MB；全仓含 `libero_90` 为 4.46 GB。数据固定使用代码注册 revision `ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64`（`cosmos_framework/inference/common/checkpoints.py:329-334`）。

公开来源：

- <https://huggingface.co/nvidia/Cosmos3-Edge/tree/main>
- <https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3>
