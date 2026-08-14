# Cosmos3 Edge Policy LIBERO 环境拼装计划

## 0. 目标与完成定义

唯一目标是把下面这条链路所需环境与资产装齐：

```text
Cosmos3-Edge HF base
        ↓ convert_model_to_dcp
Cosmos3-Edge DCP + Wan2.2 VAE + LIBERO four-suite dataset
        ↓ action_policy_libero_all_edge (8-GPU or 4-GPU preset)
trained Edge-LIBERO DCP/export
        ↓ action_policy_server_libero
LIBERO MuJoCo closed-loop client
```

当前阶段完成的定义：

1. 单一镜像中有可用的训练 venv 和隔离的 LIBERO simulator venv。
2. 实际 GPU 任务通过 CUDA/BF16/NCCL smoke。
3. Edge base、Wan VAE、Edge DCP、LIBERO 四 suite 和 normalizer 全部可读并有 manifest。
4. Edge DCP 可以多 rank 加载，LIBERO loader 可以真实解码小批量。
5. LIBERO HTTP server CLI 可启动到“等待训练 checkpoint”的边界，sim client 可独立启动。

配方已落盘，但在 G1-G8 环境与资产 gate 通过前不开始全量训练。8 卡/4 卡拓扑和 smoke 命令见 `docs/action_policy_libero_edge_posttrain.md`。

## 1. 立即决策

### 1.1 不再 clone

保留官方 clone，固定 commit：

```text
c3493f3b5f9a40d381c1d66e548ef249b8f4d64f
```

升级代码必须显式变更 commit，并重新生成依赖和资产 manifest。

### 1.2 需要一个派生训练镜像

当前 uv 命令和 Python 解释器存在，但 `.venv` 不存在；stock repo Dockerfile 也只安装 `cu130 + vllm`，不是 `cu130-train`（`Dockerfile:45-53`）。平台不能控制 `docker run`，因此运行时临时安装和 `/workspace/.venv` volume 都不可靠。

建议维护一个派生镜像：

- base：`nvcr.io/nvidia/pytorch:26.06-py3`，对应 `docs/setup.md:42-52`。
- 主 venv：`/opt/cosmos-venv`，Python 3.13。
- simulator venv：`/opt/libero-venv`，Python 3.10。
- 代码与 lock 固定到本计划 commit。
- 模型、DCP、数据和训练输出不进入镜像层。
- 不安装 `policy-server`/OpenPI；那是 RoboLab WebSocket 路径。
- 不安装 `vllm`；对当前 action post-training/HTTP server 没有必要，且扩大 torch 版本约束面。

### 1.3 主环境只选 `cu130-train`

构建时：

```bash
UV_PROJECT_ENVIRONMENT=/opt/cosmos-venv \
uv sync --frozen --no-editable --all-extras --group=cu130-train
```

这是 CUDA 13 的官方通用训练路径（`docs/setup.md:154-164`、`docs/setup.md:220-227`）。不要同时安装 `cu128*`、`cu130-torch213*` 或 `vllm`；仓库在 `pyproject.toml:315-333` 明确声明互斥关系。

## 2. 可写持久化存储是 P0 前置条件

当前 `/mnt` 和默认 `/root/.cache` 不可写；本地仅余约 86 GiB。开始下载前，平台需要给出一个所有训练 rank 可见的可写持久化根目录。下文以 `/data/cosmos-edge-libero` 表示：

```text
/data/cosmos-edge-libero/
├── cache/
│   ├── huggingface/
│   └── uv/
├── models/
│   └── Cosmos3-Edge/
├── datasets/
│   └── LIBERO_LeRobot_v3/
├── checkpoints/
│   └── Cosmos3-Edge-DCP/
├── runs/
├── exports/
└── manifests/
```

容量：

- 环境装配和 loader/model smoke：至少 150 GiB。
- 全量训练目标：至少 1 TiB，且所有节点/rank 看到同一路径。
- checkpoint 保留频率较高时按每次保存的 DCP 实际大小额外扩容。

环境变量：

```bash
export COSMOS_WRITABLE_ROOT=/data/cosmos-edge-libero
export UV_CACHE_DIR=$COSMOS_WRITABLE_ROOT/cache/uv
export HF_HOME=$COSMOS_WRITABLE_ROOT/cache/huggingface
export IMAGINAIRE_OUTPUT_ROOT=$COSMOS_WRITABLE_ROOT/runs
export EDGE_BASE_PATH=$COSMOS_WRITABLE_ROOT/models/Cosmos3-Edge
export EDGE_DCP_PATH=$COSMOS_WRITABLE_ROOT/checkpoints/Cosmos3-Edge-DCP
export LIBERO_ROOT=$COSMOS_WRITABLE_ROOT/datasets/LIBERO_LeRobot_v3
export WAN_VAE_PATH=/mnt/bos/1011/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export LD_LIBRARY_PATH=
```

不得把任何 cache/output 指向 `/mnt`。`WAN_VAE_PATH` 是唯一直接使用的只读 `/mnt` 输入。

## 3. 阶段 A：构建单一双-venv镜像

### A1. 系统层

安装：

```text
curl ffmpeg git git-lfs libx11-dev tree wget
libegl1 libglvnd0 libgl1 libglib2.0-0
```

前一组满足 Cosmos 安装；后一组为 headless MuJoCo EGL。固定 uv 版本并安装 Python 3.13、3.10。

镜像环境：

```text
PATH=/opt/cosmos-venv/bin:$PATH
LD_LIBRARY_PATH=
TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
MUJOCO_GL=egl
```

### A2. `/opt/cosmos-venv`：训练和 LIBERO HTTP server

使用当前 `uv.lock` 和 `cu130-train`。该环境需要成功导入：

```text
torch, torchvision, transformer_engine, natten, triton
cosmos_framework, transformers, diffusers
lerobot, pyarrow, av, imageio, PIL, omegaconf, tyro
```

`action_policy_server_libero.py` 使用 Python 标准库 `ThreadingHTTPServer`，不依赖 OpenPI（`action_policy_server_libero.py:43-61`、`:1198-1346`）。

### A3. `/opt/libero-venv`：独立模拟器 client

训练环境与旧版 robosuite/MuJoCo pins 冲突，按官方文档独立安装（`docs/action_policy_libero_posttrain.md:105-130`）：

```bash
uv venv --python 3.10 /opt/libero-venv
VV=/opt/libero-venv/bin/python
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO
uv pip install -p "$VV" -e /opt/LIBERO -r /opt/LIBERO/requirements.txt
uv pip install -p "$VV" \
  'robosuite==1.4.1' 'mujoco==2.3.7' 'torch<2.6' \
  loguru requests scipy pillow numpy
```

首次验证后固定 LIBERO Git commit，并写入 image manifest。将 `~/.libero/config.yaml` 和 robosuite macro 初始化结果在 image build 中准备好，避免只读 runtime root 下首次写入失败。

注意：`pyproject.toml:278-293` 另有较新的 `libero` group（MuJoCo 3.3.2），但当前仓库闭环文档明确使用兼容组合 robosuite 1.4.1 + MuJoCo 2.3.7。为保证训练/评测 parity，本计划采用独立 venv 的文档组合，不把 `--group=libero` 混入训练 venv。

### A4. 镜像 build 验收

```bash
uv pip check --python /opt/cosmos-venv/bin/python
/opt/cosmos-venv/bin/python -c \
  'import torch, transformer_engine, natten, cosmos_framework, lerobot, pyarrow, av'
/opt/cosmos-venv/bin/python -m cosmos_framework.scripts.train --help
/opt/cosmos-venv/bin/python -m cosmos_framework.scripts.action_policy_server_libero --help

MUJOCO_GL=egl /opt/libero-venv/bin/python -c \
  'import libero, robosuite, mujoco; print(robosuite.__version__, mujoco.__version__)'
```

所有命令退出码必须为 0；构建完成后任务启动不得联网补包。

## 4. 阶段 B：实际 GPU runtime 验收

在平台真正分配 4 GPU 的任务内执行：

```bash
nvidia-smi
/opt/cosmos-venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
x = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
assert (x @ x).isfinite().all()
print("nccl", torch.cuda.nccl.version())
PY
```

随后做单机 4 rank NCCL all-reduce smoke，并复验 POSIX shared memory。当前诊断会话 GPU 不可见不等于镜像失败；GPU device 注入由平台 runtime 负责，镜像无法修复。如果实际任务仍无 GPU，就在这里停下。

## 5. 阶段 C：下载并固定唯一模型 `Cosmos3-Edge`

无需 Edge-DROID，也不存在可下载的 Edge-LIBERO 成品。唯一模型下载：

```bash
hf download nvidia/Cosmos3-Edge \
  --repo-type model \
  --revision 6f58f6b4c91288838e60b6bcb2cc45d997e961de \
  --local-dir "$EDGE_BASE_PATH"
```

下载后记录：

- 解析后的完整 HF commit；
- `config.json`、`model.safetensors.index.json` 的 SHA256；
- 所有 shard 相对路径和字节数；
- 总大小；
- repo commit 与 image digest。

公开仓库约 9.18 GB，含模型配置、tokenizer、vision encoder、VAE、scheduler 和权重。Edge 训练 backbone 需要完整快照，不能只下载 reasoner 子目录（`checkpoints.py:210-226`）。

验收：

1. `config.json` 的模型类型为 Edge，且代码能加载。
2. safetensors index 引用的所有 shard 都存在。
3. processor/tokenizer 和 vision encoder 可离线加载。
4. 禁网后用本地路径完成一次模型 config/weights load。

## 6. 阶段 D：转换 Edge HF → DCP

训练基础设施读取 DCP；转换器原生支持 Edge safetensors/diffusers 布局（`convert_model_to_dcp.py:60-84`）：

```bash
/opt/cosmos-venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o "$EDGE_DCP_PATH" \
  --checkpoint-path "$EDGE_BASE_PATH"
```

输出必须在可写共享存储，不写 `/mnt`。验收：

1. `model/.metadata`、`model/config.json` 和所有 distcp shard 存在。
2. 单 rank 可从 DCP 恢复模型 state。
3. 4 rank 可恢复且参数 key/count 一致。
4. 检查 Edge action-related modules 的 key 是否与未来 policy recipe 预期一致；缺失或随机初始化策略留到 recipe 阶段明确，不能静默忽略。
5. 对 DCP metadata 和文件清单做 hash/size manifest。

## 7. 阶段 E：准备 LIBERO 四-suite训练数据

使用仓库 checkpoint catalog 固定的 revision，而不是浮动 main：

```bash
hf download nvidia/LIBERO_LeRobot_v3 \
  --repo-type dataset \
  --revision ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64 \
  --include 'libero_10/**' \
  --include 'libero_goal/**' \
  --include 'libero_object/**' \
  --include 'libero_spatial/**' \
  --local-dir "$LIBERO_ROOT"
```

不要下载 `libero_90`。四 suite 合计约 1.87 GB，目录结构必须为：

```text
$LIBERO_ROOT/
├── libero_10/{data,meta,videos}
├── libero_goal/{data,meta,videos}
├── libero_object/{data,meta,videos}
└── libero_spatial/{data,meta,videos}
```

逐 suite 验收：

- `meta/info.json` 报告 20 FPS；bundled `quantile_rot` stats 以 20 FPS 为前提（`libero_lerobot_dataset.py:17-20`）。
- raw action 为 7D，loader 转成 10D `pos3 + rot6d6 + gripper1`（`:12-15`）。
- `observation.images.image` 和 `observation.images.wrist_image` 两路视频均存在。
- parquet 可读，AV1 MP4 可由当前 PyAV/ffmpeg 解码。
- 用 `camera_mode=concat_view`, `image_size=256`, `chunk_length=16`, `action_normalization=quantile_rot` 实例化 loader。
- 每个 suite 读取首/中/尾样本，并用多 worker DataLoader 读取至少一个小批量。
- 检查输出无 NaN/Inf、shape 一致、prompt/task metadata 非空。

normalizer 已在仓库：

```text
cosmos_framework/data/generator/action/normalizer_stats/
  libero_native_frame_wise_relative_rot6d.json
```

不需要另行下载，但要把该文件 SHA256 纳入 manifest。

## 8. 阶段 F：Wan2.2 VAE 只读验收

固定路径：

```text
/mnt/bos/1011/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

当前文件大小为 2,818,839,170 bytes。验收：

1. 所有训练节点都能读取同一路径。
2. 记录 SHA256；只读计算，不复制、不修改。
3. 用 Edge model tokenizer 配置覆盖 `vae_path=$WAN_VAE_PATH` 后完成一次 VAE load。
4. 对一个 LIBERO concat sample 做 encode，并验证 latent shape/finite。

这里不需要下载整个 Wan2.2 仓库。

## 9. 阶段 G：使用正式 recipe 做组合 smoke

环境完整性的最小组合验证：

1. 从 `EDGE_DCP_PATH` 构造 Edge model config。
2. 注入只读 `WAN_VAE_PATH`。
3. 从每个 LIBERO suite 取一个真实样本，经完整 action transform/normalizer/packing 路径处理。
4. 在 1 GPU 上完成一次 forward-only 或一小步无保存训练，并立即退出。
5. 在 4 GPU 上完成 DDP/FSDP 初始化、模型加载、一个小批量 forward/backward、all-reduce，然后退出。
6. 使用正式 Edge 配方的短迭代覆盖，不保存 checkpoint、不启用 W&B。

使用 `EXTRA_TAIL_OVERRIDES` 把正式 8 卡或 4 卡配方缩短到 5 步；目标是证明模型、VAE、数据与训练栈能接通。

## 10. 阶段 H：LIBERO HTTP server 与 simulator 环境边界

已训练 checkpoint 尚不存在，因此当前只能验证接口环境：

- 主 venv 中 `action_policy_server_libero --help` 和配置解析成功。
- simulator venv 中能创建一个 headless LIBERO env、reset、执行随机/零 action、渲染 agentview+wrist。
- client 能访问一个 mock `/info`、`/predict` endpoint，证明网络/JSON/base64 路径可用。

正式训练完成后再用实际 Edge-LIBERO checkpoint 启动 HTTP server。server 接受 DCP + matching config，或导出的 safetensors（`action_policy_server_libero.py:13-33`）。评测 parity 必须保持：20 FPS、concat view、256×256 双相机、frame-wise relative rot6d、quantile_rot；见 `docs/action_policy_libero_posttrain.md:139-148`。

## 11. 完成矩阵

| Gate | 必须满足 | 当前状态 |
| --- | --- | --- |
| G0 代码 | 官方 commit；Edge config、LIBERO loader/server/eval 存在 | 通过 |
| G1 存储 | 可写共享根目录 ≥150 GiB；全量训练目标 ≥1 TiB | 未提供 |
| G2 镜像 | `/opt/cosmos-venv` + `/opt/libero-venv`，imports/pip check | 未完成 |
| G3 GPU | 实际任务 4 GPU、BF16、NCCL、shared memory | 未验证 |
| G4 Edge base | 本地完整快照 + manifest + 离线 load | 未下载 |
| G5 Edge DCP | 转换、单/4 rank load、key manifest | 未生成 |
| G6 LIBERO data | 四 suite、20 FPS、decode/loader batch | 未下载 |
| G7 VAE | `/mnt` 只读 load + sample encode | 文件存在，功能未验 |
| G8 组合 smoke | Edge DCP + VAE + 四-suite样本 + 1/4 GPU step | 未完成 |
| G9 eval client | 独立 venv、headless env/reset/render、HTTP mock | 未完成 |

G0-G9 全部通过后，环境拼装才算完成；随后冻结镜像/资产 manifest 并提交正式训练任务。

## 12. 推荐执行顺序

1. 平台确认唯一派生镜像基底和可写共享路径。
2. 构建双 venv 镜像并通过无 GPU import smoke。
3. 在真实 4-GPU runtime 通过 GPU/NCCL smoke。
4. 下载 Edge base，生成 manifest，做离线 load。
5. 转换 Edge DCP，做单/4 rank load。
6. 下载四个 LIBERO suite，完成逐 suite loader/decode smoke。
7. 加载 `/mnt` VAE，对真实样本做 encode。
8. 完成 1 GPU 和 4 GPU 的组合 smoke。
9. 完成独立 LIBERO simulator/client smoke。
10. 冻结环境基线：image digest、repo commit、uv.lock hash、HF revisions、DCP/data/VAE manifests。
11. 按 8 卡或 4 卡正式 recipe 提交全量训练。
