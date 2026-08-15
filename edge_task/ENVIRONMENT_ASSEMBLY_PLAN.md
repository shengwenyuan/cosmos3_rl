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

### 0.1 2026-08-15 执行状态

- 已创建个人持久化根目录 `/mnt/cfs/data/swy/cosmos3`，容量约 10 PiB 可用。
- 用户已明确批准从 `https://huggingface.co` 下载。固定 revision `6f58f6b4c91288838e60b6bcb2cc45d997e961de` 的官方 Edge MidTrain 已直接落到 `/mnt/cfs/data/swy/cosmos3/models/Cosmos3-Edge`：54 文件、9,198,075,487 bytes；两个 safetensors index 的全部引用均通过校验。
- 固定 revision 的第二份 Edge 快照已从官方 Hugging Face 直接下载到唯一授权的 BOS 目标 `/mnt/bos/1011/models/Cosmos3-Edge`：54 文件、9,198,075,487 bytes；与 CFS 正本的 size 清单及 11 项关键 SHA-256 完全一致。下载耗时约 4 分 47 秒，未改动其他 BOS 路径。
- uv-managed CPython 3.13.12 和完整 `cu130-train` venv 已落到个人 CFS。最终 `uv sync` 对 425 包为 1.34 秒空操作，关键训练 imports 通过；Torch 为 `2.10.0+cu130`、CUDA runtime 13.0。cache 与 venv 位于同一 NFS，必须使用 `UV_LINK_MODE=hardlink`，不能使用极慢的逐文件 copy。
- 当前 NVIDIA 官方基线为 `origin/main@c3493f3b5f9a40d381c1d66e548ef249b8f4d64f`；`edge-libero` recipe 提交为 `8e3b66b1a987e439d5bf81731fb548913919c97c`。当前 `uv.lock` SHA-256 为 `3af6bd1b80349e0e95ee056770fead8772c7d9495602116420d405c1a5587701`。
- 当前开发机真实设备视角已通过 4-GPU CUDA BF16 smoke 和 `torchrun` 4-rank NCCL all-reduce；NCCL 为 `2.28.9+cuda13.0`。正式 8-GPU RPBZZZ6 任务仍需复跑相同 gate。
- DCP 已由 `edge_task/prepare_edge_dcp.sh` 原子落到 `/mnt/cfs/data/swy/cosmos3/checkpoints/Cosmos3-Edge-DCP`：4 文件、6,740,325,663 bytes。单 rank reload 恢复 549 tensors / 3,369,657,024 parameters，4-rank warm-start 也已在真实 smoke 中通过。
- LIBERO 四套训练数据已从官方固定 revision `ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64` 下载到独立路径 `/mnt/cfs/data/swy/libero/LIBERO_LeRobot_v3`：38 个 payload 文件、1,867,646,269 bytes；四套均为 20 FPS，未下载 `libero_90`，完整 size/SHA-256 manifests 已生成。
- Wan2.2 VAE 已从 BOS 原子迁移到 `/mnt/cfs/data/swy/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`：2,818,839,170 bytes，源/目标 SHA-256 均为 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`；训练运行时不再直接读取 BOS。
- 4 卡完整 recipe smoke 已于 2026-08-15 成功完成 5/5 iterations（exit 0）：4-rank NCCL/FSDP、DCP warm-start、真实 LIBERO loader、CFS VAE、forward/backward/optimizer 均通过；首步含 compile 为 145.40 秒，后续稳态约 30–31 秒/步，loss 全部有限。
- Edge-LIBERO recipe 的 processor 已绑定 `${EDGE_BASE_PATH}`，4/8 卡 launcher 在启动前检查本地 `config.json` 与 safetensors index；训练不再按浮动 Hub `main` 获取 processor。

## 1. 立即决策

### 1.1 不再 clone

保留官方 clone，不再重复下载仓库。代码基线与 recipe 层分别固定为：

```text
NVIDIA upstream: origin/main@c3493f3b5f9a40d381c1d66e548ef249b8f4d64f
Edge-LIBERO recipes: edge-libero@8e3b66b1a987e439d5bf81731fb548913919c97c
uv.lock SHA-256: 3af6bd1b80349e0e95ee056770fead8772c7d9495602116420d405c1a5587701
```

升级代码或 recipe 必须显式变更 commit，并重新生成依赖和资产 manifest。文档收尾
修改尚在当前工作树，生成最终训练镜像时应记录届时实际 HEAD，而不是只记录上述
已提交 recipe commit。

### 1.2 需要一个派生训练镜像

stock repo Dockerfile 只安装 `cu130 + vllm`，不是 `cu130-train`（`Dockerfile:45-53`）。平台不能控制 `docker run`，因此不能依赖额外的 `/workspace/.venv` volume。当前已在自动挂载的个人 CFS 上准备持久 venv；派生镜像内的 `/opt/cosmos-venv` 仍是性能更好的生产选择。

建议维护一个派生镜像：

- base：`nvcr.io/nvidia/pytorch:26.06-py3`，对应 `docs/setup.md:42-52`。
- 主 venv：`/opt/cosmos-venv`，Python 3.13。
- 持久 fallback：`/mnt/cfs/data/swy/cosmos3/envs/cosmos-framework-cu130-train`。
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

当前 CFS 持久环境的等价命令：

```bash
export COSMOS_WRITABLE_ROOT=/mnt/cfs/data/swy/cosmos3
export UV_CACHE_DIR=$COSMOS_WRITABLE_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$COSMOS_WRITABLE_ROOT/python
export UV_PROJECT_ENVIRONMENT=$COSMOS_WRITABLE_ROOT/envs/cosmos-framework-cu130-train
export UV_LINK_MODE=hardlink

uv sync --frozen --no-editable --all-extras --group=cu130-train \
  --python $COSMOS_WRITABLE_ROOT/python/cpython-3.13.12-linux-x86_64-gnu/bin/python3.13
```

`hardlink` 仅因为 uv cache 与 venv 同在 `/mnt/cfs/data/swy/cosmos3` 才成立；若
未来把二者拆到不同文件系统，改用 `UV_LINK_MODE=copy`。训练启动阶段只激活/复用
环境，不再次运行 `uv sync`。

这是 CUDA 13 的官方通用训练路径（`docs/setup.md:154-164`、`docs/setup.md:220-227`）。不要同时安装 `cu128*`、`cu130-torch213*` 或 `vllm`；仓库在 `pyproject.toml:315-333` 明确声明互斥关系。

## 2. 可写持久化存储布局

个人 CFS 已确定为所有训练大文件的主存储。Cosmos 环境/模型/DCP/输出与 LIBERO
数据使用两个独立子树；cache、uv、DCP、数据和输出不得放 BOS。

```text
/mnt/bos/1011/models/
└── Cosmos3-Edge/                    # 已就绪：官方固定 revision 第二份副本

/mnt/cfs/data/swy/cosmos3/
├── cache/
│   ├── huggingface/
│   └── uv/
├── python/
│   └── cpython-3.13.12-linux-x86_64-gnu/
├── envs/
│   └── cosmos-framework-cu130-train/
├── models/
│   ├── Cosmos3-Edge/                # 已就绪：官方固定 revision
│   └── Wan2.2-TI2V-5B/
│       └── Wan2.2_VAE.pth           # 已迁移；训练运行时使用此 CFS 副本
├── checkpoints/
│   └── Cosmos3-Edge-DCP/
├── runs/
├── exports/
└── manifests/

/mnt/cfs/data/swy/libero/
├── LIBERO_LeRobot_v3/               # 已就绪：四 suite，不含 libero_90
└── manifests/                       # 38 文件 size + 完整 SHA-256
```

容量：

- 环境装配和 loader/model smoke：至少 150 GiB。
- 全量训练目标：至少 1 TiB，且所有节点/rank 看到同一路径。
- checkpoint 保留频率较高时按每次保存的 DCP 实际大小额外扩容。

环境变量：

```bash
export COSMOS_WRITABLE_ROOT=/mnt/cfs/data/swy/cosmos3
export LIBERO_WRITABLE_ROOT=/mnt/cfs/data/swy/libero
export UV_CACHE_DIR=$COSMOS_WRITABLE_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$COSMOS_WRITABLE_ROOT/python
export UV_PROJECT_ENVIRONMENT=$COSMOS_WRITABLE_ROOT/envs/cosmos-framework-cu130-train
export UV_LINK_MODE=hardlink
export HF_HOME=$COSMOS_WRITABLE_ROOT/cache/huggingface
export HF_ENDPOINT=https://huggingface.co
export IMAGINAIRE_OUTPUT_ROOT=$COSMOS_WRITABLE_ROOT/runs
export EDGE_BASE_PATH=$COSMOS_WRITABLE_ROOT/models/Cosmos3-Edge
export EDGE_DCP_PATH=$COSMOS_WRITABLE_ROOT/checkpoints/Cosmos3-Edge-DCP
export BASE_CHECKPOINT_PATH=$EDGE_DCP_PATH
export LIBERO_ROOT=$LIBERO_WRITABLE_ROOT/LIBERO_LeRobot_v3
export WAN_VAE_PATH=$COSMOS_WRITABLE_ROOT/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export COSMOS_PYTHON=$UV_PROJECT_ENVIRONMENT/bin/python
export PATH=$UV_PROJECT_ENVIRONMENT/bin:$PATH
export LD_LIBRARY_PATH=
```

授权边界：训练只写 `/mnt/cfs/data/swy/cosmos3` 和 `/mnt/cfs/data/swy/libero`。
BOS 只作为迁移/归档来源，不直接作为训练输入；cache、DCP、VAE、数据和训练输出
全部从个人 CFS 读取或写入。

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

随后做单机 4 rank NCCL all-reduce smoke，并复验 POSIX shared memory。受限诊断
沙箱中 GPU 可能不可见，应以平台真实设备视角为准；GPU device 注入由 runtime
负责，镜像无法修复。如果实际任务仍无 GPU，就在这里停下。

当前开发机已用 CFS venv 实测：4 张 GPU 可见、Torch CUDA BF16 通过、POSIX
shared memory 通过，且下面的 4-rank NCCL smoke 输出
`PASS nccl all-reduce world_size=4 sum=10`：

```bash
LD_LIBRARY_PATH='' NCCL_DEBUG=WARN \
  /mnt/cfs/data/swy/cosmos3/envs/cosmos-framework-cu130-train/bin/torchrun \
  --standalone --nproc-per-node=4 edge_task/nccl_smoke.py
```

这证明当前 4-GPU 开发环境的基础通信链路；正式 8-GPU 训练镜像仍需复跑，不能
把 4 卡结果外推为 8 卡验收。

## 5. 阶段 C：下载并固定唯一模型 `Cosmos3-Edge`

无需 Edge-DROID，也不存在可下载的 Edge-LIBERO 成品。用户已明确批准官方
Hugging Face 端点，并要求直接下载到个人 CFS。实际执行使用同文件系统 staging，
完整性检查通过后原子改名到正式目录：

```bash
export HF_ENDPOINT=https://huggingface.co

/mnt/cfs/data/swy/cosmos3/envs/cosmos-framework-cu130-train/bin/hf download \
  nvidia/Cosmos3-Edge \
  --repo-type model \
  --revision 6f58f6b4c91288838e60b6bcb2cc45d997e961de \
  --local-dir /mnt/cfs/data/swy/cosmos3/models/.Cosmos3-Edge.download
```

首次 8-worker 下载在最大 shard 的 HTTP 连接上停滞；保留 536,837,693-byte
断点后，以 `--max-workers 1` 成功续传。最终结果：54 文件、9,198,075,487 bytes
（8.566 GiB），`config.model_type=cosmos3_edge`；根 index 引用 3 个 shard、
transformer index 引用 2 个 shard，全部存在。

正式路径：

```text
/mnt/cfs/data/swy/cosmos3/models/Cosmos3-Edge
```

资产清单：

- `/mnt/cfs/data/swy/cosmos3/manifests/Cosmos3-Edge.files.tsv`
- `/mnt/cfs/data/swy/cosmos3/manifests/Cosmos3-Edge.sha256`

### C1. BOS 第二份官方副本

按用户要求没有从 CFS 复制，而是从官方 Hugging Face 对同一固定 revision 再下载
一份到 `/mnt/bos/1011/models/Cosmos3-Edge`。目标预检时不存在且不是符号链接；
下载仅写该新目录，约 4 分 47 秒完成。验收结果为 0 个 `.incomplete`、54 个 payload
文件、9,198,075,487 bytes，根/transformer index 分别引用 3/2 个完整 shard，且
与 CFS manifests 的文件大小和 11 项关键 SHA-256 全部一致。BOS 副本只作归档；
正式训练不得直接读取它，必须使用 CFS 正本。

公开仓库约 9.18 GB，含模型配置、tokenizer、vision encoder、VAE、scheduler 和权重。Edge 训练 backbone 需要完整快照，不能只下载 reasoner 子目录（`checkpoints.py:210-226`）。

验收：

1. `config.json` 的模型类型为 Edge，且代码能加载。
2. safetensors index 引用的所有 shard 都存在。
3. processor/tokenizer 和 vision encoder 可离线加载。
4. 禁网后用本地路径完成一次模型 config/weights load。

## 6. 阶段 D：转换 Edge HF → DCP

训练基础设施读取 DCP；转换器原生支持 Edge safetensors/diffusers 布局及本地
processor runtime（`cosmos_framework/scripts/convert_model_to_dcp.py:54-118`）：

推荐使用带输入检查、离线模式和 `.incomplete` 原子落盘保护的 launcher：

```bash
bash edge_task/prepare_edge_dcp.sh
```

最终目标固定为
`/mnt/cfs/data/swy/cosmos3/checkpoints/Cosmos3-Edge-DCP`。launcher 优先读取
个人 CFS base；输入不存在时退出且不创建空 DCP，不再回退读取 BOS。

实际转换已完成。公开 Edge 配置原本会在本地转换期间再次按 `revision=main`
解析 processor，并无必要地构造 Wan VAE；`convert_model_to_dcp.py` 现仅对 Edge
的 conversion runtime 把 processor 指向本地完整快照并跳过 VAE，保存到 DCP 的
仍是原始、无机器本地路径的 deployment config。产物为 4 文件、
6,740,325,663 bytes；单 rank DCP reload 恢复 549 tensors / 3,369,657,024
parameters。完整哈希清单位于
`/mnt/cfs/data/swy/cosmos3/manifests/Cosmos3-Edge-DCP.sha256`。

等价的底层命令为：

```bash
/opt/cosmos-venv/bin/python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o "$EDGE_DCP_PATH" \
  --checkpoint-path "$EDGE_BASE_PATH"
```

输出必须写个人 CFS checkpoint 根目录；不写 BOS 或其他未授权的 `/mnt` 路径。验收：

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
  --include 'libero_10/**' 'libero_goal/**' 'libero_object/**' 'libero_spatial/**' \
  --local-dir /mnt/cfs/data/swy/libero/.LIBERO_LeRobot_v3.download
```

不要下载 `libero_90`。四 suite 合计约 1.87 GB，目录结构必须为：

```text
$LIBERO_ROOT/
├── libero_10/{data,meta,videos}
├── libero_goal/{data,meta,videos}
├── libero_object/{data,meta,videos}
└── libero_spatial/{data,meta,videos}
```

实际下载已完成并在检查后从同文件系统暂存目录原子改名到
`/mnt/cfs/data/swy/libero/LIBERO_LeRobot_v3`。四 suite 统计为：`libero_10`
379 episodes / 101,469 frames，goal 428 / 52,042，object 454 / 66,984，spatial
432 / 52,970；合计 1,693 episodes / 273,465 frames。全部 `info.json` 为 20 FPS，
每套均有 parquet、双路 MP4 和 metadata，且没有 `.incomplete`。38 个 payload 文件
的 size 与完整 SHA-256 清单位于 `/mnt/cfs/data/swy/libero/manifests/`。训练 venv
已用 PyArrow 打开每套首个 parquet，并由 PyAV 分别解码两路首帧；8 路均为
256×256 `yuv420p`。正式 LeRobot loader/normalizer 小批量仍需在组合 smoke 中验证。

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

## 8. 阶段 F：Wan2.2 VAE 迁移与验收

固定路径：

```text
/mnt/cfs/data/swy/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

已使用 `.incomplete` 暂存、size/SHA 校验和原子改名从 BOS 迁移完成。当前文件大小为
2,818,839,170 bytes，SHA-256 为
`20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`；清单位于
`/mnt/cfs/data/swy/cosmos3/manifests/Wan2.2_VAE.sha256`。验收：

1. 所有训练节点都能读取同一 CFS 路径。
2. 记录并复验 CFS SHA256。
3. 用 Edge model tokenizer 配置覆盖 `vae_path=$WAN_VAE_PATH` 后完成一次 VAE load。
4. 对一个 LIBERO concat sample 做 encode，并验证 latent shape/finite。

这里不需要迁移整个 Wan2.2 仓库；训练运行时不读取 BOS。

## 9. 阶段 G：使用正式 recipe 做组合 smoke

环境完整性的最小组合验证：

1. 从 `EDGE_DCP_PATH` 构造 Edge model config。
2. 注入只读 `WAN_VAE_PATH`。
3. 从每个 LIBERO suite 取一个真实样本，经完整 action transform/normalizer/packing 路径处理。
4. 在 1 GPU 上完成一次 forward-only 或一小步无保存训练，并立即退出。
5. 在 4 GPU 上完成 DDP/FSDP 初始化、模型加载、一个小批量 forward/backward、all-reduce，然后退出。
6. 使用正式 Edge 配方的短迭代覆盖，不启用 W&B；把周期性 `save_iter` 调大，但注意 trainer 正常退出仍会保存 final checkpoint。

使用 `EXTRA_TAIL_OVERRIDES` 把正式 8 卡或 4 卡配方缩短到 5 步；目标是证明模型、VAE、数据与训练栈能接通。

4 卡 smoke 已实际完成，输出目录为
`/mnt/cfs/data/swy/cosmos3/runs/cosmos3_action_libero/smoke/action_policy_libero_all_edge_4gpu_smoke_20260815`。
4-rank DCP warm-start 用时 2.47 秒；5 个 optimizer iteration 全部完成，rank loss
约 15.4–16.3 且无 OOM、NaN、NCCL 或数据解码错误。首步含 `torch.compile` 为
145.40 秒，后四步约 30–31 秒/步。框架在正常退出时仍保存 final checkpoint；
本次 `iter_000000005` 约占 30 GiB，已按用户要求删除，保留 156 KiB 日志/config。

## 10. 阶段 H：LIBERO HTTP server 与 simulator 环境边界

独立 simulator 环境已安装到
`/mnt/cfs/data/swy/libero/envs/libero-eval`，使用 uv 管理的 Python 3.10.20。
官方 LIBERO 源码固定为 commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`，位于
`/mnt/cfs/data/swy/libero/LIBERO`；配置位于
`/mnt/cfs/data/swy/libero/config/config.yaml`。核心版本为 robosuite 1.4.1、
MuJoCo 2.3.7、torch 2.5.1+cu124、NumPy 1.22.4、SciPy 1.10.1 和
OpenCV 4.6.0。由于 venv 位于 CFS，`robosuite/macros_private.py` 已关闭
Numba 源码旁 cache。

当前已通过 `uv pip check`、四 suite benchmark 枚举、OffScreenRenderEnv import，
并实际创建 LIBERO-10 task 0 的 headless EGL 环境：reset 得到 agentview/wrist
两路 64×64 RGB 图像，执行一步 dummy action 后正常关闭。训练所需
`LIBERO_LeRobot_v3` 不等于 simulator 的 BDDL/init/assets；后者来自上述官方
LIBERO 仓库。

首个已训练 checkpoint `iter_000000500` 的同机 server-client smoke 已通过。
GPU 0 直接加载 8-way 训练 DCP 和 matching `config.yaml`，GPU 1 运行 EGL client；
LIBERO-10 task 0 的 2 个短 episode 共完成 4 次真实 `/predict`。每次均发送
agentview+wrist 的 256×512 concat observation，并返回 16×10 有限 action chunk；
client 转为 7D simulator action 后执行到设定的 32 steps，保存两份 33-frame GIF
和 `summary.json`。8-step sampler 的首请求含 compile 约 36 秒，稳态 HTTP 总耗时
约 0.62 秒/请求、模型推理约 0.24–0.25 秒。短 horizon 下 0/2 success 只证明链路，
不作为 checkpoint SR。server 接受 DCP + matching config，或导出的 safetensors
（`action_policy_server_libero.py:13-33`）。正式评测 parity 必须保持：20 FPS、
concat view、256×256 双相机、frame-wise relative rot6d、quantile_rot；见
`docs/action_policy_libero_posttrain.md:139-148`。

## 11. 完成矩阵

| Gate | 必须满足 | 当前状态 |
| --- | --- | --- |
| G0 代码 | 官方基线 + recipe commit；Edge config、LIBERO loader/server/eval 存在 | 通过：基线 `c3493f3`，recipe `8e3b66b`；最终镜像另记实际 HEAD |
| G1 存储 | CFS 个人根目录；全量训练目标 ≥1 TiB | 通过：CFS 约 10 PiB 可用；BOS 第二份 Edge 副本也已校验 |
| G2 训练 uv | `cu130-train` 425 包、Python 3.13.12、imports/pip check | 通过；仅 `jupyter-compare-view` 的 Python `<3.13` 元数据为非训练阻断告警 |
| G2b simulator | 独立 CFS venv，imports/headless EGL | 通过：真实 env reset、双路 render、dummy step |
| G3 GPU | 实际任务 4 GPU、BF16、NCCL、shared memory | 当前开发机通过 4 GPU/BF16/4-rank NCCL/shm；8-GPU 正式任务复验 |
| G4 Edge base | CFS 固定 revision、manifest、离线 processor/index load | 通过：CFS/BOS 各 54 文件、9,198,075,487 bytes；size/关键 SHA256 一致 |
| G5 Edge DCP | 转换、单/4 rank load、key manifest | 通过：DCP 已生成，单 rank 与 4-rank smoke warm-start 均通过 |
| G6 LIBERO data | 四 suite、20 FPS、decode/loader batch | 通过：静态/完整 SHA/PyArrow/PyAV 与真实 4-rank loader batch 均通过 |
| G7 VAE | CFS load + sample encode | 通过：VAE 已迁移并在真实训练 forward 中完成 encode |
| G8 组合 smoke | Edge DCP + VAE + 四-suite样本 + 1/4 GPU step | 4-GPU 通过：5/5 iterations、exit 0；1-GPU 独立 smoke 未做 |
| G9 eval client | 独立 venv、headless env/reset/render、实际 policy HTTP smoke | 通过：iter 500、2 episodes、4 次真实 predict、action/GIF/summary 完整 |

G0-G9 全部通过后，环境拼装才算完成；随后冻结镜像/资产 manifest 并提交正式训练任务。

## 12. 推荐执行顺序

1. **已完成**：固定个人 CFS 根目录，安装 `cu130-train` venv。
2. **已完成**：当前开发机通过 4-GPU CUDA/BF16/NCCL/shm。
3. **已完成**：从官方端点下载固定 Edge revision 到个人 CFS 并生成 manifests。
4. **已完成**：从 CFS Edge base 转换 DCP 并通过单 rank reload。
5. **已完成**：4-rank DCP warm-start。
6. **已完成**：数据下载、静态验收、逐 suite parquet/双路视频 decode 和真实 loader batch。
7. **已完成**：迁移 CFS VAE，并在真实样本训练中完成 encode。
8. **4-GPU 已完成**：5-step 组合 smoke；若需要额外隔离诊断再补 1-GPU smoke。
9. **已完成**：独立 LIBERO venv、真实 headless reset/render/step，以及 `iter_000000500` 同机 server-client smoke。
10. 冻结环境基线：image digest、repo commit、uv.lock hash、HF revisions、DCP/data/VAE manifests。
11. 按 8 卡或 4 卡正式 recipe 提交全量训练。
