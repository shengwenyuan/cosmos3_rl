# Cosmos3 Edge Policy LIBERO 环境对齐

本目录只服务一个目标：配置出 `Cosmos3-Edge → LIBERO policy post-training → LIBERO closed-loop eval` 所需的全部组件和环境。

当前目录覆盖环境拼装和 Edge-LIBERO 四-suite 配方对齐；Edge-DROID、RoboLab 和 OpenPI WebSocket server 全部排除。

## 当前结论

- 当前 clone 的 NVIDIA 官方基线是 `origin/main@c3493f3b5f9a40d381c1d66e548ef249b8f4d64f`；当前 `edge-libero` 分支在其上增加 recipe，已提交到 `8e3b66b1a987e439d5bf81731fb548913919c97c`，无需再次 clone。
- 已新增 `action_policy_libero_all_edge` 共享实验，以及 RPBZZZ6 单节点 8 卡/4 卡两套 TOML 和 launcher；两套配置保持有效全局 batch 2048。
- 已授权的持久化根目录为 `/mnt/cfs/data/swy/cosmos3` 和独立数据根 `/mnt/cfs/data/swy/libero`；训练 uv/Python、cache、模型、DCP 和 runs 放在前者，LIBERO 数据放在后者。
- 用户已明确批准改用 `HF_ENDPOINT=https://huggingface.co`。固定 revision `6f58f6b4c91288838e60b6bcb2cc45d997e961de` 的官方 Edge MidTrain 完整快照已直接落到个人 CFS：54 文件、9,198,075,487 bytes。
- Edge DCP 已生成到 `/mnt/cfs/data/swy/cosmos3/checkpoints/Cosmos3-Edge-DCP`：4 文件、6,740,325,663 bytes；单 rank reload 通过，恢复 549 tensors / 3,369,657,024 parameters。
- Edge-LIBERO recipe 的 processor 已绑定 `${EDGE_BASE_PATH}`，4/8 卡 launcher 会在训练前检查本地完整快照，避免按浮动 `main` 重访 Hub。
- `/mnt/bos/1011/models/Cosmos3-Edge` 已从官方 Hugging Face 固定 revision 独立下载：54 个 payload 文件、9,198,075,487 bytes，并与 CFS 正本的 size/SHA-256 清单完全一致；未修改任何其他 BOS 路径。
- LIBERO 四 suite 已下载到 `/mnt/cfs/data/swy/libero/LIBERO_LeRobot_v3`：38 个 payload 文件、1,867,646,269 bytes；全部为 20 FPS，未下载 `libero_90`，完整 SHA-256 清单位于同级 `manifests/`；四套的 parquet 与双路 256×256 视频首帧均已实际读取。
- Wan2.2 VAE 已迁移到 `/mnt/cfs/data/swy/cosmos3/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`：2,818,839,170 bytes，源/目标 SHA-256 一致；训练运行时不再读取 BOS。
- 4 卡完整 recipe smoke 已成功完成 5/5 iterations：4-rank DCP warm-start、LIBERO loader、CFS VAE encode、forward/backward/optimizer 全部通过；稳态约 30–31 秒/步，exit 0。
- 完整 `cu130-train` uv 环境已在个人 CFS 中按 `uv.lock`（SHA-256 `3af6bd1b80349e0e95ee056770fead8772c7d9495602116420d405c1a5587701`）安装并通过 425 包空操作同步和关键训练 imports；正式镜像内 `/opt/cosmos-venv` 仍是 I/O 性能更好的首选，CFS 环境作为可持久复用方案。
- 当前开发机真实设备视角已通过 4-GPU CUDA BF16 和 4-rank NCCL all-reduce（NCCL 2.28.9+CUDA 13.0）；8-GPU 正式任务仍需复跑同一 smoke。

## 文档索引

- [RESOURCE_INVENTORY.md](./RESOURCE_INVENTORY.md)：当前仓库、运行时、`/mnt` 和缺失组件盘点。
- [ENVIRONMENT_ASSEMBLY_PLAN.md](./ENVIRONMENT_ASSEMBLY_PLAN.md)：单镜像、存储、模型、DCP、数据和双 venv 的详细拼装计划。
- [Edge-LIBERO recipe](../docs/action_policy_libero_edge_posttrain.md)：8 卡/4 卡拓扑、资产准备、smoke 和正式启动命令。
- [env.example.sh](./env.example.sh)：不含凭据的环境变量模板。
- [configure_bashrc.sh](./configure_bashrc.sh)：校验 `/mnt/cfs/data/swy/personal/wandb` 的两行凭据、收紧权限为 `0600`，并在 `.bashrc` 顶部安装 5 行环境配置 block。
- [run_edge_libero.sh](./run_edge_libero.sh)：单节点 8 卡入口；默认正式训练，设置 `EDGE_LIBERO_MODE=smoke` 时仅跑 5 iter 无 W&B smoke。
- [run_edge_libero_4gpu.sh](./run_edge_libero_4gpu.sh)：对应的单节点 4 卡入口，使用相同的 `train|smoke` 模式。
- [verify_environment.sh](./verify_environment.sh)：不下载资源、不修改 `/mnt` 的验收脚本。
- [nccl_smoke.py](./nccl_smoke.py)：由 `torchrun` 调用的单机多卡 NCCL all-reduce smoke。
- [prepare_edge_dcp.sh](./prepare_edge_dcp.sh)：从完整 Edge HF base 离线转换并原子落盘到个人 CFS。

## 边界

1. 训练运行时只读取个人 CFS；BOS 仅保留为迁移/归档来源，不直接作为训练输入。只写 `/mnt/cfs/data/swy/cosmos3` 和 `/mnt/cfs/data/swy/libero`。
2. 不把 token 或其他凭据写进仓库、镜像层或日志。
3. 大模型、数据集、DCP 和训练输出不得提交 Git。
4. Edge base 固定使用已记录的官方 revision；训练阶段从个人 CFS 离线读取，不再使用浮动 `main` 下载权重。
