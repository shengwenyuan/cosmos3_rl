# Cosmos3 Edge Policy LIBERO 环境对齐

本目录只服务一个目标：配置出 `Cosmos3-Edge → LIBERO policy post-training → LIBERO closed-loop eval` 所需的全部组件和环境。

当前目录覆盖环境拼装和 Edge-LIBERO 四-suite 配方对齐；Edge-DROID、RoboLab 和 OpenPI WebSocket server 全部排除。

## 当前结论

- 当前 clone 来自 NVIDIA 官方仓库，固定在提交 `c3493f3b5f9a40d381c1d66e548ef249b8f4d64f`，无需再次 clone。
- 已新增 `action_policy_libero_all_edge` 共享实验，以及 RPBZZZ6 单节点 8 卡/4 卡两套 TOML 和 launcher；两套配置保持有效全局 batch 2048。
- 当前镜像有 CUDA 13.0 toolkit、uv 和 uv-managed Python 3.13.12，但没有项目 `.venv`，因此尚不能训练或加载 Edge。
- `/mnt` 只读；其中 Wan2.2 VAE 可直接复用。Edge base、Edge DCP 和 LIBERO v3 数据仍缺失。
- 平台只能选择一个镜像，因此建议在同一派生镜像中预烘焙两个隔离环境：`/opt/cosmos-venv` 用于训练/HTTP server，`/opt/libero-venv` 用于 MuJoCo/LIBERO client。

## 文档索引

- [RESOURCE_INVENTORY.md](./RESOURCE_INVENTORY.md)：当前仓库、运行时、`/mnt` 和缺失组件盘点。
- [ENVIRONMENT_ASSEMBLY_PLAN.md](./ENVIRONMENT_ASSEMBLY_PLAN.md)：单镜像、存储、模型、DCP、数据和双 venv 的详细拼装计划。
- [Edge-LIBERO recipe](../docs/action_policy_libero_edge_posttrain.md)：8 卡/4 卡拓扑、资产准备、smoke 和正式启动命令。
- [env.example.sh](./env.example.sh)：不含凭据的环境变量模板。
- [verify_environment.sh](./verify_environment.sh)：不下载资源、不修改 `/mnt` 的验收脚本。

## 边界

1. 不向 `/mnt` 写入文件、软链接、cache 或输出。
2. 不把 token 或其他凭据写进仓库、镜像层或日志。
3. 大模型、数据集、DCP 和训练输出不得提交 Git。
4. 取得可写持久化共享目录之前，不开始下载或转换。
