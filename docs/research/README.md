# Cosmos3-Edge Level 2 / Level 3 研发文档

本目录只收纳 EEF 后训练、真机迁移与部署相关的项目文档。框架通用文档继续保留在 `docs/` 顶层。

## 目录

- `shared/`：跨阶段约定、总体路线和统一接口。
- `level2/data/`：公开数据清洗、筛选、统计和 LeRobot 转换。
- `level2/training/`：DROID/RH20T 的 EEF 后训练、验收与消融计划。
- `level3/`：UR12e 真机标定、适配、在线部署和安全门禁。

## 当前主线

1. DROID `success` 数据建立通用 EEF 先验。
2. RH20T cfg4 做桌面操作域适配。
3. UR12e 少量真机数据完成机器人域迁移。
4. 以 16-step action chunk、滚动执行 8-step 为部署起点，在目标 4070 Ti 级主机上测闭环时延。

跨阶段单一事实源见 `shared/cosmos3_edge_eef_to_ur12e_staged_training_plan.md`。
