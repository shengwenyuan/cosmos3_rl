# Level 3：UR12e 真机迁移与部署

本阶段承接 Level 2 的 EEF checkpoint，范围包括：

- UR12e/夹爪/相机标定与坐标约定；
- 真机数据采集、质量门禁及小样本适配；
- action 解码、限速限位、安全停机与 closed-loop 执行；
- 4070 Ti 级主机上的 16-step 预测、8-step 执行、异步预取及时延测量；
- 仿真回归、离线回放与真机验收矩阵。

详细接口和阶段门禁以 `../shared/cosmos3_edge_eef_to_ur12e_staged_training_plan.md` 为准。Level 2 的 action 表示、归一化和相机顺序在进入本阶段后不得隐式变化。
