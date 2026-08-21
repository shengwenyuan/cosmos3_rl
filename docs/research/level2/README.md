# Level 2：公开数据 EEF 后训练

## 数据

- `data/droid_eef_data_and_filter_plan.md`：DROID 下载、Nano 对齐过滤及二次筛选。
- `data/rh20t_cfg4_data_cleaning_plan.md`：RH20T cfg4 清洗。
- `data/rh20t_cfg4_to_lerobot_plan.md`：RH20T cfg4 转 LeRobot。

## 训练

- `training/droid_eef_stage1_training_plan.md`：第一轮 DROID EEF 训练展开。
- `training/rh20t_cfg4_cosmos3_edge_training_plan.md`：RH20T cfg4 Edge 训练。

所有数据选择必须输出不可变 manifest；统计量只用 train split 计算，不能在加载器内放置不可追踪的临时筛选。
