# Mini VLA课号与模型编号对照（2026-10-23）

按用户要求，从此采用“第N课对应MN”的统一教学编号。课程、当前首页、性能汇总与后续讲解默认使用新编号。历史实验目录、模型权重、冻结协议与原始验收报告保留原编号，实验成绩不改变。

| 课号 | 当前模型编号 | 历史实验编号 | 内容 |
|---|---|---|---|
| 第1课 | **M1** | M1 | 最小单动作策略 |
| 第2课 | **M2** | M2 | 动作分块与重规划 |
| 第3课 | **M3** | M3 | 视觉编码与预训练 |
| 第4课 | **M4** | M4 | Transformer融合与动作查询 |
| 第5课 | **M5** | M6 | 四套联合训练、选点与评测 |
| 第6课 | **M6** | M7a | 空间token |
| 第7课 | **M7** | M8 | 冻结预训练VLM与回归动作头 |
| 第8课 | **M8** | M9 | 条件flow动作头 |
| 第9课 | **M9** | π0.5正式Spatial微调 | 预训练完整VLA适配与部署复盘 |

原计划中的可选M5没有运行，从新教学路线中移除，不对应一个新模型或新成绩。

- 当前第五课：新M5＝历史M6，四套联合训练模型；Spatial50为47/50，最终四套1565/2000。
- 新M6＝历史M7a，空间token；Spatial50为37/50。
- 新M7＝历史M8，冻结VLM＋回归头；Spatial50为39/50。
- 新M8＝历史M9，同冻结VLM＋flow头；Spatial50为18/50。

例如旧路径 `results/mini_vla_m6_train_20261020_run1/training/best.pt` 现在属于课程M5，路径中的m6仅表示原实验标识。

结构化课程的 `stages[].id` 与 `legacy_experiment_id` 保留历史ID以维持依赖关系；`display_model_id`、`lesson_number` 和 `canonical_course_stages` 给出教学编号。当前性能汇总JSON使用新编号，并同时保留原编号键值作为 `legacy_model` / `legacy_successes`。

[第五课：M5](MINI_VLA_LESSON_05.md) · [当前M1—M9总览](MINI_VLA_COURSE_M1_M9_CURRENT.md) · [机器可读对照](../configs/mini_vla_lesson_numbering_20261023.json)

新M9是复用已有π0.5实验的教学阶段，不对应历史文件名中的mini_vla_m9（后者是新M8）。新增VLM微调仍暂缓。
