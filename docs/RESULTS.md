# 结果与评测口径

## 四套完整评测

ResNet18 + Transformer，四套联合训练，固定选择 epoch 3，动作块 K=10，每次执行 H=5。40 个任务各评测 50 个初始状态，共 2,000 回合。

| Suite | 成功 | 回合 | 成功率 |
|---|---:|---:|---:|
| Spatial | 450 | 500 | 90.0% |
| Object | 439 | 500 | 87.8% |
| Goal | 341 | 500 | 68.2% |
| Long | 335 | 500 | 67.0% |
| 总体 | 1,565 | 2,000 | 78.25% |

[逐回合记录](../research/results/mini_vla_final2000_20261021_run3/review/episodes.csv) · [独立复核](../research/results/mini_vla_final2000_20261021_run3/review/independent_acceptance.json) · [原报告](../research/results/mini_vla_final2000_20261021_run3/REVIEWED_REPORT.md) · [机器可读汇总](libero_results.json)

`tools/check_repository.py` 检查回合键唯一性，按 suite 汇总 CSV，要求结果与此表及 JSON 一致。

## 开发实验

下表是固定 Spatial 50 回合的模型开发记录，不与上面的 2,000 回合成绩合并排名。

| 配置 | 成功 / 50 | 成功率 |
|---|---:|---:|
| CNN + 单动作 MLP | 18 | 36% |
| 动作分块 K10/H5 | 41 | 82% |
| ImageNet 预训练 ResNet18 | 44 | 88% |
| Transformer 融合 + 动作查询 | 44 | 88% |
| 四套联合训练的 ResNet18 + Transformer | 47 | 94% |
| 冻结 SmolVLM + 回归动作头 | 39 | 78% |
| 同起点冻结骨干、继续训练动作头 | 44 | 88% |
| 同起点视觉/语言 LoRA + 动作头 | 46 | 92% |

阶段原表见 [模型性能记录](../research/reports/MINI_VLA_PPT_M1_M8_PERFORMANCE.md)。最后两行是等量续训配对对照；LoRA 相对冻结续训净增加 2/50，不把从 39/50 开始的全部收益归于 LoRA。证据见 [配对分析](../research/results/mini_vla_m6_paired_dev50_20261029_run3/analysis.json) 和 [配对回合](../research/results/mini_vla_m6_paired_dev50_20261029_run3/review/paired_cases.csv)。

随后在四套各 50 回合的开发面板上，现有 SmolVLM LoRA endpoint 为 **141/200（70.5%）**。它使用不同的初态面板，不能与旧 Spatial 46/50 或完整 2,000 回合结果直接计算增益。[四套开发报告](../research/reports/MINI_VLA_SMOL_REGRESSION_DEV200_20261105.md)

## π0.5 与真机接口

仓库提供 π0.5 的数据适配、LoRA 配置、训练实现及 PiPER 推理入口。当前发布的记录不包含 π0.5 四套 2,000 回合最终成绩，因此不填入同口径排行榜。

真机三视角与 14 维状态/动作约定见 [PiPER 文档](../robots/piper/README.md)。视频提供可视化演示；视频本身不用于计算任务成功率。模型推理、真实动作执行和多次任务成功率是三个分别验证的环节。
