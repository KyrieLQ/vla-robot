# 研究代码与复现入口

这里保留模型实现、实验配置和评测记录。先在仓库根目录运行 `python examples/model_smoke.py`，可直接检查四个基础模型；完整训练和 LIBERO 闭环另需数据、权重及 Linux 仿真环境。

## 代码地图

| 内容 | 入口 |
|---|---|
| 示范划分、RGB、归一化、动作窗口 | [mini_vla/data.py](mini_vla/data.py)、[tests/test_contract.py](mini_vla/tests/test_contract.py) |
| 小 CNN 与单步动作 | [model.py](mini_vla/model.py)、[experiment_v1.py](mini_vla/experiment_v1.py) |
| 动作分块 | [chunk_model_v1.py](mini_vla/chunk_model_v1.py)、[experiment_m2_v1.py](mini_vla/experiment_m2_v1.py) |
| ResNet18 视觉编码 | [visual_model_v1.py](mini_vla/visual_model_v1.py)、[experiment_m3_v1.py](mini_vla/experiment_m3_v1.py) |
| Transformer 与动作查询 | [attention_model_v1.py](mini_vla/attention_model_v1.py)、[experiment_m4_v1.py](mini_vla/experiment_m4_v1.py) |
| SmolVLM 前端与 LoRA | [vlm_frontend_v1.py](mini_vla/vlm_frontend_v1.py)、[smolvlm_lora_v1.py](mini_vla/smolvlm_lora_v1.py) |
| 回归与 Flow 动作头 | [vlm_action_heads_v1.py](mini_vla/vlm_action_heads_v1.py)、[train_vlm_head_v1.py](mini_vla/train_vlm_head_v1.py)、[train_vlm_flow_v1.py](mini_vla/train_vlm_flow_v1.py) |
| π0.5 数据与训练 | [pi05_four_suite_v1.py](mini_vla/pi05_four_suite_v1.py)、[pi05_full_data_v1.py](mini_vla/pi05_full_data_v1.py)、[pi05_formal_worker_v3.py](mini_vla/pi05_formal_worker_v3.py) |
| LIBERO 评测协议 | [evaluation_standard_v1.py](mini_vla/evaluation_standard_v1.py)、[final_suite_client_v2.py](mini_vla/final_suite_client_v2.py) |
| 2,000 回合结果 | [REVIEWED_REPORT.md](results/mini_vla_final2000_20261021_run3/REVIEWED_REPORT.md)、[episodes.csv](results/mini_vla_final2000_20261021_run3/review/episodes.csv) |

历史实验文件名中的阶段编号曾调整，模型对应关系以实现与 [编号对照](reports/MINI_VLA_PPT_NUMBERING_20261024.md) 为准。

## 环境

基础示例使用仓库根目录的 `requirements-smoke.txt`。完整研究环境通过上游锁文件固定：

- OpenPI：`215abfb217dbac7d5f1273282331b9b1866c0479`，Python 3.11.13。
- LIBERO：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`，独立 Python 3.8.20 环境。
- 本地 OpenPI 修改见 [openpi_local.patch](reports/openpi_local.patch)。

在 Linux 上，从本目录执行以下环境准备命令。它们会下载依赖、创建项目内环境；不会开始训练。

```bash
cd research
bash scripts/bootstrap.sh
bash scripts/install_libero.sh
```

`bootstrap.sh` 校验 OpenPI 提交，应用随仓库保存的补丁，使用锁文件安装依赖。`project_env.sh` 把缓存、环境和临时文件放到本项目目录。完整依赖快照保留在 `configs/`；基础 CPU 示例不需要这些环境。

## 数据、训练和推理

训练数据和检查点体积较大，按配置中的数据来源另行准备。`configs/` 保存训练/验证划分与实验协议，`reports/` 包含预处理和归一化资产。划分以整条 episode 为单位；新任务必须重新计算训练部分的统计量。

各模型的训练入口独立于实验调度器。例如小模型入口支持：

```bash
python -m mini_vla.experiment_v1 --help
# mode: cache / train / serve
# 必填: --plan <配置 JSON> --run <本次输出目录>
# serve 另需: --checkpoint <检查点> --port <本机端口>
```

M2、M3、M4 分别使用 `experiment_m2_v1`、`experiment_m3_v1`、`experiment_m4_v1`。模型由所选 checkpoint 内的配置、预处理和参数共同恢复。训练前检查配置中的数据目录、缓存目录、模型变体和输出目录，不覆盖已发布的评测记录。

SmolVLM 的缓存提取、LoRA 训练、回归/Flow 动作头采用独立入口。π0.5 的四套数据变换和 LoRA 设置由 `pi05_four_suite_v1.py` 构建，正式 worker 还依赖已完成的资源预检与检查点绑定记录。它们是保留实验协议的研究实现，不能只把某个历史 `start_*.sh` 拷到新机器就假定所有前提已满足。

## 如何读实验记录

`reports/` 与 `results/` 保留原始实验说明、逐回合记录和必要的复核文件。部分历史报告引用未分发的大体积权重、完整视频、缓存或运行日志；本仓库的结果入口以 [docs/RESULTS.md](../docs/RESULTS.md) 为准。

运行器中的原实验路径、物理 GPU 选择和资源阈值属于历史配置。本机会话标识统一替换为 `PUBLIC_REPRODUCTION`，对应文件的原始与发布校验值均保存在源码清单中。迁移时建立自己的运行配置；与其他项目或原会话调度器耦合的包装脚本未纳入分发。核心模型、数据处理和评测实现保留在本目录。

发布时的源码清单与校验值见 [source_manifest.json](../docs/source_manifest.json)。根目录 `python tools/check_repository.py` 可验证文件完整性并重算 2,000 回合成绩。
