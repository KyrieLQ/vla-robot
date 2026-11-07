# π0.5 LoRA + LIBERO spatial 正式训练验收报告

复核时间：2026-10-13T21:08:44.580447+08:00。**授权范围已完成：双卡 30,000 步训练、两组各 50 集配对评测、完整产物审计。**

## 实测结论

| 策略 | 完成评测 | 任务成功 | 成功率 |
|---|---:|---:|---:|
| 官方 π0.5 base，使用同一份 spatial 归一化统计 | 50/50 | 0/50 | 0% |
| 本轮从 base 微调的第 30,000 步模型 | 50/50 | 42/50 | 84% |

配对差为 **+84 个百分点**；预注册的任务聚类 bootstrap（10 个任务簇、10,000 次、seed=20261013）95% 区间为 **+70 至 +94 个百分点**。42 对为基础模型失败、训练模型成功；8 对均失败，没有基础模型独自成功的场景。该区间仅反映本次有限评测的重采样结果，不包含训练种子变动或未见任务的不确定性。

![配对成功率与逐任务结果](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/paired_success.png)

这是 10 个已训练 spatial 任务、每任务 5 个预先固定初始状态、**一个训练种子（42）**的结果；不是全量 LIBERO benchmark，也未验证陌生任务或真机迁移。本轮没有建立训练演示与评测初始状态的逐条去重证明，不能称为严格未见场景泛化测试。官方论文/完整协议成绩不作为本轮直接对照，不能据此宣称达到 SOTA。

## 训练设置与耗时

- 源代码与依赖延续已验收的固定 openpi / LIBERO / LeRobot 版本，实际运行的源码、manifest 和统计量哈希全部复核一致。
- 官方 `pi05_base` 新初始化，未使用 22 步 smoke checkpoint 或官方 LIBERO expert 作为训练起点。
- 数据：固定 revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`，432 条完整轨迹、52,970 帧，10 个 spatial 任务；60,000 个样本位置约为 1.13 遍数据。
- 双卡 FSDP，物理 GPU 0/1，全局 batch=2、每卡 1；seed=42，workers=2，CPU 限定 8 个逻辑核。
- 预热 1,000 步、学习率 5e-5、EMA 关闭；action horizon=10；冻结规则沿用本项目基线。
- 总参数 3,403,421,456，可训练 466,957,072（13.7202%）；其中 LoRA 49,987,584，非 LoRA 416,969,488，包含视觉编码器。观察到的提升属于整套微调设置，不能单独归因于 LoRA 参数。
- **训练阶段 8.27 小时（约 8 小时 16 分）**；两组评测含服务启动共 **24.42 分钟**；正式流水线共 **8.67 小时（约 8 小时 40 分）**。此前数据准备和排错不计入此流水线耗时。
- 同步更新平均 0.834 秒，中位数 0.714 秒；纯更新合计 6.95 小时。训练循环另有约 4.9 分钟取 batch 和 71.6 分钟检查点/参数校验等开销。
- 前/后 1,000 步平均 loss：**0.059772 → 0.032430**。loss 与任务成功率是不同指标，不能互相替代。

![训练损失](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/training_loss.png)

## 配对评测与逐任务结果

两组均使用任务顺序 0，初始状态索引 `[2,12,22,32,42]`。环境 seed=`1000+100*task_id+init_index`，策略噪声 seed=`2000+100*task_id+init_index`，逐集重置。每集先执行 10 个固定等待动作，最多 220 个策略动作，每 5 步重规划一次。评测 checkpoint 预先固定为 30k，没有根据这些结果挑选模型。

复核 **50/50 对第一帧图像逐像素一致**；任务、初始状态、环境种子、策略种子逐对匹配。两组使用同一数值内容的 norm stats。LIBERO 固定源码的 `done` 来自 `_check_success()`，步数上限与程序异常单独记录。两组均无程序异常；base 在物理 GPU 0、训练模型在物理 GPU 1 上评测，各自实际 EGL UUID 与所选卡一致。

| task | 任务描述 | base | 本轮模型 | 差值（百分点） |
|---|---|---:|---:|---:|
| 0 | 盘子与 ramekin 之间的黑碗 → 盘子 | 0/5 | 5/5 | +100 |
| 1 | ramekin 旁的黑碗 → 盘子 | 0/5 | 4/5 | +80 |
| 2 | 桌子中央的黑碗 → 盘子 | 0/5 | 5/5 | +100 |
| 3 | 饼干盒上的黑碗 → 盘子 | 0/5 | 5/5 | +100 |
| 4 | 木柜上层抽屉内的黑碗 → 盘子 | 0/5 | 2/5 | +40 |
| 5 | ramekin 上的黑碗 → 盘子 | 0/5 | 3/5 | +60 |
| 6 | 饼干盒旁的黑碗 → 盘子 | 0/5 | 4/5 | +80 |
| 7 | 炉灶上的黑碗 → 盘子 | 0/5 | 5/5 | +100 |
| 8 | 盘子旁的黑碗 → 盘子 | 0/5 | 4/5 | +80 |
| 9 | 木柜上的黑碗 → 盘子 | 0/5 | 5/5 | +100 |

本次较弱的两项是“上层抽屉内的黑碗”（2/5）与“ramekin 上的黑碗”（3/5）。每任务只有 5 集，不能据此稳健排名任务难度或确定失败机制。以下为全部 8 个失败，不重抽初始状态、不补跑挑选成功样本。

| task | 初始状态索引 | 策略动作数 | 终止原因 |
|---|---:|---:|---|
| 1 | 22 | 220 | STEP_LIMIT_REACHED |
| 4 | 22 | 220 | STEP_LIMIT_REACHED |
| 4 | 32 | 220 | STEP_LIMIT_REACHED |
| 4 | 42 | 220 | STEP_LIMIT_REACHED |
| 5 | 12 | 220 | STEP_LIMIT_REACHED |
| 5 | 32 | 220 | STEP_LIMIT_REACHED |
| 6 | 22 | 220 | STEP_LIMIT_REACHED |
| 8 | 2 | 220 | STEP_LIMIT_REACHED |

成功示例：[task 0 / init 2](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/trained_eval/episodes/task0_init2_TASK_SUCCESS.mp4)。失败示例：[task 4 / init 22](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/trained_eval/episodes/task4_init22_STEP_LIMIT_REACHED.mp4)。全部 100 个 MP4 均已用 ffprobe 实读核验，224×224、10 FPS，帧数与对应策略步数一致。

## 参数、检查点与资源审计

- 30,000 条更新记录编号连续且唯一，loss、梯度范数和参数范数全部有限。
- 每 1,000 步留存的 30 份参数哈希均通过：20 个冻结叶子与初始状态完全一致，变化仅发生于训练范围，LoRA 叶子实际改变。
- 保留 10k、20k、30k 三个完整检查点。最终模型已在独立策略服务中重载并完成评测。正式轮次没有中途恢复；优化器断点恢复能力此前已在 20+2 步 smoke 中实测，不能写成本轮又做了一次恢复训练。
- 最终 checkpoint：`/data1/niuye/lq/pi05_lora_libero/checkpoints/pi05_libero_lora_3090x2_train/formal_spatial_20261013_v1/30000`。
- 整卡每至少 5 秒采样的峰值：GPU 0 **17.32 GiB**，GPU 1 **21.16 GiB**。这是整卡观测值，可能漏掉短峰值，并包含其他共享进程；会话中 14:47 的 NVML 查询已检测到 GPU 1 另有约 4.7 GiB 占用。因此不能把 21.16 GiB 当作本模型独占显存需求，也没有单卡训练成功结论。
- GPU 阶段窗口合计 **16.94 GPU 小时**（训练 16.53、评测 0.41）；包含 CPU 校验、I/O 和等待，不等于实测 GPU 活跃计算量或独占预留。
- 项目当前占用 **81.36 GiB / 120 GiB**，所在文件系统可用 **431.09 GiB**，满足至少保留 100 GiB 的约束。所有产物仍在数据盘。
- 主机侧已核对本轮训练、监督和两个策略服务进程退出。仅清理本轮服务，未停止其他任务；取消的定时 GPU 等卡监控保持 STOPPED。

## 证据与状态

- [完整产物审计](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/completion_audit.json)：所有更新、存盘哈希、视频哈希、配对记录、存储与资源记账。
- [初始画面配对审计](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/paired_initial_images_audit.json)；[原始分析](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/analysis.json)；[流水线阶段](/data1/niuye/lq/pi05_lora_libero/results/formal_pipeline_20261013_v2/pipeline.json)。
- [冻结方案](/data1/niuye/lq/pi05_lora_libero/configs/formal_plan_20261013.json)；[部署与正式训练复现记录](/data1/niuye/lq/pi05_lora_libero/REPRO_LOG.md)。
- 文档勘误：冻结方案 `evaluation.scope` 中 `training seed=1` 的文字应读作“一个训练种子”；实际数值字段 `training_seed`、训练代码与运行报告均为 42。原冻结文件保持不改，未变更任何实验种子。

本轮没有触发“两组均零成功”的额外阳性对照条件，无需额外下载 expert 权重。已完成原授权范围，不增加训练步数、不追加完整 benchmark、不根据结果修改阈值或重采样。后续扩量、重复训练种子或真机验证属于下一轮工作。
