# 固定适配后VLM重新训练Flow头 训练验收

2026-10-30独立验收通过。按用户选定方案，固定M6配对LoRA endpoint骨干，新Flow头从seed42随机初始化；没有从旧Flow epoch43续训，没有继续更新VLM或LoRA。本阶段没有新增闭环成功率。

完整训练50epoch、94950更新，每轮243000帧及2362230个有效目标步；1506训练/187验证episode互斥。每轮完整30465验证帧/296235目标步，用固定噪声和10步Euler生成动作计算首动作MSE，选中epoch43。initial/best/latest、优化器/RNG、全部50条历史、选点和保存的完整预测通过独立核验。

| 项目 | 数值 |
|---|---:|
| 新Flow头参数 | 432711 |
| 本轮VLM/LoRA可训练参数 | 0 |
| 初始生成首动作MSE | 2.223075 |
| 新Flow选定首动作MSE | 0.500068 |
| 原冻结VLM Flow选定首动作MSE | 0.503268 |
| 全部流水线耗时 | 8.654小时 |
| 完整新特征提取 | 7.637小时 |
| 50轮训练及每轮验证 | 0.972小时 |

新旧Flow离线MSE相对差约0.64%，仍然较小，不能据此推断闭环成绩。新Flow与同骨干回归46/50的头部训练历史不同，不称严格等预算头部消融。

全部1693个原始Parquet重新计算SHA，输入序列hash与全273465帧receipt对应，计84665278080字节BF16特征。120个固定训练输入的保存特征、文件hash与重复前向门槛通过；模型加载无missing/unexpected/mismatched/error，只有新头进入训练。初始头与原Flow initial逐张量一致。旧冻结hidden没有作为新Flow特征输入。

**完整约78.85GiB特征仅曾存在于训练进程RAM，现已随进程结束释放。**当前可以独立复核全部source/receipt/输入hash及运行时训练前后全RAM hash断言，并读取保存的120门槛特征；不能声称现在重新读取了全部79GiB新特征。保存的initial/best/latest完整验证预测已独立复算，但此审核没有重新执行全部神经网络前向。

[独立验收JSON](review/independent_acceptance.json) · [训练与验证曲线](review/flow_training_validation.png)。本阶段无新视频；旧18/50和同适配骨干回归46/50均保留。

后继为原Spatial50闭环：epoch43新头、固定原适配骨干、K10/H5、原canonical reset与初态，在线噪声SHA256(20261020,task,init,call_index)，每次由CPUfloat32私有generator生成。先核对当前在线特征与保存的适配后门槛样本、同噪声手工Euler动作1e-6、阳性动作物理重放，再采集50集。新执行器保留逐调用noisehash、动作块和逐步物理日志；中断仅重放既有动作和剩余队列，不因结局重采，不重询已记录调用。

协议`configs/mini_vla_m7_fixed_lora_flow_dev50_20261030_v1.json`，在原共享GPU调度器登记。本次不重新创建完整RAM缓存、不追加训练或最终2000。时间继续累计，存储和共享GPU边界保持。训练事件在全部闭环验收交付后才标processed。
