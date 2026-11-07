# 上游与许可

本项目基于以下工作。第三方文件保留对应版权与许可证；各许可证适用于其所覆盖的内容。

| 项目 | 用途 | 版本 / 许可 |
|---|---|---|
| [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi) | π0.5、训练配置、策略服务与数据变换 | 研究基线 `215abfb217dbac7d5f1273282331b9b1866c0479`；[Apache-2.0](LICENSES/OpenPI-Apache-2.0.txt) |
| [AgileX / OpenPI](https://github.com/agilexrobotics/openpi-agilex) | PiPER 现场适配 | `robots/piper/openpi_overlay` 为现场关键文件快照；[Apache-2.0](robots/piper/openpi_overlay/LICENSE) |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 仿真任务与评测环境 | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`；[MIT](LICENSES/LIBERO-MIT.txt) |
| [LeRobot](https://github.com/huggingface/lerobot) | 机器人数据存储与读取 | 真机导出对应 v2.1 格式；现场依赖提交 `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| [PiPER ROS](https://github.com/agilexrobotics/piper_ros) | 双臂驱动依赖与消息接口 | ROS Noetic；由设备环境提供 |
| [PyTorch](https://github.com/pytorch/pytorch) / [torchvision](https://github.com/pytorch/vision) | 基础模型、训练与视觉编码 | 由各环境的依赖文件管理 |

OpenPI 研究基线的本地修改保存在 `research/reports/openpi_local.patch`。PiPER 适配快照和研究分支不是同一套运行环境，不能直接互相覆盖。数据集和预训练权重通过原发布渠道获取，并遵守各自的使用条款。
