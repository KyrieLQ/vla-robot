# PiPER 主从示范与 π0.5 接口

本目录包含 PiPER 主从平台的采集工具，以及现场 OpenPI 项目的关键适配文件。相机和 CAN 映射来自该台设备，换设备时先核对 `collection/schema.py` 与 `collection/site_sensors.py`。

## 数据流

```text
主臂手柄 → 从臂运动
    ├─ 主臂绝对目标 ──────────→ action[14]
    ├─ 从臂实际反馈 ──────────→ observation.state[14]
    └─ 前视 + 左腕 + 右腕 RGB → 三视角观测
             ↓
       原始记录 → 时间对齐 → 三视角回看 → LeRobot → OpenPI 输入检查
```

采集器通过 SocketCAN 接收目标与反馈，通过 ROS 订阅图像。`capture.py` 不发布运动指令；手动主从跟随由平台自身的示教链路完成。

## 环境与映射

- Linux、ROS Noetic、设备现有相机与 PiPER 驱动。
- 系统 Python 用于 ROS 接收；现场 OpenPI 虚拟环境用于 LeRobot 导出与回读。
- `collection/demo.sh` 默认 OpenPI 位于 `$HOME/xmh/openpi-agilex`，实际位置不同需修改该文件。
- `site_sensors.py` 使用现场工作区、设备序列号和驱动文件校验值；它是该设备的启动配置，不是通用 USB 自动识别器。

| 字段 | 形状 / 单位 | 来源与顺序 |
|---|---|---|
| `observation.images.cam_high` | RGB，640×480 | 前视 `/camera_f/color/image_raw` |
| `observation.images.cam_left_wrist` | RGB，640×480 | 左腕 `/camera_l/color/image_raw` |
| `observation.images.cam_right_wrist` | RGB，640×480 | 右腕 `/camera_r/color/image_raw` |
| `observation.state` | 14，关节 rad / 夹爪 m | 左 J1–J6、左夹爪、右 J1–J6、右夹爪的反馈 |
| `action` | 14，关节 rad / 夹爪 m | 同一顺序的主臂绝对目标 |
| `task` | 文本 | 示范的语言指令 |

主臂目标 CAN ID 为 `0x155/156/157/159`，反馈为 `0x2A5/2A6/2A7/2A8`。当前实现沿用现场驱动的关节换算常数 `0.017444 rad/degree`；若改用精确弧度换算，应统一更新采集、统计与控制变换，避免混用。

本采集任务使用 **20 Hz**，取采样时刻之前最近的图像/状态，并为各动作分量取该时刻或之后的第一条主臂目标。原始 CAN 接收时间、图像头时间和接收时间均保留；图像帧龄、跨视角差和 CAN 组帧跨度在质量报告中检查。50 步动作块对应 2.5 秒。

## 第一条示范

在已配置并可正常手动主从跟随的设备上，准备两个终端。

```bash
# 终端 A：启动该台设备的相机和反馈配置
cd vla-robot/robots/piper/collection
bash demo.sh sensors

# 终端 B：环境检查、左右映射检查、录制、回看、导出
cd vla-robot/robots/piper/collection
bash demo.sh first
```

按提示分别小幅操作左、右主臂并开合夹爪，确认映射。摆好道具后回车开始；用主臂完成抓取、移动、释放，停止片刻后回车结束。程序打开三视角回看，确认任务完成后输入 `yes`，生成 LeRobot 数据并回读。

当前示例任务为 `put the cylinder into the box`；任务文本、数据集 ID 和输出目录定义于 `schema.py`。采集其他任务时同时修改这些约定。

| 检查 | 验收内容 |
|---|---|
| `PASS_DATA` | 图像解码、时间对齐、左右独立目标与反馈、关节/夹爪变化、完整动作窗口 |
| 人工回看确认 | 抓取对象正确、物体离桌、释放后留在容器内、三视角映射正确 |
| `PASS_LOADER` | LeRobot 重新打开，状态、图像、任务与连续动作逐值复核，OpenPI 输入变换通过 |

原始记录保存到 `$HOME/data/put_cylinder_into_box/raw/take_…/`，导出保存到 `exports/build_…/`。保留原始记录；训练集与验证集按整条示范划分，用训练部分计算本任务归一化统计。

也可单独运行：

```bash
bash demo.sh doctor
bash demo.sh record
bash demo.sh review /absolute/path/to/take
bash demo.sh export --reviewed-success /absolute/path/to/take
bash demo.sh verify /absolute/path/to/build
```

## 模型适配文件

`openpi_overlay/` 是 2026-11-06 现场项目的**关键文件快照**，需要与兼容的 [AgileX OpenPI 项目](https://github.com/agilexrobotics/openpi-agilex) 配合，不是完整 Python 包。现场分支的提交号未被固定，部署前应锁定完整依赖版本。

| 文件 | 作用 |
|---|---|
| [convert_cobot_data_to_lerobot.py](openpi_overlay/examples/aloha_real/convert_cobot_data_to_lerobot.py) | 旧 HDF5 到 LeRobot 转换实现；新采集使用 `collection/dataset.py` |
| [aloha_policy.py](openpi_overlay/src/openpi/policies/aloha_policy.py) | 三视角、状态、夹爪与动作输入输出变换 |
| [training/config.py](openpi_overlay/src/openpi/training/config.py) | 数据映射、关节 delta 变换、LoRA 与任务配置 |
| [compute_norm_stats.py](openpi_overlay/scripts/compute_norm_stats.py) | 训练数据归一化统计 |
| [serve_policy.py](openpi_overlay/scripts/serve_policy.py) | 模型加载与动作块服务 |
| [aloha_inference-joint-ros2.py](openpi_overlay/scripts/aloha_inference-joint-ros2.py) | 现场运动客户端，读取观测、请求动作并发布关节命令 |

最后一个文件虽名为 `ros2`，实际使用 `rospy`（ROS 1）。它包含固定准备位和真实运动发布逻辑，执行即可能发送动作；阅读接口时应静态查看文件。数据输入将两侧六个关节转换为相对当前状态的增量，夹爪保留绝对目标。

离线合成记录已用于验证采集解码、质量检查和锁定版本的 LeRobot v2.1 写入/回读。首条现场示范仍需按上面的数据与任务结果验收，再纳入自己的微调数据集。

不连接设备即可复核采集与质检逻辑，在仓库根目录运行：

```bash
python robots/piper/tests/test_collection.py
```

该测试使用合成 CAN 与 RGB 记录，覆盖缺帧、冻结数据、左右视角重复、图像行跨度和不完整录制的拒绝。安装兼容的 LeRobot 后，追加 `--native` 可验证真实数据写入器与读取器；这也使用合成记录和 CPU。
