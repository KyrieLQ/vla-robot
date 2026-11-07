# 发布检查

2026-11-07 在 Windows 本地完成，使用 Python 3.12、PyTorch 2.5.1、torchvision 0.20.1。以下均为 CPU 或静态检查。

| 检查 | 结果 |
|---|---|
| 四种基础模型的合成输入前向 | 通过，输出分别为 `[2,1,7]` / `[2,10,7]`，全部数值有限 |
| 原有数据约定测试 | 5 项通过：episode 划分、训练统计、窗口边界、RGB 方向、padding mask |
| PiPER 采集离线测试 | CAN 解码、状态/目标区分、时间对齐、左右映射、RGB/BGR 行跨度与损坏样例检查通过 |
| 仓库完整性 | Python 语法、JSON 解析、793 个源文件哈希、入口文档链接通过 |
| 归档逐回合记录复算 | 2,000 个回合键唯一，1,565 次成功，与该模型的四套计数一致 |
| MP4 转换 | 解码得到 600 帧、50 FPS、12 秒，与原 WebM 时长和帧数一致 |
| 首页全彩预览 | 640×480 无损 WebP，150 帧、12 秒；逐帧与原视频每隔 4 帧抽取的 RGB 比较，像素差为 0 |

重新检查：

```bash
python examples/model_smoke.py
python tools/check_repository.py
python robots/piper/tests/test_collection.py
cd research
python -m unittest mini_vla.tests.test_contract -v
```

这些检查验证代码分发、离线数据链路和已保存评测记录；不重新运行训练、LIBERO 仿真或真机运动。

预览构建入口为 `tools/build_demo_preview.py`，需安装 PyAV 和支持 WebP 的 Pillow。预览为 12.5 FPS，完整视频为 50 FPS。
