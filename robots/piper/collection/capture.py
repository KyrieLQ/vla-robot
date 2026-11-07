"""Physical leader demonstrations. Two receive-only CAN sockets and three ROS Image subscriptions."""
import argparse
import csv
from datetime import datetime
import select
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

from schema import CAMERAS, DATA_HOME, FORMAT, FPS, IDS, JOINT_NAMES, PORTS, SIDES, TASK, UNITS
from schema import frame_id, mapping_result, save_json, topic, values


def camera_image(msg, cam):
    from PIL import Image
    if (msg.width, msg.height) != (640, 480) or msg.header.frame_id.lstrip("/") != frame_id(cam):
        raise ValueError(f"{cam}: 意外的尺寸/坐标系 {msg.width}x{msg.height} {msg.header.frame_id}")
    encoding = {"rgb8": "RGB", "bgr8": "BGR"}.get(msg.encoding)
    if not encoding or msg.step < msg.width * 3 or len(msg.data) != msg.height * msg.step:
        raise ValueError(f"{cam}: 不支持的图像编码或行长度 {msg.encoding}/{msg.step}")
    return Image.frombytes("RGB", (msg.width, msg.height), msg.data, "raw", encoding, msg.step, 1)


class Listener:
    def __init__(self, folder):
        self.folder = folder
        self.stop = threading.Event()
        self.save_images = threading.Event()
        self.lock = threading.Lock()
        self.last = {}
        self.bounds = None
        self.errors = []
        self.threads = []
        self.subs = []
        self.camera_locks = {cam: threading.Lock() for cam in CAMERAS}
        self.camera_files = {}
        self.camera_writers = {}
        self.image_counts = dict.fromkeys(CAMERAS, 0)
        self.publishers = {}

    def start(self):
        import rosgraph
        import rospy
        from sensor_msgs.msg import Image
        try:
            rosgraph.Master("/cylinder_demo_probe").getPid()
        except OSError as exc:
            raise RuntimeError("本机 ROS master 未响应。先启动相机/反馈窗口，等待 READY") from exc
        rospy.init_node("cylinder_demo_listener", anonymous=True, disable_rosout=True, disable_signals=True)
        if rospy.get_param("/use_sim_time", False):
            raise RuntimeError("实机采集需要主机实时时钟，当前 /use_sim_time 已启用")
        advertised, subscriptions, _ = rosgraph.Master(rospy.get_name()).getSystemState()
        for path, nodes in advertised:
            if path in ("/master/joint_left", "/master/joint_right"):
                for node in nodes:
                    if rospy.get_param(node + "/mode", None) != 0:
                        raise RuntimeError(f"{path} 存在非 mode=0 的发布源 {node}，请恢复现场手动示教设置")
        for path, nodes in subscriptions:
            if path in ("/master/joint_left", "/master/joint_right"):
                if any(rospy.get_param(node + "/mode", None) == 1 for node in nodes):
                    raise RuntimeError("检测到外部关节控制驱动。请使用现场硬件主从示教模式")
        for cam in CAMERAS:
            nodes = dict(advertised).get(topic(cam), [])
            if len(nodes) != 1:
                raise RuntimeError(f"{topic(cam)} 需要恰好一个相机发布源，实际 {nodes}")
            self.publishers[cam] = nodes[0]
            path = self.folder / cam
            path.mkdir()
            f = (path / "index.csv").open("x", newline="", encoding="ascii")
            self.camera_files[cam] = f
            self.camera_writers[cam] = csv.writer(f)
            self.camera_writers[cam].writerow(("index", "header_ns", "receive_ns"))
            self.subs.append(rospy.Subscriber(topic(cam), Image, self.on_image, callback_args=cam,
                                              queue_size=2, buff_size=4 * 1024 * 1024))
        for side in SIDES:
            thread = threading.Thread(target=self.on_bus, args=(side,), name="can_" + side)
            self.threads.append(thread)
            thread.start()

    def on_image(self, msg, cam):
        arrival = time.time_ns()
        with self.camera_locks[cam]:
            if self.stop.is_set():
                return
            try:
                stamp = msg.header.stamp.to_nsec()
                if not -20_000_000 <= arrival - stamp <= 250_000_000:
                    if not self.save_images.is_set():
                        return  # Allow stale camera frames to drain during warm-up.
                    raise RuntimeError(f"{cam} 图像时钟和主机接收时钟相差 {(arrival-stamp)/1e6:.1f} ms")
                caller = (getattr(msg, "_connection_header", None) or {}).get("callerid")
                if caller != self.publishers[cam]:
                    raise RuntimeError(f"{cam} 相机发布源发生变化: {caller}")
                img = camera_image(msg, cam)
                if self.save_images.is_set():
                    i = self.image_counts[cam]
                    img.save(self.folder / cam / f"{i:06d}.jpg", "JPEG", quality=92, subsampling=0)
                    self.camera_writers[cam].writerow((i, stamp, arrival))
                    self.image_counts[cam] += 1
                with self.lock:
                    self.last[cam] = time.monotonic()
            except Exception as exc:
                self.fail(exc)

    def fail(self, exc):
        with self.lock:
            self.errors.append(str(exc))
        self.stop.set()

    def on_bus(self, side):
        name = "can_" + side
        try:
            if (Path("/sys/class/net") / name / "device").resolve().name != PORTS[side]:
                raise RuntimeError(f"{name} 缺失或 USB 映射改变，应为 {PORTS[side]}")
            ids = IDS["action"] + IDS["state"]
            with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock, (
                    self.folder / (name + ".csv")).open("x", encoding="ascii", newline="") as f:
                # Exact SFF ID and RTR flags. CAN_ERR_FLAG in the mask would select error frames.
                sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                                b"".join(struct.pack("=II", ident, 0xC00007FF) for ident in ids))
                if struct.calcsize("@ll") != 16:
                    raise RuntimeError("此版本针对现场 64 位 Linux 的内核时间戳")
                ts_option = getattr(socket, "SO_TIMESTAMPNS", 35)
                sock.setsockopt(socket.SOL_SOCKET, ts_option, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                sock.bind((name,))
                sock.settimeout(0.15)
                writer = csv.writer(f)
                writer.writerow(("kernel_ns", "id", "bytes"))
                while not self.stop.is_set():
                    try:
                        packet, ancillary, flags, _ = sock.recvmsg(16, socket.CMSG_SPACE(16))
                    except socket.timeout:
                        continue
                    ident, length = struct.unpack_from("=IB", packet)
                    if len(packet) != 16 or ident not in ids or length != 8:
                        raise ValueError(f"{name}: 意外 CAN 帧 {ident:#x}/{length}")
                    if flags & socket.MSG_DONTROUTE:
                        if ident in IDS["action"]:
                            raise RuntimeError(f"{name} 收到控制机本地发出的动作，不能标为物理主臂示范")
                        continue
                    clock = next((c for level, typ, c in ancillary
                                  if level == socket.SOL_SOCKET and typ == ts_option), None)
                    if clock is None or flags & socket.MSG_CTRUNC:
                        raise RuntimeError(f"{name}: 缺内核接收时间戳")
                    sec, nsec = struct.unpack_from("@ll", clock)
                    stamp = sec * 1_000_000_000 + nsec
                    payload = packet[8:]
                    writer.writerow((stamp, hex(ident), payload.hex()))
                    vals = values(ident, payload)
                    with self.lock:
                        key = (side, ident)
                        self.last[key] = time.monotonic()
                        if self.bounds is not None:
                            lo, hi = self.bounds.setdefault(key, (list(vals), list(vals)))
                            for i, value in enumerate(vals):
                                lo[i] = min(lo[i], value)
                                hi[i] = max(hi[i], value)
        except Exception as exc:
            self.fail(exc)

    def check(self):
        if self.stop.is_set():
            raise RuntimeError("; ".join(self.errors) or "采集已中止")

    def wait(self, seconds):
        self.stop.wait(seconds)
        self.check()

    def wait_enter(self, text, timeout=300):
        print(text, flush=True)
        start = time.monotonic()
        while True:
            self.check()
            if timeout is not None and time.monotonic() - start >= timeout:
                raise RuntimeError("达到时限，已保留原始数据，本条未标记完成")
            if select.select([sys.stdin], [], [], 0.15)[0]:
                if not sys.stdin.readline():
                    raise RuntimeError("终端已断开")
                return

    def wait_for_streams(self):
        keys = list(CAMERAS) + [(s, i) for s in SIDES for ids in IDS.values() for i in ids]
        for second in range(20):
            self.wait(1)
            now = time.monotonic()
            with self.lock:
                missing = [k for k in keys if now - self.last.get(k, -1e20) > 0.5]
            if not missing:
                return
            if second % 4 == 0:
                print("等待:", ", ".join(f"{k[0]} {k[1]:#x}" if isinstance(k, tuple) else k
                                         for k in missing), flush=True)
        raise RuntimeError("数据流不全。只有 0x2a* 时，请检查硬件主从跟随及主臂指令帧")

    def map_handles(self):
        result = {}
        for side, name in zip(SIDES, ("左", "右")):
            self.wait_enter(f"只测试{name}主臂：回车后有 7 秒。小幅移动手柄、张合夹爪，另一侧静止。")
            with self.lock:
                self.bounds = {}
            self.wait(7)
            with self.lock:
                bounds, self.bounds = self.bounds, None
            result[side] = mapping_result(bounds, side)
            print("目标与跟随的幅度 [rad, m]:", result[side]["max_joint_rad_and_gripper_m"], flush=True)
            if not result[side]["pass"]:
                raise RuntimeError(f"{name}侧对应关系或夹爪变化未通过，请核对实际跟随、总线与手柄后重试")
        self.wait_enter("目视确认左右各自的从臂、夹爪与同侧手柄对应后回车；不对应请 Ctrl+C。")
        return result

    def close(self):
        self.stop.set()
        for sub in self.subs:
            sub.unregister()
        for thread in self.threads:
            thread.join()
        for cam, f in self.camera_files.items():
            with self.camera_locks[cam]:
                f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_HOME / "raw")
    parser.add_argument("--receipt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not sys.platform.startswith("linux") or not sys.stdin.isatty():
        raise SystemExit("在机器人 Ubuntu 的交互终端运行采集脚本")
    from PIL import Image  # Resolve image encoding before starting a take.
    Image.init()
    args.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.root).free < 2 * 1024**3:
        raise SystemExit("原始数据目录至少需 2 GiB 可用空间")
    folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("take_%Y%m%d_%H%M%S_"), dir=args.root)).resolve()
    manifest = dict(format=FORMAT, task=TASK, fps=FPS, status="incomplete", ports=PORTS,
                    joint_names=JOINT_NAMES, units=UNITS, action_source="external_leader_can",
                    state_source="follower_feedback_can", cameras={cam: topic(cam) for cam in CAMERAS})
    save_json(folder / "capture.json", manifest)
    listen = Listener(folder)
    print("本条目录:", folder, "\n只监听相机与 CAN；请先让设备进入已配对的手动主从示教。", flush=True)
    try:
        listen.start()
        listen.wait_for_streams()
        manifest["mapping"] = listen.map_handles()
        manifest["camera_publishers"] = listen.publishers
        listen.wait_enter("把圆柱和盒子摆好（可放同色方块作干扰），回到主臂。按回车后倒数 3 秒开始本条。")
        listen.save_images.set()
        for number in (3, 2, 1):
            print(number, flush=True)
            listen.wait(1)
        manifest["start_ns"] = time.time_ns()
        listen.wait_enter("● 正在录制。放开圆柱、撤开夹爪、停一秒后回车结束（最长 90 秒）。", timeout=90)
        manifest["end_ns"] = time.time_ns()
        listen.wait(0.20)  # Future command samples for the last observation.
    except (Exception, KeyboardInterrupt) as exc:
        manifest["error"] = str(exc) or "操作员中断"
    finally:
        listen.close()
        if listen.errors:
            manifest["error"] = "; ".join(listen.errors)
        if "error" not in manifest and "end_ns" in manifest:
            manifest["status"] = "recorded"
        save_json(folder / "capture.json", manifest)
    if manifest["status"] != "recorded":
        raise SystemExit("未完成: " + manifest.get("error", "录制中断"))
    if args.receipt:
        args.receipt.write_text(str(folder) + "\n", encoding="utf8")
    print("RECORDED", folder, flush=True)
    print(f'下一步: bash demo.sh review "{folder}"')


if __name__ == "__main__":
    main()
