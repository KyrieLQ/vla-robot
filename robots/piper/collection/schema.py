"""Onsite PiPER data contract shared by the listener and offline tools."""
import json
import os
import struct
import tempfile
from pathlib import Path

FORMAT = "piper-cylinder-1"
TASK = "put the cylinder into the box"
REPO_ID = "put_cylinder_into_box"
FPS = 20
HORIZON = 50
SIDES = ("left", "right")
PORTS = {"left": "1-13:1.0", "right": "1-12:1.0"}
CAMERAS = {"cam_high": "f", "cam_left_wrist": "l", "cam_right_wrist": "r"}
MODEL_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IDS = {"action": (0x155, 0x156, 0x157, 0x159), "state": (0x2A5, 0x2A6, 0x2A7, 0x2A8)}
JOINT_NAMES = [f"{side}_{joint}" for side in SIDES
               for joint in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")]
UNITS = (["rad"] * 6 + ["m"]) * 2
DATA_HOME = Path.home() / "data" / REPO_ID


def topic(cam):
    return f"/camera_{CAMERAS[cam]}/color/image_raw"


def frame_id(cam):
    return f"camera_{CAMERAS[cam]}_color_optical_frame"


def values(ident, payload):
    if len(payload) != 8:
        raise ValueError("Expected an eight-byte PiPER CAN frame")
    if ident in (0x159, 0x2A8):
        return (struct.unpack_from(">i", payload)[0] / 1_000_000,)
    if ident not in IDS["action"][:3] + IDS["state"][:3]:
        raise ValueError(f"Unexpected PiPER ID {ident:#x}")
    # Match the installed ROS driver's conversion, including its rounded rad/degree.
    return tuple(x / 1000 * 0.017444 for x in struct.unpack(">ii", payload))


def save_json(path, obj):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf8") as f:
        temp = Path(f.name)
        try:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        except BaseException:
            f.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def mapping_result(ranges, active):
    """The moving handle AND its follower must dominate the idle side for arm and grip."""
    other = next(s for s in SIDES if s != active)
    ok = True
    report = {}
    for side in SIDES:
        for source, ids in IDS.items():
            amplitudes = []
            for ident in ids:
                bounds = ranges.get((side, ident))
                if not bounds:
                    raise ValueError(f"缺少 {side} {source} 帧 {ident:#x}")
                lo, hi = bounds
                amplitudes.extend(y - x for x, y in zip(lo, hi))
            report[f"{side}_{source}"] = [max(amplitudes[:6]), amplitudes[6]]
    for source in IDS:
        a, b = report[f"{active}_{source}"], report[f"{other}_{source}"]
        ok &= a[0] >= 0.035 and a[1] >= 0.002 and a[0] > 1.4 * b[0] and a[1] > 1.4 * b[1]
    return {"active": active, "pass": bool(ok), "max_joint_rad_and_gripper_m": report}
