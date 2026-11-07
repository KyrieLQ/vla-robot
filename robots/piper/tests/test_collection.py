"""Offline regression for this robot's pilot kit. Synthetic inputs only; no ROS or CAN."""
import argparse
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from types import SimpleNamespace

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / "collection"))
import numpy as np
from PIL import Image, ImageDraw
from capture import camera_image
import dataset as ds
from quality import inspect_take
from schema import *

T0 = 1_800_000_000_000_000_000
END = 3_300_000_000


def buses(path, moving, drop=None, freeze=False, copy=False, interval_ms=5):
    for side in SIDES:
        with (path / f"can_{side}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("kernel_ns", "id", "bytes"))
            for ms in range(-100, 3501, interval_ms):
                for source, ids in IDS.items():
                    for j, ident in enumerate(ids):
                        if ident == drop or (freeze and source == "action" and 1000 <= ms <= 1240):
                            continue
                        t = ms / 1000 - (.045 if source == "state" and not copy else 0)
                        swing = math.sin(t * 1.8) if side == moving else 0
                        vals = (round(23000 * swing) + 100 * j, round(15000 * swing) - j)
                        if j == 3:
                            vals = (round(30000 + 12000 * swing), 0)
                        stamp = T0 + ms * 1_000_000 + (0 if copy else 270_000 * j)
                        w.writerow((stamp, hex(ident), struct.pack(">ii", *vals).hex()))


def make_take(path, moving):
    path.mkdir()
    manifest = dict(format=FORMAT, task=TASK, fps=FPS, status="recorded", ports=PORTS,
                    action_source="external_leader_can", state_source="follower_feedback_can",
                    mapping={s: {"pass": True} for s in SIDES}, start_ns=T0, end_ns=T0+END,
                    joint_names=JOINT_NAMES, units=UNITS, cameras={c:topic(c) for c in CAMERAS})
    save_json(path/"capture.json", manifest)
    for c, cam in enumerate(CAMERAS):
        sub = path / cam
        sub.mkdir()
        with (sub/"index.csv").open("w", newline="") as f:
            writer=csv.writer(f)
            writer.writerow(("index", "header_ns", "receive_ns"))
            for idx in range(108):
                stamp = T0 + (idx-2)*33_333_333 + c*500_000
                writer.writerow((idx, stamp, stamp+3_000_000))
                image = Image.new("RGB",(640,480),(20+30*c,60,115))
                draw = ImageDraw.Draw(image)
                x = 21+idx+c*69
                draw.rectangle((x,100,x+95,200+c*30), fill=(255,38+30*c,60))
                image.save(sub/f"{idx:06d}.jpg", quality=92)
    buses(path, moving)


def get_status(folder, expected, error_substr=None):
    with contextlib.redirect_stdout(io.StringIO()):
        d,r = inspect_take(folder)
    assert r["technical_pass"] is expected,r
    if error_substr:
        assert any(error_substr in s for s in r["errors"]), r
    return d


def exercise(raw):
    make_take(raw, "left")
    valid = get_status(raw,True)
    assert valid["state"].shape == valid["action"].shape == (66,14)
    assert valid["action"].dtype == valid["state"].dtype == np.float32
    assert not np.array_equal(valid["state"][:, :7], valid["action"][:, :7])
    np.testing.assert_allclose(valid["action"][:, 13], 0.03)
    assert .015 < valid["action"][:, 6].min() < .020
    with np.load(raw / "alignment.npz") as aligned:
        for cam in CAMERAS:
            assert (aligned[cam+"_ns"]<=aligned["tick_ns"]).all()
        for side in SIDES:
            assert (aligned[side+"_action_ns"]>=aligned["tick_ns"][:,None]).all()
            assert (aligned[side+"_state_ns"]<=aligned["tick_ns"][:,None]).all()
    page = (raw / "review.html").read_text(encoding="utf8")
    assert "slider" in page and "左夹爪 目标" in page and "GRIPS" not in page

    for kwargs, phrase in (({"drop":0x155},"0x155"), ({"freeze":True},"组帧过旧"), ({"copy":True},"复制")):
        buses(raw, "left", **kwargs)
        get_status(raw,False,phrase)
    buses(raw, "left", interval_ms=40)
    get_status(raw,True)
    buses(raw, "left")
    idx = raw/"cam_high/index.csv"
    good_index = idx.read_text()
    lines = good_index.splitlines()
    stale = lines.copy()
    for k in range(30,37):
        i,t,a=lines[k].split(",")
        stale[k] = ",".join((i,str(int(t)-150_000_000),a))
    idx.write_text("\n".join(stale))
    get_status(raw,False,"递增")
    idx.write_text(good_index)
    blank = raw.parent/"one-white.jpg"
    Image.new("RGB",(640,480),"white").save(blank)
    other = raw/"cam_right_wrist"
    bak = raw.parent/"saved-right"
    other.rename(bak)
    shutil.copytree(raw/"cam_high",other)
    get_status(raw,False,"整段相同")
    for f in other.glob("*.jpg"):
        shutil.copyfile(blank, f)
    get_status(raw,False,"近乎单色")
    assert other.resolve().is_relative_to(raw.resolve()), "Fixture escaped test directory"
    shutil.rmtree(other)
    bak.rename(other)

    capture = raw/"capture.json"
    meta = json.loads(capture.read_text(encoding="utf8"))
    save_json(capture,dict(meta,status="incomplete"))
    get_status(raw,False,"status")
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            ds.load_takes([raw])
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete capture eligible for export")
    save_json(capture,meta)
    foreign = raw.parent/"foreign"
    foreign.mkdir()
    (foreign/"review.html").write_text("keep")
    try:
        inspect_take(foreign)
    except FileNotFoundError:
        assert (foreign/"review.html").read_text()=="keep" and not (foreign/"qa.json").exists()
    else:
        raise AssertionError("foreign directory accepted")
    get_status(raw,True)
    for side in SIDES:
        vals={}
        for s in SIDES:
            for group in IDS.values():
                for n,ident in enumerate(group):
                    m=1 if n==3 else 2
                    vals[s,ident] = ([0.]*m, ([.008 if n==3 else .08]*m) if s==side else [0.]*m)
        assert mapping_result(vals,side)["pass"]
        assert not mapping_result(vals,next(s for s in SIDES if s!=side))["pass"]
        for s in SIDES:
            vals[s,0x159]=([0.],[0.])
        assert not mapping_result(vals,side)["pass"]
    # Check row padding and both ROS color encodings.
    for enc, px in (("rgb8",b"\xfa\x05\x09"), ("bgr8",b"\x09\x05\xfa")):
        msg=SimpleNamespace(width=640,height=480,header=SimpleNamespace(frame_id=frame_id("cam_high")),
                            encoding=enc,step=1924,data=(px*640+b"\x00"*4)*480)
        img=camera_image(msg,"cam_high")
        assert img.getpixel((639,479))==(250,5,9)
    assert abs(values(0x155,struct.pack(">ii",-1000,2000))[0]+.017444)<1e-10
    print("PASS offline capture decoding, mapping, synchronization, hand review and rejected corrupt fixtures")
    return valid


def native(temp, first, second):
    os.environ["HF_HOME"]=str(temp/"hf_cache")
    os.environ["HF_DATASETS_CACHE"]=str(temp/"hf_cache/datasets")
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"]="1"
    ds.offline()
    import torch
    torch.set_num_threads(1)
    root = temp/"dataset"
    ds.write_native(root,[first,second])
    def check_types(sample):
        assert tuple(sample["action"].shape)==(50,14) and str(sample["action"].dtype)=="torch.float32"
        assert str(sample["observation.state"].dtype)=="torch.float32"
        assert all(tuple(sample["observation.images."+c].shape)==(3,480,640) for c in CAMERAS)
    result=ds.verify_native(root,[first,second],policy_check=check_types)
    assert result["episodes"]==2 and result["frames"]==132 and result["un-padded_windows_checked"]==6
    print("PASS real pinned LeRobot write/read (CPU, synthetic data)", result)
    print("FORMAT", json.loads((root/"meta/info.json").read_text())["codebase_version"])


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf8",errors="backslashreplace")
    parser=argparse.ArgumentParser()
    parser.add_argument("--native",action="store_true")
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(dir=WORK,prefix="synthetic_cylinder_") as tmp:
        t=Path(tmp)
        first=exercise(t/"take_left")
        if args.native:
            make_take(t/"take_right","right")
            native(t,first,get_status(t/"take_right",True))
