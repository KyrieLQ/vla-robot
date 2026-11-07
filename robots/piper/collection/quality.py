"""Offline alignment and data-quality checks. No ROS, CAN, or model dependencies."""
import csv
import json
from itertools import combinations
from pathlib import Path
import zlib

import numpy as np
from PIL import Image

from schema import CAMERAS, FORMAT, FPS, HORIZON, IDS, JOINT_NAMES, PORTS, SIDES, TASK, UNITS
from schema import save_json, topic, values


def read_manifest(folder):
    meta = json.loads((folder / "capture.json").read_text(encoding="utf8"))
    contract = dict(format=FORMAT, task=TASK, fps=FPS, status="recorded", ports=PORTS,
                    action_source="external_leader_can", state_source="follower_feedback_can",
                    cameras={k: topic(k) for k in CAMERAS}, joint_names=JOINT_NAMES, units=UNITS)
    for key, expected in contract.items():
        if meta.get(key) != expected:
            raise ValueError(f"采集记录字段 {key} 未通过: {meta.get(key)!r}")
    for side in SIDES:
        if meta.get("mapping", {}).get(side, {}).get("pass") is not True:
            raise ValueError("缺少已通过的左右手柄映射检验: " + side)
    return meta


def select_samples(times, ticks, future):
    times = np.asarray(times, dtype=np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("来源无样本或时间戳不严格递增")
    indices = np.searchsorted(times, ticks, side="left" if future else "right")
    if not future:
        indices -= 1
    if np.any((indices < 0) | (indices >= len(times))):
        raise ValueError("采集边缘缺少覆盖全段的原始观测或后续动作")
    return indices, times[indices]


def checked_image(folder, cam, index):
    path = folder / cam / f"{int(index):06d}.jpg"
    with Image.open(path) as opened:
        if opened.mode != "RGB" or opened.size != (640, 480):
            raise ValueError(f"图像格式/尺寸错误: {path}")
        opened.load()
        return opened.copy()


def align(folder):
    meta = read_manifest(folder)
    start, end = int(meta["start_ns"]), int(meta["end_ns"])
    if not 0 < end - start <= 91_000_000_000:
        raise ValueError("正式录制区间无效")
    ticks = np.arange(start, end, 1_000_000_000 // FPS, dtype=np.int64)
    errors, notes = [], []
    report = dict(frames=len(ticks), fps=FPS, seconds=round((end - start) / 1e9, 3),
                  errors=errors, notes=notes, cameras={}, buses={}, motion={})
    if len(ticks) < HORIZON:
        errors.append(f"仅 {len(ticks)} 帧，少于一个 {HORIZON} 步动作块")
    images, stamps, image_signatures = {}, {}, {}
    static_views = set()
    for cam in CAMERAS:
        with (folder / cam / "index.csv").open(encoding="ascii", newline="") as f:
            rows = list(csv.DictReader(f))
        if [int(r["index"]) for r in rows] != list(range(len(rows))):
            raise ValueError(f"{cam} 文件索引不连续")
        headers = np.array([int(r["header_ns"]) for r in rows], dtype=np.int64)
        arrival = np.array([int(r["receive_ns"]) for r in rows], dtype=np.int64)
        try:
            index, selected = select_samples(headers, ticks, False)
        except ValueError as exc:
            raise ValueError(f"{cam}: {exc}") from exc
        stamps[cam], images[cam] = selected, index
        age = (ticks - selected) / 1e6
        delay = (arrival[index] - selected) / 1e6
        unique = np.unique(index)
        repeats = 1 - len(unique) / len(ticks)
        if age.max() > 75 or repeats > 0.10 or delay.min() < -20 or delay.max() > 250:
            errors.append(f"{cam} 有过期、重复过多或时钟异常的图像")
        signatures = {}
        max_contrast = 0.0
        for i in unique:
            img = checked_image(folder, cam, i)
            thumb = np.asarray(img.resize((40, 30)).convert("L"))
            max_contrast = max(max_contrast, float(thumb.std()))
            signatures[int(i)] = zlib.crc32(thumb.tobytes())
        image_signatures[cam] = np.asarray([signatures[int(i)] for i in index])
        if max_contrast < 2:
            errors.append(f"{cam} 整段近乎单色，请检查相机")
        if len(set(signatures.values())) == 1:
            static_views.add(cam)
        report["cameras"][cam] = dict(raw_in_take=int(((headers >= start) & (headers < end)).sum()),
            max_age_ms=round(float(age.max()), 2), p95_age_ms=round(float(np.percentile(age, 95)), 2),
            repeated_pct=round(100 * repeats, 2), max_receipt_delay_ms=round(float(delay.max()), 2))
    for a, b in combinations(CAMERAS, 2):
        if np.array_equal(image_signatures[a], image_signatures[b]):
            errors.append(f"{a} 和 {b} 画面整段相同，核对相机映射")
    spread = np.ptp(np.stack(list(stamps.values())), axis=0) / 1e6
    report["max_view_span_ms"] = round(float(spread.max()), 2)
    if spread.max() > 50:
        errors.append("三路图像的最大时间跨度超过 50 ms")

    vectors = {source: [] for source in IDS}
    for side in SIDES:
        raw = {ident: ([], []) for ids in IDS.values() for ident in ids}
        with (folder / ("can_" + side + ".csv")).open(encoding="ascii", newline="") as f:
            for row in csv.DictReader(f):
                ident = int(row["id"], 16)
                t, v = raw[ident]
                t.append(int(row["kernel_ns"]))
                v.append(values(ident, bytes.fromhex(row["bytes"])))
        for source, ids in IDS.items():
            selected_stamps = []
            columns = []
            for ident in ids:
                times, samples = raw[ident]
                try:
                    index, selected = select_samples(times, ticks, source == "action")
                except ValueError as exc:
                    raise ValueError(f"{side} {source} {ident:#x}: {exc}") from exc
                selected_stamps.append(selected)
                columns.append(np.asarray(samples, dtype=np.float32)[index])
            matrix = np.concatenate(columns, axis=1)
            vectors[source].append(matrix)
            key = f"{side}_{source}"
            timeline = np.stack(selected_stamps, axis=1)
            stamps[key] = timeline
            age_ms = np.abs(timeline - ticks[:, None]) / 1e6
            skew = np.ptp(timeline, axis=1) / 1e6
            report["buses"][key] = {"max_age_ms": round(float(age_ms.max()), 2),
                                    "max_component_span_ms": round(float(skew.max()), 2)}
            period_ms = 1000 / FPS
            if age_ms.max() > period_ms or skew.max() > period_ms:
                errors.append(f"{key} 指令或反馈中断/组帧过旧（超过一个 {period_ms:g} ms 采样周期）")
            elif age_ms.max() > period_ms / 2 or skew.max() > 20:
                notes.append(f"{key} 的部分样本接近一个采样周期，请参阅 qa.json 中的实际时差")
            if not np.isfinite(matrix).all() or np.abs(matrix[:, :6]).max() > 2 * np.pi or np.abs(matrix[:, 6]).max() > .12:
                errors.append(f"{key} 数值或单位异常")
    data = {k: np.concatenate(v, axis=1) for k, v in vectors.items()}
    active = []
    for n, side in enumerate(SIDES):
        arm = slice(n * 7, (n + 1) * 7)
        measured, target = data["state"][:, arm], data["action"][:, arm]
        metrics = {}
        for source, array in (("state", measured), ("action", target)):
            extent = np.ptp(array, axis=0)
            metrics[source + "_joint_rad"] = round(float(extent[:6].max()), 5)
            metrics[source + "_grip_mm"] = round(float(extent[6]) * 1000, 3)
        metrics["p95_joint_tracking_deg"] = round(float(np.rad2deg(np.percentile(
            np.max(np.abs(measured[:, :6] - target[:, :6]), axis=1), 95))), 2)
        metrics["p95_grip_tracking_mm"] = round(float(np.percentile(np.abs(measured[:, 6] - target[:, 6]), 95)) * 1000, 2)
        report["motion"][side] = metrics
        if all(metrics[k + "_joint_rad"] >= .035 and metrics[k + "_grip_mm"] >= 2 for k in IDS):
            active.append(side)
            if np.array_equal(measured, target) or (len(ticks) > 1 and np.array_equal(measured[1:], target[:-1])):
                errors.append(f"{side} 活动臂整段动作逐值复制了同期或下一帧反馈，需复查标签来源")
    if not active:
        errors.append("未发现同一侧同时有主从关节运动及夹爪开合")
    expected_motion = {"cam_high"} | {f"cam_{side}_wrist" for side in active}
    for cam in sorted(static_views):
        if cam in expected_motion:
            errors.append(f"{cam} 在示教期间画面完全静止，请检查图像是否冻结")
        else:
            notes.append(f"{cam} 位于静止侧，画面全程相同，请在回看中核对")
    report["active_sides"] = active
    report["technical_pass"] = not errors
    return dict(folder=folder, ticks=ticks, images=images, stamps=stamps, **data), report


def write_review(folder, data, report):
    rows = np.stack(list(data["images"].values()), axis=1).tolist()
    keys = json.dumps(list(CAMERAS))
    grippers = np.stack([data[source][:, i] for i in (6, 13) for source in ("action", "state")], axis=1)
    grippers_mm = (grippers.astype(np.float64) * 1000).round(1).tolist()
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>圆柱入盒 · 示范回看</title>
<style>body{font:16px system-ui;margin:2rem;background:#16202a;color:#f3f5f6}main{display:flex;gap:1%}
figure{margin:0;flex:1;min-width:0}img{width:100%;display:block}input{width:65%}button{padding:.45rem 1rem}
figcaption,p{margin:.65rem 0}output{font-variant-numeric:tabular-nums}</style>
<h1>圆柱入盒 · 示范回看</h1><p>STATUS</p><main><figure><img><figcaption>前视</figcaption></figure>
<figure><img><figcaption>左腕</figcaption></figure><figure><img><figcaption>右腕</figcaption></figure></main>
<p><button id="play">播放</button> <input id="slider" type="range" min="0" value="0"> <output></output></p>
<p id="grip"></p>
<p>检查选中圆柱、离桌、入盒、松爪后留在盒中，以及三视角的左右对应。</p>
<script>const rows=ROWS, keys=KEYS, fps=FPS, grips=GRIPS; const bar=document.querySelector('#slider'), button=document.querySelector('#play');
let timer=null; bar.max=rows.length-1; function show(i){bar.value=i;document.querySelectorAll('img').forEach((im,k)=>{
im.src=keys[k]+'/'+String(rows[i][k]).padStart(6,'0')+'.jpg'});document.querySelector('output').textContent=
(i+1)+' / '+rows.length+' · '+(i/fps).toFixed(2)+' s';const [la,ls,ra,rs]=grips[i];
document.querySelector('#grip').textContent='左夹爪 目标 '+la+' / 实际 '+ls+' mm　　右夹爪 目标 '+ra+' / 实际 '+rs+' mm'}
function stop(){clearInterval(timer);timer=null;button.textContent='播放'}
bar.oninput=()=>{stop();show(+bar.value)}; button.onclick=()=>{if(timer){stop();return}
if(+bar.value===rows.length-1)show(0);button.textContent='暂停';timer=setInterval(()=>{const n=+bar.value+1;
if(n>=rows.length){stop();return}show(n)},1000/fps)}; show(0)</script></html>"""
    status = "数据检查通过，任务结果待目视确认。" if report["technical_pass"] else "数据检查未通过，详见 qa.json。"
    page = page.replace("STATUS", status).replace("ROWS", json.dumps(rows)).replace("GRIPS", json.dumps(grippers_mm))
    (folder / "review.html").write_text(page.replace("KEYS", keys).replace("FPS", str(FPS)), encoding="utf8")
    # Keep raw source stamps separately from the nominal LeRobot grid.
    np.savez_compressed(folder / "alignment.npz", tick_ns=data["ticks"],
                        **{k + "_ns": v for k, v in data["stamps"].items()})


def inspect_take(folder):
    manifest = json.loads((folder / "capture.json").read_text(encoding="utf8"))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("请选择本工具生成的原始 take 目录")
    try:
        data, report = align(folder)
        write_review(folder, data, report)
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        data, report = None, {"technical_pass": False, "errors": [str(exc)]}
        # Prevent a stale successful preview from surviving a subsequently invalid take.
        (folder / "review.html").unlink(missing_ok=True)
        (folder / "alignment.npz").unlink(missing_ok=True)
    save_json(folder / "qa.json", report)
    print(("PASS_DATA" if report["technical_pass"] else "FAIL_DATA"), folder, flush=True)
    for msg in report.get("errors", []) + report.get("notes", []):
        print("  " + msg)
    return data, report
