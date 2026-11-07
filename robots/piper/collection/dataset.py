"""Review raw demonstrations and build a new, offline LeRobot dataset."""
import argparse
import inspect
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from quality import checked_image, inspect_take
from schema import CAMERAS, DATA_HOME, FPS, HORIZON, JOINT_NAMES, MODEL_KEYS, REPO_ID, TASK, save_json


def offline():
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", HF_HUB_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")


def native_dataset():
    offline()
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def features():
    spec = {key: {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES.copy()}
            for key in ("observation.state", "action")}
    for cam in CAMERAS:
        spec[f"observation.images.{cam}"] = {"dtype": "image", "shape": (3, 480, 640),
                                             "names": ["channels", "height", "width"]}
    return spec


def write_native(root, takes):
    cls = native_dataset()
    writer = cls.create(repo_id=REPO_ID, root=root, robot_type="piper_bimanual", fps=FPS,
                        features=features(), use_videos=False,
                        image_writer_processes=0, image_writer_threads=0)
    old_task_api = "task" in inspect.signature(writer.save_episode).parameters
    for d in takes:
        for i in range(len(d["ticks"])):
            row = {"observation.state": d["state"][i], "action": d["action"][i]}
            for cam in CAMERAS:
                row["observation.images." + cam] = np.asarray(checked_image(d["folder"], cam, d["images"][cam][i]))
            if not old_task_api:
                row["task"] = TASK
            writer.add_frame(row)
        if old_task_api:
            writer.save_episode(task=TASK)
        else:
            writer.save_episode()
    if old_task_api and hasattr(writer, "consolidate"):
        writer.consolidate()


def check_policy_inputs(sample):
    from openpi import transforms
    from openpi.policies.aloha_policy import AlohaInputs
    repack = transforms.RepackTransform({"images": {key: f"observation.images.{cam}"
                                                   for key, cam in zip(MODEL_KEYS, CAMERAS)},
                                        "state": "observation.state", "actions": "action", "prompt": "task"})
    packed = AlohaInputs(adapt_to_pi=True)(repack(sample))
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    joint_dims = np.asarray(mask, dtype=bool)
    expected = np.asarray(packed["actions"]).copy()
    expected[:, joint_dims] -= np.asarray(packed["state"])[joint_dims]
    prepared = transforms.DeltaActions(mask)(packed)
    if prepared["prompt"] != TASK:
        raise ValueError("OpenPI prompt mismatch")
    for key, shape in (("actions", (HORIZON, 14)), ("state", (14,))):
        array = np.asarray(prepared[key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("OpenPI transform returned invalid " + key)
    np.testing.assert_allclose(np.asarray(prepared["actions"]), expected, rtol=0, atol=1e-6)
    for key in MODEL_KEYS:
        img = np.asarray(prepared["image"][key])
        if img.shape != (480, 640, 3) or img.dtype != np.uint8 or not prepared["image_mask"][key]:
            raise ValueError(f"OpenPI 输入缺少真实 RGB 视角 {key}")


def verify_native(root, takes, policy_check=check_policy_inputs):
    cls = native_dataset()
    dataset = cls(repo_id=REPO_ID, root=root, delta_timestamps={"action": [k / FPS for k in range(HORIZON)]})
    info = json.loads((root / "meta/info.json").read_text(encoding="utf8"))
    if len(dataset) != sum(len(d["ticks"]) for d in takes) or info["fps"] != FPS or info["total_episodes"] != len(takes):
        raise ValueError("LeRobot 回读的帧数、频率或 episode 数不符")
    for key in ("action", "observation.state"):
        if info["features"][key]["names"] != JOINT_NAMES:
            raise ValueError(f"{key} 左右/夹爪维度名不一致")
    start, checked = 0, 0
    for ep, d in enumerate(takes):
        n = len(d["ticks"])
        for local in sorted({0, (n - HORIZON) // 2, n - HORIZON}):
            row = dataset[start + local]
            if row["task"] != TASK or int(row["episode_index"]) != ep or int(row["frame_index"]) != local:
                raise ValueError("LeRobot 任务文本、episode 或帧序不符")
            if "action_is_pad" in row and np.asarray(row["action_is_pad"]).any():
                raise ValueError("段内完整动作块被填充")
            np.testing.assert_allclose(np.asarray(row["action"]), d["action"][local:local+HORIZON], rtol=0, atol=1e-6)
            np.testing.assert_allclose(np.asarray(row["observation.state"]), d["state"][local], rtol=0, atol=1e-6)
            for cam in CAMERAS:
                expected = np.asarray(checked_image(d["folder"], cam, d["images"][cam][local])).transpose(2, 0, 1)
                observed = np.asarray(row["observation.images." + cam])
                if observed.shape != expected.shape:
                    raise ValueError(f"{cam} 回读形状不符")
                np.testing.assert_allclose(observed, expected.astype(np.float32) / 255, rtol=0, atol=1e-5)
            policy_check(row)
            checked += 1
        start += n
    return {"episodes": len(takes), "frames": len(dataset), "fps": FPS, "un-padded_windows_checked": checked,
            "action_window_shape": [HORIZON, 14], "task": TASK}


def load_takes(paths):
    paths = [Path(path).expanduser().resolve(strict=True) for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("同一条示范被重复选入")
    takes = []
    for folder in paths:
        data, report = inspect_take(folder)
        if not report["technical_pass"]:
            raise ValueError(f"请查看 {folder / 'qa.json'}，此条未进入训练集")
        takes.append(data)
    return takes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("review").add_argument("episode", type=Path)
    export = sub.add_parser("export")
    export.add_argument("episodes", nargs="+", type=Path)
    export.add_argument("--reviewed-success", action="store_true",
                        help="已回看每条示范并确认正确物体、成功松爪入盒、无辅助手干预")
    export.add_argument("--root", type=Path, default=DATA_HOME / "exports")
    verify = sub.add_parser("verify")
    verify.add_argument("export_dir", type=Path, help="含 selection.json 的导出目录")
    args = ap.parse_args()
    if args.command == "review":
        folder = args.episode.expanduser().resolve(strict=True)
        _, report = inspect_take(folder)
        if (folder / "review.html").exists():
            print("打开三视角回看:", (folder / "review.html").as_uri())
        if not report["technical_pass"]:
            raise SystemExit(2)
        return
    if args.command == "export":
        if not args.reviewed_success:
            ap.error("先回看各条数据。确认成功后使用 --reviewed-success")
        takes = load_takes(args.episodes)
        native_dataset()  # Detect missing deps before creating a destination.
        args.root.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="build_", dir=args.root)).resolve()
        save_json(run / "selection.json", {"task": TASK, "reviewed_success": True,
                                           "episodes": [str(d["folder"]) for d in takes]})
    else:
        run = args.export_dir.resolve(strict=True)
        selection = json.loads((run / "selection.json").read_text(encoding="utf8"))
        if selection["task"] != TASK or selection["reviewed_success"] is not True:
            raise ValueError("此导出没有本任务的人工成功确认")
    root = run / REPO_ID
    print("导出/回读目录:", run, flush=True)
    try:
        save_json(run / "validation.json", {"status": "checking"})
        if args.command == "export":
            write_native(root, takes)
        else:
            takes = load_takes(selection["episodes"])
        result = verify_native(root, takes)
    except Exception as exc:
        save_json(run / "validation.json", {"status": "FAILED", "error": str(exc)})
        raise
    save_json(run / "validation.json", {"status": "PASS_LOADER", **result})
    print("PASS_LOADER", root, json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
