"""CPU-only audit of all fixed preflight inputs and independent resource arithmetic."""
from pathlib import Path
import hashlib
import io
import json
import math
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/mini_vla_m8_preflight_20261020_run2"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 2**20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    start = time.monotonic()
    import pyarrow.parquet as pq
    from PIL import Image
    from mini_vla.preflight_m8_v2 import make_processor

    out = RUN / "review"
    out.mkdir(exist_ok=True)
    assert not (out / "independent_acceptance.json").exists()
    plan_path = ROOT / "configs/mini_vla_m8_preflight_20261020_v2.json"
    plan = json.loads(plan_path.read_text())
    report = json.loads((RUN / "feasibility.json").read_text())
    for f, h in json.loads((RUN / "source_manifest.json").read_text()).items():
        assert sha(ROOT / f) == h, f
    for f, h in plan["input_sha256"].items():
        assert sha(ROOT / f) == h, f
    dl = json.loads((RUN / "download_manifest.json").read_text())
    previous_dl = json.loads((ROOT / plan["resume_from"] / "download_manifest.json").read_text())
    assert dl == previous_dl
    for f, info in dl["files"].items():
        p = ROOT / plan["model_directory"] / f
        assert p.stat().st_size == info["bytes"] and sha(p) == info["sha256"]
    with (ROOT / plan["model_directory"] / "model.safetensors").open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        assert header_size < 10 * 2**20
        header = json.loads(f.read(header_size))
    weight_elements = sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
    assert weight_elements == report["model_parameters"] == 507482304
    split = json.loads((ROOT / plan["split"]).read_text())
    train = {e["episode_id"]: e for e in split["train"]}
    validation = {e["episode_id"] for e in split["validation"]}
    samples = json.loads((ROOT / plan["sample_manifest"]).read_text())["samples"]
    assert len(samples) == len(report["input_hashes"]) == len(report["sequence_lengths"]) == 120
    assert all(s["episode_id"] in train and s["episode_id"] not in validation for s in samples)
    # Independently regenerate the frozen evenly distributed selection.
    groups = {}
    for e in split["train"]:
        groups.setdefault((e["suite"], e["task_id"]), []).append(e)
    expected = []
    for group, episodes in sorted(groups.items()):
        episodes.sort(key=lambda e: e["episode_id"])
        for number, ratio in zip([0, len(episodes)//2, len(episodes)-1], [0, .5, 1]):
            e = episodes[number]
            expected.append((e["episode_id"], int((e["length"]-1)*ratio)))
    assert [(s["episode_id"], s["frame_index"]) for s in samples] == expected
    processor = make_processor(ROOT / plan["model_directory"])
    task_lengths = {}
    for i, (sample, observed) in enumerate(zip(samples, report["input_hashes"])):
        eid = sample["episode_id"]
        path = ROOT / plan["dataset_root"] / f"data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet"
        row = pq.read_table(path, columns=["image", "wrist_image", "episode_index", "frame_index"], use_threads=False).to_pylist()[sample["frame_index"]]
        assert row["episode_index"] == eid and row["frame_index"] == sample["frame_index"]
        images = [Image.open(io.BytesIO(row[k]["bytes"])).convert("RGB") for k in ["image", "wrist_image"]]
        assert all(im.size == (256, 256) for im in images)
        text = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": "External camera:"}, {"type": "image"},
            {"type": "text", "text": "Wrist camera:"}, {"type": "image"},
            {"type": "text", "text": sample["instruction"]},
        ]}], tokenize=False, add_generation_prompt=True)
        actual = hashlib.sha256(b"".join(im.tobytes() for im in images) + text.encode()).hexdigest()
        assert observed == dict(episode_id=eid, frame_index=sample["frame_index"], input_sha256=actual)
        x = processor(text=[text], images=[images], return_tensors="pt", padding=True)
        assert list(x["pixel_values"].shape) == [1, 2, 3, 512, 512]
        assert int((x["input_ids"] == 49190).sum()) == 128
        length = int(x["attention_mask"].sum())
        assert length == report["sequence_lengths"][i]
        key = (sample["suite"], sample["task_id"])
        assert task_lengths.get(key, length) == length
        task_lengths[key] = length
        if i % 30 == 0:
            print("independent input audit", i+1, "/120", flush=True)
    assert len(task_lengths) == 40
    measurements = report["measurements"]
    for b in [1, 2, 4, 8]:
        m = measurements[str(b)]
        assert m["samples"] == 120 and m["timed_batches"] == 120 // b
        assert abs(m["seconds_per_sample"] * 120 - m["seconds"]) < 1e-10
        assert m["total_valid_tokens"] == sum(report["sequence_lengths"])
    best = min(m["seconds_per_sample"] for m in measurements.values())
    n = sum(e["length"] for e in split["train"] + split["validation"])
    assert n == 273465
    assert abs(best*n/3600 - report["one_pass_forward_only_hours"]) < 1e-10
    assert abs(best*n*50/3600 - report["optimistic_50_epoch_forward_only_hours"]) < 1e-10
    # Text/token count is constant within each task with fixed two-image512/no-split processing.
    # This sizes all train/validation frames from metadata only, without reading validation images.
    token_count = sum(e["length"] * task_lengths[(e["suite"], e["task_id"])]
                      for e in split["train"] + split["validation"])
    bytes_hidden = token_count * 960 * 2
    task_table = [dict(suite=s, task_id=t, tokens=v) for (s, t), v in sorted(task_lengths.items())]
    result = dict(
        status="PASS", scope="engineering feasibility, no policy performance evaluation",
        checked_inputs=120, total_inputs=120, tasks=40, all_download_and_source_hashes_verified=True,
        all_input_hashes_verified=True, sample_selection_reproduced=True,
        total_pretrained_weight_elements=weight_elements, frozen_vlm=True,
        processor_config_consumed=True, visual_tokens_per_two_camera_input=128,
        repeated_input_max_error=report["repeated_input_max_error"], readout_shape=report["readout_shape"],
        measurements=measurements, timer_audit_boundary="Per-batch raw timing arrays were not saved; aggregate arithmetic verified, wall-clock measurements not independently remeasured.",
        exact_metadata_sized_hidden_bytes=bytes_hidden,
        exact_metadata_sized_hidden_gib=bytes_hidden / 2**30,
        planned_cache_frames=n, planned_cache_tokens=token_count, per_task_token_lengths=task_table,
        cache_size_boundary="Exact given fixed processor and known40 instruction strings; must validate lengths during full extraction. Excludes indices/masks/manifests/checkpoints. BF16 raw uint16 storage preserves bits.",
        one_pass_forward_only_hours=best*n/3600, fifty_epoch_forward_only_hours=best*n*50/3600,
        cost_boundary="120 preloaded training samples only; excludes full I/O, head backward and eval. No formal training ETA guarantee.",
        new_training_updates=0, new_simulation_episodes=0, confirmation_accessed=False,
        elapsed_seconds=time.monotonic()-start,
    )
    assert result["repeated_input_max_error"] == 0 and result["readout_shape"] == [1, 10, 7]
    (out / "independent_acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_task_token_lengths"}, indent=2))


if __name__ == "__main__":
    main()
