"""Bounded frozen-VLM interface/timing preparation; no optimizer or simulation."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def file_hash(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 2**20), b""):
            h.update(data)
    return h.hexdigest()


def download(plan, run):
    import requests
    metadata = json.loads((ROOT / plan["metadata"]).read_text())
    target = ROOT / plan["model_directory"]
    target.mkdir(parents=True, exist_ok=True)
    assert sum(x["size"] for x in metadata["files"]) < 3 * 2**30
    verified = {}
    for item in metadata["files"]:
        name, size = item["rfilename"], item["size"]
        dest = target / name
        expected = item.get("lfs", {}).get("sha256", item["blobId"])
        git = "lfs" not in item
        if dest.exists():
            assert dest.stat().st_size == size and file_hash(dest, git) == expected
        else:
            part = dest.with_suffix(dest.suffix + ".part")
            url = f'https://huggingface.co/{metadata["repo_id"]}/resolve/{metadata["revision"]}/{name}'
            for attempt in range(3):
                try:
                    offset = part.stat().st_size if part.exists() else 0
                    if offset < size:
                        headers = {"Range": f"bytes={offset}-"} if offset else {}
                        with requests.get(url, headers=headers, stream=True, timeout=(15, 30)) as response:
                            response.raise_for_status()
                            if offset and response.status_code == 206:
                                assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
                            elif offset:
                                offset = 0  # Server returned full object; replace only this partial file.
                            mode = "ab" if offset else "wb"
                            last = time.monotonic()
                            with part.open(mode) as f:
                                for chunk in response.iter_content(4 * 2**20):
                                    f.write(chunk)
                                    offset += len(chunk)
                                    assert offset <= size, "Remote object exceeds frozen size"
                                    if time.monotonic() - last > 5:
                                        write(run / "progress.json", dict(stage="download", file=name, bytes=offset, total_bytes=size))
                                        print("download", name, offset, "/", size, flush=True)
                                        last = time.monotonic()
                    assert part.stat().st_size == size and file_hash(part, git) == expected
                    part.replace(dest)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(3)
            assert dest.exists()
        verified[name] = dict(bytes=size, sha256=file_hash(dest))
    write(run / "download_manifest.json", dict(revision=metadata["revision"], files=verified))
    # This small missing processor dependency is isolated; existing environments remain unchanged.
    vendor = ROOT / plan["dependency_directory"]
    vendor.mkdir(parents=True, exist_ok=True)
    if not (vendor / "num2words").exists() or not (vendor / "docopt.py").exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check",
             "--target", str(vendor), "num2words==0.5.14", "docopt==0.6.2"],
            check=True, timeout=120,
        )


def gpu_probe(plan, run):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    model_path = ROOT / plan["model_directory"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
    )
    assert not any(loading.get(k) for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]), loading
    model = model.cuda().eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    split = json.loads((ROOT / plan["split"]).read_text())
    allowed = {e["episode_id"] for e in split["train"]}
    forbidden = {e["episode_id"] for e in split["validation"]}
    samples = json.loads((ROOT / plan["sample_manifest"]).read_text())["samples"]
    assert len(samples) == 120 and all(s["episode_id"] in allowed - forbidden for s in samples)
    prepared, hashes = [], []
    pre = json.loads((ROOT / plan["preprocessing"]).read_text())
    stats = pre["normalization"]["state"]
    for i, sample in enumerate(samples):
        eid = sample["episode_id"]
        path = ROOT / plan["dataset_root"] / f"data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet"
        rows = pq.read_table(path, columns=["image", "wrist_image", "state", "episode_index", "frame_index"], use_threads=False).to_pylist()
        row = rows[sample["frame_index"]]
        assert row["episode_index"] == eid and row["frame_index"] == sample["frame_index"]
        # Embedded demonstration RGB already has the orientation used by the existing training data.
        images = [Image.open(io.BytesIO(row[k]["bytes"])).convert("RGB") for k in ["image", "wrist_image"]]
        assert all(im.size == (256, 256) for im in images)
        state = (np.asarray(row["state"], dtype=np.float32) - stats["mean"]) / stats["scale"]
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "External camera:"}, {"type": "image"},
            {"type": "text", "text": "Wrist camera:"}, {"type": "image"},
            {"type": "text", "text": sample["instruction"]},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prepared.append(dict(images=images, text=text, state=state))
        hashes.append(dict(episode_id=eid, frame_index=sample["frame_index"],
                           input_sha256=hashlib.sha256(b"".join(im.tobytes() for im in images) + text.encode()).hexdigest()))
        if i % 20 == 0:
            write(run / "progress.json", dict(stage="input_preparation", samples=i + 1, total=120))

    def process(batch):
        inputs = processor(text=[s["text"] for s in batch], images=[s["images"] for s in batch],
                           return_tensors="pt", padding=True,
                           size={"longest_edge": 512}, do_image_splitting=False)
        assert inputs["pixel_values"].shape == (len(batch), 2, 3, 512, 512)
        assert ((inputs["input_ids"] == model.config.image_token_id).sum(1) == 128).all()
        return inputs

    def move(inputs):
        return {k: v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda() for k, v in inputs.items()}

    summary, length_by_sample = {}, {}
    first_hidden = None
    with torch.inference_mode():
        for batch_size in [1, 2, 4, 8]:
            warm = move(process(prepared[:batch_size]))
            for _ in range(2):
                model.model(**warm, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            latencies, token_count = [], 0
            for offset in range(0, len(prepared), batch_size):
                torch.cuda.synchronize()
                t = time.perf_counter()
                cpu = process(prepared[offset:offset + batch_size])
                inputs = move(cpu)
                hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                assert hidden.shape[:2] == inputs["input_ids"].shape and hidden.shape[-1] == 960
                assert torch.isfinite(hidden).all()
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - t)
                lengths = cpu["attention_mask"].sum(1).tolist()
                token_count += sum(lengths)
                for j, length in enumerate(lengths):
                    if offset + j in length_by_sample:
                        assert length_by_sample[offset + j] == length
                    length_by_sample[offset + j] = length
                if batch_size == 1 and offset == 0:
                    first_hidden = hidden.float().clone()
                del hidden
            summary[str(batch_size)] = dict(
                samples=120, timed_batches=len(latencies), seconds=sum(latencies),
                seconds_per_sample=sum(latencies) / 120,
                batch_median_seconds=float(np.median(latencies)), total_valid_tokens=token_count,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            )
            write(run / "progress.json", dict(stage="throughput", completed_batch_size=batch_size, measurements=summary))
            print(json.dumps(summary[str(batch_size)]), flush=True)
        # Same-input repeat is recorded, not a claim of bitwise equality across batch sizes.
        repeat = model.model(**move(process(prepared[:1])), use_cache=False, return_dict=True).last_hidden_state.float()
        difference = float((first_hidden - repeat).abs().max())
        assert difference == 0, ("same-input-repeat mismatch", difference)

    # An untrained readout checks the interface only; no optimizer, loss-based selection, or success claim.
    with torch.no_grad():
        projection = torch.nn.Linear(960, 128).cuda()
        decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(128, 4, 512, dropout=0, batch_first=True), 1
        ).cuda().eval()
        state_encoder = torch.nn.Linear(8, 128).cuda()
        output = torch.nn.Linear(128, 7).cuda()
        state = torch.as_tensor(prepared[0]["state"], dtype=torch.float32, device="cuda")[None, None]
        memory = torch.cat([projection(first_hidden), state_encoder(state)], 1)
        action = output(decoder(torch.zeros(1, 10, 128, device="cuda"), memory))
        assert action.shape == (1, 10, 7) and torch.isfinite(action).all()
    lengths = list(length_by_sample.values())
    best = min(v["seconds_per_sample"] for v in summary.values())
    n = 273465
    result = dict(
        status="PASS", kind="ENGINEERING_FEASIBILITY_ONLY", samples=120,
        measurements=summary, input_hashes=hashes, model_parameters=sum(p.numel() for p in model.parameters()),
        trainable_vlm_parameters=0, model_dtype="bfloat16", backend="sdpa",
        processor_overrides=plan["processor_overrides"], sequence_lengths=lengths,
        raw_hidden_cache_estimate_gib=dict(
            minimum_observed=n * min(lengths) * 960 * 2 / 2**30,
            mean_observed=n * float(np.mean(lengths)) * 960 * 2 / 2**30,
            maximum_observed=n * max(lengths) * 960 * 2 / 2**30),
        optimistic_50_epoch_forward_only_hours=best * (243000 + 30465) * 50 / 3600,
        one_pass_forward_only_hours=best * n / 3600,
        timing_boundary="120 train frames, frozen backbone and processor only; excludes full I/O, action training/backward, all evaluation and head optimizer costs. Extrapolations are not guarantees.",
        repeated_input_max_error=difference, readout_shape=list(action.shape),
        new_training_updates=0, new_simulation_episodes=0, confirmation_accessed=False,
        elapsed_seconds=time.monotonic() - started,
    )
    write(run / "feasibility.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ["input_hashes", "sequence_lengths"]}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--gpu-probe", action="store_true")
    args = p.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    run = Path(args.run)
    for path, sha in plan["input_sha256"].items():
        assert file_hash(ROOT / path) == sha, path
    if args.gpu_probe:
        gpu_probe(plan, run)
        return 0
    download(plan, run)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / plan["dependency_directory"]) + ":" + env.get("PYTHONPATH", "")
    command = [sys.executable, str(ROOT / "scripts/gpu_run.py"), "--", sys.executable,
               "-m", "mini_vla.preflight_m8_v1", "--plan", str(plan_path),
               "--run", str(run), "--gpu-probe"]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
