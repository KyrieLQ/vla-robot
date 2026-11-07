"""Proposed batch1 cache: same online computation; explicit batch revision approval required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from .vlm_frontend_v1 import FrozenFrontend, decode_record, bits_bf16

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 2**20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def atomic_npy(path, array):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_rows(plan, episode):
    eid = episode["episode_id"]
    name = f"data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet"
    p = ROOT / plan["dataset_root"] / name
    rows = pq.read_table(p, columns=["image", "wrist_image", "episode_index", "frame_index"], use_threads=False).to_pylist()
    assert len(rows) == episode["length"]
    assert all(row["episode_index"] == eid and row["frame_index"] == i for i, row in enumerate(rows))
    return p, rows


def gate(plan, run, frontend, binding, cache):
    """Fixed120 train inputs: lossless disk round-trip and configured cache batch/online1 parity."""
    target = cache / "gate"
    target.mkdir(exist_ok=True)
    done = target / "complete.json"
    if done.exists():
        result = json.loads(done.read_text())
        assert result["binding"] == binding and result["status"] == "PASS"
        for name, h in result["hashes"].items():
            assert sha(target / name) == h
        write(run / "frontend_gate.json", result)
        return
    samples = json.loads((ROOT / plan["sample_manifest"]).read_text())["samples"]
    original = json.loads((ROOT / plan["preflight_result"]).read_text())
    split = json.loads((ROOT / plan["split"]).read_text())
    train = {e["episode_id"]: e for e in split["train"]}
    records = []
    for s in samples:
        _, rows = load_rows(plan, train[s["episode_id"]])
        records.append(decode_record(rows[s["frame_index"]], s["instruction"]))
    assert len(records) == 120
    comparisons, hashes = [], {}
    started = time.monotonic()
    batch_size = plan["cache_binding"]["batch_size"]
    for offset in range(0, 120, batch_size):
        batch = records[offset:offset+batch_size]
        first, repeated = frontend.encode(batch), frontend.encode(batch)
        for j, (a, b, record) in enumerate(zip(first, repeated, batch)):
            index = offset+j
            np.testing.assert_array_equal(a["bits"], b["bits"])
            np.testing.assert_array_equal(a["token_ids"], b["token_ids"])
            assert a["input_sha256"] == original["input_hashes"][index]["input_sha256"]
            path = target / f"input_{index:03d}.npy"
            atomic_npy(path, a["bits"])
            loaded = np.load(path, allow_pickle=False)
            np.testing.assert_array_equal(loaded, a["bits"])
            online = frontend.encode([record])[0]
            np.testing.assert_array_equal(online["token_ids"], a["token_ids"])
            x, y = bits_bf16(a["bits"]).float(), bits_bf16(online["bits"]).float()
            relative = float(torch.linalg.vector_norm(x-y) / torch.linalg.vector_norm(y).clamp_min(1e-12))
            comparisons.append(dict(sample=index, episode_id=samples[index]["episode_id"],
                                    frame_index=samples[index]["frame_index"], relative_L2=relative,
                                    input_sha256=a["input_sha256"], disk_bits_exact=True))
            write(target / "partial_review.json", dict(binding=binding, checked=len(comparisons),
                                                        comparisons=comparisons))
            assert relative <= plan["online_relative_l2_max"], comparisons[-1]
            hashes[path.name] = sha(path)
        write(run / "progress.json", dict(stage="frontend_gate", checked=offset+len(batch), total=120))
    result = dict(status="PASS", binding=binding, inputs=120, comparisons=comparisons,
                  max_relative_L2=max(x["relative_L2"] for x in comparisons),
                  identical_batch_repeat_exact=True, disk_roundtrip_exact=True, hashes=hashes,
                  elapsed_seconds=time.monotonic()-started, new_optimizer_updates=0)
    write(done, result)
    write(run / "frontend_gate.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    run = Path(args.run)
    revision = json.loads((ROOT/plan["batch_size_authorization"]).read_text())
    assert revision["status"] == "APPROVED" and revision["cache_batch_size"] == 1 and revision["thread"] == plan["thread"]
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    for path, h in plan["input_sha256"].items():
        assert sha(ROOT/path) == h, path
    # Weights were frozen and checked in preflight; independently verify local files before load.
    downloads = json.loads((ROOT/plan["download_manifest"]).read_text())
    for name, info in downloads["files"].items():
        f = ROOT/plan["model_directory"]/name
        assert f.stat().st_size == info["bytes"] and sha(f) == info["sha256"], name
    binding = plan["cache_binding"]
    cache = ROOT/plan["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    binding_path = cache/"binding.json"
    if binding_path.exists():
        assert json.loads(binding_path.read_text()) == binding
    else:
        write(binding_path, binding)
    split = json.loads((ROOT/plan["split"]).read_text())
    assert len(split["train"]) == 1506 and len(split["validation"]) == 187
    assert not {e["episode_id"] for e in split["train"]} & {e["episode_id"] for e in split["validation"]}
    audit = json.loads((ROOT/plan["preflight_acceptance"]).read_text())
    task_lengths = {(e["suite"], e["task_id"]): e["tokens"] for e in audit["per_task_token_lengths"]}
    source_info = json.loads((ROOT/plan["source_integrity"]).read_text())
    source_hashes = {e["path"]: e["sha256"] for e in source_info["files"]}
    frontend = FrozenFrontend(ROOT/plan["model_directory"])
    write(run/"loading.json", dict(status="PASS", loading_info=frontend.loading,
                                  parameter_count=sum(p.numel() for p in frontend.model.parameters()),
                                  trainable_parameters=0))
    gate(plan, run, frontend, binding, cache)
    results, completed_frames, newly_encoded_frames = {}, 0, 0
    compute_started = time.monotonic()
    for part in ("train", "validation"):
        index, offset = [], 0
        for e in split[part]:
            eid, length = e["episode_id"], e["length"]
            seq = task_lengths[(e["suite"], e["task_id"])]
            directory = cache/part/f"episode_{eid:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            receipt_path = directory/"complete.json"
            datafile = directory/"hidden.npy"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                assert receipt["binding"] == binding and receipt["episode"] == e
                assert receipt["shape"] == [length, seq, 960]
                for name, h in receipt["hashes"].items():
                    assert sha(directory/name) == h, directory/name
            else:
                source, rows = load_rows(plan, e)
                name = str(source.relative_to(ROOT/plan["dataset_root"]))
                source_hash = sha(source)
                assert source_hash == source_hashes[name]
                # If interrupted, recompute only this uncommitted deterministic episode.
                # Completed receipts are immutable and always reused above.
                temp = directory/"hidden.partial.npy"
                values = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint16,
                                                  shape=(length, seq, 960))
                inputs, fixed_ids = [], None
                batch_size = plan["cache_binding"]["batch_size"]
                for first in range(0, length, batch_size):
                    records = [decode_record(row, e["instruction"]) for row in rows[first:first+batch_size]]
                    features = frontend.encode(records)
                    for j, feature in enumerate(features):
                        assert feature["bits"].shape == (seq, 960)
                        if fixed_ids is None:
                            fixed_ids = feature["token_ids"]
                        np.testing.assert_array_equal(feature["token_ids"], fixed_ids)
                        values[first+j] = feature["bits"]
                        inputs.append(dict(frame_index=first+j, input_sha256=feature["input_sha256"]))
                    newly_encoded_frames += len(features)
                    if first % 64 == 0:
                        elapsed = time.monotonic()-compute_started
                        write(run/"progress.json", dict(stage="cache", part=part, episode_id=eid,
                              completed_frames=completed_frames, total_frames=273465,
                              frame_in_episode=first+len(features), newly_encoded_frames=newly_encoded_frames,
                              elapsed_seconds=time.monotonic()-started,
                              estimated_remaining_seconds=(273465-completed_frames-first-len(features))
                                  * elapsed/max(newly_encoded_frames, 1)))
                values.flush()
                del values
                with temp.open("rb") as f:
                    os.fsync(f.fileno())
                temp.replace(datafile)
                atomic_npy(directory/"token_ids.npy", fixed_ids)
                write(directory/"input_hashes.json", inputs)
                receipt = dict(status="PASS", binding=binding, episode=e, part=part,
                               shape=[length, seq, 960], dtype="uint16_raw_bfloat16",
                               source_parquet_sha256=source_hash,
                               hashes={f:sha(directory/f) for f in ("hidden.npy", "token_ids.npy", "input_hashes.json")})
                write(receipt_path, receipt)
            index.append(dict(episode_id=eid, task_id=e["task_id"], suite=e["suite"],
                              start=offset, end=offset+length, tokens=seq,
                              directory=str(directory.relative_to(cache)), receipt_sha256=sha(receipt_path)))
            offset += length
            completed_frames += length
            print(f"cache {part} episode {eid}: {completed_frames}/273465", flush=True)
        assert offset == (243000 if part == "train" else 30465)
        write(cache/part/"index.json", index)
        results[part] = dict(frames=offset, episodes=len(index), index_sha256=sha(cache/part/"index.json"))
    assert completed_frames == 273465
    raw_bytes = sum(e["length"] * task_lengths[(e["suite"], e["task_id"])] * 960 * 2
                    for part in ("train", "validation") for e in split[part])
    assert raw_bytes == plan["expected_hidden_bytes"] == 84665278080
    result = dict(status="COMPLETE_REVIEW_REQUIRED", binding=binding, parts=results,
                  frames=completed_frames, hidden_payload_bytes=raw_bytes, episodes=1693,
                  elapsed_seconds=time.monotonic()-started, newly_encoded_frames=newly_encoded_frames,
                  frontend_gate_sha256=sha(cache/"gate/complete.json"),
                  new_training_updates=0, new_simulation_episodes=0, confirmation_accessed=False)
    write(cache/"complete.json", result)
    write(run/"cache_result.json", result)
    write(run/"progress.json", dict(stage="complete", completed_frames=273465, total_frames=273465))


if __name__ == "__main__":
    main()
