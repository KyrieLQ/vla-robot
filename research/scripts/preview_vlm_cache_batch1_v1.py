"""Validate the proposed engineering change on original120 inputs, never full-cache."""
import json
from pathlib import Path
import time
import torch
from mini_vla.vlm_frontend_v1 import FrozenFrontend
from mini_vla.vlm_cache_v2 import gate, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/mini_vla_cache_batch1_preview_20261020_v1"


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=False)
    plan = json.loads((ROOT/"configs/mini_vla_modern_cache_batch1_proposal_20261020_v1.json").read_text())
    assert plan["status"] == "DRAFT_AWAITING_BATCH1_APPROVAL"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    timings = []
    class TimedFrontend(FrozenFrontend):
        def encode(self, records):
            assert len(records) == 1
            t = time.perf_counter()
            result = super().encode(records)
            timings.append(time.perf_counter()-t)
            return result
    frontend = TimedFrontend(ROOT/plan["model_directory"])
    cache = ROOT/plan["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    assert not any(cache.glob("*/index.json"))
    gate(plan, OUT, frontend, plan["cache_binding"], cache)
    result = json.loads((OUT/"frontend_gate.json").read_text())
    assert result["status"] == "PASS" and result["inputs"] == 120
    assert len(timings) == 360
    steady = timings[3:]
    seconds_per_frame = sum(steady)/len(steady)
    summary = dict(
        status="PASS", inputs=120, repeated_encodes=360,
        max_relative_L2=result["max_relative_L2"], all_same_batch_repeats_exact=True,
        all_disk_roundtrips_exact=True, measured_seconds_per_frame=seconds_per_frame,
        fullcache_forward_estimate_hours=273465*seconds_per_frame/3600,
        fullcache_estimate_with_20percent_margin_hours=1.2*273465*seconds_per_frame/3600,
        raw_call_seconds=timings, elapsed_seconds=time.monotonic()-start,
        production_cache_started=False, new_training_updates=0, new_simulation_episodes=0,
        boundary="Same original120 gate inputs only; fullcache batch-size change still awaits user approval.",
    )
    assert not any(cache.glob("*/index.json"))
    write(OUT/"result.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="raw_call_seconds"},indent=2))


if __name__ == "__main__":
    main()
