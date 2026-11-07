"""Bounded attribution on the original eight inputs; no production cache mutation."""
import json
from pathlib import Path
import time
import torch
from mini_vla.vlm_frontend_v1 import FrozenFrontend, decode_record, prompt_for
from mini_vla.vlm_cache_v1 import load_rows, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/mini_vla_cache_parity_diagnostic_20261020_v1"


def relative(x, y):
    return float(torch.linalg.vector_norm(x.float()-y.float()) /
                 torch.linalg.vector_norm(y.float()).clamp_min(1e-12))


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    p = json.loads((ROOT/"configs/mini_vla_modern_cache_20261020_v1.json").read_text())
    diagnostic = json.loads((ROOT/"configs/mini_vla_cache_parity_diagnostic_20261020_v1.json").read_text())
    split = json.loads((ROOT/p["split"]).read_text())
    episodes = {e["episode_id"]:e for e in split["train"]}
    records = []
    for s in diagnostic["samples"]:
        _, rows = load_rows(p, episodes[s["episode_id"]])
        records.append(decode_record(rows[s["frame_index"]], s["instruction"]))
    frontend = FrozenFrontend(ROOT/p["model_directory"])
    model, processor = frontend.model.model, frontend.processor

    def prepare(records):
        return processor(text=[prompt_for(processor, r["instruction"]) for r in records],
                         images=[r["images"] for r in records], return_tensors="pt", padding=True)
    batch = prepare(records)
    single = [prepare([r]) for r in records]
    assert all(torch.equal(batch["pixel_values"][i], s["pixel_values"][0]) for i, s in enumerate(single))
    assert all(torch.equal(batch["input_ids"][i, batch["attention_mask"][i].bool()],
                           s["input_ids"][0, s["attention_mask"][0].bool()]) for i, s in enumerate(single))
    def device(cpu):
        return {k: v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda() for k, v in cpu.items()}
    b = device(batch)
    ones = [device(s) for s in single]
    def valid(result, inputs, i=0):
        return result.last_hidden_state[i, inputs["attention_mask"][i].bool()]
    def position(inputs):
        ids = inputs["attention_mask"].long().cumsum(-1)-1
        return ids.masked_fill(inputs["attention_mask"] == 0, 0)
    result = dict(status="RUNNING", samples=8, original_gate=.01, input_pixels_equal=True,
                  valid_token_ids_equal=True, padding_side=processor.tokenizer.padding_side,
                  batch_lengths=batch["attention_mask"].sum(1).tolist(), comparisons={})
    with torch.inference_mode():
        out8 = model(**b, use_cache=False)
        refs = [valid(model(**s, use_cache=False), s).clone() for s in ones]
        result["comparisons"]["bf16_sdpa_full"] = [relative(valid(out8, b, i), ref) for i, ref in enumerate(refs)]
        repeat = valid(model(**ones[0], use_cache=False), ones[0])
        result["single_repeat_max_abs"] = float((repeat-refs[0]).abs().max())
        out8pos = model(**b, position_ids=position(b), use_cache=False)
        posrefs = [valid(model(**s, position_ids=position(s), use_cache=False), s) for s in ones]
        result["comparisons"]["bf16_mask_position_ids"] = [relative(valid(out8pos, b, i), ref) for i, ref in enumerate(posrefs)]
        result["position_fix_single_change"] = [relative(x, y) for x,y in zip(posrefs,refs)]
        images8 = model.get_image_features(b["pixel_values"], b["pixel_attention_mask"])
        images1 = [model.get_image_features(s["pixel_values"], s["pixel_attention_mask"]) for s in ones]
        result["comparisons"]["bf16_vision"] = [relative(images8[2*i:2*i+2], x) for i,x in enumerate(images1)]
        fixedrefs = []
        for i,s in enumerate(ones):
            fixedrefs.append(valid(model(input_ids=s["input_ids"], attention_mask=s["attention_mask"],
                                         image_hidden_states=images8[2*i:2*i+2], use_cache=False),s))
        result["comparisons"]["bf16_language_fixed_batch8_vision"] = [relative(valid(out8,b,i),x) for i,x in enumerate(fixedrefs)]
        write(OUT/"partial.json",result)
        # Keep the same BF16-rounded values while changing only arithmetic precision.
        model.float()
        b32 = {k:v.float() if v.is_floating_point() else v for k,v in b.items()}
        ones32 = [{k:v.float() if v.is_floating_point() else v for k,v in s.items()} for s in ones]
        out32 = model(**b32, use_cache=False)
        refs32 = [valid(model(**s,use_cache=False),s) for s in ones32]
        result["comparisons"]["fp32_arithmetic_bf16_rounded_weights"] = [relative(valid(out32,b32,i),x) for i,x in enumerate(refs32)]
    result.update(status="COMPLETE", elapsed_seconds=time.monotonic()-start,
                  new_training_updates=0,new_simulation_episodes=0,
                  production_configuration_changed=False,
                  interpretation="Attribution only; production still BF16/SDPA,batch8 cached vsbatch1online,gate1e-2")
    write(OUT/"result.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
