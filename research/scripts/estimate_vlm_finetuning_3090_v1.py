"""Read-only model/header analysis and explicitly assumed timing scenarios. No torch/GPU."""
from pathlib import Path
import collections
import hashlib
import json
import math
import struct

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'cache/models/smolvlm2_500m_7b375e1'
PRE = ROOT/'results/mini_vla_m8_preflight_20261020_run2/review/independent_acceptance.json'


def main():
    tensors = {}
    for file in MODEL.glob('*.safetensors'):
        with file.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            headers = json.loads(stream.read(length))
        tensors.update({k:v for k,v in headers.items() if k!='__metadata__'})
    count = collections.Counter()
    for key, spec in tensors.items():
        group = 'unused_lm_head' if key.startswith('lm_head.') else 'vision' if 'vision_model' in key else 'connector' if 'connector' in key else 'text'
        elements = 1
        for dimension in spec['shape']:
            elements *= dimension
        count[group] += elements
    loaded = sum(count.values()); active = loaded-count['unused_lm_head']; action_head=398663
    assert loaded==507482304 and active==460173504
    lora=[]
    for rank in [8,16]:
        per_group=collections.Counter()
        for key,spec in tensors.items():
            if not key.endswith('.weight') or '.self_attn.' not in key:continue
            module=key.split('.')[-2]
            if module not in ['q_proj','k_proj','v_proj','o_proj','out_proj']:continue
            group='vision' if 'vision_model' in key else 'text'
            per_group[group]+=rank*sum(spec['shape'])
        trained=sum(per_group.values())+count['connector']+action_head
        lora.append(dict(rank=rank,attention_adapters=dict(per_group),connector_trainable=count['connector'],action_head_trainable=action_head,total_trainable=trained,
                         conservative_parameter_state_gib=(active*2+trained*18)/2**30,
                         note='BF16 frozen base +18B/trainable parameter conservatively double-counts existing connector base storage; not activation-inclusive VRAM'))
    probe=json.loads(PRE.read_text())
    low=probe['measurements']['8']['seconds_per_sample'];high=probe['measurements']['1']['seconds_per_sample']
    train=243000;val=30465
    scenarios={}
    for name,multipliers in [('lora_all_attention',(2.,3.)),('full_active_vlm',(2.5,4.))]:
        # Multipliers, IO overhead and two-GPU speedup are assumptions, not measurements.
        lower=(train*low*multipliers[0]+val*low)*1.1/3600
        upper=(train*high*multipliers[1]+val*high)*1.2/3600
        scenarios[name]=dict(assumed_training_to_old_forward_multiplier=list(multipliers),assumed_data_runtime_overhead=[1.1,1.2],assumed_two_gpu_speedup=[1.4,1.7],one_gpu_hours_per_epoch=[lower,upper],two_gpu_hours_per_epoch=[lower/1.7,upper/1.4],one_gpu_5epoch_hours=[lower*5,upper*5],two_gpu_5epoch_hours=[lower*5/1.7,upper*5/1.4],one_gpu_50epoch_days=[lower*50/24,upper*50/24],two_gpu_50epoch_days=[lower*50/1.7/24,upper*50/1.4/24])
    output=dict(status='ANALYTICAL_ESTIMATE_NOT_TRAINING_BENCHMARK',date='2026-10-23',canonical_models='M7/M8 use this frozenVLM; proposed finetuning is a new unexecuted comparison',parameter_counts=dict(count),loaded_parameters=loaded,active_vlm_parameters=active,regression_action_head=action_head,
                loaded_bf16_weight_gib=loaded*2/2**30,full_active_plus_head_parameter_states_gib_16_to_18bytes=[(active+action_head)*b/2**30 for b in [16,18]],
                lora_cases=lora,old_forward_seconds_per_sample=[low,high],old_forward_scope='120 preloaded training samples; CPU image processor+H2D+BF16 SDPA forward+finite check; no backward, optimizer or source decoding; GPU0 RTX3090',
                forward_only_50_training_epochs_hours=[train*50*low/3600,train*50*high/3600],scenarios=scenarios,
                uncertainty='Timing ranges are scenario assumptions, not confidence intervals or guarantees; optimized preprocessing/kernel/batching can invalidate old forward extrapolation. Full backward and activation peaks not measured.',
                source_sha256={str(PRE.relative_to(ROOT)):hashlib.sha256(PRE.read_bytes()).hexdigest(),'cache/models/smolvlm2_500m_7b375e1/config.json':hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest()},gpu_operations=0,training_updates=0)
    dest=ROOT/'reports/VLM_FINETUNING_3090_ESTIMATE_20261023.json'
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['active_vlm_parameters','full_active_plus_head_parameter_states_gib_16_to_18bytes','scenarios']},indent=2))


if __name__=='__main__':main()
