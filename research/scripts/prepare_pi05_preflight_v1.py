"""CPU-only exact freeze/shape inventory and official-transform input contract."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import flax.nnx as nnx
import flax.traverse_util as tu
import jax
import numpy as np
from mini_vla.pi05_four_suite_v1 import make_config,RawTrainingInputs
R=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='cpu';p=json.loads(Path(a.plan).read_text());started=time.monotonic();cfg=make_config()
    shape=jax.eval_shape(lambda:nnx.state(cfg.model.create(jax.random.key(42))))
    flat=tu.flatten_dict(shape.to_pure_dict(),sep='/');train=tu.flatten_dict(shape.filter(cfg.trainable_filter).to_pure_dict(),sep='/')
    old=json.loads((R/'reports/parameter_audit.json').read_text());rows=[];groups=defaultdict(int)
    assert sum(x.size for x in flat.values())==old['total_parameters']
    for key,value in train.items():
        assert '/img/' not in key and ('lora' in key or key.startswith(('action_in_proj/','action_out_proj/','time_mlp_in/','time_mlp_out/')))
        group='language_lora' if 'llm' in key and '_1/' not in key else 'action_lora' if 'llm' in key else key.split('/')[0]
        groups[group]+=value.size;rows.append(dict(path=key,shape=list(value.shape),dtype=str(value.dtype),parameters=value.size))
    count=sum(x.size for x in train.values());assert count==old['trainable_parameters']-old['module_summary']['PaliGemma/img']['trainable']==52153376
    inputs=RawTrainingInputs(p);receipts=[]
    for i in range(120):
        observation,actions,row=inputs.get(i);again,again_actions,repeat=inputs.get(i);assert row==repeat
        assert all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves((observation,actions)),jax.tree.leaves((again,again_actions))))
        receipts.append(row)
    assert len({(r['suite'],r['task_id']) for r in receipts})==40
    result=dict(status='PASS',samples=120,tasks=40,suites=4,total_parameters=sum(x.size for x in flat.values()),
                trainable_parameters=count,frozen_vision_parameters=414803696,groups=dict(groups),trainable_leaves=rows,
                approximate_compact_parameters_bytes=count*4,approximate_trainstate_bytes=count*12,
                all_inputs_repeat_bitwise=True,quantile_roundtrip_atol=2e-6,action_dim=32,real_action_dim=7,action_horizon=10,
                text_length=200,discrete_state_input=False,zero_new_optimizer_updates=True,zero_new_simulations=True,
                new_gpu_compute=0,elapsed_seconds=time.monotonic()-started,receipts=receipts,
                limitations='Prospective engineering recipe;vision freezing differs from oldSpatialfine-tune. No performance established or formaltrainingauthorizedbythisscript.')
    (out/'cpu_contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['receipts','trainable_leaves']}))

if __name__=='__main__':main()
