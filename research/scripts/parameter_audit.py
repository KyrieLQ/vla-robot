#!/usr/bin/env python3
"""Abstract-shape audit: no model allocation, weights loading or GPU work."""
import collections, json, math, os, pathlib
import flax.nnx as nnx
import flax.traverse_util as tu
import jax
from openpi.training import config, optimizer
from openpi.models import gemma

def main():
    root=pathlib.Path(os.environ['PROJECT_ROOT']);cfg=config.get_config('pi05_libero_lora_3090_smoke')
    state=jax.eval_shape(lambda k:nnx.state(cfg.model.create(k)),jax.random.key(cfg.seed))
    full=tu.flatten_dict(state.to_pure_dict(),sep='/')
    train=tu.flatten_dict(state.filter(cfg.trainable_filter).to_pure_dict(),sep='/')
    frozen=tu.flatten_dict(state.filter(cfg.freeze_filter).to_pure_dict(),sep='/')
    assert set(train).isdisjoint(frozen) and set(train)|set(frozen)==set(full)
    rows=[];groups=collections.defaultdict(lambda:{'trainable':0,'frozen':0})
    for name,v in full.items():
        n=math.prod(v.shape);module='/'.join(name.split('/')[:2]) if name.startswith('PaliGemma/') else name.split('/')[0]
        is_train=name in train;groups[module]['trainable' if is_train else 'frozen']+=n
        rows.append({'path':name,'shape':list(v.shape),'dtype':str(v.dtype),'parameters':n,'trainable':is_train,'lora':'lora' in name})
    tx=optimizer.create_optimizer(cfg.optimizer,cfg.lr_schedule)
    opt_shape=jax.eval_shape(tx.init,state.filter(cfg.trainable_filter))
    total=sum(r['parameters'] for r in rows);train_total=sum(r['parameters'] for r in rows if r['trainable'])
    lora={}
    for variant in [cfg.model.paligemma_variant,cfg.model.action_expert_variant]:
        lora[variant]={name:{'rank':c.rank,'alpha':c.alpha,'rslora':c.rslora,'scaling':c.scaling_value} for name,c in gemma.get_config(variant).lora_configs.items()}
    result={'status':'PASS','audit_type':'abstract model and optimizer shapes; actual GPU initialization not yet validated',
            'total_parameters':total,'trainable_parameters':train_total,'trainable_fraction':train_total/total,
            'lora_trainable_parameters':sum(r['parameters'] for r in rows if r['trainable'] and r['lora']),
            'non_lora_trainable_parameters':sum(r['parameters'] for r in rows if r['trainable'] and not r['lora']),
            'module_summary':dict(groups),'lora':lora,'optimizer_parameter_paths':sorted(train),
            'optimizer_state_shapes':[{'shape':list(v.shape),'dtype':str(v.dtype)} for v in jax.tree.leaves(opt_shape)],
            'parameters':rows,'freeze_rule':'same Pi0Config instance .get_freeze_filter(); no additional freezing',
            'model':{'pi05':cfg.model.pi05,'action_dim':cfg.model.action_dim,'action_horizon':cfg.model.action_horizon,
                     'max_token_len':cfg.model.max_token_len,'discrete_state_input':cfg.model.discrete_state_input},
            'ema_decay':cfg.ema_decay,'batch_size':cfg.batch_size,'fsdp_devices':cfg.fsdp_devices}
    (root/'reports/parameter_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['parameters','optimizer_parameter_paths','optimizer_state_shapes']},indent=2))
if __name__=='__main__':main()
