#!/usr/bin/env python3
"""CPU only: zero identity, adapter-only updates, real-meta target coverage and admission."""
import copy,json,time
from pathlib import Path
import torch
from torch import nn
from mini_vla.smolvlm_lora_v1 import LoRALinear,inject_attention_lora,adapter_parameters
from mini_vla.vlm_action_heads_v1 import RegressionHead,HeadConfig
from mini_vla.vlm_cached_data_v2 import VLMCacheData
from mini_vla_lora_preflight_supervisor_v1 import eligibility,consumed
R=Path(__file__).resolve().parents[1]
def main():
    began=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42);checks={}
    base=nn.Linear(7,11);ref=copy.deepcopy(base);m=LoRALinear(base);x=torch.randn(3,7)
    assert torch.equal(m(x),ref(x));checks['zero_adapter_exact']=True
    frozen=base.weight.detach().clone();frozenbias=base.bias.detach().clone();opt=torch.optim.AdamW([m.lora_a,m.lora_b],lr=.01)
    a0=m.lora_a.detach().clone()
    for _ in range(3):
        opt.zero_grad();m(x).square().mean().backward();assert base.weight.grad is None;opt.step()
    assert torch.equal(base.weight,frozen) and torch.equal(base.bias,frozenbias)
    assert not torch.equal(a0,m.lora_a) and m.lora_b.abs().sum()>0;checks['adapter_gradient_and_base_frozen']=True
    restored=LoRALinear(copy.deepcopy(ref));restored.load_state_dict(m.state_dict());assert torch.equal(m(x),restored(x));checks['adapter_roundtrip']=True
    # Ensure masked gradient accumulation equals a full logical-batch mean, including short tails.
    a=nn.Linear(7,7);b=copy.deepcopy(a);inp=torch.randn(3,10,7);target=torch.randn(3,10,7)
    mask=torch.tensor([[1]*10,[1]*3+[0]*7,[1]*7+[0]*3],dtype=torch.bool)
    total=7*mask.sum();((a(inp)-target).square()*mask[...,None]).sum().div(total).backward()
    for i in range(3):(((b(inp[i:i+1])-target[i:i+1]).square()*mask[i:i+1,:,None]).sum()/total).backward()
    assert torch.allclose(a.weight.grad,b.weight.grad,atol=1e-7);checks['valid_target_weighted_accumulation']=True
    from transformers import AutoConfig,AutoModelForImageTextToText
    cfg=AutoConfig.from_pretrained(R/'cache/models/smolvlm2_500m_7b375e1',local_files_only=True)
    with torch.device('meta'):full=AutoModelForImageTextToText.from_config(cfg,trust_remote_code=False)
    targets=inject_attention_lora(full)
    adapters=adapter_parameters(full)
    assert len(targets)==176 and sum(p.numel() for p in adapters.values())==2228224
    assert all(p.requires_grad==n.endswith(('.lora_a','.lora_b')) for n,p in full.named_parameters())
    assert not any('connector' in n or 'embed' in n or 'lm_head' in n for n in adapters)
    checks['real_smolvlm_meta_176_targets_2228224_parameters']=True
    del full
    head=RegressionHead();ck=torch.load(R/'results/mini_vla_m8_regression_train_20261021_run1/training/best.pt',map_location='cpu',weights_only=False)
    head.load_state_dict(ck['state_dict']);assert ck['epoch']==12 and sum(p.numel() for p in head.parameters())==398663
    checks['unchanged_epoch12_regression_head']=True
    dataset=VLMCacheData(R/'datasets/mini_vla_vlm_fulltokens_20261020_batch1_v2',R/'datasets/mini_vla_four_suite_128_20261019_v1','train',max_open_episodes=1)
    e=dataset.index[0];sample=dataset[e['end']-1];assert sample['action_mask'].sum()==1 and sample['actions'].shape==(10,7)
    assert sample['state'].shape==(8,) and sample['hidden'].dtype==torch.bfloat16
    checks['original_numeric_tail_mask_and_state']=True
    plan={'phase':'modern_m6_lora_preflight','resources':{'stage_cumulative_seconds':900,'additional_artifact_bytes':64*2**20}}
    ledger=json.loads((R/'results/mini_vla_m6_20261019_ledger.json').read_text());alloc=215840034816
    assert eligibility(plan,ledger,99*2**30,alloc)['status']=='WAITING_FOR_STORAGE_RESERVE'
    assert eligibility(plan,ledger,120*2**30,alloc)['status']=='DISK_READY'
    exhausted=copy.deepcopy(ledger);exhausted['additional_entries'].append({'stage_id':'modern_m6_lora_preflight','elapsed_seconds':900})
    assert eligibility(plan,exhausted,120*2**30,alloc)['status']=='BUDGET_BLOCKED'
    checks['disk_and_cumulative_budget_fail_closed']=True
    result=dict(status='PASS',checks=checks,elapsed_seconds=time.monotonic()-began,gpu_operations=0,retained_optimizer_updates=0,disposable_cpu_updates=3,
                limitation='Real CUDA forward/backward, online parity and activation peaks remain unmeasured until preflight.')
    (R/'reports/smolvlm_lora_cpu_contract_20261027.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
