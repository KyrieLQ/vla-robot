import copy,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.paired_continuation_v1 import RawData,lr_factor,sample_order,loss_numerator
from mini_vla.vlm_cached_data_v2 import VLMCacheData
from mini_vla.vlm_action_heads_v1 import HeadConfig,RegressionHead
from mini_vla.smolvlm_lora_v1 import LoRALinear
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);checks={}
 order=sample_order(243000);assert len(set(order))==243000 and np.array_equal(order,sample_order(243000));assert len(order[1898*128:])==56
 checks['full_order_1899_updates_last56_and_same_order']=True
 assert lr_factor(0)==.01 and lr_factor(99)==1 and abs(lr_factor(1898)-.1)<1e-12
 checks['schedule_warmup_and_endpoint']=True
 a=RegressionHead(HeadConfig(hidden_size=8,width=8,attention_heads=2,feedforward=16));b=copy.deepcopy(a)
 obs=dict(hidden=torch.randn(3,6,8),hidden_mask=torch.tensor([[1]*6,[1]*4+[0]*2,[1]*5+[0]],dtype=torch.bool),state=torch.randn(3,8));y=torch.randn(3,10,7);mask=torch.arange(10)[None]<torch.tensor([10,3,1])[:,None];den=7*mask.sum()
 (loss_numerator(a(**obs),y,mask)/den).backward()
 for i in range(3):(loss_numerator(b(**{k:v[i:i+1] for k,v in obs.items()}),y[i:i+1],mask[i:i+1])/den).backward()
 assert all(torch.allclose(x.grad,z.grad,atol=2e-6,rtol=2e-5) for x,z in zip(a.parameters(),b.parameters()))
 checks['actual_decoder_masked_accumulation_matches_full_logical_batch']=True
 # Real target windows / raw path match original cache, including each split's episode tails.
 p=json.loads((R/'configs/mini_vla_m6_lora_preflight_20261027_v1.json').read_text())
 for part in ['train','validation']:
  raw=RawData(p,part);cached=VLMCacheData(R/p['cache'],R/p['numeric_cache'],part,max_open_episodes=1)
  for i in [0,raw.index[0]['end']-1,raw.index[1]['start'],len(raw)-1]:
   r,c=raw[i],cached[i]
   for k in ['state','actions','action_mask']:assert torch.equal(r[k],c[k]),(part,i,k)
   assert 'hidden' not in r and r['record']['images'][0].size==(256,256)
  assert sum(min(10,e['end']-i) for e in raw.index for i in range(e['start'],e['end']))==({'train':2362230,'validation':296235}[part])
 checks['raw_dataset_no_stale_features_full_tail_counts_and_boundaries']=True
 # Optimizer state and adapter export survive a committed-step restart exactly on CPU.
 m=LoRALinear(torch.nn.Linear(7,9));opt=torch.optim.AdamW([m.lora_a,m.lora_b],lr=1e-3);inp=torch.randn(2,7)
 def step(model,optimizer):optimizer.zero_grad();model(inp).square().sum().backward();optimizer.step()
 step(m,opt)
 with tempfile.TemporaryDirectory(dir=R/'tmp') as tmp:
  f=Path(tmp)/'latest.pt';torch.save(dict(model=m.state_dict(),opt=opt.state_dict()),f);c=torch.load(f,weights_only=False)
  n=copy.deepcopy(m);n.load_state_dict(c['model']);op2=torch.optim.AdamW([n.lora_a,n.lora_b],lr=1e-3);op2.load_state_dict(c['opt']);step(m,opt);step(n,op2)
  assert all(torch.equal(x,z) for x,z in zip(m.state_dict().values(),n.state_dict().values()))
 checks['adapter_optimizer_exact_step_resume']=True
 res=dict(status='PASS',checks=checks,elapsed_seconds=time.monotonic()-start,gpu_operations=0,formal_updates=0)
 (R/'reports/paired_continuation_cpu_contract_20261028.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
if __name__=='__main__':main()
