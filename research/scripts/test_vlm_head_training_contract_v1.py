"""Real-cache transport parity and checkpoint-resume controls; no persistent training."""
from pathlib import Path
import json,tempfile,time
import numpy as np
import torch
from mini_vla.vlm_cached_data_v1 import VLMCacheData as Original,collate_cached as collate_original
from mini_vla.vlm_cached_data_v2 import VLMCacheData,collate_cached
from mini_vla.vlm_action_heads_v1 import RegressionHead,HeadConfig
from mini_vla.model import masked_mse
from mini_vla.train_vlm_head_v1 import save
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(42);cache=R/'datasets/mini_vla_vlm_fulltokens_20261020_batch1_v2';numeric=R/'datasets/mini_vla_four_suite_128_20261019_v1';old=Original(cache,numeric,'train');new=VLMCacheData(cache,numeric,'train');ids=[]
 for t in [0,9,10,19,20,29,30,39]:
  locations=np.flatnonzero(new.numeric['task_id']==t);ids.append(int(locations[len(locations)//2]))
 a=collate_original([old[i] for i in ids]);b=collate_cached([new[i] for i in ids]);assert b['hidden'].dtype==torch.bfloat16
 for k in a:torch.testing.assert_close(a[k],b[k].to(a[k].dtype),rtol=0,atol=0)
 model=RegressionHead().eval();obs_a={k:a[k] for k in ['hidden','hidden_mask','state']};obs_b={k:b[k] for k in ['hidden','hidden_mask','state']}
 with torch.no_grad():torch.testing.assert_close(model(**obs_a),model(**obs_b),rtol=0,atol=0)
 assert sum(p.numel() for p in model.parameters())==398663
 # Two additional updates after serialization must match an uninterrupted continuation.
 model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 def step(m,o):
  o.zero_grad(set_to_none=True);loss=masked_mse(m(**obs_b),b['actions'],b['action_mask']);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step();return float(loss)
 first=step(model,optimizer)
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'resume.pt';save(p,dict(model=model.state_dict(),optimizer=optimizer.state_dict()));step(model,optimizer);expected=step(model,optimizer)
  ck=torch.load(p,map_location='cpu',weights_only=False);restored=RegressionHead();restored.load_state_dict(ck['model']);opt=torch.optim.AdamW(restored.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(ck['optimizer']);step(restored,opt);actual=step(restored,opt)
  assert actual==expected
  for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
 # Endpoints preserve original source action windows and never cross a discontinuous episodeID.
 for e in new.index:
  idx,valid=new.window_indices(e['end']-1);assert idx.tolist()==[e['end']-1]*10 and valid.tolist()==[True]+[False]*9
 report=dict(status='PASS',real_frames=8,suites=4,parameter_count=398663,BF16_transport_exact_to_FP32=True,CPU_forward_exact=True,optimizer_resume_exact=True,episode_end_checks=len(new.index),disposable_CPU_optimizer_updates=5,weights_discarded=True,gpu_operations=0,elapsed_seconds=time.monotonic()-start)
 (R/'reports/m8_regression_training_contract_20261021.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
