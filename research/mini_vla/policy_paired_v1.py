"""Frozen or LoRA continued M6 regression head; identical observation/action interface."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from .policy_m8_v1 import M8Policy,sha,R
from .smolvlm_lora_v1 import DifferentiableFrontend,adapter_parameters
from .vlm_frontend_v1 import FrozenFrontend,decode_record,prompt_for,bf16_bits,bits_bf16
from .vlm_action_heads_v1 import RegressionHead,HeadConfig
from .paired_continuation_v1 import RawData
from .data import normalize

class AdaptedFrontend:
 def __init__(self,path,weights):
  self.front=DifferentiableFrontend(str(path),rank=8,alpha=8,checkpointing=False)
  expected=adapter_parameters(self.front.model);assert set(expected)==set(weights)
  with torch.no_grad():
   for k,p in expected.items():
    assert p.shape==weights[k].shape and torch.isfinite(weights[k]).all();p.copy_(weights[k])
  self.front.eval().requires_grad_(False);self.model=self.front.model;self.processor=self.front.processor;self.loading=self.front.loading
 @torch.inference_mode()
 def encode(self,records):
  assert len(records)==1
  inputs=self.front.inputs(records);hidden,mask=self.front(inputs);result=[]
  for i,record in enumerate(records):
   h=hidden[i,mask[i]];assert torch.isfinite(h).all()
   result.append(dict(bits=bf16_bits(h),input_sha256=hashlib.sha256(b''.join(im.tobytes() for im in record['images'])+prompt_for(self.processor,record['instruction']).encode()).hexdigest()))
  return result

def load_policy(plan,arm):
 chosen=plan['arms'][arm];assert sha(R/chosen['checkpoint'])==chosen['checkpoint_sha256']
 ck=torch.load(R/chosen['checkpoint'],map_location='cpu',weights_only=False)
 assert ck['architecture']=='smolvlm_paired_regression' and ck['arm']==arm and ck['updates']==1899
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 front=FrozenFrontend(str(R/plan['model_directory'])) if arm=='frozen' else AdaptedFrontend(R/plan['model_directory'],ck['adapters'])
 head=RegressionHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict']);head.cuda().eval()
 return M8Policy(front,head,ck['preprocessing']['normalization']),ck

@torch.inference_mode()
def check(plan,arm,policy,output):
 started=time.monotonic();data=RawData(plan,'train');index={e['episode_id']:e for e in data.index}
 samples=json.loads((R/plan['sample_manifest']).read_text())['samples'];split=json.loads((R/plan['split']).read_text());train={e['episode_id'] for e in split['train']};val={e['episode_id'] for e in split['validation']};assert len(samples)==120
 old=json.loads((R/plan['cache']/'gate/complete.json').read_text())['comparisons'];rows=[];latencies=[]
 for i,s in enumerate(samples):
  eid=s['episode_id'];frame=s['frame_index'];assert eid in train-val
  e=index[eid];num=data[e['start']+frame];record=num['record']
  source=R/plan['dataset_root']/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet'
  raw=pq.read_table(source,columns=['state'],use_threads=False).slice(frame,1).to_pylist()[0]['state']
  obs={'observation/image':np.asarray(record['images'][0]),'observation/wrist_image':np.asarray(record['images'][1]),'observation/state':np.asarray(raw),'prompt':s['instruction']}
  tick=time.monotonic();deployed=policy.infer(obs);latencies.append(time.monotonic()-tick)
  assert deployed['vlm_input_sha256']==old[i]['input_sha256'] and (old[i]['episode_id'],old[i]['frame_index'])==(eid,frame)
  if arm=='frozen':
   stored=np.load(R/plan['cache']/e['directory']/'hidden.npy',mmap_mode='r')[frame]
   h=bits_bf16(stored)[None].cuda();mask=torch.ones(h.shape[:2],dtype=torch.bool,device=h.device)
  else:
   # Independent training-interface route: current adapted model, original raw record.
   # Never use old cached hidden features as a LoRA reference.
   h,mask=policy.frontend.front(policy.frontend.front.inputs([record]));stored=bf16_bits(h[0,mask[0]])
  assert deployed['hidden_sha256']==hashlib.sha256(stored.tobytes()).hexdigest()
  np.testing.assert_allclose(normalize(np.asarray(raw),policy.normalization['state']).numpy(),num['state'].numpy(),atol=1e-6,rtol=0)
  expected=policy.head(hidden=h,hidden_mask=mask,state=num['state'][None].cuda())[0].cpu().numpy()
  np.testing.assert_allclose(deployed['normalized_actions'],expected,atol=1e-6,rtol=0)
  stats=policy.normalization['actions'];commands=np.clip(expected*np.asarray(stats['scale'],dtype=np.float32)+np.asarray(stats['mean'],dtype=np.float32),-1,1)
  np.testing.assert_allclose(deployed['actions'],commands,atol=1e-6,rtol=0)
  rows.append(dict(episode_id=eid,frame_index=frame,input_sha256=deployed['vlm_input_sha256'],hidden_sha256=deployed['hidden_sha256'],action_max_error=float(np.max(np.abs(expected-deployed['normalized_actions'])))))
 result=dict(status='PASS',arm=arm,samples=120,rows=rows,features_bitwise_equal_to='original_frozen_cache' if arm=='frozen' else 'current_adapted_raw_training_interface',maximum_action_error=max(x['action_max_error'] for x in rows),loading=policy.frontend.loading,checkpoint_sha256=plan['arms'][arm]['checkpoint_sha256'],all_deployed_parameters_frozen=not any(p.requires_grad for p in policy.frontend.model.parameters()),inference_seconds_p95=float(np.quantile(latencies,.95)),elapsed_seconds=time.monotonic()-started,new_simulation_episodes=0)
 assert result['all_deployed_parameters_frozen'];Path(output).write_text(json.dumps(result,indent=2)+'\n')
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--arm',choices=['frozen','lora'],required=True);ap.add_argument('--port',type=int,required=True);ap.add_argument('--contract-output',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());policy,ck=load_policy(plan,a.arm)
 check(plan,a.arm,policy,a.contract_output)
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 metadata=dict(model='paired_smolvlm_regression',arm=a.arm,updates=ck['updates'],checkpoint_sha256=plan['arms'][a.arm]['checkpoint_sha256'],action_horizon=10,adapter_sha256=sha(Path(__file__)))
 WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
if __name__=='__main__':main()
