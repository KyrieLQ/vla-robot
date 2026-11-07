"""Online frozenVLM+M9 conditional flow adapter, preserving the exact cached input contract."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .vlm_frontend_v1 import FrozenFrontend,records_from_observation,bits_bf16
from .vlm_action_heads_v1 import FlowHead,HeadConfig
from .data import normalize,denormalize
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def observation_digest(image,wrist,state,prompt):
 return hashlib.sha256(image.tobytes()+wrist.tobytes()+np.asarray(state,dtype='<f4').tobytes()+prompt.encode()).hexdigest()
class M9Policy:
 def __init__(self,frontend,head,normalization):self.frontend=frontend;self.head=head.eval();self.normalization=normalization
 @torch.inference_mode()
 def infer(self,observation):
  image=observation['observation/image'];wrist=observation['observation/wrist_image'];state=np.asarray(observation['observation/state']);prompt=observation['prompt']
  if state.shape!=(8,) or not np.isfinite(state).all() or not isinstance(prompt,str):raise ValueError('Invalid state/instruction')
  record=records_from_observation(image,wrist,prompt);before=time.perf_counter();feature=self.frontend.encode([record])[0];feature_seconds=time.perf_counter()-before
  device=next(self.head.parameters()).device;hidden=bits_bf16(feature['bits'])[None].to(device);robot=normalize(state,self.normalization['state'])[None].to(device)
  ids=[observation['sampling/'+k] for k in ['task_id','initial_state_index','call_index']]
  if any(not isinstance(x,(int,np.integer)) or isinstance(x,bool) or x<0 for x in ids):raise ValueError('Nonnegative sampling identities required')
  raw=(':'.join(str(x) for x in [20261020]+ids)).encode();noise_seed=int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')%(2**63-1)
  noise=torch.randn((10,7),generator=torch.Generator().manual_seed(noise_seed),dtype=torch.float32)
  normalized=self.head.sample(hidden=hidden,hidden_mask=torch.ones(hidden.shape[:2],dtype=torch.bool,device=device),state=robot,noise=noise[None].to(device))[0]
  commands=denormalize(normalized,self.normalization['actions']);assert commands.shape==(10,7) and torch.isfinite(commands).all()
  result=dict(noise_seed=noise_seed,noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),actions=commands.clamp(-1,1).cpu().numpy().astype(np.float32),normalized_actions=normalized.cpu().numpy().astype(np.float32),hidden_sha256=hashlib.sha256(feature['bits'].tobytes()).hexdigest(),vlm_input_sha256=feature['input_sha256'],observation_sha256=observation_digest(image,wrist,state,prompt),component_timing=dict(vlm_seconds=feature_seconds,total_seconds=time.perf_counter()-before))
  return result

def load_policy(plan):
 file=R/plan['checkpoint'];assert sha(file)==plan['checkpoint_sha256'];ck=torch.load(file,map_location='cpu',weights_only=False);assert ck['epoch']==plan['selected_epoch'] and ck['vlm_frozen'] and ck['architecture']=='frozen_vlm_flow' and ck['model_config']['euler_steps']==10
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 frontend=FrozenFrontend(R/plan['model_directory']);head=FlowHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict']);head.cuda().eval()
 return M9Policy(frontend,head,ck['preprocessing']['normalization']),ck

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--contract-output');a=p.parse_args();plan=json.loads(Path(a.plan).read_text());policy,ck=load_policy(plan)
 if a.contract_output:
  from .check_m9_online_v1 import check
  check(plan,a.contract_output,policy)
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 metadata=dict(model='frozen_vlm_flow_m9',epoch=ck['epoch'],action_horizon=10,euler_steps=10,noise_seed_rule="SHA256(20261020,task,init,call_index)",checkpoint_sha256=plan['checkpoint_sha256'],frontend_binding=ck['cache_binding'],adapter_sha256=sha(Path(__file__)),head_parameters=sum(x.numel() for x in policy.head.parameters()))
 WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
if __name__=='__main__':main()
