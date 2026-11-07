"""Online frozenVLM+M8 regression adapter, preserving the exact cached input contract."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .vlm_frontend_v1 import FrozenFrontend,records_from_observation,bits_bf16
from .vlm_action_heads_v1 import RegressionHead,HeadConfig
from .data import normalize,denormalize
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def observation_digest(image,wrist,state,prompt):
 return hashlib.sha256(image.tobytes()+wrist.tobytes()+np.asarray(state,dtype='<f4').tobytes()+prompt.encode()).hexdigest()
class M8Policy:
 def __init__(self,frontend,head,normalization):self.frontend=frontend;self.head=head.eval();self.normalization=normalization
 @torch.inference_mode()
 def infer(self,observation):
  image=observation['observation/image'];wrist=observation['observation/wrist_image'];state=np.asarray(observation['observation/state']);prompt=observation['prompt']
  if state.shape!=(8,) or not np.isfinite(state).all() or not isinstance(prompt,str):raise ValueError('Invalid state/instruction')
  record=records_from_observation(image,wrist,prompt);before=time.perf_counter();feature=self.frontend.encode([record])[0];feature_seconds=time.perf_counter()-before
  device=next(self.head.parameters()).device;hidden=bits_bf16(feature['bits'])[None].to(device);robot=normalize(state,self.normalization['state'])[None].to(device)
  normalized=self.head(hidden=hidden,hidden_mask=torch.ones(hidden.shape[:2],dtype=torch.bool,device=device),state=robot)[0]
  commands=denormalize(normalized,self.normalization['actions']);assert commands.shape==(10,7) and torch.isfinite(commands).all()
  result=dict(actions=commands.clamp(-1,1).cpu().numpy().astype(np.float32),normalized_actions=normalized.cpu().numpy().astype(np.float32),hidden_sha256=hashlib.sha256(feature['bits'].tobytes()).hexdigest(),vlm_input_sha256=feature['input_sha256'],observation_sha256=observation_digest(image,wrist,state,prompt),component_timing=dict(vlm_seconds=feature_seconds,total_seconds=time.perf_counter()-before))
  return result

def load_policy(plan):
 file=R/plan['checkpoint'];assert sha(file)==plan['checkpoint_sha256'];ck=torch.load(file,map_location='cpu',weights_only=False);assert ck['epoch']==plan['selected_epoch'] and ck['vlm_frozen']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 frontend=FrozenFrontend(R/plan['model_directory']);head=RegressionHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict']);head.cuda().eval()
 return M8Policy(frontend,head,ck['preprocessing']['normalization']),ck

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--contract-output');a=p.parse_args();plan=json.loads(Path(a.plan).read_text());policy,ck=load_policy(plan)
 if a.contract_output:
  from .check_m8_online_v1 import check
  check(plan,a.contract_output,policy)
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 metadata=dict(model='frozen_vlm_regression_m8',epoch=ck['epoch'],action_horizon=10,checkpoint_sha256=plan['checkpoint_sha256'],frontend_binding=ck['cache_binding'],adapter_sha256=sha(Path(__file__)),head_parameters=sum(x.numel() for x in policy.head.parameters()))
 WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
if __name__=='__main__':main()
