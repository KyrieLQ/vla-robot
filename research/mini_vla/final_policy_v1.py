"""Selected M6 policy, same adapter/weights, bounded single-thread CPU preprocessing."""
import argparse,hashlib,json
from pathlib import Path
import torch
from .attention_model_v1 import AttentionMiniVLA,AttentionConfig
from .policy import MiniVLAPolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text());file=R/plan['checkpoint']
 assert hashlib.sha256(file.read_bytes()).hexdigest()==plan['checkpoint_sha256']
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False
 ck=torch.load(file,map_location='cpu',weights_only=False);assert ck['epoch']==3
 model=AttentionMiniVLA(AttentionConfig(**ck['model_config'])).cuda();model.load_state_dict(ck['state_dict'],strict=True);model.eval()
 policy=MiniVLAPolicy(model,ck['preprocessing']['vocabulary'],ck['preprocessing']['normalization'])
 metadata=dict(model='mini_vla_m6_final',epoch=3,action_horizon=10,checkpoint_sha256=plan['checkpoint_sha256'],cpu_threads=1,adapter_sha256=hashlib.sha256((R/'mini_vla/policy.py').read_bytes()).hexdigest())
 WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
if __name__=='__main__':main()
