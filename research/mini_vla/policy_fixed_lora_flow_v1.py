"""Selected newly trained Flow head on fixed regression-adapted SmolVLM."""
import argparse,json
from pathlib import Path
import torch
from .policy_m9_v1 import M9Policy,R,sha
from .policy_paired_v1 import AdaptedFrontend
from .vlm_action_heads_v1 import FlowHead,HeadConfig

def load_policy(plan):
 assert sha(R/plan['checkpoint'])==plan['checkpoint_sha256'] and sha(R/plan['backbone_checkpoint'])==plan['backbone_checkpoint_sha256']
 ck=torch.load(R/plan['checkpoint'],map_location='cpu',weights_only=False);base=torch.load(R/plan['backbone_checkpoint'],map_location='cpu',weights_only=False)
 assert ck['architecture']=='fixed_adapted_vlm_flow' and ck['epoch']==plan['selected_epoch'] and ck['vlm_and_adapters_frozen'] and ck['frontend_checkpoint_sha256']==plan['backbone_checkpoint_sha256']
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 front=AdaptedFrontend(R/plan['model_directory'],base['adapters']);head=FlowHead(HeadConfig(**ck['model_config']));head.load_state_dict(ck['state_dict']);head.cuda().eval()
 assert head.config.euler_steps==10 and not any(x.requires_grad for x in front.model.parameters())
 return M9Policy(front,head,ck['preprocessing']['normalization']),ck

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--port',type=int,required=True);ap.add_argument('--contract-output',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());policy,ck=load_policy(p)
 from .check_fixed_lora_flow_online_v1 import check
 check(p,a.contract_output,policy)
 from openpi.serving.websocket_policy_server import WebsocketPolicyServer
 metadata=dict(model='fixed_adapted_vlm_flow',arm='flow',epoch=ck['epoch'],updates=ck['updates'],action_horizon=10,euler_steps=10,noise_seed_rule='SHA256(20261020,task,init,call_index)',checkpoint_sha256=p['checkpoint_sha256'],backbone_checkpoint_sha256=p['backbone_checkpoint_sha256'],adapter_sha256=sha(Path(__file__)))
 WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=metadata).serve_forever()
if __name__=='__main__':main()
