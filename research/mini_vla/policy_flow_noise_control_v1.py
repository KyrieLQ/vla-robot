"""Same fixed Flow weights; isolate noise persistence across replanning calls."""
import argparse, hashlib, json, time
import numpy as np
import torch
from .policy_m9_v1 import R, observation_digest
from .policy_fixed_lora_flow_v1 import load_policy as load_original
from .vlm_frontend_v1 import records_from_observation, bits_bf16
from .data import normalize, denormalize

def noise_identity(task, initial, call, mode):
    ids=[task,initial,call]
    if any(not isinstance(x,(int,np.integer)) or isinstance(x,bool) or x<0 for x in ids):
        raise ValueError('Nonnegative sampling identities required')
    if mode not in ('fresh','episode_fixed'):raise ValueError(mode)
    source_call=call if mode=='fresh' else 0
    seed=int.from_bytes(hashlib.sha256(f'20261020:{task}:{initial}:{source_call}'.encode()).digest()[:8],'little')%(2**63-1)
    noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32)
    return noise,seed,hashlib.sha256(noise.numpy().tobytes()).hexdigest()

class NoiseControlPolicy:
    def __init__(self,base,mode):
        self.frontend,self.head,self.normalization=base.frontend,base.head,base.normalization
        self.mode=mode

    @torch.inference_mode()
    def infer(self,observation):
        image=observation['observation/image'];wrist=observation['observation/wrist_image']
        state=np.asarray(observation['observation/state']);prompt=observation['prompt']
        if state.shape!=(8,) or not np.isfinite(state).all() or not isinstance(prompt,str):raise ValueError('Invalid state/instruction')
        ids=[observation['sampling/'+k] for k in ['task_id','initial_state_index','call_index']]
        noise,seed,noise_hash=noise_identity(*ids,self.mode)
        record=records_from_observation(image,wrist,prompt);before=time.perf_counter()
        feature=self.frontend.encode([record])[0];feature_seconds=time.perf_counter()-before
        device=next(self.head.parameters()).device
        hidden=bits_bf16(feature['bits'])[None].to(device)
        robot=normalize(state,self.normalization['state'])[None].to(device)
        obs=dict(hidden=hidden,hidden_mask=torch.ones(hidden.shape[:2],dtype=torch.bool,device=device),state=robot)
        normalized=self.head.sample(**obs,noise=noise[None].to(device))[0]
        commands=denormalize(normalized,self.normalization['actions']).clamp(-1,1)
        # This counterfactual uses exactly the same observation features. Never execute or select it.
        shadow_noise,shadow_seed,shadow_hash=noise_identity(*ids,'fresh')
        shadow=normalized if self.mode=='fresh' or ids[-1]==0 else self.head.sample(**obs,noise=shadow_noise[None].to(device))[0]
        shadow_commands=denormalize(shadow,self.normalization['actions']).clamp(-1,1)
        assert torch.isfinite(normalized).all() and torch.isfinite(shadow).all()
        return dict(actions=commands.cpu().numpy().astype(np.float32),normalized_actions=normalized.cpu().numpy().astype(np.float32),
                    noise_seed=seed,noise_sha256=noise_hash,noise_mode=self.mode,source_call_index=0 if self.mode=='episode_fixed' else int(ids[-1]),
                    shadow_fresh_actions=shadow_commands.cpu().numpy().astype(np.float32),
                    shadow_fresh_normalized_actions=shadow.cpu().numpy().astype(np.float32),
                    shadow_fresh_noise_seed=shadow_seed,shadow_fresh_noise_sha256=shadow_hash,
                    hidden_sha256=hashlib.sha256(feature['bits'].tobytes()).hexdigest(),vlm_input_sha256=feature['input_sha256'],
                    observation_sha256=observation_digest(image,wrist,state,prompt),
                    component_timing=dict(vlm_seconds=feature_seconds,total_seconds=time.perf_counter()-before))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--port',type=int,required=True);ap.add_argument('--contract-output',required=True)
    args=ap.parse_args();p=json.loads(open(args.plan).read());assert p['noise_mode']=='episode_fixed'
    base,ck=load_original(p);policy=NoiseControlPolicy(base,p['noise_mode'])
    from .check_flow_noise_control_v1 import check
    check(p,args.contract_output,policy)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    metadata=dict(model='flow_noise_temporal_control',arm='flow',epoch=ck['epoch'],updates=ck['updates'],action_horizon=10,euler_steps=10,
                  noise_mode=p['noise_mode'],noise_seed_rule='SHA256(20261020,task,init,0)',
                  checkpoint_sha256=p['checkpoint_sha256'],backbone_checkpoint_sha256=p['backbone_checkpoint_sha256'])
    WebsocketPolicyServer(policy,host='127.0.0.1',port=args.port,metadata=metadata).serve_forever()

if __name__=='__main__':main()
