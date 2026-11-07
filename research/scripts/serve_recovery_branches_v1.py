#!/usr/bin/env python3
"""Original policy, with explicit finite replay of its PRNG carried key only."""
import argparse,logging,os,pathlib
import jax
from openpi.training import config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

class ReplayPolicy:
    def __init__(self,policy):self.policy=policy
    def infer(self,obs):
        obs=dict(obs);seed=obs.pop('_policy_seed',None);advance=obs.pop('_policy_rng_advance',0)
        assert isinstance(advance,int) and 0<=advance<=44
        if seed is not None:
            key=jax.random.key(int(seed))
            for _ in range(advance):key,_=jax.random.split(key)
            self.policy._rng=key
        else:assert advance==0
        assert set(obs)=={'observation/image','observation/wrist_image','observation/state','prompt'}
        return self.policy.infer(obs)

def main():
    logging.basicConfig(level=logging.INFO)
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
    assert jax.default_backend()=='gpu' and jax.device_count()==1
    ck=pathlib.Path(a.checkpoint).resolve();assert ck.is_relative_to(ROOT/'checkpoints')
    cfg=config.get_config('pi05_libero_lora_3090x2_train')
    policy=policy_config.create_trained_policy(cfg,ck,sample_kwargs={'num_steps':10})
    websocket_policy_server.WebsocketPolicyServer(ReplayPolicy(policy),host='127.0.0.1',port=a.port,metadata={'action_horizon':10,'rng_advance_supported':True,'privileged_policy_inputs':False}).serve_forever()
if __name__=='__main__':main()
