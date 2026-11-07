#!/usr/bin/env python3
import argparse, dataclasses, json, logging, os, pathlib
import jax
from openpi.training import config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize

ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
class SeededPolicy:
    def __init__(self,policy):self.policy=policy
    def infer(self,obs):
        obs=dict(obs);seed=obs.pop('_policy_seed',None)
        if seed is not None:self.policy._rng=jax.random.key(int(seed))
        return self.policy.infer(obs)

def main():
    logging.basicConfig(level=logging.INFO)
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--variant',choices=['base','trained'],required=True);p.add_argument('--checkpoint');a=p.parse_args()
    assert jax.default_backend()=='gpu' and jax.device_count()==1
    cfg=config.get_config('pi05_libero_lora_3090x2_train')
    if a.variant=='base':
        cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m'))
        ck=ROOT/'cache/openpi/openpi-assets/checkpoints/pi05_base'
        norm=normalize.load(ROOT/'configs/assets/pi05_libero_lora_3090_train/physical-intelligence/libero')
    else:
        ck=pathlib.Path(a.checkpoint).resolve();assert ck.is_relative_to(ROOT/'checkpoints')
        norm=None  # Reload the stats saved with the actual trained checkpoint.
    policy=policy_config.create_trained_policy(cfg,ck,norm_stats=norm,sample_kwargs={'num_steps':10})
    server=websocket_policy_server.WebsocketPolicyServer(SeededPolicy(policy),host='127.0.0.1',port=a.port,metadata={'variant':a.variant,'action_horizon':10,'episode_rng_reset_supported':True})
    server.serve_forever()
if __name__=='__main__':main()
