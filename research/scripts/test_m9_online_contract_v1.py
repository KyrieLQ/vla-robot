"""Stateless sampling IDs, private RNG, cache interface and control conversion."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.policy_m9_v1 import M9Policy
started=time.monotonic();torch.set_num_threads(4)
class Frontend:
 def encode(self,records):
  assert len(records)==1
  assert np.asarray(records[0]['images'][0])[0,0,0]==17 and np.asarray(records[0]['images'][1])[0,0,0]==29
  return [dict(bits=np.zeros((155,960),np.uint16),input_sha256='a'*64)]
class Head(torch.nn.Module):
 def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
 def sample(self,*,hidden,hidden_mask,state,noise):
  assert hidden.shape==(1,155,960) and hidden_mask.all()
  torch.testing.assert_close(state,torch.ones(1,8),atol=0,rtol=0)
  return noise
stats=dict(state=dict(mean=list(range(8)),scale=[2.]*8),actions=dict(mean=[.1]*7,scale=[.5]*7))
p=M9Policy(Frontend(),Head(),stats)
obs={'observation/image':np.full((256,256,3),17,np.uint8),'observation/wrist_image':np.full((256,256,3),29,np.uint8),'observation/state':np.arange(8,dtype=np.float64)+2,'prompt':'pick up the bowl','sampling/task_id':3,'sampling/initial_state_index':12,'sampling/call_index':4}
before=torch.get_rng_state().clone();r=p.infer(obs);assert torch.equal(before,torch.get_rng_state())
seed=int.from_bytes(hashlib.sha256(b'20261020:3:12:4').digest()[:8],'little')%(2**63-1)
noise=torch.randn((10,7),generator=torch.Generator().manual_seed(seed),dtype=torch.float32).numpy()
assert r['noise_seed']==seed and r['noise_sha256']==hashlib.sha256(noise.tobytes()).hexdigest()
np.testing.assert_array_equal(r['normalized_actions'],noise)
np.testing.assert_allclose(r['actions'],np.clip(noise*.5+.1,-1,1),atol=1e-7,rtol=0)
other=dict(obs);other['sampling/call_index']=5;different=p.infer(other);assert different['noise_seed']!=seed and not np.array_equal(r['actions'],different['actions'])
np.testing.assert_array_equal(p.infer(obs)['actions'],r['actions'])
extra=dict(obs,target_actions=np.full((10,7),1e9),action_mask=np.zeros(10,bool));np.testing.assert_array_equal(p.infer(extra)['actions'],r['actions'])
bad=dict(obs);bad['sampling/call_index']=-1
try:p.infer(bad)
except ValueError:pass
else:raise AssertionError('Negative call index accepted')
from openpi_client import msgpack_numpy
packed=msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(r));assert packed['noise_seed']==seed
result=dict(status='PASS',sampling_seed_rule_independently_verified=True,noise_hash_verified=True,stateless_call_order=True,global_RNG_unchanged=True,targets_and_target_mask_not_consumed=True,camera_order_state_norm_and_action_inverse_clip=True,transport_seed_verified=True,elapsed_seconds=time.monotonic()-started,new_training_updates=0,new_simulation_episodes=0)
Path('reports/m9_online_cpu_contract_20261021.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
