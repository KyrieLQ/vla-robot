"""CPU contracts for observation-only inference, camera order, standardization and clipping."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from mini_vla.policy_m8_v1 import M8Policy,observation_digest

started=time.monotonic();torch.set_num_threads(4)
class Frontend:
    def encode(self,records):
        assert len(records)==1
        r=records[0]
        assert np.asarray(r['images'][0])[0,0,0]==17
        assert np.asarray(r['images'][1])[0,0,0]==29
        assert r['instruction']=='pick up the bowl'
        return [dict(bits=np.zeros((155,960),dtype=np.uint16),input_sha256='a'*64)]
class Head(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
    def forward(self,hidden,hidden_mask,state):
        assert hidden.dtype==torch.bfloat16 and hidden.shape==(1,155,960)
        assert hidden_mask.all()
        torch.testing.assert_close(state,torch.ones(1,8),atol=0,rtol=0)
        return torch.tensor([-4.,-1.,0.,.5,1.,2.,4.]).repeat(1,10,1)
stats=dict(state=dict(mean=list(range(8)),scale=[2.]*8),actions=dict(mean=[.1]*7,scale=[.5]*7))
policy=M8Policy(Frontend(),Head(),stats)
obs={'observation/image':np.full((256,256,3),17,dtype=np.uint8),
     'observation/wrist_image':np.full((256,256,3),29,dtype=np.uint8),
     'observation/state':np.arange(8,dtype=np.float64)+2,'prompt':'pick up the bowl'}
result=policy.infer(obs)
np.testing.assert_allclose(result['actions'][0],[-1,-.4,.1,.35,.6,1,1],atol=1e-7,rtol=0)
assert result['actions'].dtype==np.float32 and result['actions'].shape==(10,7)
assert result['observation_sha256']==observation_digest(obs['observation/image'],obs['observation/wrist_image'],obs['observation/state'],obs['prompt'])
# Added targets cannot affect inference; they are never read by the observation path.
extra=dict(obs,target_actions=np.full((10,7),1e9))
np.testing.assert_array_equal(result['actions'],policy.infer(extra)['actions'])
for key,value in [('observation/image',np.zeros((128,128,3),dtype=np.uint8)),('observation/state',np.zeros(7))]:
    bad=dict(obs);bad[key]=value
    try:policy.infer(bad)
    except ValueError:pass
    else:raise AssertionError('Malformed observation accepted')
out=dict(status='PASS',camera_order=True,state_normalization=True,action_denormalization_and_clip=True,
         inference_ignores_future_targets=True,invalid_inputs_rejected=True,new_training_updates=0,
         elapsed_seconds=time.monotonic()-started)
Path('reports/m8_online_cpu_contract_20261021.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
