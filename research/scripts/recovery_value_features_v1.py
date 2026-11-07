"""Frozen VLA prefix features; no policy sampling, simulation or parameter updates."""
import argparse,json,os,pathlib,time
import numpy as np
from PIL import Image
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.training import config
from openpi.shared import nnx_utils
from recovery_value_data_v1 import load_records
from paired_live_audit_v1 import image_hash,sha
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
class PrefixTap(nnx.Module):
    def __init__(self,model):self.model=model
    def extract(self,obs):
        obs=model_lib.preprocess_observation(None,obs,train=False)
        tokens,mask,ar=self.model.embed_prefix(obs)
        outputs,_=self.model.PaliGemma.llm([tokens,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1)
        hidden=outputs[0].astype(jnp.float32)
        text_length=obs.tokenized_prompt.shape[1]
        image_length=(tokens.shape[1]-text_length)//len(obs.images)
        keys=list(obs.images);pools=[]
        for name in ('base_0_rgb','left_wrist_0_rgb'):
            index=keys.index(name);pools.append(hidden[:,index*image_length:(index+1)*image_length].mean(axis=1))
        text=hidden[:,-text_length:];text_mask=obs.tokenized_prompt_mask
        pools.append(jnp.where(text_mask[...,None],text,0).sum(axis=1)/jnp.maximum(text_mask.sum(axis=1,keepdims=True),1))
        return jnp.concatenate(pools,axis=-1)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--run',required=True);a=parser.parse_args()
    run=pathlib.Path(a.run);plan=json.loads(pathlib.Path(a.plan).read_text());start=time.monotonic()
    assert jax.default_backend()=='gpu' and jax.device_count()==1 and plan['vla_parameters_frozen']
    records=load_records(plan)
    policy=policy_config.create_trained_policy(config.get_config('pi05_libero_lora_3090x2_train'),ROOT/plan['checkpoint'],sample_kwargs={'num_steps':10})
    tap=PrefixTap(policy._model);extract=nnx_utils.module_jit(tap.extract);features=[];latencies=[];repeat_error=None
    for i,row in enumerate(records):
        directory=pathlib.Path(row['directory']);images=[np.asarray(Image.open(directory/f'root_{camera}.png').convert('RGB')).copy() for camera in ('external','wrist')]
        assert image_hash(*images)==row['image_sha256']
        request={'observation/image':images[0],'observation/wrist_image':images[1],'observation/state':np.asarray(row['state']),'prompt':row['prompt']}
        assert set(request)=={'observation/image','observation/wrist_image','observation/state','prompt'}
        inputs=policy._input_transform(request);inputs=jax.tree.map(lambda x:jnp.asarray(x)[None,...],inputs)
        obs=model_lib.Observation.from_dict(inputs)
        tick=time.monotonic();value=np.asarray(extract(obs))[0];latencies.append(time.monotonic()-tick)
        assert value.shape==(6144,) and np.isfinite(value).all()
        if i==0:
            repeated=np.asarray(extract(obs))[0];repeat_error=float(np.abs(value-repeated).max())
            np.testing.assert_allclose(value,repeated,atol=1e-6,rtol=0)
        features.append(value.astype(np.float16))
        if i%20==0:
            (run/'feature_progress.json').write_text(json.dumps({'status':'RUNNING','completed':i+1,'expected':269,'elapsed_seconds':time.monotonic()-start})+'\n')
            print(json.dumps({'feature_roots':i+1,'expected':269}),flush=True)
    np.savez_compressed(run/'features.npz',prefix=np.stack(features),robot_history=np.asarray([r['robot_history'] for r in records],dtype=np.float32),action_history=np.asarray([r['action_history'] for r in records],dtype=np.float32))
    np.savez_compressed(run/'targets.npz',outcomes=np.asarray([r['outcomes'] for r in records],dtype=np.float32),calls=np.asarray([r['calls'] for r in records],dtype=np.float32),progress=np.asarray([r['progress_targets'] for r in records],dtype=np.float32),cost=np.asarray([r['normalized_positive_extra_calls'] for r in records],dtype=np.float32))
    # IDs are provenance/grouping fields only; models never receive these records.
    (run/'roots.json').write_text(json.dumps([{k:r[k] for k in ('root_id','pair_id','task_id','step','partition','directory','prompt','image_sha256')} for r in records],indent=2)+'\n')
    result={'status':'COMPLETE','roots':269,'fit_roots':186,'calibration_roots':83,'feature_shape':[269,6144],'feature_dtype':'float16','repeat_max_error':repeat_error,
            'elapsed_seconds':time.monotonic()-start,'encoder_seconds_including_first_compile':sum(latencies),'encoder_seconds_excluding_first_compile':sum(latencies[1:]),
            'first_compile_and_forward_seconds':latencies[0],'parameters_updated':0,'policy_actions_sampled':0,'simulator_steps':0,'confirmation_used':0,
            'feature_sha256':sha(run/'features.npz'),'target_sha256':sha(run/'targets.npz'),'root_metadata_sha256':sha(run/'roots.json'),
            'cost_note':'This offline extraction is extra compute. No live VLA-cache sharing has been implemented or credited.'}
    (run/'feature_progress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
