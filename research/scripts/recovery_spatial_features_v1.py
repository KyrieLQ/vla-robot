"""Fixed checkpoint spatial-prefix tap, no action sampling or simulation."""
import argparse,json,os,pathlib,time
import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy_config
from openpi.training import config
from openpi.shared import nnx_utils
from paired_live_audit_v1 import arrays,image_hash,sha
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
class SpatialTap(nnx.Module):
    def __init__(self,model):self.model=model
    def extract(self,obs):
        obs=model_lib.preprocess_observation(None,obs,train=False);tokens,mask,ar=self.model.embed_prefix(obs)
        outputs,_=self.model.PaliGemma.llm([tokens,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1);hidden=outputs[0].astype(jnp.float32)
        length=obs.tokenized_prompt.shape[1];n=(tokens.shape[1]-length)//len(obs.images);assert n==256 and hidden.shape[-1]==2048
        keys=list(obs.images);grids=[];pools=[]
        for name in ('base_0_rgb','left_wrist_0_rgb'):
            idx=keys.index(name);t=hidden[:,idx*n:(idx+1)*n];pools.append(t.mean(axis=1));grid=t.reshape(t.shape[0],4,4,4,4,2048).mean(axis=(2,4)).reshape(t.shape[0],16,2048);grids.append(grid)
        text=hidden[:,-length:];m=obs.tokenized_prompt_mask;pools.append(jnp.where(m[...,None],text,0).sum(1)/jnp.maximum(m.sum(1,keepdims=True),1))
        return jnp.stack(grids,axis=1),jnp.concatenate(pools,-1)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();plan=json.loads(pathlib.Path(a.plan).read_text());run=pathlib.Path(a.run);start=time.monotonic()
    assert jax.default_backend()=='gpu' and jax.device_count()==1
    data=arrays(run/'prepared_arrays.npz');samples=json.loads((run/'samples.json').read_text());root_lookup={int(s):i for i,s in enumerate(data['root_indices'])}
    matrices=np.stack([np.random.default_rng(seed).normal(size=(2048,32)).astype('float32')/np.sqrt(32) for seed in plan['critic']['seeds']]);policy=policy_config.create_trained_policy(config.get_config('pi05_libero_lora_3090x2_train'),ROOT/plan['checkpoint'],sample_kwargs={'num_steps':10});tap=SpatialTap(policy._model);extract=nnx_utils.module_jit(tap.extract)
    feature=np.empty((3,len(samples),2,16,32),dtype=np.float32);pools=[];current=None;obsdata=None;latencies=[];root_checks=0;repeat_error=None
    for i,row in enumerate(samples):
        if current!=row['directory']:current=row['directory'];obsdata=arrays(pathlib.Path(current)/'policy_inputs.npz')
        j=row['call_index'];assert int(obsdata['call_step'][j])==row['step'];external=obsdata['external'][j];wrist=obsdata['wrist'][j];assert image_hash(external,wrist)==row['image_sha256']
        request={'observation/image':external,'observation/wrist_image':wrist,'observation/state':obsdata['robot_state'][j],'prompt':row['prompt']}
        transformed=policy._input_transform(request);obs=model_lib.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None,...],transformed));tick=time.monotonic();grids,pooled=extract(obs);grids=np.asarray(grids)[0];pooled=np.asarray(pooled)[0];latencies.append(time.monotonic()-tick)
        assert grids.shape==(2,16,2048) and pooled.shape==(6144,) and np.isfinite(grids).all()
        if i==0:
            again=np.asarray(extract(obs)[0])[0];repeat_error=float(abs(grids-again).max());np.testing.assert_allclose(grids,again,atol=1e-6,rtol=0)
        if i in root_lookup:
            expected=data['original_prefix'][root_lookup[i]];actual=pooled.astype('float16')
            if not np.array_equal(actual,expected):
                evidence=dict(index=i,root_index=root_lookup[i],max_abs_float16_difference=float(abs(actual.astype('float32')-expected.astype('float32')).max()));(run/'feature_identity_failure.json').write_text(json.dumps(evidence)+'\n');raise AssertionError('Frozen original root feature identity failed')
            root_checks+=1;pools.append(actual)
        for s in range(3):feature[s,i]=grids@matrices[s]
        if i%25==0:
            elapsed=time.monotonic()-start;state=dict(status='RUNNING',inputs_completed=i+1,expected=len(samples),elapsed_seconds=elapsed,remaining_seconds=(len(samples)-i-1)*np.mean(latencies[-25:]));(run/'feature_progress.json').write_text(json.dumps(state)+'\n');print(json.dumps(state),flush=True)
    assert root_checks==186;np.savez_compressed(run/'spatial_features.npz',grid=feature,projection=matrices,verified_root_pooled=np.stack(pools))
    result=dict(status='COMPLETE',inputs_completed=len(samples),fit_episodes=140,root_identity_checks=root_checks,token_layout=[2,16,16,2048],pooled_grid_shape=[2,4,4,2048],saved_shape=list(feature.shape),seeds=plan['critic']['seeds'],repeat_error=repeat_error,elapsed_seconds=time.monotonic()-start,encoder_seconds_including_compile=sum(latencies),encoder_seconds_excluding_compile=sum(latencies[1:]),extra_encoder_compute=True,free_cache_sharing_claim=False,vla_updates=0,policy_actions_sampled=0,simulator_steps=0,calibration_inputs=0,confirmation_inputs=0,feature_sha256=sha(run/'spatial_features.npz'))
    (run/'feature_progress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
