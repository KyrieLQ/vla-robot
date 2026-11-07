"""Complete the missing Euler-generation timing gate on existing120 TRAIN inputs.

No optimizer invocation, no validation labels, no simulations, no retained model.
"""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import flax.nnx as nnx
import jax
import numpy as np
import torch
from openpi.training import sharding
from .pi05_four_suite_v1 import R,make_config,RawTrainingInputs
from .pi05_compact_state_v1 import sha,leaf_digest

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());run=Path(a.run).absolute();start=time.monotonic()
    assert jax.default_backend()=='gpu' and jax.device_count()==4
    cfg=make_config(4,4);data=RawTrainingInputs(p);contract=json.loads((R/p['cpu_contract']).read_text());torch.set_num_threads(1)
    def progress(stage,**kw):write(run/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-start,**kw))
    progress('official_base_initialization')
    spec=importlib.util.spec_from_file_location('generation_probe_upstream',R/'repos/openpi/scripts/train.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    mesh=sharding.make_mesh(4);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    _,init_rng=jax.random.split(jax.random.key(42));initial,layout=upstream.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(initial)
    before=leaf_digest(initial.params);reference=json.loads((R/p['completed_preflight']/'gpu4/initial_parameters.json').read_text());assert before==reference['all_parameters']
    metrics=[];predictions=[];receipts=[]
    def batch(offset):
        examples=[data.get(i) for i in range(offset,offset+4)];noise=[]
        for i,example in enumerate(examples):
            receipt=example[2];assert receipt==contract['receipts'][offset+i]
            seed=int.from_bytes(hashlib.sha256(('20261105:%d:%d'%(receipt['episode_id'],receipt['frame_index'])).encode()).digest()[:8],'little')%(2**63-1)
            z=torch.randn((10,32),generator=torch.Generator().manual_seed(seed),dtype=torch.float32).numpy();noise.append(z)
            receipts.append(dict(**receipt,noise_seed=seed,noise_sha256=hashlib.sha256(z.tobytes()).hexdigest()))
        observation=jax.tree.map(lambda *xs:np.stack(xs),*[x[0] for x in examples])
        return jax.device_put(observation,ds),jax.device_put(np.stack(noise),ds)
    def generate(params,observation,noise):
        model=nnx.merge(initial.model_def,params);model.eval()
        return model.sample_actions(jax.random.key(0),observation,num_steps=10,noise=noise)
    first_obs,first_noise=batch(0);progress('compile_euler10_generation')
    fn=jax.jit(generate,in_shardings=(layout.params,ds,ds),out_shardings=ds)
    tick=time.monotonic()
    with sharding.set_mesh(mesh):compiled=fn.lower(initial.params,first_obs,first_noise).compile()
    compile_seconds=time.monotonic()-tick
    for offset in range(0,120,4):
        load=time.monotonic();obs,z=(first_obs,first_noise) if offset==0 else batch(offset);load_seconds=time.monotonic()-load
        tick=time.monotonic()
        with sharding.set_mesh(mesh):result=compiled(initial.params,obs,z)
        jax.block_until_ready(result);seconds=time.monotonic()-tick;array=np.asarray(jax.device_get(result));assert array.shape==(4,10,32) and np.isfinite(array).all();predictions.append(array)
        metrics.append(dict(first_input=offset,compute_seconds=seconds,input_seconds=load_seconds));write(run/'metrics.json',metrics);progress('generation_throughput',completed=offset+4,expected=120)
    with sharding.set_mesh(mesh):repeat=compiled(initial.params,first_obs,first_noise)
    np.testing.assert_array_equal(np.asarray(jax.device_get(repeat)),predictions[0]);assert before==leaf_digest(initial.params)
    np.save(run/'training_input_generation.npy',np.concatenate(predictions));write(run/'input_receipts.json',receipts)
    timed=[m['compute_seconds']+m['input_seconds'] for m in metrics[2:]];mean=float(np.mean(timed));report=dict(status='PASS',devices=4,batch=4,training_inputs=120,training_input_presentations=124,optimizer_updates=0,new_validation_frames=0,new_simulation_episodes=0,all_parameters_unchanged=True,repeat_first4_bitwise=True,full_generation_shape=[120,10,32],euler_steps=10,compile_seconds=compile_seconds,mean_batch_compute_io_seconds=mean,p95_batch_seconds=float(np.quantile(timed,.95)),projected_full30465_validation_seconds=mean*((30465+3)//4),device_memory_stats=[d.memory_stats() for d in jax.devices()],elapsed_seconds=time.monotonic()-start,limits='Timing on120traininginputs;notvalidationperformance. Fullvalfinalbatch needs maskedpadding of3rows;no target/endmask passed to generation.')
    write(run/'report.json',report);progress('complete');print(json.dumps(report))

if __name__=='__main__':main()
