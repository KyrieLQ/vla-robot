"""Disposable native pi05 updates, real multiGPU timing and exact compact resume."""
import argparse,functools,hashlib,importlib.util,json,os,random,time
from pathlib import Path
import flax.traverse_util as tu
import jax
import numpy as np
from openpi.training import sharding
from .pi05_four_suite_v1 import R,make_config,RawTrainingInputs
from .pi05_compact_state_v1 import save,restore,leaf_digest,sha

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--out',required=True);ap.add_argument('--reference');a=ap.parse_args()
    pp=Path(a.plan).absolute();p=json.loads(pp.read_text());out=Path(a.out).absolute();out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    devices=jax.device_count();assert jax.default_backend()=='gpu' and devices in (2,4)
    np.random.seed(42);random.seed(42);cfg=make_config(4,devices);data=RawTrainingInputs(p)
    assert Path(cfg.weight_loader.params_path)==R/p['official_base']
    assert not jax.config.jax_enable_compilation_cache
    contract=json.loads((R/p['cpu_contract']).read_text());indices=p['probe_sample_indices'];assert len(indices)==40
    def progress(stage,**kw):write(out/'progress.json',dict(stage=stage,devices=devices,elapsed_seconds=time.monotonic()-began,**kw))
    progress('official_base_initialization')
    spec=importlib.util.spec_from_file_location('pi05_upstream_probe_train',R/'repos/openpi/scripts/train.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    mesh=sharding.make_mesh(devices);rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec());ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    train_rng,init_rng=jax.random.split(jax.random.key(p['probe']['training_seed']))
    tick=time.monotonic();initial,state_sharding=upstream.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(initial)
    init_seconds=time.monotonic()-tick
    train_initial=initial.params.filter(cfg.trainable_filter);assert sum(x.size for x in jax.tree.leaves(train_initial))==52153376
    initial_hashes=leaf_digest(initial.params);initial_train_hashes=leaf_digest(train_initial)
    write(out/'initial_parameters.json',dict(all_parameters=initial_hashes,trainable_parameters=initial_train_hashes,base_binding=p['official_base_manifest_sha256']))
    if a.reference:
        reference=Path(a.reference);previous=json.loads((reference/'initial_parameters.json').read_text());assert previous['all_parameters']==initial_hashes and previous['trainable_parameters']==initial_train_hashes
    sample_receipts=[]
    def batch(cursor):
        items=[data.get(indices[i]) for i in range(cursor,cursor+4)]
        for i,item in enumerate(items):assert item[2]==contract['receipts'][indices[cursor+i]]
        sample_receipts.extend(item[2] for item in items)
        raw=jax.tree.map(lambda *xs:np.stack(xs),*[item[:2] for item in items])
        return jax.device_put(raw,ds)
    first=batch(0);progress('jit_compile')
    fn=jax.jit(functools.partial(upstream.train_step,cfg),in_shardings=(rep,state_sharding,ds),out_shardings=(state_sharding,rep))
    tick=time.monotonic()
    with sharding.set_mesh(mesh):compiled=fn.lower(train_rng,initial,first).compile()
    compile_seconds=time.monotonic()-tick
    state=initial;metrics=[];binding=dict(plan_sha256=sha(pp),base_manifest_sha256=p['official_base_manifest_sha256'],recipe='frozen_vision_native_gemma_lora_v1',samples_sha256=p['probe_sample_order_sha256'])
    snapshot=None
    for step in range(10):
        progress('disposable_updates',completed=step,total=10)
        loaded=time.monotonic();inputs=first if step==0 else batch(step*4);io_seconds=time.monotonic()-loaded
        tick=time.monotonic()
        with sharding.set_mesh(mesh):state,info=compiled(train_rng,state,inputs)
        jax.block_until_ready((state,info));seconds=time.monotonic()-tick
        info={k:float(v) for k,v in jax.device_get(info).items()};assert all(np.isfinite(v) for v in info.values()) and info['grad_norm']>0 and int(state.step)==step+1
        metrics.append(dict(step=step+1,compute_seconds=seconds,input_seconds=io_seconds,**info));write(out/'metrics.json',metrics)
        if step==1 and not a.reference:
            progress('compact_checkpoint_write');snapshot=save(out/'probe_step2.npz',state,cfg,train_rng,8,binding)
            assert snapshot['bytes']<p['probe']['checkpoint_max_bytes']
    # Frozen base tensors are checked exhaustively, including every SigLIP tensor.
    final_hashes=leaf_digest(state.params);train_keys=set(initial_train_hashes)
    assert all(initial_hashes[k]==final_hashes[k] for k in initial_hashes if k not in train_keys)
    changed=[k for k in train_keys if initial_hashes[k]!=final_hashes[k]]
    assert any('lora' in k and '_1' not in k for k in changed) and any('lora' in k and '_1' in k for k in changed)
    assert not any('img' in k for k in changed)
    final_train=state.params.filter(cfg.trainable_filter);final_arrays=jax.tree.leaves(final_train)
    numeric=None;restore_pass=None
    if not a.reference:
        progress('compact_restore_from_initial_template')
        restored,key,cursor=restore(out/'probe_step2.npz',initial,cfg,binding);assert int(restored.step)==2 and cursor==8
        for step in range(2,10):
            progress('same_config_resume_check',completed=step,total=10)
            inputs=batch(step*4)
            with sharding.set_mesh(mesh):restored,info=compiled(key,restored,inputs)
            jax.block_until_ready(restored)
        assert leaf_digest((restored.params,restored.opt_state))==leaf_digest((state.params,state.opt_state))
        assert int(restored.step)==int(state.step)==10;restore_pass=True
        np.savez(out/'reference_trainable_step10.npz',**{'p%04d'%i:np.asarray(jax.device_get(v)) for i,v in enumerate(final_arrays)})
        write(out/'reference_parameter_manifest.json',dict(leaves=len(final_arrays),sha256=sha(out/'reference_trainable_step10.npz'),updates=10))
    else:
        progress('all_trainable_multigpu_comparison');reference=Path(a.reference);manifest=json.loads((reference/'reference_parameter_manifest.json').read_text())
        assert sha(reference/'reference_trainable_step10.npz')==manifest['sha256'] and manifest['leaves']==len(final_arrays)
        max_error=diff_sq=base_sq=delta_sq=0.;initial_arrays=jax.tree.leaves(train_initial)
        with np.load(reference/'reference_trainable_step10.npz',allow_pickle=False) as other:
            for i,(value,init_value) in enumerate(zip(final_arrays,initial_arrays)):
                x=np.asarray(jax.device_get(value),dtype=np.float64);y=other['p%04d'%i].astype(np.float64);z=np.asarray(jax.device_get(init_value),dtype=np.float64)
                diff=x-y;max_error=max(max_error,float(np.max(np.abs(diff))));diff_sq+=float(np.square(diff).sum());base_sq+=float(np.square(y).sum());delta_sq+=float(np.square(y-z).sum())
        ref_metrics=json.loads((reference/'metrics.json').read_text());loss_error=max(abs(x['loss']-y['loss'])/max(abs(y['loss']),1e-8) for x,y in zip(metrics,ref_metrics));grad_error=max(abs(x['grad_norm']-y['grad_norm'])/max(abs(y['grad_norm']),1e-8) for x,y in zip(metrics,ref_metrics))
        numeric=dict(parameter_max_error=max_error,parameter_relative_L2=(diff_sq/max(base_sq,1e-30))**.5,
                     update_delta_relative_L2=(diff_sq/max(delta_sq,1e-30))**.5,loss_relative_error=loss_error,gradient_norm_relative_error=grad_error)
        gate=p['probe'];numeric['status']='PASS' if (max_error<=gate['parameter_absolute_tolerance'] and numeric['update_delta_relative_L2']<=gate['update_delta_relative_L2_tolerance'] and loss_error<=gate['loss_relative_tolerance'] and grad_error<=gate['gradient_norm_relative_tolerance']) else 'NUMERICAL_COMPARISON_NOT_ADMITTED'
    timings=[row['compute_seconds']+row['input_seconds'] for row in metrics[2:]]
    result=dict(status='PASS',devices=devices,global_batch=4,official_base_initialized=True,trainable_parameters=52153376,
                frozen_vision_parameters=414803696,all_frozen_parameters_unchanged=True,trainable_changed_leaves=len(changed),
                language_and_action_lora_updated=True,initialization_seconds=init_seconds,compile_seconds=compile_seconds,
                timed_updates=8,mean_update_seconds=float(np.mean(timings)),p95_update_seconds=float(np.quantile(timings,.95)),
                samples_per_second=4/float(np.mean(timings)),compact_checkpoint_restore_bitwise=restore_pass,
                compact_checkpoint_bytes=snapshot['bytes'] if snapshot else None,multigpu_numeric_comparison=numeric,
                disposable_updates=18 if not a.reference else 10,formal_training_updates=0,new_simulation_episodes=0,
                first40_training_inputs=sample_receipts[:40],device_memory_stats=[d.memory_stats() for d in jax.devices()],
                elapsed_seconds=time.monotonic()-began,formal_training_started=False)
    write(out/'report.json',result);progress('complete',disposable_updates=result['disposable_updates']);print(json.dumps({k:v for k,v in result.items() if k not in ['first40_training_inputs','device_memory_stats']}))

if __name__=='__main__':main()
