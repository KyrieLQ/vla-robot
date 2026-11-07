#!/usr/bin/env python3
"""Instrument the pinned upstream JAX init/update/checkpoint functions, without changing the model or loss."""
import argparse, dataclasses, datetime, faulthandler, functools, hashlib, importlib.util, json, logging, os, pathlib
import subprocess, threading, time
import flax.traverse_util as tu
import jax
import numpy as np
from openpi.training import config, data_loader, checkpoints, sharding
from storage_guard import check as check_storage

ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def fingerprint(tree):
    flat=tu.flatten_dict(tree.to_pure_dict(),sep='/')
    out={}
    for key,v in flat.items():
        arr=np.asarray(jax.device_get(v))
        assert np.isfinite(arr).all(),f'Nonfinite parameter: {key}'
        out[key]=hashlib.sha256(arr.tobytes()).hexdigest()
    return out

def opt_fingerprint(tree):
    h=hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        a=np.asarray(jax.device_get(leaf));assert np.isfinite(a).all()
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    faulthandler.enable(all_threads=True)
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],default='smoke')
    p.add_argument('--exp-name',required=True);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--fsdp-devices',type=int,default=1);a=p.parse_args()
    if a.steps<=0 or not 1<=a.fsdp_devices<=4:p.error('Invalid steps or FSDP size')
    if a.mode=='smoke' and a.steps not in [20,22]:p.error('Smoke scope is 20 updates, then resume to 22')
    logging.basicConfig(level=logging.INFO)
    check_storage()
    assert jax.default_backend()=='gpu', 'CPU fallback is not a GPU validation'
    assert jax.device_count()==a.fsdp_devices, 'Visible GPU count differs from requested configuration'
    config_name='pi05_libero_lora_3090'+('x2' if a.fsdp_devices==2 else '')+'_'+a.mode
    cfg=dataclasses.replace(config.get_config(config_name),exp_name=a.exp_name,
                            num_train_steps=a.steps,resume=a.resume,fsdp_devices=a.fsdp_devices,
                            batch_size=a.fsdp_devices,keep_period=1 if a.mode=='smoke' else 5000)
    jax.config.update('jax_compilation_cache_dir',os.environ['JAX_COMPILATION_CACHE_DIR'])
    spec=importlib.util.spec_from_file_location('upstream_train',ROOT/'repos/openpi/scripts/train.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    out=ROOT/'results'/a.exp_name;out.mkdir(parents=True,exist_ok=a.resume)
    result={'status':'RUNNING','mode':a.mode,'config':cfg.name,'exp_name':a.exp_name,'resume':a.resume,
            'target_updates':a.steps,'batch_size':cfg.batch_size,'fsdp_devices':cfg.fsdp_devices,
            'seed':cfg.seed,'gpu_devices':[str(d) for d in jax.devices()],
            'allocator':{'preallocate':os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),'memory_fraction':os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')},
            'started_at':datetime.datetime.now().astimezone().isoformat(),'updates':[]}
    phase=['initialization'];stop=threading.Event();samples=[]
    def monitor():
        with (out/('memory_resume.jsonl' if a.resume else 'memory_smoke.jsonl')).open('x') as f:
            while not stop.is_set():
                try:
                    text=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                      '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
                    row={'time':time.time(),'phase':phase[0],'gpus':[[int(v.strip()) for v in line.split(',')] for line in text.splitlines()]}
                    samples.append(row);f.write(json.dumps(row)+'\n');f.flush()
                except Exception as exc:samples.append({'phase':phase[0],'error':type(exc).__name__})
                stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    report_path=out/('resume_report.json' if a.resume else 'training_report.json')
    mgr=None;loader=None;iterator=None
    try:
        rng=jax.random.key(cfg.seed);train_rng,init_rng=jax.random.split(rng)
        mesh=sharding.make_mesh(cfg.fsdp_devices)
        ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        mgr,resuming=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,keep_period=cfg.keep_period,overwrite=False,resume=a.resume)
        assert resuming==a.resume,'Requested resume did not restore an existing checkpoint'
        loader=data_loader.create_data_loader(cfg,sharding=ds,shuffle=True);iterator=iter(loader);batch=next(iterator)
        started=time.monotonic();state,state_sharding=upstream.init_train_state(cfg,init_rng,mesh,resume=resuming)
        if resuming:state=checkpoints.restore_state(mgr,state,loader)
        jax.block_until_ready(state);result['initialization_seconds']=time.monotonic()-started
        start_step=int(state.step);result['start_step']=start_step
        assert start_step==(20 if a.resume and a.mode=='smoke' else start_step)
        before=fingerprint(state.params);before_opt=opt_fingerprint(state.opt_state)
        train_paths=set(tu.flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(),sep='/'))
        if a.resume:
            previous=json.loads((out/f'state_{start_step}.json').read_text())
            assert before==previous['parameter_sha256'],'Restored parameters differ from saved training state'
            assert before_opt==previous['optimizer_sha256'],'Optimizer state was not faithfully restored'
            result['checkpoint_reload']='PASS';result['optimizer_restore']='PASS'
            # The pinned upstream checkpoint does not store sampler state. Recreate
            # its deterministic seeded stream and advance to the restored update.
            for _ in range(start_step):batch=next(iterator)
            result['data_batches_skipped']=start_step
        phase[0]='compilation';t=time.monotonic()
        fn=jax.jit(functools.partial(upstream.train_step,cfg),
                   in_shardings=(rep,state_sharding,ds),out_shardings=(state_sharding,rep),donate_argnums=(1,))
        with sharding.set_mesh(mesh):compiled=fn.lower(train_rng,state,batch).compile()
        result['compile_seconds']=time.monotonic()-t
        phase[0]='training'
        with (out/('metrics_resume.jsonl' if a.resume else 'metrics_smoke.jsonl')).open('x') as f:
            for step in range(start_step,a.steps):
                t=time.monotonic()
                with sharding.set_mesh(mesh):state,info=compiled(train_rng,state,batch)
                jax.block_until_ready((state,info));elapsed=time.monotonic()-t
                metrics={k:float(v) for k,v in jax.device_get(info).items()}
                assert all(np.isfinite(v) for v in metrics.values()),'Nonfinite loss, gradient norm, or parameter norm'
                assert int(state.step)==step+1,'Optimizer update counter failed'
                row={'update':int(state.step),'seconds':elapsed,**metrics};result['updates'].append(row)
                f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
                if a.mode=='train' and int(state.step)%cfg.save_interval==0:
                    reserve=int(sum(v.size*v.dtype.itemsize for v in jax.tree.leaves((state.params,state.opt_state)))*1.15)+20*1024**2
                    check_storage(reserve)
                    checkpoints.save_state(mgr,state,loader,int(state.step));mgr.wait_until_finished()
                if step+1<a.steps:batch=next(iterator)
        phase[0]='parameter_verification';after=fingerprint(state.params)
        changed=[k for k in before if before[k]!=after[k]]
        assert all(k in train_paths for k in changed),'A frozen parameter changed'
        assert any('lora' in k for k in changed),'No LoRA parameter changed'
        result.update(frozen_unchanged=True,changed_parameter_paths=changed,finished_updates=int(state.step),
                      optimizer_updates_this_process=int(state.step)-start_step,finite_parameters=True)
        phase[0]='checkpoint_save'
        reserve=int(sum(v.size*v.dtype.itemsize for v in jax.tree.leaves((state.params,state.opt_state)))*1.15)+20*1024**2
        result['storage_before_save']=check_storage(reserve)
        checkpoints.save_state(mgr,state,loader,int(state.step));mgr.wait_until_finished()
        latest=mgr.latest_step();assert latest is not None
        checkpoint=cfg.checkpoint_dir/str(latest);assert (checkpoint/'params').is_dir()
        result['checkpoint_path']=str(checkpoint);result['checkpoint_label']=latest
        (out/f'state_{int(state.step)}.json').write_text(json.dumps({'parameter_sha256':after,'optimizer_sha256':opt_fingerprint(state.opt_state)},indent=2)+'\n')
        (out/'checkpoint.json').write_text(json.dumps({'path':str(checkpoint),'config':cfg.name,'updates':int(state.step)},indent=2)+'\n')
        result['status']='PASS'
    except Exception as exc:
        result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        stop.set();thread.join(timeout=8)
        peaks={}
        for s in samples:
            for gpu,used,util in s.get('gpus',[]):peaks[s['phase']+':gpu'+str(gpu)]=max(peaks.get(s['phase']+':gpu'+str(gpu),0),used)
        result['observed_peak_device_memory_mib']=peaks
        result['memory_sampling']='nvidia-smi device-wide every >=0.5 s; includes driver/other occupancy and may miss short peaks'
        result['ended_at']=datetime.datetime.now().astimezone().isoformat()
        report_path.write_text(json.dumps(result,indent=2)+'\n')
        if mgr is not None:
            mgr.close();print('CLEANUP checkpoint_manager_closed',flush=True)
        if iterator is not None:iterator.close()
        # The pinned loader uses persistent spawned Torch workers. Join them
        # before Python/JAX native finalization instead of relying on atexit.
        if loader is not None:
            torch_loader=loader._data_loader.torch_loader
            worker_iterator=getattr(torch_loader,'_iterator',None)
            if worker_iterator is not None:worker_iterator._shutdown_workers()
        print('CLEANUP data_workers_closed',flush=True)
    print('TRAINER main_returned',flush=True)
if __name__=='__main__':main()
