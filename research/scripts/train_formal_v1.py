#!/usr/bin/env python3
"""Fixed-plan formal training using the already validated upstream JAX update."""
import argparse, dataclasses, datetime, faulthandler, functools, importlib.util, json, os, pathlib, statistics, subprocess, threading, time
import flax.traverse_util as tu
import jax
import numpy as np
from openpi.training import config, data_loader, checkpoints, sharding
from storage_guard import check as check_storage
from train_verified import fingerprint, opt_fingerprint

ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def now():return datetime.datetime.now().astimezone().isoformat()
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    faulthandler.enable(all_threads=True)
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=json.loads((ROOT/'configs/formal_plan_20261013.json').read_text())
    assert plan['steps']==30000 and plan['fsdp_devices']==2 and plan['global_batch']==2
    assert json.loads((ROOT/'reports/formal_data_audit.json').read_text())['status']=='PASS'
    assert jax.default_backend()=='gpu' and jax.device_count()==2
    cfg=dataclasses.replace(config.get_config(plan['config']),exp_name=a.experiment,num_train_steps=plan['steps'],
        seed=plan['training_seed'],batch_size=2,fsdp_devices=2,num_workers=2,ema_decay=None,
        save_interval=plan['checkpoint_interval'],keep_period=plan['checkpoint_keep_period'],resume=a.resume)
    assert cfg.lr_schedule.peak_lr==plan['learning_rate'] and cfg.lr_schedule.warmup_steps==plan['warmup_steps']
    jax.config.update('jax_compilation_cache_dir',os.environ['JAX_COMPILATION_CACHE_DIR'])
    out=ROOT/'results'/a.experiment;out.mkdir(exist_ok=a.resume)
    attempt=out/('attempt_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));attempt.mkdir()
    report={'status':'RUNNING','experiment':a.experiment,'attempt':str(attempt),'config':cfg.name,'started_at':now(),'pid':os.getpid(),
      'resume':a.resume,'target_updates':plan['steps'],'global_batch':2,'fsdp_devices':2,'seed':cfg.seed,'gpu_selection':os.environ.get('VLA_GPU_SELECTION'),
      'parameter_source':'official pi05_base (fresh initialization) or exact same-run checkpoint','initialization_seconds':None,'compile_seconds':None}
    phase=['initialization'];stop=threading.Event();peaks={}
    progress={'status':'RUNNING','phase':phase[0],'pid':os.getpid(),'attempt':str(attempt),'step':0,'target_steps':plan['steps'],'started_at':now()}
    write(out/'progress.json',progress)
    def memory():
        with (attempt/'memory.jsonl').open('x') as f:
            while not stop.is_set():
                try:
                    raw=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
                    devices=[[int(x.strip()) for x in s.split(',')] for s in raw.splitlines()]
                    row={'time':time.time(),'phase':phase[0],'gpus':devices};f.write(json.dumps(row)+'\n');f.flush()
                    for gpu,used,_ in devices:peaks[str(gpu)]=max(peaks.get(str(gpu),0),used)
                except Exception as exc:f.write(json.dumps({'time':time.time(),'error':type(exc).__name__})+'\n');f.flush()
                stop.wait(5)
    monitor=threading.Thread(target=memory,daemon=True);monitor.start()
    spec=importlib.util.spec_from_file_location('upstream_train',ROOT/'repos/openpi/scripts/train.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    mgr=None;loader=None;iterator=None
    try:
        check_storage();rng=jax.random.key(cfg.seed);train_rng,init_rng=jax.random.split(rng)
        mesh=sharding.make_mesh(2);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        mgr,resuming=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,keep_period=cfg.keep_period,overwrite=False,resume=a.resume)
        assert resuming==a.resume
        loader=data_loader.create_data_loader(cfg,sharding=ds,shuffle=True);iterator=iter(loader);batch=next(iterator)
        t=time.monotonic();state,state_sharding=upstream.init_train_state(cfg,init_rng,mesh,resume=resuming)
        if resuming:state=checkpoints.restore_state(mgr,state,loader)
        jax.block_until_ready(state);report['initialization_seconds']=time.monotonic()-t
        start_step=int(state.step);assert 0<=start_step<plan['steps'];report['start_step']=start_step
        before=fingerprint(state.params);train_paths=set(tu.flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict(),sep='/'))
        train_count=sum(x.size for x in jax.tree.leaves(state.params.filter(cfg.trainable_filter)))
        assert train_count==466957072;report['trainable_parameters']=train_count
        if resuming:
            saved=json.loads((out/f'state_{start_step}.json').read_text())
            assert before==saved['parameter_sha256'] and opt_fingerprint(state.opt_state)==saved['optimizer_sha256']
            report['checkpoint_reload']='PASS';report['optimizer_restore']='PASS'
            phase[0]='restore_data_position'
            for i in range(start_step):
                batch=next(iterator)
                if i%100==0:progress.update(phase=phase[0],replayed_batches=i+1,step=start_step,updated_at=now());write(out/'progress.json',progress)
        else:
            write(out/'initial_parameters.json',{'parameter_sha256':before,'trainable_paths':sorted(train_paths)})
        initial=json.loads((out/'initial_parameters.json').read_text())['parameter_sha256']
        phase[0]='compilation';progress.update(phase=phase[0],step=start_step,updated_at=now());write(out/'progress.json',progress)
        fn=jax.jit(functools.partial(upstream.train_step,cfg),in_shardings=(rep,state_sharding,ds),out_shardings=(state_sharding,rep),donate_argnums=(1,))
        t=time.monotonic()
        with sharding.set_mesh(mesh):compiled=fn.lower(train_rng,state,batch).compile()
        report['compile_seconds']=time.monotonic()-t;write(attempt/'report.json',report)
        def save(step):
            phase[0]='checkpoint_save';progress.update(phase=phase[0],step=step,updated_at=now());write(out/'progress.json',progress)
            reserve=int(sum(x.size*x.dtype.itemsize for x in jax.tree.leaves((state.params,state.opt_state)))*1.15)+20*2**20
            check_storage(reserve)
            hashes=fingerprint(state.params);opt=opt_fingerprint(state.opt_state)
            assert all(hashes[k]==initial[k] for k in initial if k not in train_paths),'Frozen parameter changed'
            assert any('lora' in k and hashes[k]!=initial[k] for k in train_paths),'LoRA did not change'
            checkpoints.save_state(mgr,state,loader,step);mgr.wait_until_finished()
            assert mgr.latest_step()==step
            ck=cfg.checkpoint_dir/str(step);assert (ck/'params').is_dir() and (ck/'train_state').is_dir()
            write(out/f'state_{step}.json',{'parameter_sha256':hashes,'optimizer_sha256':opt})
            write(out/'checkpoint.json',{'path':str(ck),'config':cfg.name,'updates':step,'saved_at':now(),'frozen_unchanged':True})
            print('CHECKPOINT_SAVED',step,str(ck),flush=True)
        wall=time.monotonic();times=[]
        phase[0]='training'
        with (attempt/'metrics.jsonl').open('x') as f:
            for step in range(start_step,plan['steps']):
                t=time.monotonic()
                with sharding.set_mesh(mesh):state,info=compiled(train_rng,state,batch)
                jax.block_until_ready((state,info));compute=time.monotonic()-t
                values={k:float(v) for k,v in jax.device_get(info).items()};assert all(np.isfinite(x) for x in values.values())
                assert int(state.step)==step+1
                if (step+1)%cfg.save_interval==0:save(step+1)
                phase[0]='training';load_start=time.monotonic()
                if step+1<plan['steps']:batch=next(iterator)
                load_seconds=time.monotonic()-load_start
                elapsed=time.monotonic()-wall;times.append(compute)
                row={'update':step+1,'update_seconds':compute,'next_batch_seconds':load_seconds,'training_wall_seconds':elapsed,**values}
                f.write(json.dumps(row)+'\n');f.flush()
                if (step+1)%10==0 or step==start_step:
                    progress.update(status='RUNNING',phase=phase[0],step=step+1,target_steps=plan['steps'],updated_at=now(),loss=values['loss'],
                      update_seconds_mean_100=statistics.mean(times[-100:]),wall_seconds_per_update=elapsed/(step+1-start_step),
                      remaining_seconds_estimate=elapsed/(step+1-start_step)*(plan['steps']-step-1),latest_checkpoint_step=mgr.latest_step())
                    write(out/'progress.json',progress)
                if (step+1)%100==0 or step==start_step:print(json.dumps(progress),flush=True)
        report.update(status='PASS',finished_updates=int(state.step),optimizer_updates_this_attempt=int(state.step)-start_step,frozen_unchanged=True,
          checkpoint=json.loads((out/'checkpoint.json').read_text()),training_wall_seconds=time.monotonic()-wall,update_seconds_mean=statistics.mean(times),finite_parameters=True)
        write(out/'training_report.json',report)
    except Exception as exc:
        report.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        stop.set();monitor.join(timeout=8)
        report.update(ended_at=now(),observed_peak_device_memory_mib=peaks,memory_sampling='whole selected device NVML every >=5s; may miss brief peaks')
        write(attempt/'report.json',report)
        if report['status']=='PASS':write(out/'training_report.json',report)
        if mgr is not None:mgr.close()
        if iterator is not None:iterator.close()
        if loader is not None:
            workers=loader._data_loader.torch_loader._iterator
            if workers is not None:workers._shutdown_workers()
        progress.update(status=report['status'],phase='finished',updated_at=now());write(out/'progress.json',progress)
        print('CLEANUP_COMPLETE',report['status'],flush=True)
if __name__=='__main__':main()
