#!/usr/bin/env python3
"""Finite end-to-end smoke orchestration. Exit 75 means wait for idle resources, not user permission."""
import argparse, datetime, fcntl, hashlib, json, os, pathlib, signal, subprocess, sys, time

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu-only',action='store_true');a=p.parse_args()
    root=pathlib.Path(os.environ['PROJECT_ROOT']);py=str(root/'envs/openpi/bin/python');sim=str(root/'envs/libero/bin/python')
    lock=(root/'results/validation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    train_gpus=int(os.environ.get('TRAIN_GPUS','1'))
    assert train_gpus in (1,2), 'This validation entry supports the registered one- and two-GPU recipes'
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');exp='smoke_'+stamp+('_2gpu' if train_gpus==2 else '')
    previous_train=os.environ.get('VLA_REUSE_TRAIN_RUN')
    if previous_train:
        prior=pathlib.Path(previous_train).resolve();assert prior.is_relative_to(root/'results')
        old=json.loads((prior/'status.json').read_text());exp=old['experiment']
    out=root/'results'/('pipeline_'+stamp);out.mkdir()
    result={'thread':os.environ.get('CODEX_THREAD_ID'),'experiment':exp,'status':'RUNNING','stages':[],'training_gpus':train_gpus}
    (out/'status.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'reports/latest_pipeline.json').write_text(json.dumps({'path':str(out),'status':'RUNNING'},indent=2)+'\n')
    cpu=os.environ.copy();cpu.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    gpu=os.environ.copy();gpu.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    def run(name,cmd,env):
        print('STAGE',name,flush=True);start=time.monotonic()
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result.update(active_stage=name,active_pid=child.pid,stage_started_at=datetime.datetime.now().astimezone().isoformat())
            (out/'status.json').write_text(json.dumps(result,indent=2)+'\n')
            try:code=child.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        row={'stage':name,'returncode':code,'status':'PASS' if code==0 else ('WAITING_FOR_IDLE_GPU' if code==75 else 'FAIL'),
             'elapsed_seconds':time.monotonic()-start,'log':str(out/(name+'.log'))}
        result['stages'].append(row);(out/'status.json').write_text(json.dumps(result,indent=2)+'\n')
        if code:raise subprocess.CalledProcessError(code,cmd)
    def gated(command,count=1):return [py,str(root/'scripts/gpu_run.py'),'--gpu',os.environ.get('GPU','auto'),'--gpus',str(count),
         '--multi-card-reason',os.environ.get('VLA_MULTI_GPU_REASON','Explicit user-authorized parallel validation'),'--']+command
    rc=0
    try:
        if not a.gpu_only:run('cpu_checks',['bash',str(root/'scripts/cpu_checks.sh')],cpu)
        for name in ['assets_integrity','parameter_audit','data_audit','batch_validation','base_weight_audit']:
            assert json.loads((root/'reports'/(name+'.json')).read_text())['status']=='PASS',name
        run('jax_gpu',gated([py,str(root/'scripts/gpu_probe.py')],train_gpus),gpu)
        previous_egl=os.environ.get('VLA_REUSE_EGL_RUN')
        if previous_egl:
            prior=pathlib.Path(previous_egl).resolve();assert prior.is_relative_to(root/'results')
            evidence=json.loads((prior/'egl_render/result.json').read_text())
            hashes=json.loads((prior/'source_manifest.json').read_text())['sha256']
            assert evidence['status']=='PASS' and evidence.get('actual_egl_render_uuid')
            for name in ['scripts/egl_device.py','scripts/libero_validate.py']:
                assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes[name], 'EGL implementation changed since validation'
            result['stages'].append({'stage':'egl_render','status':'PASS','returncode':0,'elapsed_seconds':0,
                  'reused':True,'evidence':str(prior/'egl_render/result.json')})
            (out/'status.json').write_text(json.dumps(result,indent=2)+'\n')
        else:
            run('egl_render',gated([sim,str(root/'scripts/libero_validate.py'),'--render-only','--episodes','1','--output',str(out/'egl_render')]),gpu)
        if previous_train:
            evidence=root/'results'/exp/'training_report.json'
            saved=json.loads(evidence.read_text())
            assert saved['status']=='PASS' and saved['finished_updates']==20 and saved['fsdp_devices']==train_gpus
            assert saved['frozen_unchanged'] and saved['finite_parameters']
            ck20=pathlib.Path(saved['checkpoint_path']).resolve()
            assert ck20.is_relative_to(root/'checkpoints') and (ck20/'params').is_dir() and (ck20/'train_state').is_dir()
            assert not (root/'results'/exp/'resume_report.json').exists(), 'Prior resume must be inspected before any retry'
            row={'stage':'train20','status':'CHECKPOINT_AWAITING_RELOAD','reused':True,'evidence':str(evidence),
                 'source_pipeline':str(pathlib.Path(previous_train).resolve()),'original_process_returncode':next(x['returncode'] for x in old['stages'] if x['stage']=='train20'),
                 'note':'20 verified updates saved before native process exit failure; require independent resume verification'}
            result['stages'].append(row)
        else:
            run('train20',['bash',str(root/'scripts/smoke_train.sh'),exp],gpu)
        run('resume22',['bash',str(root/'scripts/resume.sh'),exp],gpu)
        if previous_train:
            restored=json.loads((root/'results'/exp/'resume_report.json').read_text())
            assert restored['checkpoint_reload']=='PASS' and restored['optimizer_restore']=='PASS' and restored['finished_updates']==22
            row.update(status='PASS',validated_by='resume22 full parameter and optimizer hashes')
        ck=json.loads((root/'results'/exp/'checkpoint.json').read_text())
        run('closed_loop',['bash',str(root/'scripts/evaluate.sh'),ck['path'],'2','0',ck['config']],gpu)
        result['status']='SUCCESS'
    except subprocess.CalledProcessError as exc:
        rc=exc.returncode;result['status']='WAITING_FOR_IDLE_GPU' if rc==75 else 'FAIL'
    except Exception as exc:
        rc=1;result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
    finally:
        result['finished_at']=datetime.datetime.now().astimezone().isoformat()
        result['active_stage']=None;result['active_pid']=None
        (out/'status.json').write_text(json.dumps(result,indent=2)+'\n')
        (root/'reports/latest_pipeline.json').write_text(json.dumps({'path':str(out),'status':result['status']},indent=2)+'\n')
        summary=subprocess.run([py,str(root/'scripts/update_report.py')],env=cpu)
        (out/'completion_event.json').write_text(json.dumps({'experiment_finished':True,'report_processed':summary.returncode==0,
              'event_delivery':'local finite pipeline; no external message or persistent polling','thread':result['thread']},indent=2)+'\n')
    return rc
if __name__=='__main__':raise SystemExit(main())
