#!/usr/bin/env python3
"""Finite data validation -> training -> paired base/final evaluation -> analysis."""
import argparse, datetime, hashlib, json, os, pathlib, signal, subprocess, time
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--experiment',required=True)
    recovery=p.add_mutually_exclusive_group();recovery.add_argument('--resume-training',action='store_true');recovery.add_argument('--reuse-trained',action='store_true');a=p.parse_args()
    out=pathlib.Path(a.run_dir);assert out.is_dir()
    assert json.loads((ROOT/'reports/spatial_assets_integrity.json').read_text())['status']=='PASS'
    plan=json.loads((ROOT/'configs/formal_plan_20261013.json').read_text())
    assert os.environ['CODEX_THREAD_ID']==plan['thread']
    py=str(ROOT/'envs/openpi/bin/python');cpu=os.environ.copy();cpu.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    gpu=os.environ.copy();gpu.update(JAX_PLATFORMS='cuda',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',NCCL_DEBUG='WARN',PYTHONFAULTHANDLER='1')
    paths=['configs/formal_plan_20261013.json','scripts/formal_pipeline_v1.py','scripts/formal_stage_supervisor.py','scripts/formal_data_v1.py','scripts/train_formal_v1.py','scripts/formal_eval_v1.py','scripts/serve_formal_v1.py','scripts/evaluate_spatial_v1.py','scripts/analyze_formal_v1.py','scripts/gpu_run.py','scripts/egl_device.py','repos/openpi/src/openpi/training/config.py','repos/openpi/src/openpi/shared/array_typing.py']
    paths+=['scripts/train_verified.py','scripts/project_env.sh','scripts/storage_guard.py','repos/openpi/scripts/train.py','repos/openpi/src/openpi/training/data_loader.py','reports/openpi_local.patch']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    for name in paths:
        target=out/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    write(out/'source_manifest.json',{'recorded_at':now(),'sha256':hashes})
    status={'status':'RUNNING','started_at':now(),'experiment':a.experiment,'thread':plan['thread'],'stages':[]}
    write(out/'pipeline.json',status)
    def run(stage,command,env,timeout):
        for name,expected in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, 'Frozen implementation changed: '+name
        start=time.monotonic();print('STAGE',stage,flush=True)
        with (out/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status.update(active_stage=stage,active_pid=child.pid,active_pgid=child.pid,stage_started_at=now());write(out/'pipeline.json',status)
            write(ROOT/'reports/project_status.json',{'phase':'formal_training','status':'RUNNING','active_stage':stage,'pipeline':str(out),'experiment':a.experiment,'updated_at':now(),'formal_training_authorized':True})
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
            finally:
                # This group belongs exclusively to this stage, including the
                # inference server and data workers; reap any leftovers.
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        row={'stage':stage,'returncode':code,'status':'PASS' if code==0 else ('WAITING_FOR_IDLE_GPU' if code==75 else 'FAIL'),'elapsed_seconds':time.monotonic()-start,'log':str(out/(stage+'.log'))}
        status['stages'].append(row);status['active_pgid']=None;write(out/'pipeline.json',status)
        if code:raise subprocess.CalledProcessError(code,command)
    def gated(command,count):return [py,str(ROOT/'scripts/gpu_run.py'),'--gpu',','.join(map(str,plan['allowed_physical_gpus'])),'--gpus',str(count),'--multi-card-reason','Explicit user-authorized formal dual-GPU training on 2026-10-13','--']+command
    rc=0
    try:
        if not (ROOT/'reports/formal_data_audit.json').exists():run('data_audit_norm',[py,'-u',str(ROOT/'scripts/formal_data_v1.py')],cpu,7200)
        assert json.loads((ROOT/'reports/formal_data_audit.json').read_text())['status']=='PASS'
        if a.reuse_trained:
            previous=ROOT/'results'/a.experiment/'training_report.json';verified=json.loads(previous.read_text())
            ck=json.loads((previous.parent/'checkpoint.json').read_text())
            assert verified['status']=='PASS' and verified['finished_updates']==30000 and verified['frozen_unchanged']
            assert ck['updates']==30000 and (pathlib.Path(ck['path'])/'params').is_dir()
            assert (previous.parent/'state_30000.json').is_file()
            status['stages'].append({'stage':'train30000','status':'PASS','reused':True,'evidence':str(previous),'note':'Final policy evaluation independently reloads the retained checkpoint'})
            write(out/'pipeline.json',status)
        else:
            command=[py,'-u',str(ROOT/'scripts/train_formal_v1.py'),'--experiment',a.experiment]
            if a.resume_training:command.append('--resume')
            run('train30000',gated(command,2),gpu,18*3600)
        for variant in ['base','trained']:
            run(variant+'_eval',gated([py,'-u',str(ROOT/'scripts/formal_eval_v1.py'),'--variant',variant,'--experiment',a.experiment,'--output',str(out/(variant+'_eval'))],1),gpu,7500)
        run('analysis',[py,'-u',str(ROOT/'scripts/analyze_formal_v1.py'),'--run-dir',str(out),'--experiment',a.experiment],cpu,600)
        status['status']='SUCCESS'
    except subprocess.CalledProcessError as exc:rc=exc.returncode;status['status']='WAITING_FOR_IDLE_GPU' if rc==75 else 'FAIL'
    except KeyboardInterrupt:rc=130;status['status']='INTERRUPTED'
    except Exception as exc:rc=1;status.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
    finally:
        status.update(finished_at=now(),active_stage=None,active_pid=None,active_pgid=None);write(out/'pipeline.json',status)
        write(ROOT/'reports/project_status.json',{'phase':'formal_training','status':status['status'],'active_stage':None,'pipeline':str(out),'experiment':a.experiment,'updated_at':now(),'formal_training_authorized':True})
    return rc
if __name__=='__main__':raise SystemExit(main())
