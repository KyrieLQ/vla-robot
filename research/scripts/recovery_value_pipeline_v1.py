"""Bounded frozen-feature pilot; one GPU for extraction, CPU for module fitting."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
from storage_guard_v3 import check,du_paths_bytes
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    path=pathlib.Path(a.plan).resolve();plan=json.loads(path.read_text());run=pathlib.Path(a.output).resolve()
    assert plan['originating_thread']==os.environ['CODEX_THREAD_ID']
    assert json.loads((ROOT/plan['parent_run']/'review/acceptance.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'reports/recovery_value_cpu_checks_20261014.json').read_text())['status']=='PASS'
    frozen=run/'frozen_source';frozen.mkdir()
    names=['recovery_value_pipeline_v1.py','recovery_value_supervisor_v1.py','recovery_value_features_v1.py','recovery_value_fit_v1.py','recovery_value_data_v1.py','recovery_value_cpu_checks_v1.py','paired_live_audit_v1.py','storage_guard_v2.py','storage_guard_v3.py','research_gpu_admission_v1.py','gpu_run.py','project_env.sh']
    files=[ROOT/'scripts'/n for n in names]+[path,ROOT/plan['root_manifest'],ROOT/plan['development_split']]
    parent=json.loads((ROOT/plan['parent_run']/'source_manifest.json').read_text())['sha256']
    files += [ROOT/f for f in parent if f.startswith('repos/')]
    hashes={}
    for source in files:
        rel=source.relative_to(ROOT);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);hashes[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write(run/'source_manifest.json',{'created_at':now(),'sha256':hashes})
    state={'status':'RUNNING','phase':'preflight','started_at':now(),'vla_updates':0,'confirmation_used':0};start=time.monotonic()
    scripts=frozen/'scripts';fplan=frozen/path.relative_to(ROOT);env=os.environ.copy();env['PYTHONPATH']=str(scripts)+os.pathsep+env.get('PYTHONPATH','')
    def guard():
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        resource=check(reserve_bytes=64*2**20)
        allocated=du_paths_bytes([ROOT/p for p in plan['resources']['all_recovery_artifact_roots']]+[run])
        assert allocated+64*2**20<=4*2**30
        return dict(resource,phase_allocated_bytes=allocated)
    try:
        state['storage']=guard();state['phase']='frozen_prefix_feature_extraction';write(run/'pipeline.json',state)
        command=[str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'research_gpu_admission_v1.py'),'--gpu','0,1,2,3','--gpus','1','--',str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'recovery_value_features_v1.py'),'--plan',str(fplan),'--run',str(run)]
        tick=time.monotonic();result=subprocess.run(command,env=env);state['feature_gpu_window_seconds']=time.monotonic()-tick
        if result.returncode:state.update(status='GPU_UNAVAILABLE' if result.returncode==75 else 'FAILED',returncode=result.returncode);return result.returncode
        guard();state['phase']='fit_only_module_training_and_calibration';write(run/'pipeline.json',state)
        cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
        subprocess.run([str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'recovery_value_fit_v1.py'),'--plan',str(fplan),'--run',str(run)],env=cpu_env,check=True)
        result=json.loads((run/'PILOT_RESULTS.json').read_text());assert result['models_trained']==15 and result['confirmation_used']==0 and result['vla_updates']==0
        state.update(status='COMPLETE',storage_after=guard(),offline_gate=result['primary_offline_learnability_gate_passed']);return 0
    except BaseException as exc:
        state.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-start);write(run/'pipeline.json',state)
if __name__=='__main__':raise SystemExit(main())
