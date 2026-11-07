"""Immutable approved paired-live collection, full audit and paired statistics."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=pathlib.Path(a.output).resolve();plan_path=pathlib.Path(a.plan).resolve();plan=json.loads(plan_path.read_text())
    assert plan['paired_live_amendment_approved'] and plan['originating_thread']==os.environ['CODEX_THREAD_ID']
    assert plan['expected_branches']==1345 and plan['workers']==2
    frozen=out/'frozen_source';frozen.mkdir(exist_ok=False)
    names=['paired_live_pipeline_v2.py','paired_live_supervisor_v2.py','paired_live_workers_v2.py','paired_live_worker_v2.py',
           'paired_live_collect_v2.py','paired_live_scene_v1.py','paired_live_audit_v1.py','paired_live_analyze_v2.py',
           'paired_live_cpu_checks_v1.py','paired_live_analyze_v1.py','verify_continuation_v4.py','storage_guard.py','paired_live_prepare_v1.py','paired_live_resume_v1.py','paired_live_resume_cpu_checks_v1.py','root_observation_snapshot_v1.py',
           'serve_recovery_branches_v1.py','research_telemetry_v1.py','egl_device.py','research_gpu_admission_v1.py','gpu_run.py','storage_guard_v2.py','project_env.sh']
    files=[ROOT/'scripts'/n for n in names]+[plan_path,ROOT/plan['root_manifest'],ROOT/plan['development_split'],ROOT/plan['approval_file']]
    upstream=json.loads((ROOT/'results/research_diagnosis_20261014_v3/source_manifest.json').read_text())['sha256']
    files += [ROOT/path for path in upstream if path.startswith('repos/')]
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(out/'source_manifest.json',{'sha256':hashes,'created_at':now()})
    fplan=frozen/plan_path.relative_to(ROOT);scripts=frozen/'scripts';env=os.environ.copy()
    env['PYTHONPATH']=str(scripts)+os.pathsep+env.get('PYTHONPATH','')
    state={'status':'RUNNING','phase':'admission_and_paired_collection','started_at':now(),'pid':os.getpid(),
           'expected_branches':1345,'confirmation_used':0,'new_training_updates':0,'plan':str(fplan)}
    write(out/'pipeline.json',state);start=time.monotonic()
    from storage_guard_v2 import check
    def guard():
        assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h for path,h in hashes.items()),'Frozen source changed'
        return check(reserve_bytes=128*2**20)
    try:
        state['storage']=guard();write(out/'pipeline.json',state)
        command=[str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'research_gpu_admission_v1.py'),'--gpu','0,1,2,3',
                 '--gpus','2','--multi-card-reason','Explicitly approved uniform paired round; maximum two GPUs; no old/live queue overlap','--',
                 str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'paired_live_workers_v2.py'),'--run',str(out),'--plan',str(fplan)]
        tick=time.monotonic();child=subprocess.Popen(command,env=env);state['collection_pid']=child.pid;write(out/'pipeline.json',state)
        code=child.wait();state['collection_window_seconds']=time.monotonic()-tick
        state['selected_gpu_window_hours_upper_bound']=2*state['collection_window_seconds']/3600
        if code:
            state.update(status='GPU_UNAVAILABLE' if code==75 else 'FAILED',returncode=code);return code
        state['storage_after_collection']=guard();state['phase']='full_audit_and_paired_statistics';write(out/'pipeline.json',state)
        subprocess.run([str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'paired_live_analyze_v2.py'),
                        '--run',str(out),'--plan',str(fplan)],env=env,check=True)
        guard();state.update(status='COMPLETE',automatic_statistics='COMPLETE',interpretation_review='pending');return 0
    except BaseException as exc:
        state.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-start);write(out/'pipeline.json',state)
if __name__=='__main__':raise SystemExit(main())
