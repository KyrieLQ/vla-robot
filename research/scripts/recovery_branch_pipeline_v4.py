#!/usr/bin/env python3
"""Freeze code and execute one finite two-GPU branch experiment plus analysis."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--plan',required=True);a=p.parse_args();out=pathlib.Path(a.output);plan_path=pathlib.Path(a.plan);plan=json.loads(plan_path.read_text())
    assert plan['originating_thread']==os.environ['CODEX_THREAD_ID']
    assert plan.get('prefix_replay_acceptance_amendment_approved') is True, 'User approval for the proposed engineering acceptance change is still required'
    assert plan.get('unconsumed_suffix_image_amendment_approved') is True
    frozen=out/'frozen_source';frozen.mkdir(exist_ok=False)
    names=['recovery_branch_pipeline_v4.py','recovery_branch_supervisor_v4.py','run_recovery_branch_workers_v4.py','recovery_branch_worker_v4.py','evaluate_recovery_branches_v4.py','serve_recovery_branches_v1.py','analyze_recovery_branches_v4.py','verify_continuation_v4.py','research_telemetry_v1.py','egl_device.py','research_gpu_admission_v1.py','gpu_run.py','storage_guard.py','project_env.sh']
    files=[ROOT/'scripts'/n for n in names]+[plan_path,ROOT/plan['root_manifest'],ROOT/plan['source_diagnosis'],ROOT/plan['development_split']]
    # Carry forward guards for all fixed upstream components used by the parent experiment.
    parent=json.loads((ROOT/'results/research_diagnosis_20261014_v3/source_manifest.json').read_text())['sha256']
    files+=[ROOT/p for p in parent if p.startswith('repos/')]
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);hashes[str(rel)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(out/'source_manifest.json',{'sha256':hashes,'created_at':now()})
    fplan=frozen/plan_path.relative_to(ROOT);scripts=frozen/'scripts';env=os.environ.copy();env['PYTHONPATH']=str(scripts)+os.pathsep+env.get('PYTHONPATH','')
    state={'status':'RUNNING','phase':'admission_and_branch_collection','started_at':now(),'pid':os.getpid(),'expected_branches':1076,'confirmation_used':0,'plan':str(fplan)};write(out/'pipeline.json',state);start=time.monotonic()
    from storage_guard import check
    def guard():
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'Frozen source changed'
        return check(reserve_bytes=512*2**20)
    try:
        state['storage']=guard();write(out/'pipeline.json',state)
        command=[str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'research_gpu_admission_v1.py'),'--gpu','0,1,2,3','--gpus',str(plan['workers']),'--multi-card-reason','User approved at most two GPUs; finish only remaining paired branches','--',str(ROOT/'envs/openpi/bin/python'),'-u',str(scripts/'run_recovery_branch_workers_v4.py'),'--run',str(out),'--plan',str(fplan)]
        child=subprocess.Popen(command,env=env);state['collection_pid']=child.pid;write(out/'pipeline.json',state);tick=time.monotonic();code=child.wait()
        state['collection_window_seconds']=time.monotonic()-tick;state['selected_gpu_window_hours_upper_bound']=plan['workers']*state['collection_window_seconds']/3600
        if code:
            state['status']='GPU_UNAVAILABLE' if code==75 else 'FAILED';state['returncode']=code;return code
        state['storage_after_collection']=guard();state['phase']='analysis';write(out/'pipeline.json',state)
        workers=[json.loads((out/'workers'/f'worker{w}'/'result.json').read_text()) for w in range(plan['workers'])]
        assert all(w['status'] in ('PASS','INCOMPLETE_RENDER_REVIEW') for w in workers)
        deferred=[d for w in workers for d in w.get('deferred_roots',[])]
        if deferred:
            summary={'status':'INCOMPLETE_RENDER_REVIEW','created_at':now(),'completed_branches':sum(w['completed_branches'] for w in workers),'expected_branches':plan['expected_branches'],'deferred_roots':deferred,'root_count':len(deferred),'confirmation_used':0,'new_training_updates':0,'performance_analysis':'NOT_RUN_INCOMPLETE_INTEGRITY_COVERAGE','next_action':'Resolve pending engineering acceptance for unconsumed suffix images. Preserve completed branches; do not resample or train a module from this incomplete set.'}
            write(out/'incomplete_collection.json',summary)
            state.update(status='INCOMPLETE_RENDER_REVIEW',automatic_statistics='NOT_RUN_INCOMPLETE',deferred_roots=len(deferred))
            return 78
        subprocess.run([str(ROOT/'envs/openpi/bin/python'),str(scripts/'analyze_recovery_branches_v4.py'),'--run',str(out),'--plan',str(fplan)],env=env,check=True)
        state.update(status='COMPLETE',automatic_statistics='COMPLETE',interpretation_review='pending');return 0
    except BaseException as e:
        state.update(status='FAILED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());raise
    finally:
        state.update(finished_at=now(),elapsed_seconds=time.monotonic()-start);write(out/'pipeline.json',state)
if __name__=='__main__':raise SystemExit(main())
