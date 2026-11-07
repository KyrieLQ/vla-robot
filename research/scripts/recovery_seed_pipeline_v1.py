"""Approved bounded seed repeat study, frozen sources and event-supervised cleanup."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
from storage_guard_v3 import check,du_paths_bytes
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--plan',required=True);a=p.parse_args();run=pathlib.Path(a.output).resolve();path=pathlib.Path(a.plan).resolve();plan=json.loads(path.read_text());r=plan['resources'];assert plan['originating_thread']==os.environ['CODEX_THREAD_ID'] and plan['expected_branches']==130 and plan['workers']==2
    assert r['paired_limit_approved'] and r['paired_limit_seconds']==16200 and r['recovery_limit_seconds']==28800
    assert r['prior_paired_active_seconds']+plan['queue_timeout_seconds']<=16200 and r['prior_recovery_active_seconds']+plan['queue_timeout_seconds']<=28800
    assert plan['seed_stability_round_prior_active_seconds']+plan['queue_timeout_seconds']<=2100
    assert hashlib.sha256((ROOT/plan['seed_approval_file']).read_bytes()).hexdigest()==plan['seed_approval_sha256'];assert json.loads((ROOT/plan['cpu_checks_report']).read_text())['status']=='PASS'
    names=['recovery_seed_pipeline_v1.py','recovery_seed_supervisor_v1.py','recovery_seed_workers_v1.py','recovery_seed_worker_v1.py','recovery_seed_collect_v1.py','recovery_seed_analyze_v1.py','recovery_seed_cpu_checks_v1.py','paired_live_scene_v1.py','paired_live_audit_v1.py','paired_live_cpu_checks_v1.py','paired_live_analyze_v1.py','serve_recovery_branches_v1.py','research_telemetry_v1.py','root_observation_snapshot_v1.py','egl_device.py','research_gpu_admission_v1.py','gpu_run.py','storage_guard_v2.py','storage_guard_v3.py','project_env.sh']
    upstream=json.loads((ROOT/'results/recovery_paired_live_20261014_v3/source_manifest.json').read_text())['sha256'];files=[ROOT/'scripts'/n for n in names]+[path,ROOT/plan['root_manifest'],ROOT/plan['proposal_file'],ROOT/plan['seed_approval_file'],ROOT/plan['approval_file'],ROOT/plan['development_split'],ROOT/plan['cpu_checks_report']]+[ROOT/p for p in upstream if p.startswith('repos/')]
    frozen=run/'frozen_source';frozen.mkdir();hashes={}
    for source in files:
        rel=source.relative_to(ROOT);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);hashes[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write(run/'source_manifest.json',{'sha256':hashes});scripts=frozen/'scripts';env=os.environ.copy();env['PYTHONPATH']=str(scripts)+os.pathsep+env.get('PYTHONPATH','');fplan=frozen/path.relative_to(ROOT);python=str(ROOT/'envs/openpi/bin/python');start=time.monotonic();state=dict(status='RUNNING',phase='gpu_admission_preflight_and_seed_collection',expected_branches=130,calibration_used=0,confirmation_used=0,new_training_updates=0);write(run/'pipeline.json',state)
    def guard():
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items());storage=check(reserve_bytes=64*2**20);stage=du_paths_bytes([ROOT/p for p in plan['recovery_artifact_roots']]+[run]);local=du_paths_bytes([run]);assert stage+64*2**20<=4*2**30 and local<=320*2**20;return dict(storage,stage_allocated_bytes=stage,run_allocated_bytes=local)
    try:
        state['storage']=guard();tick=time.monotonic();command=[python,'-u',str(scripts/'research_gpu_admission_v1.py'),'--gpu','0,1,2,3','--gpus','2','--multi-card-reason','Approved fixed20fit cases130 branches; at most2 GPUs and35min, cumulative4.5/8h limits','--',python,'-u',str(scripts/'recovery_seed_workers_v1.py'),'--run',str(run),'--plan',str(fplan)];child=subprocess.Popen(command,env=env);state['collection_pid']=child.pid;write(run/'pipeline.json',state);code=child.wait();state['collection_window_seconds']=time.monotonic()-tick
        if code:state.update(status='GPU_UNAVAILABLE' if code==75 else 'FAILED',returncode=code);return code
        state.update(phase='complete_artifact_audit_and_seed_statistics',storage_after_collection=guard());write(run/'pipeline.json',state);subprocess.run([python,'-u',str(scripts/'recovery_seed_analyze_v1.py'),'--plan',str(fplan),'--run',str(run)],env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True);state.update(status='COMPLETE',storage_after=guard(),independent_review='pending');return 0
    except BaseException as exc:state.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:state['elapsed_seconds']=time.monotonic()-start;write(run/'pipeline.json',state)
if __name__=='__main__':raise SystemExit(main())
