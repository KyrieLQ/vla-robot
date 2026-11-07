"""Finite spatial representation pilot, durable parent supervises the whole process group."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
from storage_guard_v3 import check,du_paths_bytes
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();path=pathlib.Path(a.plan).resolve();plan=json.loads(path.read_text());run=pathlib.Path(a.output).resolve();start=time.monotonic()
    assert plan['originating_thread']==os.environ['CODEX_THREAD_ID'];assert json.loads((ROOT/'reports/recovery_spatial_cpu_checks_20261014.json').read_text())['status']=='PASS'
    assert hashlib.sha256((ROOT/plan['parent_action_critic']/'review/acceptance.json').read_bytes()).hexdigest()==plan['parent_review_sha256']
    assert plan['resources']['prior_paired_active_seconds']+plan['resources']['max_active_seconds']<=14400 and plan['resources']['prior_recovery_active_seconds']+plan['resources']['max_active_seconds']<=28800
    names=['recovery_spatial_data_v1.py','recovery_spatial_features_v1.py','recovery_spatial_fit_v1.py','recovery_spatial_cpu_checks_v1.py','recovery_spatial_pipeline_v1.py','recovery_spatial_supervisor_v1.py','recovery_value_fit_v1.py','recovery_fit_audit_v1.py','paired_live_audit_v1.py','storage_guard_v2.py','storage_guard_v3.py','research_gpu_admission_v1.py','gpu_run.py','project_env.sh']
    parent_manifest=json.loads((ROOT/plan['parent_pilot']/'source_manifest.json').read_text())['sha256'];files=[ROOT/'scripts'/n for n in names]+[path,ROOT/'reports/recovery_spatial_cpu_checks_20261014.json']+[ROOT/p for p in parent_manifest if p.startswith('repos/')]
    files += [ROOT/'repos/openpi/src/openpi/models'/n for n in ('pi0.py','siglip.py','gemma.py','pi0_config.py')]
    frozen=run/'frozen_source';frozen.mkdir();hashes={}
    for source in files:
        rel=source.relative_to(ROOT);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);hashes[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write(run/'source_manifest.json',{'sha256':hashes});scripts=frozen/'scripts';fplan=frozen/path.relative_to(ROOT);env=dict(os.environ);env['PYTHONPATH']=str(scripts)+os.pathsep+env.get('PYTHONPATH','');cpu=dict(env,CUDA_VISIBLE_DEVICES='');python=str(ROOT/'envs/openpi/bin/python');state=dict(status='RUNNING',phase='prepare_existing_fit_samples',new_rollouts=0,confirmation_used=0);write(run/'pipeline.json',state)
    def guard():
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items());result=check(reserve_bytes=128*2**20);used=du_paths_bytes([ROOT/p for p in plan['resources']['all_recovery_artifact_roots']]+[run]);local=du_paths_bytes([run]);assert used+64*2**20<=4*2**30 and local<=128*2**20
        return dict(result,phase_allocated_bytes=used,run_allocated_bytes=local)
    try:
        state['storage']=guard();subprocess.run([python,'-u',str(scripts/'recovery_spatial_data_v1.py'),'--plan',str(fplan),'--run',str(run)],env=cpu,check=True);guard();state['phase']='frozen_spatial_feature_extraction';write(run/'pipeline.json',state)
        result=subprocess.run([python,'-u',str(scripts/'research_gpu_admission_v1.py'),'--gpu','0,1,2,3','--gpus','1','--',python,'-u',str(scripts/'recovery_spatial_features_v1.py'),'--plan',str(fplan),'--run',str(run)],env=env)
        if result.returncode:state.update(status='GPU_UNAVAILABLE' if result.returncode==75 else 'FAILED',returncode=result.returncode);return result.returncode
        guard();state['phase']='fit_only_spatial_geometry_and_joint_critics';write(run/'pipeline.json',state);subprocess.run([python,'-u',str(scripts/'recovery_spatial_fit_v1.py'),'--plan',str(fplan),'--run',str(run)],env=cpu,check=True)
        r=json.loads((run/'SPATIAL_RESULTS.json').read_text());assert r['fit_episodes']==140 and r['fit_roots']==186 and r['calibration_used']==r['confirmation_used']==r['vla_updates']==0
        state.update(status='COMPLETE',storage_after=guard(),preparation_gate=r['preparation_gate_passed']);return 0
    except BaseException as exc:state.update(status='FAILED',error=str(exc),traceback=traceback.format_exc());raise
    finally:state['elapsed_seconds']=time.monotonic()-start;write(run/'pipeline.json',state)
if __name__=='__main__':raise SystemExit(main())
