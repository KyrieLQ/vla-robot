"""CPU-only grouped audit with frozen sources, bounded storage and no calibration scoring."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
from storage_guard_v3 import check,du_paths_bytes
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    path=pathlib.Path(a.plan).resolve();plan=json.loads(path.read_text());run=pathlib.Path(a.output).resolve();start=time.monotonic()
    assert plan['originating_thread']==os.environ['CODEX_THREAD_ID']
    parent=ROOT/plan['parent_pilot'];assert json.loads((ROOT/plan['parent_fit_audit']/'review/acceptance.json').read_text())['status']=='PASS'
    assert not json.loads((parent/'PILOT_RESULTS.json').read_text())['primary_offline_learnability_gate_passed']
    names=['recovery_action_critic_v1.py','recovery_action_critic_pipeline_v1.py','recovery_action_critic_supervisor_v1.py','recovery_fit_audit_v1.py','recovery_value_fit_v1.py','paired_live_audit_v1.py','storage_guard_v2.py','storage_guard_v3.py','project_env.sh']
    frozen=run/'frozen_source';frozen.mkdir();hashes={}
    for source in [ROOT/'scripts'/name for name in names]+[path]:
        rel=source.relative_to(ROOT);dest=frozen/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);hashes[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write(run/'source_manifest.json',{'sha256':hashes})
    state={'status':'RUNNING','phase':'fit_only_action_conditioned_critic','confirmation_used':0,'calibration_predictions_generated':False,'gpu_jobs':0};write(run/'pipeline.json',state)
    def guard():
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        resource=check(reserve_bytes=64*2**20);used=du_paths_bytes([ROOT/p for p in plan['resources']['all_recovery_artifact_roots']]+[run]);assert used+64*2**20<=4*2**30
        return dict(resource,phase_allocated_bytes=used)
    try:
        state['storage']=guard();env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env['PYTHONPATH']=str(frozen/'scripts')+os.pathsep+env.get('PYTHONPATH','')
        subprocess.run([str(ROOT/'envs/openpi/bin/python'),'-u',str(frozen/'scripts/recovery_action_critic_v1.py'),'--plan',str(frozen/path.relative_to(ROOT)),'--run',str(run)],env=env,check=True)
        result=json.loads((run/'ACTION_CRITIC_RESULTS.json').read_text());assert result['calibration_used']==0 and result['fit_episodes']==140
        state.update(status='COMPLETE',storage_after=guard());return 0
    except BaseException as exc:state.update(status='FAILED',error=str(exc),traceback=traceback.format_exc());raise
    finally:state['elapsed_seconds']=time.monotonic()-start;write(run/'pipeline.json',state)
if __name__=='__main__':raise SystemExit(main())
