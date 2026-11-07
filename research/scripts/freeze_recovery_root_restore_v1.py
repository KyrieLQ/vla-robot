#!/usr/bin/env python3
"""Audit a reviewed failed parent and preserve both outcomes and root measurements."""
import datetime,hashlib,json,math,os,pathlib,subprocess
import numpy as np
from PIL import Image
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=ROOT/'results/recovery_branches_20261014_v3'
    old_path=ROOT/'configs/recovery_branch_plan_20261014_v3.json';plan=json.loads(old_path.read_text())
    approval_path=ROOT/'configs/recovery_suffix_acceptance_20261014_APPROVED.json';approval=json.loads(approval_path.read_text());assert approval['approved']
    state=json.loads((parent/'status.json').read_text());assert state['status']=='FAILED'
    failure=json.loads((parent/'workers/worker0/progress.json').read_text())
    assert failure['active_root']=='task7_init44_t60' and failure['active_method']=='retreat5_fine20' and 'root image' in failure['error']
    manifest=json.loads((ROOT/plan['root_manifest']).read_text());roots={r['root_id']:r for r in manifest['roots']};valid={(r,m['name']) for r in roots for m in plan['methods']}
    references={}
    def source(pair):
        if pair not in references:
            p=ROOT/plan['source_episodes']/pair
            with np.load(p/'policy_trace.npz') as f:tr={k:f[k] for k in f.files}
            with np.load(p/'oracle.npz') as f:oracle={k:f[k] for k in f.files}
            references[pair]=(tr,oracle)
        return references[pair]
    kept={}
    for file in sorted((parent/'workers').glob('worker*/progress.json')):
        for row in json.loads(file.read_text())['branches']:
            key=(row['root_id'],row['method']);assert key in valid and key not in kept
            directory=pathlib.Path(row['output']);saved=json.loads((directory/'result.json').read_text());assert saved['status']==row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED')
            with np.load(directory/'policy_trace.npz') as tr,np.load(directory/'oracle.npz') as oracle:
                assert tr['action'].shape==(row['steps'],7)
                assert np.array_equal(tr['done'],oracle['goal_predicate'][1:,0]) and bool(tr['done'].any())==row['task_success']
                assert all(np.isfinite(tr[k]).all() for k in ('action','robot_state','action_chunk'))
            kept[key]={'root_id':key[0],'method':key[1],'output':str(directory),'source_result_sha256':sha(directory/'result.json'),'trace_sha256':sha(directory/'policy_trace.npz'),'accepted_under':'prior original/approved image gates'}
    assert len(kept)==781
    snapshots={};method_order={m['name']:i for i,m in enumerate(plan['methods'])}
    entries=sorted(kept.values(),key=lambda r:(r['root_id'],method_order[r['method']]))
    for entry in entries:
        rid=entry['root_id']
        if rid in snapshots:continue
        root=roots[rid];ref,ref_oracle=source(root['pair_id']);step=root['branch_step'];directory=pathlib.Path(entry['output'])
        external=directory/'root_external.png';wrist=directory/'root_wrist.png'
        images=[np.asarray(Image.open(p).convert('RGB')).copy() for p in (external,wrist)]
        assert all(im.shape==(224,224,3) for im in images)
        digest=hashlib.sha256(images[0].tobytes()+images[1].tobytes()).hexdigest();assert digest==ref['image_sha256'][step]
        with np.load(directory/'oracle.npz') as data:
            for key in ('qpos','qvel','control','object_position','object_quaternion'):
                np.testing.assert_allclose(data[key][0],ref_oracle[key][step],atol=1e-7,rtol=0)
        with np.load(directory/'policy_trace.npz') as data:
            np.testing.assert_allclose(data['robot_state'][0],ref['robot_state'][step],atol=1e-6,rtol=0)
            assert data['image_sha256'][0]==digest
        snapshots[rid]={'source_directory':str(directory),'source_method':entry['method'],'external_path':str(external),'external_file_sha256':sha(external),'wrist_path':str(wrist),'wrist_file_sha256':sha(wrist),'policy_image_sha256':digest,'source_trace_sha256':entry['trace_sha256'],'source_oracle_sha256':sha(directory/'oracle.npz'),'measurement_time':'root before any candidate action','selection':'fixed method priority; task success is not used'}
    assert 'task7_init44_t60' in snapshots
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    plan.update(version=4,implementation_revision=5,created_at=now,parent_protocol=str(old_path),parent_protocol_sha256=sha(old_path),
                unconsumed_suffix_image_amendment_approved=True,suffix_approval_file=str(approval_path.relative_to(ROOT)),suffix_approval_sha256=sha(approval_path),
                suffix_approval_answer=approval['user_answer'],resume_runs=[str(parent.relative_to(ROOT))],resume_audit=list(kept.values()),
                retained_complete_branches=len(kept),remaining_new_branches=len(valid)-len(kept),known_unconsumed_render_blocks=[],root_observation_snapshots=snapshots,workers=2)
    plan['parent_failure_review']={'path':'results/recovery_branches_20261014_v3/root_render_failure_audit.json','review':'Root policy observation cache was being regenerated for each candidate. Preserve and restore the identical measured root observation after strict physical-state reconstruction. No pixel editing, future observation replay, or success-based selection; root and inference-input hashes remain exact.'}
    plan['state_reconstruction']+=' Restore the exact already-measured root policy image cache where it exists; otherwise validate and capture the first root observation before any candidate action. Subsequent observations remain live. Verify current robot/simulator state against the fixed reference before restoration.'
    plan['acceptance']['image_hashes']='Restored root observation and all actual inference-input images remain exactly equal to the original hashes. No changes to image contents. Approved unconsumed prefix/suffix images including terminal record frames are diagnostic only, with strict numeric checks.'
    plan['root_cache_rules']={'values':'unaltered lossless 224x224 policy-camera observations measured at the identical root','no_future_data':True,'no_outcome_conditioning':True,'raw_rerender':'retain independently regenerated images/differences for audit; they are not the restored root measurement','state_gate':'original qpos/qvel/control/object-pose 1e-7 and robot-state 1e-6 before using snapshot','cache_missing':'original strict image gate, then save the validated root measurement before any candidate action; never accept an unmatched root'}
    plan['failure_containment']='Any restored root/policy-input image mismatch, or numeric/action/outcome violation remains fatal. The two approved unconsumed-image exceptions are active; no root-image tolerance is introduced.'
    for regression in [{'pair_id':'task1_init22','branch_step':120},{'pair_id':'task7_init44','branch_step':60}]:
        if regression not in plan['preflight_roots']:plan['preflight_roots'].append(regression)
    plan['preflight_protocol']='Each worker validates five continue suffixes including prior prefix/suffix cases and the current restored-root case; all original consumed-image/numeric/outcome gates must pass before new interventions.'
    elapsed=plan['prior_resource_accounting']['cumulative_active_window_seconds']+state['elapsed_seconds']
    gpu_hours=plan['prior_resource_accounting']['selected_gpu_window_hours']+2*state['elapsed_seconds']/3600
    probe_windows=[]
    for file in (ROOT/'reports').glob('gpu_selection_20261014_12*.json'):
        data=json.loads(file.read_text())
        if data.get('status')=='COMMAND_FINISHED':
            sec=file.stat().st_mtime-datetime.datetime.fromisoformat(data['time']).timestamp()
            probe_windows.append({'record':str(file.relative_to(ROOT)),'seconds':sec,'gpus':data['count']});elapsed+=sec;gpu_hours+=sec*data['count']/3600
    plan['prior_resource_accounting'].update(parent_v3_elapsed_seconds=state['elapsed_seconds'],root_probe_windows=probe_windows,cumulative_active_window_seconds=elapsed,selected_gpu_window_hours=gpu_hours)
    plan['queue_timeout_seconds']=28800-math.ceil(elapsed);assert plan['queue_timeout_seconds']>600;plan['job_timeout_seconds']=min(25200,plan['queue_timeout_seconds']-120)
    plan['prior_branch_artifact_bytes']=sum(int(subprocess.check_output(['du','-s','-B1',str(ROOT/'results'/name)],text=True).split()[0]) for name in ['recovery_branches_20261014_v1','recovery_branches_20261014_v2','recovery_branches_20261014_v3'])
    path=ROOT/'configs/recovery_branch_plan_20261014_v4.json'
    with path.open('x') as f:json.dump(plan,f,indent=2,ensure_ascii=False);f.write('\n')
    report={'created_at':now,'retained_branches':len(kept),'remaining_branches':len(valid)-len(kept),'verified_root_observation_snapshots':len(snapshots),'failed_root_snapshot_verified':True,'no_trial_resampled':True,'no_acceptance_threshold_changed':True,'plan':str(path),'gpu_preflight':'PENDING'}
    (parent/'root_snapshot_repair_prelaunch.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
