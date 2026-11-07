"""Freeze unchanged acceptance for remaining roots, with rejected input isolated."""
import datetime,hashlib,json,math,os,pathlib,subprocess
import numpy as np
from PIL import Image
from verify_continuation_v4 import verify_continuation
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(p):
    with np.load(p) as f:return {k:f[k] for k in f.files}
def main():
    parent=ROOT/'results/recovery_branches_20261014_v4'
    old_path=ROOT/'configs/recovery_branch_plan_20261014_v4.json'
    plan=json.loads(old_path.read_text());state=json.loads((parent/'status.json').read_text())
    assert state['status']=='FAILED'
    failure=json.loads((parent/'workers/worker1/progress.json').read_text())
    blocked='task7_init48_t60'
    assert failure['active_root']==blocked and failure['active_method']=='continue'
    evidence=parent/'workers/worker1/roots'/blocked/'continue/render_mismatch.json'
    mismatch=json.loads(evidence.read_text())
    assert mismatch['policy_consumes_frame'] and mismatch['step']==175
    manifest=json.loads((ROOT/plan['root_manifest']).read_text())
    roots={r['root_id']:r for r in manifest['roots']}
    valid={(r,m['name']) for r in roots for m in plan['methods']}
    refs={};kept={};control_audits=[]
    def source(pair):
        if pair not in refs:
            p=ROOT/plan['source_episodes']/pair
            refs[pair]=(arrays(p/'policy_trace.npz'),arrays(p/'oracle.npz'))
        return refs[pair]
    for file in sorted((parent/'workers').glob('worker*/progress.json')):
        for row in json.loads(file.read_text())['branches']:
            key=(row['root_id'],row['method']);assert key in valid and key not in kept
            directory=pathlib.Path(row['output']);saved=json.loads((directory/'result.json').read_text())
            assert saved['status']==row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED')
            tr=arrays(directory/'policy_trace.npz');oracle=arrays(directory/'oracle.npz')
            assert tr['action'].shape==(row['steps'],7)
            assert np.array_equal(tr['done'],oracle['goal_predicate'][1:,0])
            assert bool(tr['done'].any())==row['task_success']
            assert all(np.isfinite(tr[k]).all() for k in ('action','robot_state','action_chunk'))
            if row['method']=='continue':
                reference,ref_oracle=source(row['pair_id'])
                control_audits.append(verify_continuation(row,tr,oracle,reference,ref_oracle,plan))
            kept[key]={'root_id':key[0],'method':key[1],'output':str(directory),
                       'source_result_sha256':sha(directory/'result.json'),'trace_sha256':sha(directory/'policy_trace.npz'),
                       'accepted_under':'unchanged consumed-input, action, physics and outcome gates; approved unconsumed-image exceptions'}
    assert len(kept)==857 and not any(key[0]==blocked for key in kept)
    snapshots=dict(plan['root_observation_snapshots'])
    priority={m['name']:i for i,m in enumerate(plan['methods'])}
    for entry in sorted(kept.values(),key=lambda r:(r['root_id'],priority[r['method']])):
        rid=entry['root_id']
        if rid in snapshots:continue
        root=roots[rid];reference,ref_oracle=source(root['pair_id']);step=root['branch_step']
        directory=pathlib.Path(entry['output']);paths=[directory/f'root_{cam}.png' for cam in ('external','wrist')]
        images=[np.asarray(Image.open(p).convert('RGB')).copy() for p in paths]
        assert all(im.shape==(224,224,3) for im in images)
        digest=hashlib.sha256(images[0].tobytes()+images[1].tobytes()).hexdigest()
        assert digest==reference['image_sha256'][step]
        oracle=arrays(directory/'oracle.npz');tr=arrays(directory/'policy_trace.npz')
        for k in ('qpos','qvel','control','object_position','object_quaternion'):
            np.testing.assert_allclose(oracle[k][0],ref_oracle[k][step],atol=1e-7,rtol=0)
        np.testing.assert_allclose(tr['robot_state'][0],reference['robot_state'][step],atol=1e-6,rtol=0)
        assert tr['image_sha256'][0]==digest
        snapshots[rid]={'source_directory':str(directory),'source_method':entry['method'],
                        'external_path':str(paths[0]),'external_file_sha256':sha(paths[0]),
                        'wrist_path':str(paths[1]),'wrist_file_sha256':sha(paths[1]),
                        'policy_image_sha256':digest,'source_trace_sha256':entry['trace_sha256'],
                        'source_oracle_sha256':sha(directory/'oracle.npz'),
                        'measurement_time':'root before any candidate action','selection':'fixed method priority, never task outcome'}
    now=datetime.datetime.now().astimezone().isoformat()
    plan.update(version=5,implementation_revision=6,created_at=now,parent_protocol=str(old_path),
                parent_protocol_sha256=sha(old_path),resume_runs=[str(parent.relative_to(ROOT))],
                resume_audit=list(kept.values()),retained_complete_branches=len(kept),remaining_new_branches=len(valid)-len(kept),
                root_observation_snapshots=snapshots,known_unconsumed_render_blocks=[])
    plan['known_strict_render_blocks']=[{'root_id':blocked,'method':'continue','step':175,
        'reason':'Actual policy-input image differs; independent inference also exceeds original action tolerance. Gate remains failed.',
        'source':str(evidence.relative_to(ROOT)),'source_sha256':sha(evidence),'new_attempt':False}]
    plan['failure_containment']='Reject and retain any new root/consumed-image failure; stop that root without retrying, continue independent roots. Numeric/action/outcome violations remain fatal. No denominator or acceptance change.'
    plan['parent_failure_review']={'source':str(evidence.relative_to(ROOT)),
        'sensitivity_probe':'reports/policy_image_effect_20261014/result.json',
        'review':'Policy-input exception is NOT approved. Known failed root stays unattempted and in the denominator; only independent remaining work is admitted.'}
    accounting=plan['prior_resource_accounting']
    elapsed=accounting['cumulative_active_window_seconds']+state['elapsed_seconds']
    gpu_hours=accounting['selected_gpu_window_hours']+2*state['elapsed_seconds']/3600
    probe=ROOT/'reports/gpu_selection_20261014_132542.json';p=json.loads(probe.read_text())
    assert p['status']=='COMMAND_FINISHED' and p['count']==1
    sec=probe.stat().st_mtime-datetime.datetime.fromisoformat(p['time']).timestamp()
    elapsed+=sec;gpu_hours+=sec/3600
    accounting.update(parent_v4_elapsed_seconds=state['elapsed_seconds'],image_sensitivity_probe_window_seconds=sec,
                      cumulative_active_window_seconds=elapsed,selected_gpu_window_hours=gpu_hours)
    plan['queue_timeout_seconds']=28800-math.ceil(elapsed)
    plan['job_timeout_seconds']=min(25200,plan['queue_timeout_seconds']-120)
    assert plan['queue_timeout_seconds']>600
    plan['prior_branch_artifact_bytes']=sum(int(subprocess.check_output(['du','-s','-B1',str(ROOT/'results'/f'recovery_branches_20261014_v{v}')],text=True).split()[0]) for v in range(1,5))
    target=ROOT/'configs/recovery_branch_plan_20261014_v5.json'
    with target.open('x') as f:json.dump(plan,f,indent=2,ensure_ascii=False);f.write('\n')
    report={'created_at':now,'retained_complete_branches':len(kept),'expected_branches':len(valid),
            'held_branches':4,'independent_remaining_branches':len(valid)-len(kept)-4,
            'strict_continue_audits':control_audits,'policy_input_gate_changed':False,
            'accepted_invalid_outcomes':0,'plan':str(target),'gpu_preflight':'PENDING'}
    (parent/'unaffected_resume_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='strict_continue_audits'}))
    print('Independent continuation audits:',len(control_audits))
if __name__=='__main__':main()
