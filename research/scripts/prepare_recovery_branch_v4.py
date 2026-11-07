#!/usr/bin/env python3
"""Freeze a successor only after the parent has terminated and its rows are audited."""
import argparse,datetime,hashlib,json,math,os,pathlib,subprocess
import numpy as np
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(parent,approval_path,output,inspect_only=False):
    parent=pathlib.Path(parent);approval_path=pathlib.Path(approval_path)
    state=json.loads((parent/'status.json').read_text());old_path=pathlib.Path(state['plan']);plan=json.loads(old_path.read_text())
    approval=json.loads(approval_path.read_text())
    assert approval['approved'] is True and approval['status']=='APPROVED'
    assert plan['originating_thread']==os.environ['CODEX_THREAD_ID']
    assert sha(old_path)==approval['parent_sha256']
    if not inspect_only:assert state['status'] in ('INCOMPLETE_RENDER_REVIEW','COMPLETE'), 'Parent requires failure review or is still running'
    for hashkey,pathkey in [('root_manifest_sha256','root_manifest'),('development_split_sha256','development_split'),('source_diagnosis_sha256','source_diagnosis')]:assert sha(ROOT/plan[pathkey])==plan[hashkey]
    roots=json.loads((ROOT/plan['root_manifest']).read_text())['roots']
    valid={(root['root_id'],method['name']) for root in roots for method in plan['methods']}
    kept={}
    for file in sorted((parent/'workers').glob('worker*/progress.json')):
        worker=json.loads(file.read_text())
        if not inspect_only:assert worker['status'] in ('PASS','INCOMPLETE_RENDER_REVIEW')
        for row in worker['branches']:
            key=(row['root_id'],row['method']);assert key in valid and key not in kept
            directory=pathlib.Path(row['output']);saved=json.loads((directory/'result.json').read_text())
            assert saved['status']==row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED')
            with np.load(directory/'policy_trace.npz') as trace,np.load(directory/'oracle.npz') as oracle:
                assert trace['action'].shape==(row['steps'],7) and trace['robot_state'].shape==(row['steps']+1,8)
                assert bool(trace['done'].any())==row['task_success']
                assert np.array_equal(trace['done'],oracle['goal_predicate'][1:,0])
                for name in ('action','action_chunk','robot_state'):assert np.isfinite(trace[name]).all()
            kept[key]={'root_id':key[0],'method':key[1],'output':str(directory),
                       'source_result_sha256':sha(directory/'result.json'),'trace_sha256':sha(directory/'policy_trace.npz'),
                       'accepted_under':'prior completed branch with unchanged or stricter observation gates'}
    missing=sorted(valid-set(kept))
    summary={'parent_status':state['status'],'retained_complete_branches':len(kept),'missing_branches':len(missing),'expected_branches':len(valid),'missing_keys':missing}
    if inspect_only:return summary
    if not missing:return dict(summary,next_action='NO_ADDITIONAL_EVALUATION')
    plan.update(version=4,created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),parent_protocol=str(old_path),parent_protocol_sha256=sha(old_path),
                unconsumed_suffix_image_amendment_approved=True,suffix_approval_file=str(approval_path.relative_to(ROOT)),suffix_approval_sha256=sha(approval_path),
                suffix_approval_answer=approval['user_answer'],resume_runs=[str(parent.relative_to(ROOT))],resume_audit=list(kept.values()),
                retained_complete_branches=len(kept),remaining_new_branches=len(missing),known_unconsumed_render_blocks=[])
    plan['workers']=1 if len(missing)<=32 else 2
    plan['worker_scheduling_rule']='Use one worker for at most 32 missing branches, otherwise two. This changes only execution scheduling; all root/method IDs, seeds, fit/calibration membership and metrics stay frozen.'
    plan['acceptance']['image_hashes']='Root and actual inference-input images exactly match. Unconsumed prefix/suffix images, including terminal record frames, are diagnostic only under the two explicit approvals. Physical state, action, predicted chunks and success/step gates stay unchanged.'
    plan['failure_containment']='Both unconsumed-image exceptions are approved; any root/inference-image or numeric/action/final-result mismatch still stops affected work. Never change or resample root/method IDs due to outcomes.'
    regression={'pair_id':'task1_init22','branch_step':120}
    if regression not in plan['preflight_roots']:plan['preflight_roots'].append(regression)
    plan['preflight_protocol']='Each execution worker checks four continue suffixes: success, failure and both previously exposed render cases; actual inference inputs and all numeric/outcome gates must pass before new interventions.'
    cumulative=plan['prior_resource_accounting']['cumulative_active_window_seconds']+state['elapsed_seconds']
    plan['prior_resource_accounting'].update(parent_v3_elapsed_seconds=state['elapsed_seconds'],cumulative_active_window_seconds=cumulative)
    plan['prior_resource_accounting']['selected_gpu_window_hours']+=2*state['elapsed_seconds']/3600
    plan['queue_timeout_seconds']=int(plan['original_queue_active_window_budget_seconds']-math.ceil(cumulative))
    assert plan['queue_timeout_seconds']>600, 'Remaining approved active-window budget too small; requires review'
    plan['job_timeout_seconds']=min(25200,plan['queue_timeout_seconds']-120)
    names=['recovery_branches_20261014_v1','recovery_branches_20261014_v2','recovery_branches_20261014_v3']
    plan['prior_branch_artifact_bytes']=sum(int(subprocess.check_output(['du','-s','-B1',str(ROOT/'results'/name)],text=True).split()[0]) for name in names)
    assert plan['prior_branch_artifact_bytes']+512*2**20<plan['new_artifact_budget_gib']*2**30
    plan['completion_audit']='Verify all 1076 branches, every actual continue-policy input image, predicted chunk, action, robot/simulator state and final outcome. Report approved record-only image differences separately.'
    output=pathlib.Path(output)
    with output.open('x') as f:json.dump(plan,f,indent=2,ensure_ascii=False);f.write('\n')
    return dict(summary,next_action='LAUNCH_FROZEN_SUCCESSOR',plan=str(output),workers=plan['workers'],remaining_seconds=plan['queue_timeout_seconds'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--output',required=True);parser.add_argument('--inspect-only',action='store_true');a=parser.parse_args()
    print(json.dumps(prepare(a.parent,a.approval,a.output,a.inspect_only),indent=2))
if __name__=='__main__':main()
