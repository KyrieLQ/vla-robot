"""Freeze the approved new round only after old queue cleanup and full old-row audit."""
import argparse,datetime,fcntl,json,math,os,pathlib,subprocess
import numpy as np
from PIL import Image
from paired_live_audit_v1 import arrays,sha,checked_close,image_hash,NUMERIC_FIELDS,METHODS
from verify_continuation_v4 import verify_continuation
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def alive(pid):
    try:
        os.kill(int(pid),0)
        stat=pathlib.Path('/proc')/str(pid)/'stat'
        return stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

def prepare(parent,output):
    state=json.loads((parent/'status.json').read_text())
    assert state['status'] in ('INCOMPLETE_RENDER_REVIEW','COMPLETE'),'Current strict queue must finish normally or with explicit held roots'
    assert not alive(state['supervisor_pid']),'Parent supervisor cleanup still in progress'
    for p in (parent/'services').glob('worker*/service.json'):
        service=json.loads(p.read_text());assert service['cleanup_completed']
        assert not any(alive(service[k]) for k in ('server_pid','client_pid') if service.get(k))
    lock=(ROOT/'results/formal_active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_path=pathlib.Path(state['plan']).resolve();old=json.loads(old_path.read_text())
    approval_path=ROOT/'configs/recovery_paired_live_20261014_APPROVED.json';approval=json.loads(approval_path.read_text())
    assert approval['approved'] and approval['originating_thread']==os.environ['CODEX_THREAD_ID']
    checks_path=ROOT/'reports/paired_live_cpu_checks_20261014.json';assert json.loads(checks_path.read_text())['status']=='PASS'
    manifest=json.loads((ROOT/old['root_manifest']).read_text());roots={r['root_id']:r for r in manifest['roots']}
    expected={(rid,m['name']) for rid in roots for m in old['methods']};kept={};continuations=[];reference={};held=[]
    def source(root):
        if root['pair_id'] not in reference:
            d=ROOT/old['source_episodes']/root['pair_id']
            assert sha(d/'policy_trace.npz')==root['source_trace_sha256'] and sha(d/'oracle.npz')==root['source_oracle_sha256']
            reference[root['pair_id']]=(arrays(d/'policy_trace.npz'),arrays(d/'oracle.npz'))
        return reference[root['pair_id']]
    for path in sorted((parent/'workers').glob('worker*/result.json')):
        data=json.loads(path.read_text());assert data['status'] in ('PASS','INCOMPLETE_RENDER_REVIEW');held.extend(data.get('deferred_roots',[]))
        for row in data['branches']:
            key=(row['root_id'],row['method']);assert key in expected and key not in kept
            d=pathlib.Path(row['output']);saved=json.loads((d/'result.json').read_text())
            assert saved['status']==row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED')
            trace=arrays(d/'policy_trace.npz');oracle=arrays(d/'oracle.npz')
            assert trace['action'].shape==(row['steps'],7) and np.array_equal(trace['done'],oracle['goal_predicate'][1:,0])
            if row['method']=='continue':
                ref,ref_oracle=source(roots[row['root_id']])
                continuations.append(verify_continuation(row,trace,oracle,ref,ref_oracle,old))
            kept[key]={'root_id':key[0],'method':key[1],'output':str(d),'result_sha256':sha(d/'result.json'),
                       'trace_sha256':sha(d/'policy_trace.npz'),'oracle_sha256':sha(d/'oracle.npz')}
    assert len(kept)+sum(4-sum(key[0]==r['root_id'] for key in kept) for r in held)==1076
    assert len({r['root_id'] for r in held})==len(held)
    snapshots={};priority=['continue','fine20','resample5','retreat5_fine20']
    for rid,root in roots.items():
        entries=[pathlib.Path(kept[(rid,m)]['output']) for m in priority if (rid,m) in kept]
        if rid in old['root_observation_snapshots']:entries.append(pathlib.Path(old['root_observation_snapshots'][rid]['source_directory']))
        # These are already-recorded measurements, including roots whose later
        # inference failed. Their outcomes are never reused in the new round.
        for v in (6,4):entries.append(ROOT/'results'/f'recovery_branches_20261014_v{v}'/'workers'/f"worker{root['worker']}"/'roots'/rid/'continue')
        ref,ref_oracle=source(root);step=root['branch_step']
        for d in entries:
            paths=[d/f'root_{cam}.png' for cam in ('external','wrist')]
            if not all(p.is_file() for p in paths) or not (d/'policy_trace.npz').is_file() or not (d/'oracle.npz').is_file():continue
            images=[np.asarray(Image.open(p).convert('RGB')).copy() for p in paths]
            digest=image_hash(*images)
            assert digest==ref['image_sha256'][step],'Archived reference candidate is not the verified original root measurement'
            tr=arrays(d/'policy_trace.npz');oracle=arrays(d/'oracle.npz')
            assert tr['image_sha256'][0]==digest
            checked_close(tr['robot_state'][0],ref['robot_state'][step],1e-6)
            for key in NUMERIC_FIELDS:checked_close(oracle[key][0],ref_oracle[key][step],1e-7)
            snapshots[rid]={'source_directory':str(d),'external_path':str(paths[0]),'external_file_sha256':sha(paths[0]),
                            'wrist_path':str(paths[1]),'wrist_file_sha256':sha(paths[1]),'policy_image_sha256':digest,
                            'selection':'fixed method priority; recorded root only, never downstream outcome',
                            'historical_observation_verified':True,'old_outcome_reused':False}
            break
    prior=dict(old['prior_resource_accounting'])
    elapsed=prior['cumulative_active_window_seconds']+state['elapsed_seconds']
    prior.update(strict_v6_elapsed_seconds=state['elapsed_seconds'],cumulative_active_window_seconds=elapsed,
                 selected_gpu_window_hours=prior['selected_gpu_window_hours']+2*state['elapsed_seconds']/3600)
    cap=min(14400,28800-math.ceil(elapsed));assert cap>1800,'Insufficient remaining time; do not exceed approved cumulative budget'
    prior_dirs=[ROOT/'results'/f'recovery_branches_20261014_v{v}' for v in range(1,7)]
    prior_dirs += [ROOT/'reports/policy_image_effect_20261014']
    size=sum(int(subprocess.check_output(['du','-s','-B1',str(d)],text=True).split()[0]) for d in prior_dirs if d.exists())
    from storage_guard import check
    storage=check(reserve_bytes=128*2**20);assert size+128*2**20<4*2**30
    plan={key:old[key] for key in ['checkpoint','source_episodes','root_manifest','root_manifest_sha256','development_split',
                                 'max_total_steps','fine_steps','retreat_steps','retreat_action','reset_protocol','wait_steps','analysis']}
    plan.update(version=1,phase='recovery_paired_live',created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                originating_thread=approval['originating_thread'],paired_live_amendment_approved=True,approval_file=str(approval_path.relative_to(ROOT)),
                approval_sha256=sha(approval_path),parent_run=str(parent.relative_to(ROOT)),parent_plan=str(old_path),parent_plan_sha256=sha(old_path),
                methods=list(METHODS),expected_branches=1345,expected_roots=269,expected_unique_episodes=200,workers=2,
                maximum_simultaneous_gpus=2,allowed_physical_gpus=[0,1,2,3],queue_timeout_seconds=cap,job_timeout_seconds=cap-300,
                project_budget_gib=120,filesystem_reserve_gib=100,new_artifact_budget_gib=4,prior_branch_artifact_bytes=size,
                prior_resource_accounting=prior,root_observation_snapshots=snapshots,resume_runs=[],reuse_old_outcomes=False,
                cpu_checks_file=str(checks_path.relative_to(ROOT)),cpu_checks_sha256=sha(checks_path),
                preflight_specs=[{'root_id':'task0_init2_t60','method':'continue_a'},
                                 {'root_id':'task4_init22_t120','method':'continue_a'},
                                 {'root_id':'task7_init48_t60','method':'continue_a'},
                                 {'root_id':'task4_init22_t120','method':'resample5'},
                                 {'root_id':'task0_init2_t60','method':'retreat5_fine20'}],
                periodic_chat_notifications=False,persistent_idle_monitor=False,new_training_updates=0,confirmation_used=0,
                model_replay='all actual calls on all five branches; same original 1e-6 action tolerance',
                physics_replay='all 538 new A/B suffixes, actual recorded actions; 1e-7 physics, 1e-6 proprioception, exact goal and terminal step',
                historical_suffix_equality_required=False,primary_control='continue_a',repeat_control='continue_b',
                feasibility_min_distinct_robust_rescues=8,robust_rescue_rule=approval['robust_rescue_rule'],
                root_acquisition='Verified historical root observation when available; otherwise a single pre-action measurement, identical across all five methods. Never outcome-selected or repeatedly sampled.')
    plan['analysis']['feasibility_gate']='At least 8 distinct originally failed episodes with A and B both failing and at least one fixed candidate succeeding, and all new model/physics reconstruction audits passed.'
    plan['analysis']['noise_limit']='One same-seed repeat B for each fixed A; report disagreements and sensitivity, no claim of zero stochastic variance.'
    assert all(spec['root_id'] in snapshots for spec in plan['preflight_specs'])
    with output.open('x') as f:json.dump(plan,f,indent=2,ensure_ascii=False);f.write('\n')
    audit={'created_at':plan['created_at'],'old_complete_branches_retained':len(kept),'old_expected_branches':1076,
           'old_held_roots':held,'old_control_audits':continuations,'old_result_inventory':list(kept.values()),
           'new_round_branches':1345,'old_outcomes_reused_in_new_statistics':False,'verified_root_observations':len(snapshots),
           'budget_seconds':cap,'prior_active_seconds':elapsed,'prior_artifact_bytes':size,'storage':storage,'plan':str(output)}
    (parent/'approved_paired_successor_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('old_control_audits','old_result_inventory','old_held_roots')}))
    return plan
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    prepare(pathlib.Path(args.parent).resolve(),pathlib.Path(args.output).resolve())
