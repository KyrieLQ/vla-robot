"""Hash-bound resumption within the same approved paired round, never old strict outcomes."""
import argparse,datetime,fcntl,json,math,os,pathlib,subprocess,time,collections
from paired_live_audit_v1 import check_saved_branch,verify_audit_bindings,sha,METHODS
from storage_guard_v3 import du_bytes,du_paths_bytes,check
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
SCIENCE_KEYS=('approval_sha256','checkpoint','source_episodes','root_manifest','root_manifest_sha256','development_split',
              'methods','max_total_steps','fine_steps','retreat_steps','retreat_action','reset_protocol','wait_steps',
              'root_observation_snapshots','primary_control','repeat_control','model_replay','physics_replay',
              'historical_suffix_equality_required','feasibility_min_distinct_robust_rescues','robust_rescue_rule','analysis')

def within(path,parent):
    try:pathlib.Path(path).resolve().relative_to(pathlib.Path(parent).resolve());return True
    except ValueError:return False

def verify_entry(entry,root,plan):
    directory=pathlib.Path(entry['output'])
    assert entry['root_id']==root['root_id'] and entry['method'] in METHODS
    for name,digest in entry['files_sha256'].items():
        assert pathlib.Path(name).name==name and sha(directory/name)==digest,'Retained artifact changed: '+name
    row=json.loads((directory/'result.json').read_text())
    assert row['root_id']==entry['root_id'] and row['method']==entry['method'] and row['output']==str(directory)
    check_saved_branch(directory,row,root,plan)
    if entry['mode']=='verified':
        assert row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED') and row['verification_status']=='PASS'
        verify_audit_bindings(directory,row)
    else:
        assert entry['mode']=='audit_only'
        assert row['status']=='RUNNING' and row['verification_status']=='PENDING'
        assert row['rollout_status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED')
        assert not row.get('error'),'A failed numeric verification may not be silently retried'
    return row

def alive(pid):
    try:
        os.kill(int(pid),0);p=pathlib.Path('/proc')/str(pid)/'stat'
        return p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (ProcessLookupError,FileNotFoundError):return False

def prepare(parent,output):
    preparation_started=time.monotonic()
    state=json.loads((parent/'status.json').read_text());assert state['status']=='FAILED'
    assert not alive(state['supervisor_pid']) and not alive(state['child_pid'])
    cleanup=[]
    for p in (parent/'services').glob('worker*/service.json'):
        service=json.loads(p.read_text());pids=[service[k] for k in ('server_pid','client_pid') if service.get(k)]
        assert not any(alive(pid) for pid in pids)
        cleanup.append({'service':str(p),'reported_cleanup_completed':service['cleanup_completed'],'recorded_processes_confirmed_exited':True})
    lock=(ROOT/'results/formal_active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old_path=pathlib.Path(state['plan']);old=json.loads(old_path.read_text())
    assert old['paired_live_amendment_approved'] and old['expected_branches']==1345
    assert old['originating_thread']==os.environ['CODEX_THREAD_ID']
    assert sha(ROOT/old['approval_file'])==old['approval_sha256']
    manifest=json.loads((ROOT/old['root_manifest']).read_text());roots={r['root_id']:r for r in manifest['roots']}
    directories={}
    for path in (parent/'workers').glob('worker*/progress.json'):
        for row in json.loads(path.read_text())['branches']:directories[(row['root_id'],row['method'])]=pathlib.Path(row['output'])
    for path in (parent/'workers').glob('worker*/roots/*/*/result.json'):
        row=json.loads(path.read_text());key=(row['root_id'],row['method'])
        assert key not in directories or directories[key]==path.parent;directories[key]=path.parent
    prior_entries={(e['root_id'],e['method']):e for e in old.get('resume_entries',[])}
    entries=[];incomplete=[]
    for key,directory in sorted(directories.items()):
        assert key[0] in roots and key[1] in METHODS
        if not within(directory,parent/'workers'):
            assert key in prior_entries and directory==pathlib.Path(prior_entries[key]['output'])
            verify_entry(prior_entries[key],roots[key[0]],old)
        row=json.loads((directory/'result.json').read_text())
        if row['verification_status']=='PASS':mode='verified'
        elif row.get('rollout_status') and row['status']=='RUNNING' and row['verification_status']=='PENDING':mode='audit_only'
        elif row.get('error'):
            raise AssertionError('Unreviewed verification/program error at '+str(directory))
        else:
            incomplete.append({'root_id':key[0],'method':key[1],'output':str(directory),'reason':'No complete committed rollout; source preserved'});continue
        files={p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file() and not p.name.endswith('.tmp')}
        entry={'root_id':key[0],'method':key[1],'output':str(directory),'mode':mode,'files_sha256':files,
               'same_paired_round':True,'source_run':str(parent),'source_plan_sha256':sha(old_path)}
        verify_entry(entry,roots[key[0]],old)
        for camera in ('external','wrist'):
            p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(directory/(camera+'.mp4'))],capture_output=True,text=True,check=True,timeout=30)
            stream=json.loads(p.stdout)['streams'][0];assert (stream['width'],stream['height'],int(stream['nb_frames']))==(224,224,row['steps']+1)
        entries.append(entry)
    complete=sum(e['mode']=='verified' for e in entries);audit_only=sum(e['mode']=='audit_only' for e in entries)
    elapsed=state['elapsed_seconds']
    dedup_path=parent/'deduplication_result.json'
    dedup_seconds=json.loads(dedup_path.read_text())['elapsed_seconds'] if dedup_path.exists() else 0.0
    maintenance_seconds=dedup_seconds+time.monotonic()-preparation_started
    paired_elapsed=old.get('paired_round_prior_active_seconds',0.0)+elapsed+maintenance_seconds
    prior=dict(old['prior_resource_accounting']);prior['cumulative_active_window_seconds']+=elapsed+maintenance_seconds
    prior['selected_gpu_window_hours']+=2*elapsed/3600
    prior.setdefault('paired_attempts',[]).append({'run':str(parent.relative_to(ROOT)),'elapsed_seconds':elapsed,'selected_gpu_window_hours_upper_bound':2*elapsed/3600})
    prior.setdefault('non_gpu_maintenance',[]).append({'parent':str(parent.relative_to(ROOT)),'seconds':maintenance_seconds,'scope':'byte-preserving deduplication plus completed-artifact inspection before cap freeze'})
    cap=min(14400-math.ceil(paired_elapsed),28800-math.ceil(prior['cumulative_active_window_seconds']))
    assert cap>600,'Approved active-time budget exhausted'
    plan=dict(old);plan.update(version=old['version']+1,implementation_revision=3,
        created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        parent_run=str(parent.relative_to(ROOT)),parent_plan=str(old_path),parent_plan_sha256=sha(old_path),
        scientific_round_id=old.get('scientific_round_id','recovery_paired_live_20261014'),
        resume_entries=entries,resume_runs=[str(parent.relative_to(ROOT))],retained_verified_branches=complete,
        retained_audit_only_rollouts=audit_only,remaining_new_rollouts=1345-complete-audit_only,
        prior_resource_accounting=prior,paired_round_prior_active_seconds=paired_elapsed,
        queue_timeout_seconds=cap,job_timeout_seconds=cap-300,
        prior_branch_artifact_bytes=old['prior_branch_artifact_bytes']+du_bytes(parent),
        execution_repair='Bounded retry only when du reports a disappeared atomic JSON temporary. Never use partial du counts; other disk errors still fail. Fully recorded interrupted rollout is audited, not re-simulated.',
        resumption_rule='Reuse only immutable hash-bound artifacts from this approved paired round. Completed historical strict-round outcomes are never allowed.')
    if old.get('recovery_artifact_roots'):
        artifact_paths=[ROOT/path for path in old['recovery_artifact_roots']]+[parent]
    else:
        dedup=json.loads((parent/'deduplication_result.json').read_text())
        assert dedup['status']=='COMPLETE' and dedup['all_content_hashes_unchanged'] and dedup['all_pathnames_retained']
        inventory=json.loads((parent/'deduplication_inventory.json').read_text())
        artifact_paths=[pathlib.Path(path) for path in inventory['stage_paths']]
    artifact_paths=list(dict.fromkeys(p.resolve() for p in artifact_paths))
    assert all(within(p,ROOT) for p in artifact_paths)
    actual_stage_bytes=du_paths_bytes(artifact_paths)
    assert actual_stage_bytes+128*2**20<=4*2**30,'Actual phase allocation still lacks branch headroom'
    plan.update(recovery_artifact_roots=[str(path.relative_to(ROOT)) for path in artifact_paths],
                prior_branch_artifact_bytes=actual_stage_bytes,
                artifact_accounting='One du invocation over all recovery-attempt roots; count shared immutable inodes once, retain 128 MiB concurrent-write reserve; 4 GiB limit unchanged.',
                execution_repair='Deduplicate byte-identical immutable NPZ/PNG with every original pathname and hash preserved; resume unchanged cases and count physical phase allocation consistently across attempts.')
    # Scheduling only: leave every started episode on its original worker;
    # balance wholly unstarted episodes using counts, never task outcomes.
    by_pair=collections.defaultdict(list)
    for root in roots.values():by_pair[root['pair_id']].append(root)
    present={(entry['root_id'],entry['method']) for entry in entries}
    assignment={};remaining_load=[0,0];unstarted=[]
    for pair,group in sorted(by_pair.items()):
        workers={root['worker'] for root in group};assert len(workers)==1
        original=next(iter(workers));existing=sum((root['root_id'],method) in present for root in group for method in METHODS)
        missing=len(group)*5-existing
        if existing:
            assignment[pair]=original;remaining_load[original]+=missing
        else:unstarted.append((missing,pair))
    for missing,pair in sorted(unstarted,key=lambda item:(-item[0],item[1])):
        worker=min(range(2),key=lambda w:(remaining_load[w],w));assignment[pair]=worker;remaining_load[worker]+=missing
    assert len(assignment)==200 and sum(remaining_load)==1345-len(entries)
    plan.update(execution_worker_by_pair=assignment,remaining_rollouts_by_worker=remaining_load,
                execution_assignment_rule='Only wholly unstarted episodes may move; all roots and five methods of one episode share its assigned worker. Original manifest, IDs, inputs, seeds, parameters, thresholds and existing outcomes unchanged.')
    assert all(plan[key]==old[key] for key in SCIENCE_KEYS)
    plan['source_scientific_identity_sha256']=sha(old_path)
    check(reserve_bytes=128*2**20)
    report={'created_at':plan['created_at'],'verified_complete_branches':complete,'audit_only_complete_rollouts':audit_only,
            'remaining_new_rollouts':plan['remaining_new_rollouts'],'expected_branches':1345,'numeric_or_scientific_threshold_changed':False,
            'outcomes_resampled':False,'incomplete_artifacts_preserved':incomplete,'cleanup_evidence':cleanup,
            'this_round_remaining_seconds':cap,'paired_active_seconds_spent':paired_elapsed,
            'cumulative_active_seconds_spent':prior['cumulative_active_window_seconds'],
            'actual_prior_stage_allocated_bytes':actual_stage_bytes,'artifact_cap_bytes':4*2**30,
            'non_gpu_maintenance_charged_seconds':maintenance_seconds,'remaining_rollouts_by_worker':remaining_load,'plan':str(output)}
    with output.open('x') as f:json.dump(plan,f,indent=2,ensure_ascii=False);f.write('\n')
    (parent/'resume_inventory_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    return plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    prepare(pathlib.Path(a.parent).resolve(),pathlib.Path(a.output).resolve())
