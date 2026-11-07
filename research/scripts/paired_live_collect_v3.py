"""Five fixed live branches per root, each with lossless independent verification."""
import argparse,collections,datetime,json,os,pathlib,subprocess,time,traceback
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--worker',type=int,required=True);a=p.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());assert plan['paired_live_amendment_approved'] is True
    from paired_live_audit_v1 import arrays,sha,image_hash,replay_model_inputs,verify_audit_bindings,METHODS
    assert sha(ROOT/plan['approval_file'])==plan['approval_sha256']
    assert json.loads((ROOT/plan['approval_file']).read_text())['approved'] is True
    assert sha(ROOT/plan['root_manifest'])==plan['root_manifest_sha256']
    manifest=json.loads((ROOT/plan['root_manifest']).read_text());all_roots=manifest['roots']
    assignment=plan['execution_worker_by_pair']
    assert set(assignment)=={r['pair_id'] for r in all_roots} and set(assignment.values())<={0,1}
    roots=[r for r in all_roots if assignment[r['pair_id']]==a.worker]
    split=json.loads((ROOT/plan['development_split']).read_text())
    allowed={r['pair_id'] for r in split['development']};assert all(r['pair_id'] in allowed for r in all_roots)
    assert tuple(plan['methods'])==METHODS and len(all_roots)==269
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
    from egl_device import select_egl
    mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import imageio
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from paired_live_scene_v1 import LiveScene
    from paired_live_resume_v2 import verify_entry
    client=WebsocketClientPolicy('127.0.0.1',a.port)
    start=time.monotonic()
    status={'status':'RUNNING','worker':a.worker,'expected_roots':len(roots),'expected_branches':len(roots)*5,
            'completed_roots':0,'completed_branches':0,'phase':'preflight','started_at':now(),'render_mapping':mapping,
            'active_root':None,'active_method':None,'branches':[],'model_calls_verified':0,'physics_controls_verified':0,
            'reused_branches':0,'audit_only_finished':0,'new_rollouts_completed':0}
    def progress(operation=None,done=None,total=None):
        if operation:status.update(operation=operation,operation_done=done,operation_total=total)
        status.update(heartbeat_at=now(),elapsed_seconds=time.monotonic()-start)
        n=status['completed_branches'];status['estimated_remaining_seconds']=(status['elapsed_seconds']/n*(status['expected_branches']-n) if n else None)
        write(out/'progress.json',status)
    scene=LiveScene(plan,mapping,progress)
    def budget_guard():
        # Bound simultaneous unflushed branch writes before creating video/NPZs.
        from storage_guard_v3 import check,du_paths_bytes
        check(reserve_bytes=128*2**20)
        paths=[ROOT/path for path in plan['recovery_artifact_roots']]+[out.parents[1]]
        used=du_paths_bytes(paths)
        assert used+128*2**20 <= plan['new_artifact_budget_gib']*2**30,'Recovery artifact budget reached'
    def branch(root,method,directory):
        budget_guard();directory.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
        task,obs,recorder,old,audit=scene.prepare(root,directory)
        step0=root['branch_step'];replay_seconds=time.monotonic()-begin
        row={'root_id':root['root_id'],'pair_id':root['pair_id'],'task_id':root['task_id'],'branch_step':step0,'method':method,
             'status':'RUNNING','verification_status':'PENDING','language':task.language,'policy_calls':0,'steps':0,
             'baseline_success':old['success_at_220'],'task_success':False,'first_success_step':None,
             'root_image_sha256':audit['root_policy_image_sha256'],'root_max_abs_errors':audit['physical_max_errors'],
             'prefix_replay_seconds':replay_seconds,'prefix_replay_actions':step0,'module_partition':root['module_partition'],
             'output':str(directory),'manual_steps':0}
        metadata={'prompt':task.language,'policy_seed':root['resample_seed'] if method=='resample5' else root['policy_seed'],
                  'policy_rng_advance':0 if method=='resample5' else step0//5,'actual_inputs_only':True,'oracle_sent_to_policy':False}
        write(directory/'policy_inputs.json',metadata)
        writers=[imageio.get_writer(str(directory/(c+'.mp4')),fps=10,codec='libx264',ffmpeg_params=['-crf','18']) for c in ('external','wrist')]
        queue=collections.deque();actions=[];states=[];hashes=[];chunks=[];calls=[];latencies=[];dones=[]
        external_inputs=[];wrist_inputs=[];state_inputs=[];suffix_start=time.monotonic()
        try:
            for step in range(step0,plan['max_total_steps']):
                external,wrist,state=scene.inputs(obs);relative=step-step0
                writers[0].append_data(external);writers[1].append_data(wrist)
                hashes.append(image_hash(external,wrist));states.append(state.copy())
                if method=='retreat5_fine20' and relative<plan['retreat_steps']:
                    action=np.asarray(plan['retreat_action'],dtype=np.float64);row['manual_steps']+=1
                else:
                    if not queue:
                        request={'observation/image':external,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language}
                        if row['policy_calls']==0:request.update(_policy_seed=metadata['policy_seed'],_policy_rng_advance=metadata['policy_rng_advance'])
                        # Save exactly the real inference inputs, before any simulator action.
                        external_inputs.append(external.copy());wrist_inputs.append(wrist.copy());state_inputs.append(state.copy())
                        tick=time.monotonic();chunk=np.asarray(client.infer(request)['actions']);latencies.append(time.monotonic()-tick)
                        assert chunk.shape==(10,7) and np.isfinite(chunk).all()
                        chunks.append(chunk.copy());calls.append(step);row['policy_calls']+=1
                        fine=(method=='fine20' and relative<plan['fine_steps']) or (method=='retreat5_fine20' and relative<plan['retreat_steps']+plan['fine_steps'])
                        queue.extend(chunk[:1 if fine else 5])
                    action=queue.popleft()
                assert action.shape==(7,) and np.isfinite(action).all()
                actions.append(action.copy());obs,_,done,_=scene.env.step(action.tolist());dones.append(bool(done))
                assert recorder.record(audit=(relative==0))==bool(done)
                row['steps']=relative+1
                if relative%20==0:progress('live_rollout',relative+1,plan['max_total_steps']-step0)
                if done:row.update(task_success=True,first_success_step=step+1);break
            external,wrist,state=scene.inputs(obs)
            writers[0].append_data(external);writers[1].append_data(wrist);hashes.append(image_hash(external,wrist));states.append(state.copy())
            row.update(rollout_status='TASK_SUCCESS' if row['task_success'] else 'STEP_LIMIT_REACHED',
                       inference_seconds=sum(latencies),suffix_wall_seconds=time.monotonic()-suffix_start)
        except BaseException as exc:
            row.update(status='PROGRAM_EXCEPTION',error_type=type(exc).__name__,error=str(exc));raise
        finally:
            for writer in writers:writer.close()
            np.savez_compressed(directory/'policy_trace.npz',action=np.asarray(actions).reshape(-1,7),robot_state=np.asarray(states).reshape(-1,8),
                                action_chunk=np.asarray(chunks).reshape(-1,10,7),call_step=np.asarray(calls,dtype=np.int64),
                                inference_seconds=np.asarray(latencies),done=np.asarray(dones,dtype=bool),image_sha256=np.asarray(hashes))
            np.savez_compressed(directory/'policy_inputs.npz',external=np.asarray(external_inputs,dtype=np.uint8).reshape(-1,224,224,3),
                                wrist=np.asarray(wrist_inputs,dtype=np.uint8).reshape(-1,224,224,3),robot_state=np.asarray(state_inputs).reshape(-1,8),
                                call_step=np.asarray(calls,dtype=np.int64))
            recorder.save(directory/'oracle.npz');write(directory/'result.json',row)
        try:
            tick=time.monotonic()
            model_audit=replay_model_inputs(client,directory,row,root,plan,lambda n,t:progress('model_replay',n,t))
            write(directory/'model_replay.json',model_audit);row['model_replay_seconds']=time.monotonic()-tick
            if method in ('continue_a','continue_b'):
                tick=time.monotonic();physics=scene.replay_physics(root,directory,row)
                write(directory/'physics_replay.json',physics);row['physics_replay_seconds']=time.monotonic()-tick
            else:row['physics_replay_seconds']=0.0
            verify_audit_bindings(directory,row)
            row.update(status=row['rollout_status'],verification_status='PASS',elapsed_seconds=time.monotonic()-begin,
                       model_calls_verified=model_audit['calls_checked'],physics_replay_verified=method in ('continue_a','continue_b'))
        except BaseException as exc:
            row.update(status='PROGRAM_EXCEPTION',verification_status='FAIL',error_type=type(exc).__name__,error=str(exc));raise
        finally:write(directory/'result.json',row)
        return row
    retained={(e['root_id'],e['method']):e for e in plan.get('resume_entries',[])}
    assert len(retained)==len(plan.get('resume_entries',[]))
    def finish_saved_rollout(root,method,directory,entry):
        row=dict(verify_entry(entry,root,plan));source=pathlib.Path(entry['output'])
        budget_guard();directory.mkdir(parents=True,exist_ok=False)
        # Link only immutable recorded data, never result/audit JSON that will be
        # newly committed. Failed-run evidence remains byte-for-byte unchanged.
        for name in entry['files_sha256']:
            if name in ('result.json','model_replay.json','physics_replay.json'):continue
            os.link(source/name,directory/name)
        row.update(output=str(directory),rollout_preserved_from=str(source),status='RUNNING',verification_status='PENDING')
        write(directory/'result.json',row)
        tick=time.monotonic()
        try:
            audit=replay_model_inputs(client,directory,row,root,plan,lambda n,t:progress('retained_model_replay',n,t))
            write(directory/'model_replay.json',audit);row['model_replay_seconds']=time.monotonic()-tick
            if method in ('continue_a','continue_b'):
                step=time.monotonic();physics=scene.replay_physics(root,directory,row)
                write(directory/'physics_replay.json',physics);row['physics_replay_seconds']=time.monotonic()-step
            else:row['physics_replay_seconds']=0.0
            verify_audit_bindings(directory,row)
            row.update(status=row['rollout_status'],verification_status='PASS',model_calls_verified=audit['calls_checked'],
                       physics_replay_verified=method in ('continue_a','continue_b'),resume_verification_seconds=time.monotonic()-tick,
                       elapsed_seconds=row['prefix_replay_seconds']+row['suffix_wall_seconds']+time.monotonic()-tick)
            return row
        except BaseException as exc:
            row.update(status='PROGRAM_EXCEPTION',verification_status='FAIL',error_type=type(exc).__name__,error=str(exc));raise
        finally:write(directory/'result.json',row)
    def execute(root,method,directory):
        entry=retained.get((root['root_id'],method))
        if entry is None:return branch(root,method,directory),'new'
        if entry['mode']=='verified':return verify_entry(entry,root,plan),'reused'
        return finish_saved_rollout(root,method,directory,entry),'audit_only'
    try:
        checks=[]
        for spec in plan['preflight_specs']:
            root=next(r for r in all_roots if r['root_id']==spec['root_id'])
            assert root['root_id'] in plan['root_observation_snapshots'],'Shared preflight roots require frozen common input bytes'
            status.update(active_root=root['root_id'],active_method=spec['method']);progress()
            checks.append(branch(root,spec['method'],out/'preflight'/root['root_id']/spec['method']))
        controls=[r for r in checks if r['method']=='continue_a']
        assert {r['task_success'] for r in controls}=={True,False},'Preflight must include working positive and negative controls'
        write(out/'preflight.json',{'status':'PASS','worker':a.worker,'checked_at':now(),'rows':checks})
        status['phase']='waiting_for_preflight_gate';progress()
        gate=out.parents[1]/'COLLECTION_GATE.json';deadline=time.monotonic()+900
        while not gate.is_file():
            if time.monotonic()>deadline:raise TimeoutError('Cross-worker preflight deadline')
            time.sleep(1)
        assert json.loads(gate.read_text())['status']=='PASS'
        status['phase']='collection';progress()
        for root in roots:
            root_rows=[]
            for method in plan['methods']:
                status.update(active_root=root['root_id'],active_method=method);progress()
                row,mode=execute(root,method,out/'roots'/root['root_id']/method)
                root_rows.append(row);status['branches'].append(row);status['completed_branches']+=1
                status[{'new':'new_rollouts_completed','reused':'reused_branches','audit_only':'audit_only_finished'}[mode]]+=1
                status['model_calls_verified']+=row['model_calls_verified']
                status['physics_controls_verified']+=int(row['physics_replay_verified']);progress()
                print(json.dumps({'worker':a.worker,'completed':status['completed_branches'],'expected':status['expected_branches'],
                                  'root':root['root_id'],'method':method,'success':row['task_success'],'verification':'PASS'}),flush=True)
            assert len({r['root_image_sha256'] for r in root_rows})==1
            status['completed_roots']+=1;progress()
        assert status['completed_branches']==len(roots)*5 and status['physics_controls_verified']==len(roots)*2
        status['status']='PASS'
    except BaseException as exc:
        status.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        scene.close();progress();write(out/'result.json',status)
if __name__=='__main__':main()
