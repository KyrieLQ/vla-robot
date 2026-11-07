"""Frozen three-seed fit cohort; no model fitting, no outcome-dependent retries."""
import argparse,collections,datetime,json,os,pathlib,time,traceback
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--worker',type=int,required=True);a=p.parse_args();plan=json.loads(pathlib.Path(a.plan).read_text())
    from paired_live_audit_v1 import arrays,sha,image_hash,replay_model_inputs,verify_audit_bindings,check_saved_branch
    assert sha(ROOT/plan['seed_approval_file'])==plan['seed_approval_sha256'] and json.loads((ROOT/plan['seed_approval_file']).read_text())['approved']
    assert sha(ROOT/plan['root_manifest'])==plan['root_manifest_sha256'];roots={r['root_id']:r for r in json.loads((ROOT/plan['root_manifest']).read_text())['roots']};jobs=[j for j in plan['jobs'] if j['worker']==a.worker];assert len(roots)==60 and len(jobs)==65
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False);run=out.parents[1]
    from egl_device import select_egl
    mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from paired_live_scene_v1 import LiveScene
    from storage_guard_v3 import check,du_paths_bytes
    client=WebsocketClientPolicy('127.0.0.1',a.port);start=time.monotonic();state=dict(status='RUNNING',phase='existing_trace_positive_negative_preflight',worker=a.worker,expected_branches=65,completed_branches=0,model_calls_verified=0,physics_controls_verified=0,branches=[],render_mapping=mapping,started_at=now())
    def progress(operation=None,done=None,total=None):
        if operation:state.update(operation=operation,operation_done=done,operation_total=total)
        elapsed=time.monotonic()-start;state.update(heartbeat_at=now(),elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/state['completed_branches']*(65-state['completed_branches']) if state['completed_branches'] else None);write(out/'progress.json',state)
    scene=LiveScene(plan,mapping,progress)
    def guard():
        check(reserve_bytes=64*2**20);used=du_paths_bytes([ROOT/p for p in plan['recovery_artifact_roots']]+[run]);local=du_paths_bytes([run]);assert used+64*2**20<=4*2**30,'Recovery4GiB cap';assert local+32*2**20<=320*2**20,'Seed study320MiB cap including concurrent unflushed branch reserve'
    def branch(root,method,directory):
        assert method in ('continue_a','continue_b','resample5');guard();directory.mkdir(parents=True,exist_ok=False);begin=time.monotonic();task,obs,recorder,old,audit=scene.prepare(root,directory);step0=root['branch_step'];prefix_seconds=time.monotonic()-begin
        row=dict(root_id=root['root_id'],source_root_id=root['source_root_id'],pair_id=root['pair_id'],task_id=root['task_id'],branch_step=step0,replicate_index=root['replicate_index'],replicate_seed=root['replicate_seed'],method=method,status='RUNNING',verification_status='PENDING',language=task.language,policy_calls=0,steps=0,baseline_success=old['success_at_220'],task_success=False,first_success_step=None,root_image_sha256=audit['root_policy_image_sha256'],root_max_abs_errors=audit['physical_max_errors'],prefix_replay_seconds=prefix_seconds,prefix_replay_actions=step0,module_partition='fit',output=str(directory),manual_steps=0)
        metadata=dict(prompt=task.language,policy_seed=root['resample_seed'] if method=='resample5' else root['policy_seed'],policy_rng_advance=0 if method=='resample5' else step0//5,actual_inputs_only=True,oracle_sent_to_policy=False,replicate_index=root['replicate_index']);write(directory/'policy_inputs.json',metadata)
        queue=collections.deque();actions=[];states=[];hashes=[];chunks=[];calls=[];latencies=[];dones=[];external_inputs=[];wrist_inputs=[];state_inputs=[];suffix_start=time.monotonic()
        try:
            for step in range(step0,plan['max_total_steps']):
                external,wrist,state=scene.inputs(obs);hashes.append(image_hash(external,wrist));states.append(state.copy())
                if not queue:
                    request={'observation/image':external,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language}
                    if not row['policy_calls']:request.update(_policy_seed=metadata['policy_seed'],_policy_rng_advance=metadata['policy_rng_advance'])
                    external_inputs.append(external.copy());wrist_inputs.append(wrist.copy());state_inputs.append(state.copy());tick=time.monotonic();chunk=np.asarray(client.infer(request)['actions']);latencies.append(time.monotonic()-tick);assert chunk.shape==(10,7) and np.isfinite(chunk).all();chunks.append(chunk.copy());calls.append(step);row['policy_calls']+=1;queue.extend(chunk[:5])
                action=queue.popleft();assert action.shape==(7,) and np.isfinite(action).all();actions.append(action.copy());obs,_,done,_=scene.env.step(action.tolist());dones.append(bool(done));assert recorder.record(audit=(step==step0))==bool(done);row['steps']=step-step0+1
                if (step-step0)%20==0:progress('live_rollout',step-step0+1,220-step0)
                if done:row.update(task_success=True,first_success_step=step+1);break
            external,wrist,state=scene.inputs(obs);hashes.append(image_hash(external,wrist));states.append(state.copy());row.update(rollout_status='TASK_SUCCESS' if row['task_success'] else 'STEP_LIMIT_REACHED',inference_seconds=sum(latencies),suffix_wall_seconds=time.monotonic()-suffix_start)
        except BaseException as exc:row.update(status='PROGRAM_EXCEPTION',error_type=type(exc).__name__,error=str(exc));raise
        finally:
            np.savez_compressed(directory/'policy_trace.npz',action=np.asarray(actions).reshape(-1,7),robot_state=np.asarray(states).reshape(-1,8),action_chunk=np.asarray(chunks).reshape(-1,10,7),call_step=np.asarray(calls,dtype=np.int64),inference_seconds=np.asarray(latencies),done=np.asarray(dones,dtype=bool),image_sha256=np.asarray(hashes))
            np.savez_compressed(directory/'policy_inputs.npz',external=np.asarray(external_inputs,dtype=np.uint8).reshape(-1,224,224,3),wrist=np.asarray(wrist_inputs,dtype=np.uint8).reshape(-1,224,224,3),robot_state=np.asarray(state_inputs).reshape(-1,8),call_step=np.asarray(calls,dtype=np.int64));recorder.save(directory/'oracle.npz');write(directory/'result.json',row)
        try:
            tick=time.monotonic();audit=replay_model_inputs(client,directory,row,root,plan,lambda n,t:progress('model_replay',n,t));write(directory/'model_replay.json',audit);row['model_replay_seconds']=time.monotonic()-tick
            if method in ('continue_a','continue_b'):
                tick=time.monotonic();physics=scene.replay_physics(root,directory,row);write(directory/'physics_replay.json',physics);row['physics_replay_seconds']=time.monotonic()-tick
            else:row['physics_replay_seconds']=0.0
            verify_audit_bindings(directory,row);row.update(status=row['rollout_status'],verification_status='PASS',elapsed_seconds=time.monotonic()-begin,model_calls_verified=audit['calls_checked'],physics_replay_verified=method in ('continue_a','continue_b'));return row
        except BaseException as exc:row.update(status='PROGRAM_EXCEPTION',verification_status='FAIL',error_type=type(exc).__name__,error=str(exc));raise
        finally:write(directory/'result.json',row)
    try:
        controls=[]
        for spec in plan['preflight_existing_controls']:
            guard();root=spec['root'];directory=pathlib.Path(spec['directory']);row=json.loads((directory/'result.json').read_text())
            for name,h in spec['sha256'].items():assert sha(directory/name)==h
            assert root['module_partition']=='fit' and row['task_success']==spec['expected_success'];state.update(active_root=root['root_id'],active_method='existing_trace_audit');progress();model=replay_model_inputs(client,directory,row,root,plan,lambda n,t:progress('preflight_input_replay',n,t));physics=scene.replay_physics(root,directory,row);controls.append(dict(root_id=root['root_id'],task_success=row['task_success'],model_replay=model,physics_replay=physics))
        assert {c['task_success'] for c in controls}=={True,False};write(out/'preflight.json',dict(status='PASS',worker=a.worker,rows=controls,new_closed_loop_episodes=0));state['phase']='waiting_for_preflight_gate';progress();gate=run/'COLLECTION_GATE.json';deadline=time.monotonic()+300
        while not gate.is_file():
            if time.monotonic()>deadline:raise TimeoutError('Cross-worker preflight deadline')
            time.sleep(1)
        assert json.loads(gate.read_text())['status']=='PASS';state['phase']='collection'
        for job in jobs:
            root=roots[job['root_id']];assert root['worker']==a.worker and root['module_partition']=='fit';state.update(active_root=root['root_id'],active_method=job['method']);progress();row=branch(root,job['method'],out/'roots'/root['root_id']/job['method']);state['branches'].append(row);state['completed_branches']+=1;state['model_calls_verified']+=row['model_calls_verified'];state['physics_controls_verified']+=int(row['physics_replay_verified']);progress();print(json.dumps(dict(worker=a.worker,completed=state['completed_branches'],expected=65,root=root['root_id'],method=job['method'],verification='PASS')),flush=True)
        assert state['completed_branches']==65 and state['physics_controls_verified']==35;state['status']='PASS'
    except BaseException as exc:state.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:scene.close();progress();write(out/'result.json',state)
if __name__=='__main__':main()
