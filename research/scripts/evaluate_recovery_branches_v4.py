#!/usr/bin/env python3
"""Paired continuations from verified native-state replay; no oracle policy input."""
import argparse,collections,datetime,hashlib,importlib.util,json,os,pathlib,time,traceback
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
class UnconsumedRenderMismatch(RuntimeError):
    pass
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,required=True);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);parser.add_argument('--worker',type=int,required=True);a=parser.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
    assert plan.get('unconsumed_suffix_image_amendment_approved') is True
    manifest_path=ROOT/plan['root_manifest'];assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==plan['root_manifest_sha256']
    manifest=json.loads(manifest_path.read_text());all_roots=manifest['roots'];roots=all_roots if plan['workers']==1 else [r for r in all_roots if r['worker']==a.worker]
    split=json.loads((ROOT/plan['development_split']).read_text());allowed={c['pair_id'] for c in split['development']};assert all(r['pair_id'] in allowed for r in all_roots)
    from egl_device import select_egl,verify_current_egl
    mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import imageio
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from research_telemetry_v1 import OracleRecorder
    spec=importlib.util.spec_from_file_location('upstream_libero',ROOT/'repos/openpi/examples/libero/main.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0);client=WebsocketClientPolicy('127.0.0.1',a.port)
    source=ROOT/plan['source_episodes'];cache={};env=None
    status={'status':'RUNNING','worker':a.worker,'expected_roots':len(roots),'expected_branches':len(roots)*4,'completed_roots':0,'completed_branches':0,'reused_branches':0,'new_branches':0,'active_root':None,'active_method':None,'phase':'preflight','started_at':now(),'render_mapping':mapping,'branches':[],'deferred_roots':[]}
    start=time.monotonic()
    def progress():
        status.update(heartbeat_at=now(),elapsed_seconds=time.monotonic()-start)
        n=status['new_branches'];status['estimated_remaining_seconds']=(status['elapsed_seconds']/n*(status['expected_branches']-status['completed_branches']) if n else None)
        write(out/'progress.json',status)
    def inputs(obs):
        im=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wr=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
        assert im.shape==wr.shape==(256,256,3) and im.std()>1 and wr.std()>1
        im=image_tools.convert_to_uint8(image_tools.resize_with_pad(im,224,224));wr=image_tools.convert_to_uint8(image_tools.resize_with_pad(wr,224,224))
        st=np.concatenate([obs['robot0_eef_pos'],helper._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]);assert st.shape==(8,) and np.isfinite(st).all()
        return im,wr,st
    def digest(im,wr):return hashlib.sha256(im.tobytes()+wr.tobytes()).hexdigest()
    def load(root):
        ep=source/root['pair_id']
        assert hashlib.sha256((ep/'policy_trace.npz').read_bytes()).hexdigest()==root['source_trace_sha256']
        assert hashlib.sha256((ep/'oracle.npz').read_bytes()).hexdigest()==root['source_oracle_sha256']
        with np.load(ep/'policy_trace.npz') as f:trace={k:f[k] for k in f.files}
        with np.load(ep/'oracle.npz') as f:oracle={k:f[k] for k in f.files}
        row=json.loads((ep/'episode.json').read_text());return trace,oracle,row
    def reset(root):
        nonlocal env
        if env is not None:env.close();env=None
        t=root['task_id'];idx=root['initial_state_index']
        if t not in cache:
            task=suite.get_task(t);states=suite.get_task_init_states(t)
            bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            cache[t]=(task,states,bddl)
        task,states,bddl=cache[t]
        np.random.seed(root['env_seed'])
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index'])
        for previous in plan['reset_protocol']['anchor_indices']:
            if previous>=idx:break
            env.seed(1000+100*t+previous);env.reset();env.set_init_state(states[previous])
            for _ in range(plan['wait_steps']):env.step(helper.LIBERO_DUMMY_ACTION)
        np.random.seed(root['env_seed']);env.seed(root['env_seed']);env.reset();obs=env.set_init_state(states[idx])
        status['actual_egl_uuid']=verify_current_egl(mapping['selected_gpu']['uuid'])
        for _ in range(plan['wait_steps']):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
        return task,obs
    def branch(root,method,directory,trace,old_oracle,old_row):
        directory.mkdir(parents=True,exist_ok=False)
        branch_start=time.monotonic();step0=root['branch_step'];assert step0%5==0
        task,obs=reset(root)
        prefix_render_differences=[]
        prefix_max_errors={field:0.0 for field in ('qpos','qvel','control')}
        for step in range(step0):
            im,wr,st=inputs(obs)
            # Engineering amendment explicitly approved by user on 2026-10-14: no inference
            # consumes these replay images. Validate physical state at every step;
            # retain render differences instead of changing any numeric tolerance.
            assert plan.get('prefix_replay_acceptance_amendment_approved') is True
            state_fields={'qpos':env.sim.data.qpos,'qvel':env.sim.data.qvel,'control':env.sim.data.ctrl}
            for field,value in state_fields.items():
                np.testing.assert_allclose(value,old_oracle[field][step],atol=plan['acceptance']['root_simulator_atol'],rtol=0,err_msg=root['root_id']+'/prefix/'+field)
                prefix_max_errors[field]=max(prefix_max_errors[field],float(np.max(np.abs(value-old_oracle[field][step]))))
            observed_hash=digest(im,wr)
            if observed_hash!=trace['image_sha256'][step]:
                prefix_render_differences.append({'step':step,'actual_hash':observed_hash,'reference_hash':str(trace['image_sha256'][step]),'policy_inference_consumed_image':False})
            np.testing.assert_allclose(st,trace['robot_state'][step],atol=1e-6,rtol=0)
            obs,_,done,_=env.step(trace['action'][step].tolist());assert not done
        im,wr,st=inputs(obs)
        write(directory/'prefix_replay_audit.json',{'root_id':root['root_id'],'steps_checked':step0,'max_physical_errors':prefix_max_errors,'image_hash_differences':prefix_render_differences,'images_consumed_by_policy':False,'root_actual_image_sha256':digest(im,wr),'root_expected_image_sha256':str(trace['image_sha256'][step0])})
        imageio.imwrite(str(directory/'root_external.png'),im);imageio.imwrite(str(directory/'root_wrist.png'),wr)
        assert digest(im,wr)==trace['image_sha256'][step0],(root['root_id'],'root image')
        recorder=OracleRecorder(env);assert not recorder.record(audit=True)
        root_errors={}
        for key in ('qpos','qvel','control','object_position','object_quaternion'):
            current=recorder.rows[0][key];expected=old_oracle[key][step0]
            np.testing.assert_allclose(current,expected,atol=plan['acceptance']['root_simulator_atol'],rtol=0,err_msg=root['root_id']+'/'+key)
            root_errors[key]=float(np.max(np.abs(current-expected))) if current.size else 0.0
        root_hash=digest(im,wr)
        write(directory/'oracle_metadata.json',recorder.meta)
        replay_seconds=time.monotonic()-branch_start
        suffix_render_differences=[]
        queue=collections.deque();actions=[];robot_states=[];chunks=[];call_steps=[];latencies=[];hashes=[];dones=[];manual_steps=0
        row={'root_id':root['root_id'],'pair_id':root['pair_id'],'task_id':root['task_id'],'branch_step':step0,'method':method,'status':'RUNNING','policy_calls':0,'steps':0,'baseline_success':old_row['success_at_220'],'task_success':False,'first_success_step':None,'root_image_sha256':root_hash,'root_max_abs_errors':root_errors,'prefix_replay_seconds':replay_seconds,'prefix_replay_actions':step0,'prefix_render_differences':prefix_render_differences,'suffix_render_differences':suffix_render_differences,'module_partition':root['module_partition'],'output':str(directory)}
        writers=[imageio.get_writer(str(directory/(c+'.mp4')),fps=10,codec='libx264',ffmpeg_params=['-crf','18']) for c in ('external','wrist')]
        suffix_start=time.monotonic();continuation_errors={'action':0.0,'robot_state':0.0,'chunk':0.0}
        try:
            for step in range(step0,plan['max_total_steps']):
                im,wr,st=inputs(obs);image_hash=digest(im,wr)
                if method=='continue':
                    for key in ('qpos','qvel','control','object_position','object_quaternion'):
                        np.testing.assert_allclose(recorder.rows[-1][key],old_oracle[key][step],atol=plan['acceptance']['root_simulator_atol'],rtol=0,err_msg=root['root_id']+'/continued_state/'+key)
                    if image_hash!=trace['image_sha256'][step]:
                        consumed=not queue
                        errors={}
                        for key in ('qpos','qvel','control','object_position','object_quaternion'):
                            value=recorder.rows[-1][key];reference=old_oracle[key][step]
                            np.testing.assert_allclose(value,reference,atol=plan['acceptance']['root_simulator_atol'],rtol=0,err_msg=root['root_id']+'/suffix/'+key)
                            errors[key]=float(np.max(np.abs(value-reference)))
                        evidence={'root_id':root['root_id'],'step':step,'policy_consumes_frame':consumed,'terminal_record_frame':False,'actual_hash':image_hash,'reference_hash':str(trace['image_sha256'][step]),'physical_max_errors':errors}
                        write(directory/'render_mismatch.json',evidence)
                        imageio.imwrite(str(directory/f'render_mismatch_{step}_external.png'),im);imageio.imwrite(str(directory/f'render_mismatch_{step}_wrist.png'),wr)
                        if consumed:raise AssertionError((root['root_id'],'policy input image',step))
                        if not plan.get('unconsumed_suffix_image_amendment_approved',False):raise UnconsumedRenderMismatch((root['root_id'],'unconsumed continue image',step))
                        suffix_render_differences.append(evidence)
                    np.testing.assert_allclose(st,trace['robot_state'][step],atol=1e-6,rtol=0)
                    continuation_errors['robot_state']=max(continuation_errors['robot_state'],float(np.max(np.abs(st-trace['robot_state'][step]))))
                writers[0].append_data(im);writers[1].append_data(wr);hashes.append(image_hash);robot_states.append(st)
                relative=step-step0
                if method=='retreat5_fine20' and relative<plan['retreat_steps']:
                    action=np.asarray(plan['retreat_action'],dtype=np.float64);manual_steps+=1
                else:
                    if not queue:
                        req={'observation/image':im,'observation/wrist_image':wr,'observation/state':st,'prompt':task.language}
                        if row['policy_calls']==0:
                            req['_policy_seed']=root['resample_seed'] if method=='resample5' else root['policy_seed']
                            req['_policy_rng_advance']=0 if method=='resample5' else step0//5
                        tick=time.monotonic();chunk=client.infer(req)['actions'];latencies.append(time.monotonic()-tick)
                        assert chunk.shape==(10,7) and np.isfinite(chunk).all()
                        if method=='continue':
                            expected=trace['action_chunk'][step//5]
                            np.testing.assert_allclose(chunk,expected,atol=1e-6,rtol=0)
                            continuation_errors['chunk']=max(continuation_errors['chunk'],float(np.max(np.abs(chunk-expected))))
                        chunks.append(chunk.copy());call_steps.append(step);row['policy_calls']+=1
                        fine=(method=='fine20' and relative<plan['fine_steps']) or (method=='retreat5_fine20' and relative<plan['retreat_steps']+plan['fine_steps'])
                        queue.extend(chunk[:1 if fine else 5])
                    action=queue.popleft()
                assert action.shape==(7,) and np.isfinite(action).all()
                if method=='continue':
                    np.testing.assert_allclose(action,trace['action'][step],atol=1e-6,rtol=0)
                    continuation_errors['action']=max(continuation_errors['action'],float(np.max(np.abs(action-trace['action'][step]))))
                actions.append(action.copy());obs,_,done,_=env.step(action.tolist());dones.append(bool(done))
                assert recorder.record(audit=(relative==0))==bool(done)
                row['steps']=step-step0+1
                if relative%20==0:status['active_action_step']=step+1;progress()
                if done:row['task_success']=True;row['first_success_step']=step+1;break
            im,wr,st=inputs(obs);writers[0].append_data(im);writers[1].append_data(wr);hashes.append(digest(im,wr));robot_states.append(st)
            if method=='continue':
                end=min(old_row['steps'],220)
                assert step0+row['steps']==end
                assert row['task_success']==old_row['success_at_220']
                if hashes[-1]!=trace['image_sha256'][end]:
                    errors={}
                    for key in ('qpos','qvel','control','object_position','object_quaternion'):
                        value=recorder.rows[-1][key];reference=old_oracle[key][end]
                        np.testing.assert_allclose(value,reference,atol=plan['acceptance']['root_simulator_atol'],rtol=0)
                        errors[key]=float(np.max(np.abs(value-reference)))
                    evidence={'root_id':root['root_id'],'step':end,'policy_consumes_frame':False,'terminal_record_frame':True,'actual_hash':hashes[-1],'reference_hash':str(trace['image_sha256'][end]),'physical_max_errors':errors}
                    write(directory/'render_mismatch.json',evidence)
                    imageio.imwrite(str(directory/f'render_mismatch_{end}_external.png'),im);imageio.imwrite(str(directory/f'render_mismatch_{end}_wrist.png'),wr)
                    if not plan.get('unconsumed_suffix_image_amendment_approved',False):raise UnconsumedRenderMismatch((root['root_id'],'terminal record image',end))
                    suffix_render_differences.append(evidence)
            row.update(status='TASK_SUCCESS' if row['task_success'] else 'STEP_LIMIT_REACHED',manual_steps=manual_steps,
                       inference_seconds=sum(latencies),suffix_wall_seconds=time.monotonic()-suffix_start,elapsed_seconds=time.monotonic()-branch_start,
                       continuation_verified=(method=='continue'),continuation_max_abs_errors=continuation_errors if method=='continue' else None)
        except BaseException as exc:
            row.update(status='PROGRAM_EXCEPTION',error_type=type(exc).__name__,error=str(exc));raise
        finally:
            for w in writers:w.close()
            np.savez_compressed(str(directory/'policy_trace.npz'),action=np.asarray(actions),robot_state=np.asarray(robot_states),action_chunk=np.asarray(chunks),call_step=np.asarray(call_steps),inference_seconds=np.asarray(latencies),done=np.asarray(dones),image_sha256=np.asarray(hashes))
            recorder.save(directory/'oracle.npz');write(directory/'result.json',row)
        return row
    resume={}
    resume_expected={(r['root_id'],r['method']):r for r in plan['resume_audit']}
    for prior_run in plan.get('resume_runs',[]):
        prior=ROOT/prior_run
        for progress_path in sorted((prior/'workers').glob('worker*/progress.json')):
            data=json.loads(progress_path.read_text())
            for row in data.get('branches',[]):
                if row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED'):
                    key=(row['root_id'],row['method']);expected=resume_expected[key];old_dir=pathlib.Path(row['output'])
                    assert hashlib.sha256((old_dir/'result.json').read_bytes()).hexdigest()==expected['source_result_sha256']
                    assert hashlib.sha256((old_dir/'policy_trace.npz').read_bytes()).hexdigest()==expected['trace_sha256']
                    stored=json.loads((old_dir/'result.json').read_text())
                    assert stored['status']==row['status'] and stored['task_success']==row['task_success']
                    assert key not in resume
                    resume[key]=dict(row,reused_from=str(prior))
    assert set(resume)==set(resume_expected)
    def execute(root,method,directory,trace,oracle,old):
        key=(root['root_id'],method)
        if '/preflight/' not in str(directory) and key in resume:return resume[key]
        return branch(root,method,directory,trace,oracle,old)
    try:
        checks=[]
        for spec in plan['preflight_roots']:
            root=next(r for r in all_roots if r['pair_id']==spec['pair_id'] and r['branch_step']==spec['branch_step'])
            trace,oracle,old=load(root)
            status.update(active_root=root['root_id'],active_method='continue');progress()
            checks.append(execute(root,'continue',out/'preflight'/root['root_id'],trace,oracle,old))
        assert {r['task_success'] for r in checks}=={True,False}
        write(out/'preflight.json',{'status':'PASS','worker':a.worker,'checked_at':now(),'rows':checks})
        status['phase']='waiting_for_preflight_gate';progress()
        gate=out.parents[1]/'COLLECTION_GATE.json';deadline=time.monotonic()+600
        while not gate.is_file():
            if time.monotonic()>deadline:raise TimeoutError('Cross-worker preflight gate not opened')
            time.sleep(1)
        assert json.loads(gate.read_text())['status']=='PASS'
        status['phase']='collection';progress()
        known_deferred={x['root_id']:x for x in plan.get('known_unconsumed_render_blocks',[])}
        for root in roots:
            if root['root_id'] in known_deferred and not plan.get('unconsumed_suffix_image_amendment_approved',False):
                status['deferred_roots'].append(dict(known_deferred[root['root_id']],new_attempt=False));progress();continue
            trace,oracle,old=load(root);root_rows=[];deferred=False
            for m in plan['methods']:
                method=m['name'];status.update(active_root=root['root_id'],active_method=method);progress()
                try:row=execute(root,method,out/'roots'/root['root_id']/method,trace,oracle,old)
                except UnconsumedRenderMismatch as exc:
                    record={'root_id':root['root_id'],'method':method,'reason':str(exc),'new_attempt':True,'exception_type':type(exc).__name__}
                    status['deferred_roots'].append(record);write(out/'deferred_roots.json',status['deferred_roots']);deferred=True;progress()
                    print(json.dumps({'worker':a.worker,'deferred':record}),flush=True);break
                root_rows.append(row);status['branches'].append(row);status['completed_branches']+=1
                status['reused_branches' if row.get('reused_from') else 'new_branches']+=1;progress()
                print(json.dumps({'worker':a.worker,'completed':status['completed_branches'],'expected':status['expected_branches'],'root':root['root_id'],'method':method,'success':row['task_success']}),flush=True)
            if deferred:continue
            assert len({r['root_image_sha256'] for r in root_rows})==1
            status['completed_roots']+=1;progress()
            if status['completed_roots']%10==0:
                from storage_guard import check
                check(reserve_bytes=256*2**20)
                import subprocess
                used=int(subprocess.check_output(['du','-s','-B1',str(out.parents[1])],text=True).split()[0])
                assert used+plan.get('prior_branch_artifact_bytes',0)+256*2**20<plan['new_artifact_budget_gib']*2**30
        assert status['completed_roots']+len(status['deferred_roots'])==len(roots)
        if not status['deferred_roots']:assert status['completed_branches']==len(roots)*4
        status['status']='INCOMPLETE_RENDER_REVIEW' if status['deferred_roots'] else 'PASS'
    except BaseException as exc:
        status.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        progress();write(out/'result.json',status)
if __name__=='__main__':main()
