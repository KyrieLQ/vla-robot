"""Paired continuation: unchanged canonical50 reset with durable prefix replay."""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import imageio

ROOT=Path(__file__).resolve().parents[1]

def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arm',choices=['flow'],required=True);p.add_argument('--port',type=int);p.add_argument('--variant',choices=['h1','h5','preflight'],required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());ev=plan['evaluation'];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert a.variant in ['h5','preflight'] and ev['episodes_per_variant']==5*len(ev['task_ids']) and set(ev['task_ids']).issubset(set(range(10))) and not ev['confirmation_access']
    scientific_hash=plan['scientific_plan_sha256']
    selected=plan['arms'][a.arm]
    from .paired_eval_journal_v1 import append as journal_append,load as journal_load
    from egl_device import select_egl,verify_current_egl
    mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    spec=importlib.util.spec_from_file_location('libero_helper',ROOT/'repos/openpi/examples/libero/main.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    languages={t['task_id']:t['language'] for t in json.loads((ROOT/'reports/spatial_full_manifest.json').read_text())['tasks']}
    def reset(task_id,index):
        task=suite.get_task(task_id);assert task.language==languages[task_id]
        states=suite.get_task_init_states(task_id);seed=1000+100*task_id+index
        np.random.seed(seed)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index'])
        for previous in ev['initial_state_indices']:
            if previous>=index:break
            env.seed(1000+100*task_id+previous);env.reset();env.set_init_state(states[previous])
            for _ in range(ev['wait_steps']):env.step(helper.LIBERO_DUMMY_ACTION)
        np.random.seed(seed);env.seed(seed);env.reset();obs=env.set_init_state(states[index])
        verify_current_egl(mapping['selected_gpu']['uuid'])
        for _ in range(ev['wait_steps']):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
        assert not env.check_success(),'Initial state already successful'
        return env,task,obs
    if a.variant=='preflight':
        source=ROOT/ev['preflight_positive'];row=json.loads((source/'episode.json').read_text())
        trace=np.load(source/'policy_trace.npz');oracle=np.load(source/'oracle.npz')
        assert row['task_success'] and row['canonical_reset_history']==[]
        env,task,obs=reset(row['task_id'],row['initial_state_index'])
        try:
            np.testing.assert_allclose(env.sim.data.qpos,oracle['qpos'][0],atol=1e-7,rtol=0)
            np.testing.assert_allclose(env.sim.data.qvel,oracle['qvel'][0],atol=1e-7,rtol=0)
            actual=[];max_error=0.
            for i,action in enumerate(trace['action']):
                obs,_,done,_=env.step(action.tolist());actual.append(bool(done));assert bool(done)==bool(env.check_success())
                max_error=max(max_error,float(np.max(np.abs(env.sim.data.qpos-oracle['qpos'][i+1]))))
                assert max_error<1e-7
            assert actual==trace['done'].tolist() and actual[-1] and len(actual)==row['steps']
            write(out/'result.json',{'status':'PASS','source':str(source),'source_trace_sha256':digest(source/'policy_trace.npz'),
                                    'steps':len(actual),'max_qpos_error':max_error,'success_verified':True,'mapping':mapping,
                                    'note':'Replay of existing successful trace; not a new policy evaluation sample'})
        finally:env.close()
        return
    client=WebsocketClientPolicy('127.0.0.1',a.port);metadata=client.get_server_metadata()
    from .flow_time_film_metadata_v1 import validate_metadata
    validate_metadata(metadata,plan,a.arm)
    rows=[];started=time.monotonic();env=None
    for task_id in ev['task_ids']:
        for index in ev['initial_state_indices']:
            pair='task%d_init%d'%(task_id,index);ep=out/pair
            if (ep/'episode.json').exists():
                record=json.loads((ep/'episode.json').read_text())
                assert record['checkpoint_sha256']==metadata['checkpoint_sha256'] and record['plan_sha256']==scientific_hash
                assert digest(ep/'trace.npz')==record['trace_sha256']
                assert digest(ep/'external.mp4')==record['video_sha256'];assert digest(ep/'trajectory.jsonl')==record['journal_sha256'];rows.append(record);continue
            ep.mkdir(exist_ok=True)
            journal=ep/'trajectory.jsonl';old_calls,old_steps=journal_load(journal)
            if (ep/'external.mp4').exists():
                (ep/'external.mp4').rename(ep/('external.partial.%d.mp4'%time.time_ns()))
            env,task,obs=reset(task_id,index)
            initial_qpos=env.sim.data.qpos.copy();initial_qvel=env.sim.data.qvel.copy()
            actions=[];robot=[];dones=[];latencies=[];server_ms=[];success=False;queue=deque();chunks=[];call_steps=[];normalized_chunks=[];observation_hashes=[];vlm_hashes=[];noise_seeds=[];noise_hashes=[]
            execute=1 if a.variant=='h1' else 5
            writer=imageio.get_writer(str(ep/'external.mp4'),fps=10,codec='libx264',ffmpeg_params=['-crf','20'])
            try:
                for step in range(ev['max_steps']):
                    external=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
                    wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                    assert external.shape==wrist.shape==(256,256,3) and external.dtype==wrist.dtype==np.uint8
                    state=np.concatenate([obs['robot0_eef_pos'],helper._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']])
                    assert state.shape==(8,) and np.isfinite(state).all()
                    if step==0:
                        imageio.imwrite(str(ep/'first.png'),external)
                        initial_image=hashlib.sha256(external.tobytes()+wrist.tobytes()).hexdigest()
                        baseline=json.loads((ROOT/plan['control_flow_evaluation']/pair/'episode.json').read_text());assert initial_image==baseline['initial_image_sha256']
                        start_record={'pair_id':pair,'task_id':task_id,'initial_state_index':index,'language':task.language,'checkpoint_sha256':metadata['checkpoint_sha256'],'plan_sha256':scientific_hash,'execution_plan_sha256':digest(Path(a.plan)),'initial_image_sha256':initial_image}
                        if (ep/'episode_start.json').exists():
                            assert json.loads((ep/'episode_start.json').read_text())==start_record
                        else:write(ep/'episode_start.json',start_record)
                    writer.append_data(external)
                    if not queue:
                        if step in old_calls:
                            result=old_calls[step];latencies.append(result['latency']);server_ms.append(result['server_ms'])
                        else:
                            tick=time.monotonic();result=client.infer({'observation/image':external,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language,'sampling/task_id':task_id,'sampling/initial_state_index':index,'sampling/call_index':len(chunks)})
                            latencies.append(time.monotonic()-tick);server_ms.append(result['server_timing']['infer_ms'])
                        chunk=np.asarray(result['actions'],dtype=np.float32);assert chunk.shape==(10,7) and np.isfinite(chunk).all() and np.max(np.abs(chunk))<=1.000001
                        expected_input=hashlib.sha256(external.tobytes()+wrist.tobytes()+np.asarray(state,dtype='<f4').tobytes()+task.language.encode()).hexdigest()
                        assert result['observation_sha256']==expected_input
                        seed=int.from_bytes(hashlib.sha256(('20261020:%d:%d:%d'%(task_id,index,len(chunks))).encode()).digest()[:8],'little')%(2**63-1)
                        assert result['noise_seed']==seed and len(result['noise_sha256'])==64
                        noise_seeds.append(seed);noise_hashes.append(result['noise_sha256'])
                        normalized=np.asarray(result['normalized_actions'],dtype=np.float32);assert normalized.shape==(10,7) and np.isfinite(normalized).all()
                        observation_hashes.append(expected_input);vlm_hashes.append(result['vlm_input_sha256']);normalized_chunks.append(normalized.copy())
                        if step not in old_calls:
                            journal_append(journal,dict(kind='call',step=step,actions=chunk.tolist(),normalized_actions=normalized.tolist(),observation_sha256=expected_input,vlm_input_sha256=result['vlm_input_sha256'],latency=latencies[-1],server_ms=server_ms[-1],noise_seed=seed,noise_sha256=result['noise_sha256']))
                        chunks.append(chunk.copy());call_steps.append(step);queue.extend(chunk[:execute])
                    action=queue.popleft();actions.append(action);robot.append(state)
                    obs,_,done,_=env.step(action.tolist());assert bool(done)==bool(env.check_success());dones.append(bool(done))
                    if step<len(old_steps):
                        prior=old_steps[step];np.testing.assert_array_equal(action,prior['action']);np.testing.assert_allclose(state,prior['robot_state'],atol=1e-7,rtol=0)
                        np.testing.assert_allclose(env.sim.data.qpos,prior['qpos'],atol=1e-7,rtol=0);np.testing.assert_allclose(env.sim.data.qvel,prior['qvel'],atol=1e-7,rtol=0);assert bool(done)==prior['done']
                    else:
                        journal_append(journal,dict(kind='step',step=step,action=action.tolist(),robot_state=state.tolist(),done=bool(done),qpos=env.sim.data.qpos.tolist(),qvel=env.sim.data.qvel.tolist()))
                    if done:success=True;break
                last=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);writer.append_data(last);imageio.imwrite(str(ep/'last.png'),last)
            finally:
                writer.close();env.close();env=None
            if (ep/'trace.npz').exists():(ep/'trace.npz').rename(ep/('trace.partial.%d.npz'%time.time_ns()))
            np.savez_compressed(ep/'trace.npz',actions=np.asarray(actions),robot_state=np.asarray(robot),done=np.asarray(dones),
                                latency=np.asarray(latencies),server_ms=np.asarray(server_ms),initial_qpos=initial_qpos,initial_qvel=initial_qvel,action_chunks=np.asarray(chunks),call_steps=np.asarray(call_steps),normalized_action_chunks=np.asarray(normalized_chunks),observation_sha256=np.asarray(observation_hashes),vlm_input_sha256=np.asarray(vlm_hashes),noise_seed=np.asarray(noise_seeds,dtype=np.int64),noise_sha256=np.asarray(noise_hashes))
            record={'pair_id':pair,'task_id':task_id,'initial_state_index':index,'language':task.language,'variant':a.variant,
                    'task_success':success,'status':'TASK_SUCCESS' if success else 'STEP_LIMIT_REACHED','steps':len(actions),'policy_calls':len(latencies),'action_horizon':10,'execute_steps':execute,
                    'checkpoint_sha256':metadata['checkpoint_sha256'],'plan_sha256':scientific_hash,'execution_plan_sha256':digest(Path(a.plan)),
                    'initial_image_sha256':initial_image,'arm':a.arm,'journal_sha256':digest(journal),'replayed_prefix_steps':len(old_steps),'trace_sha256':digest(ep/'trace.npz'),'video_sha256':digest(ep/'external.mp4')}
            write(ep/'episode.json',record);rows.append(record)
            elapsed=time.monotonic()-started
            progress={'stage':'evaluation','variant':a.variant,'completed':len(rows),'expected':ev['episodes_per_variant'],
                      'elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed/max(1,len(rows))*(ev['episodes_per_variant']-len(rows))}
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert len(rows)==ev['episodes_per_variant']
    write(out/'result.json',{'status':'PASS','variant':a.variant,'expected':ev['episodes_per_variant'],'episodes':rows,
                            'elapsed_seconds':time.monotonic()-started,'mapping':mapping,'metadata':metadata})

if __name__=='__main__':main()
