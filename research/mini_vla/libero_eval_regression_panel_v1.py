"""Frozen four-suite development panel. Complete-task shards; durable action queues."""
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import imageio
import numpy as np
from .regression_panel_contract_v1 import case_key, validate_metadata
from .paired_eval_journal_v1 import append as journal_append, load as journal_load

ROOT=Path(__file__).resolve().parents[1]
def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bits_hash(value):return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--port',type=int);parser.add_argument('--preflight',action='store_true');parser.add_argument('--assets-only',action='store_true')
    args=parser.parse_args();pp=Path(args.plan);plan=json.loads(pp.read_text());ev=plan['evaluation']
    panel=json.loads((ROOT/plan['panel']).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert not panel['final2000_access'] and panel['K']==10 and panel['H']==5
    from egl_device import select_egl, verify_current_egl
    mapping=None if args.assets_only else select_egl()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from libero.libero import benchmark, get_libero_path
    suites={name:benchmark.get_benchmark_dict()[name](task_order_index=0) for name in panel['suites']}
    assets=json.loads((ROOT/panel['source_manifest']).read_text())
    assert digest(ROOT/panel['source_manifest'])==panel['source_manifest_sha256']
    for path, expected in assets['asset_sha256'].items():assert digest(ROOT/path)==expected
    split=json.loads((ROOT/plan['split']).read_text());instructions={name:set() for name in suites}
    for entry in split['train']:instructions[entry['suite']].add(entry['instruction'])
    contracts=[]
    for suite_name,suite in suites.items():
        for task_id in range(10):
            task=suite.get_task(task_id);states=suite.get_task_init_states(task_id)
            cases=[c for c in panel['cases'] if c['suite']==suite_name and c['task_id']==task_id]
            assert len(cases)==5 and all(c['task_name']==task.name and c['partition']=='development' for c in cases)
            assert task.language in instructions[suite_name] and len(states)>=50
            for c in cases:
                initial=np.asarray(states[c['initial_state_index']]);assert initial.ndim==1 and np.isfinite(initial).all()
                contracts.append(dict(case=case_key(c),language=task.language,initial_asset_sha256=bits_hash(initial)))
    write(out/'asset_contract.json',dict(status='PASS',tasks=40,cases=200,assets=80,rows=contracts,new_policy_calls=0))
    if args.assets_only:return
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    spec=importlib.util.spec_from_file_location('libero_helper',ROOT/'repos/openpi/examples/libero/main.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    def reset(case):
        name,tid,index=case['suite'],case['task_id'],case['initial_state_index']
        suite=suites[name];task=suite.get_task(tid);states=suite.get_task_init_states(tid)
        seed=case['development_environment_seed'];np.random.seed(seed)
        env=OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(tid),camera_heights=256,camera_widths=256,
                               render_gpu_device_id=mapping['matched_egl_index'])
        for previous in case['development_anchor_history']:
            env.seed(1000+10000*panel['suites'].index(name)+100*tid+previous)
            env.reset();env.set_init_state(states[previous])
            for _ in range(panel['wait_steps']):env.step(helper.LIBERO_DUMMY_ACTION)
        np.random.seed(seed);env.seed(seed);env.reset();obs=env.set_init_state(states[index])
        verify_current_egl(mapping['selected_gpu']['uuid'])
        for _ in range(panel['wait_steps']):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
        assert not env.check_success(),'Initial state already successful'
        return env,task,obs
    if args.preflight:
        source=ROOT/ev['preflight_positive'];row=json.loads((source/'episode.json').read_text())
        trace=np.load(source/'policy_trace.npz');oracle=np.load(source/'oracle.npz')
        assert row['task_success'] and row['canonical_reset_history']==[]
        # Replay the previously accepted positive under its own exact empty-anchor reset.
        case=dict(suite='libero_spatial',task_id=row['task_id'],initial_state_index=row['initial_state_index'],
                  development_environment_seed=1000+100*row['task_id']+row['initial_state_index'],development_anchor_history=[])
        tick=time.monotonic();env,task,obs=reset(case);physical_start=time.monotonic();actual=[];error=0.
        writer=imageio.get_writer(str(out/'positive.mp4'),fps=10,codec='libx264',ffmpeg_params=['-crf','28','-threads','1'])
        try:
            np.testing.assert_allclose(env.sim.data.qpos,oracle['qpos'][0],atol=1e-7,rtol=0)
            np.testing.assert_allclose(env.sim.data.qvel,oracle['qvel'][0],atol=1e-7,rtol=0)
            for i,action in enumerate(trace['action']):
                writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-2,::-2]))
                obs,_,done,_=env.step(action.tolist());assert bool(done)==bool(env.check_success());actual.append(bool(done))
                error=max(error,float(np.max(np.abs(env.sim.data.qpos-oracle['qpos'][i+1]))))
                assert error<1e-7
            writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-2,::-2]))
            assert actual==trace['done'].tolist() and actual[-1] and len(actual)==73
        finally:writer.close();env.close()
        write(out/'result.json',dict(status='PASS',steps=len(actual),max_qpos_error=error,success_verified=True,
              source=str(source),source_trace_sha256=digest(source/'policy_trace.npz'),mapping=mapping,
              physical_seconds=time.monotonic()-physical_start,elapsed_seconds=time.monotonic()-tick,
              video_bytes=(out/'positive.mp4').stat().st_size,new_policy_calls=0,new_episodes=0))
        return
    client=WebsocketClientPolicy('127.0.0.1',args.port);metadata=client.get_server_metadata();validate_metadata(metadata,plan)
    scientific_hash=plan['scientific_plan_sha256'];execution_hash=digest(pp)
    selected_cases=[c for c in panel['cases'] if 10*panel['suites'].index(c['suite'])+c['task_id'] in ev['global_task_ids']]
    rows=[];started=time.monotonic()
    for case in selected_cases:
        pair=case_key(case)
        for arm in ev['arm_order']:
            selected=plan['arms'][arm];ep=out/arm/pair;ep.mkdir(parents=True,exist_ok=True)
            video=case['initial_state_index']==panel['initial_state_indices_per_task'][case['suite']+':'+str(case['task_id'])][0]
            if (ep/'episode.json').exists():
                complete=json.loads((ep/'episode.json').read_text())
                assert complete['checkpoint_sha256']==selected['checkpoint_sha256'] and complete['plan_sha256']==scientific_hash
                assert digest(ep/'trace.npz')==complete['trace_sha256'] and digest(ep/'trajectory.jsonl')==complete['journal_sha256']
                assert complete['video_selected']==video
                if video:assert digest(ep/'external.mp4')==complete['video_sha256']
                rows.append(complete);continue
            journal=ep/'trajectory.jsonl';old_calls,old_steps=journal_load(journal)
            if (ep/'external.mp4').exists():(ep/'external.mp4').rename(ep/('external.partial.%d.mp4'%time.time_ns()))
            env,task,obs=reset(case);initial_qpos=env.sim.data.qpos.copy();initial_qvel=env.sim.data.qvel.copy()
            queue=deque();actions=[];robot=[];dones=[];latencies=[];server_ms=[];chunks=[];normalized=[];call_steps=[];obs_hashes=[];vlm_hashes=[];hidden_hashes=[]
            writer=imageio.get_writer(str(ep/'external.mp4'),fps=10,codec='libx264',ffmpeg_params=['-crf','28','-threads','1']) if video else None
            max_steps=panel['max_action_steps'][case['suite']]
            try:
                for step in range(max_steps):
                    external=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                    state=np.concatenate([obs['robot0_eef_pos'],helper._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']])
                    assert external.shape==wrist.shape==(256,256,3) and external.dtype==wrist.dtype==np.uint8
                    assert state.shape==(8,) and np.isfinite(state).all()
                    if step==0:
                        identity=dict(initial_image_sha256=hashlib.sha256(external.tobytes()+wrist.tobytes()).hexdigest(),
                                      initial_qpos_sha256=bits_hash(initial_qpos),initial_qvel_sha256=bits_hash(initial_qvel),
                                      initial_state_sha256=bits_hash(state),seed=case['development_environment_seed'],history=case['development_anchor_history'])
                        if arm!='existing':
                            control=json.loads((out/'existing'/pair/'episode_start.json').read_text());assert identity==control['initial_identity']
                        start=dict(pair_id=pair,arm=arm,checkpoint_sha256=selected['checkpoint_sha256'],plan_sha256=scientific_hash,
                                   initial_identity=identity,language=task.language,case=case,video_selected=video)
                        if (ep/'episode_start.json').exists():assert json.loads((ep/'episode_start.json').read_text())==start
                        else:write(ep/'episode_start.json',start)
                        imageio.imwrite(str(ep/'first.png'),external[::2,::2])
                    if writer:writer.append_data(external[::2,::2])
                    if not queue:
                        if step in old_calls:result=old_calls[step];latency=result['latency'];ms=result['server_ms']
                        else:
                            tick=time.monotonic();result=client.infer({'policy/arm':arm,'observation/image':external,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language})
                            latency=time.monotonic()-tick;ms=result['server_timing']['infer_ms']
                        assert result['arm']==arm and result['checkpoint_sha256']==selected['checkpoint_sha256']
                        obs_hash=hashlib.sha256(external.tobytes()+wrist.tobytes()+np.asarray(state,dtype='<f4').tobytes()+task.language.encode()).hexdigest()
                        assert result['observation_sha256']==obs_hash
                        chunk=np.asarray(result['actions'],dtype=np.float32);norm=np.asarray(result['normalized_actions'],dtype=np.float32)
                        assert chunk.shape==norm.shape==(10,7) and np.isfinite(chunk).all() and np.isfinite(norm).all() and np.max(np.abs(chunk))<=1.000001
                        if step not in old_calls:
                            journal_append(journal,dict(kind='call',step=step,actions=chunk.tolist(),normalized_actions=norm.tolist(),
                               observation_sha256=obs_hash,vlm_input_sha256=result['vlm_input_sha256'],hidden_sha256=result['hidden_sha256'],
                               arm=arm,checkpoint_sha256=selected['checkpoint_sha256'],latency=latency,server_ms=ms))
                        chunks.append(chunk);normalized.append(norm);call_steps.append(step);latencies.append(latency);server_ms.append(ms)
                        obs_hashes.append(obs_hash);vlm_hashes.append(result['vlm_input_sha256']);hidden_hashes.append(result['hidden_sha256']);queue.extend(chunk[:5])
                    action=queue.popleft();actions.append(action);robot.append(state)
                    obs,_,done,_=env.step(action.tolist());assert bool(done)==bool(env.check_success());dones.append(bool(done))
                    if step<len(old_steps):
                        prior=old_steps[step];np.testing.assert_array_equal(action,prior['action'])
                        np.testing.assert_allclose(state,prior['robot_state'],atol=1e-7,rtol=0)
                        np.testing.assert_allclose(env.sim.data.qpos,prior['qpos'],atol=1e-7,rtol=0)
                        np.testing.assert_allclose(env.sim.data.qvel,prior['qvel'],atol=1e-7,rtol=0);assert bool(done)==prior['done']
                    else:journal_append(journal,dict(kind='step',step=step,action=action.tolist(),robot_state=state.tolist(),done=bool(done),qpos=env.sim.data.qpos.tolist(),qvel=env.sim.data.qvel.tolist()))
                    if done:break
                last=np.ascontiguousarray(obs['agentview_image'][::-2,::-2]);imageio.imwrite(str(ep/'last.png'),last)
                if writer:writer.append_data(last)
            finally:
                if writer:writer.close()
                env.close()
            assert len(old_steps)<=len(actions)
            if (ep/'trace.npz').exists():(ep/'trace.npz').rename(ep/('trace.partial.%d.npz'%time.time_ns()))
            np.savez_compressed(ep/'trace.npz',actions=np.asarray(actions),robot_state=np.asarray(robot),done=np.asarray(dones),
              latency=np.asarray(latencies),server_ms=np.asarray(server_ms),initial_qpos=initial_qpos,initial_qvel=initial_qvel,
              action_chunks=np.asarray(chunks),normalized_action_chunks=np.asarray(normalized),call_steps=np.asarray(call_steps),
              observation_sha256=np.asarray(obs_hashes),vlm_input_sha256=np.asarray(vlm_hashes),hidden_sha256=np.asarray(hidden_hashes))
            complete=dict(start,task_success=bool(dones[-1]),steps=len(actions),policy_calls=len(chunks),max_steps=max_steps,
              action_horizon=10,execute_steps=5,execution_plan_sha256=execution_hash,journal_sha256=digest(journal),
              trace_sha256=digest(ep/'trace.npz'),video_sha256=digest(ep/'external.mp4') if video else None,replayed_prefix_steps=len(old_steps))
            write(ep/'episode.json',complete);rows.append(complete)
            elapsed=time.monotonic()-started
            progress=dict(stage='four_suite_development',completed=len(rows),expected=3*len(selected_cases),
                          elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/len(rows)*(3*len(selected_cases)-len(rows)))
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert len(rows)==3*len(selected_cases)
    write(out/'result.json',dict(status='PASS',expected=len(rows),episodes=rows,metadata=metadata,mapping=mapping,elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
