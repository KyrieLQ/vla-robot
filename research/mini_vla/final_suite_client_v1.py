"""One full suite in upstream order; no canonical-anchor results are reused."""
import argparse,hashlib,importlib.util,json,os,signal,time
from pathlib import Path
import imageio
import numpy as np
from .final_episode_v1 import run_episode,load_journal,validate_records
R=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--suite',required=True);p.add_argument('--port',required=True,type=int);a=p.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
    assert plan['confirmation_released_for_frozen_final_only'] and plan['selected_model']=='M6'
    science=json.loads((R/plan['scientific_plan']).read_text());limit=next(s['max_action_steps'] for s in science['suites'] if s['name']==a.suite)
    from egl_device import select_egl,verify_current_egl
    mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    spec=importlib.util.spec_from_file_location('helper',R/'repos/openpi/examples/libero/main.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    client=WebsocketClientPolicy('127.0.0.1',a.port);meta=client.get_server_metadata();assert meta['checkpoint_sha256']==plan['checkpoint_sha256'] and meta['epoch']==3
    cases=json.loads((R/science['case_manifest']).read_text())['cases'];case_map={(c['task_id'],c['initial_state_index']):c for c in cases if c['suite']==a.suite};assert len(case_map)==500
    # Match upstream: seed once per suite; each task environment is seeded once.
    np.random.seed(7);suite=benchmark.get_benchmark_dict()[a.suite](task_order_index=0);started=time.monotonic();rows=[]
    def terminate(_sig,_frame):raise InterruptedError('Supervised stage interruption; journal retained')
    signal.signal(signal.SIGTERM,terminate)
    for task_id in range(10):
        task=suite.get_task(task_id);initials=suite.get_task_init_states(task_id)
        assert task.name==case_map[(task_id,0)]['task_name']
        env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index']);env.seed(7)
        try:
            for index in range(50):
                directory=out/f'task{task_id}_init{index}';directory.mkdir(exist_ok=True)
                env.reset();obs=env.set_init_state(initials[index]);verify_current_egl(mapping['selected_gpu']['uuid'])
                for _ in range(10):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
                def observation(o):
                    image=np.ascontiguousarray(o['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(o['robot0_eye_in_hand_image'][::-1,::-1]);state=np.concatenate([o['robot0_eef_pos'],helper._quat2axisangle(o['robot0_eef_quat'].copy()),o['robot0_gripper_qpos']]);return image,wrist,state,task.language
                image,wrist,_,_=observation(obs)
                identity=dict(suite=a.suite,task_id=task_id,initial_state_index=index,partition=case_map[(task_id,index)]['partition'],language=task.language,checkpoint_sha256=plan['checkpoint_sha256'],scientific_plan_sha256=sha(R/plan['scientific_plan']),initial_qpos=env.sim.data.qpos.tolist(),initial_qvel=env.sim.data.qvel.tolist(),initial_image_sha256=hashlib.sha256(image.tobytes()+wrist.tobytes()).hexdigest())
                journal=directory/'trajectory.jsonl';chosen_video=index in science['videos']['indices_per_task'];writer=None
                if journal.exists():
                    # No blind re-query after interruption: reconstruct only complete recorded outcomes.
                    records=load_journal(journal);summary=validate_records(records,limit)
                    prior=json.loads((directory/'start.json').read_text());assert prior['checkpoint_sha256']==identity['checkpoint_sha256'] and prior['scientific_plan_sha256']==identity['scientific_plan_sha256']
                    for key in ('initial_qpos','initial_qvel'):np.testing.assert_allclose(identity[key],prior[key],atol=1e-7,rtol=0)
                    if chosen_video and not (directory/'episode.json').exists():writer=imageio.get_writer(str(directory/'external.mp4'),fps=10,codec='libx264',ffmpeg_params=['-crf','20'])
                    try:
                        for step in (r for r in records if r['kind']=='step'):
                            if writer is not None:writer.append_data(observation(obs)[0])
                            obs,_,done,_=env.step(step['action']);assert bool(done)==step['done']
                            for key in ('qpos','qvel'):np.testing.assert_allclose(getattr(env.sim.data,key),step[key],atol=1e-7,rtol=0)
                        if writer is not None:writer.append_data(observation(obs)[0])
                    finally:
                        if writer is not None:writer.close()
                    if (directory/'episode.json').exists():
                        row=json.loads((directory/'episode.json').read_text());assert row['trace_sha256']==sha(journal)
                        if chosen_video:assert row['video_sha256']==sha(directory/'external.mp4')
                        rows.append(row);continue
                else:
                    write(directory/'start.json',identity)
                    if chosen_video:writer=imageio.get_writer(str(directory/'external.mp4'),fps=10,codec='libx264',ffmpeg_params=['-crf','20'])
                    try:
                        with journal.open('x') as stream:summary=run_episode(env,obs,client.infer,observation,limit,stream,writer)
                    finally:
                        if writer is not None:writer.close()
                    assert validate_records(load_journal(journal),limit)==summary
                row=dict(identity,**summary,trace_sha256=sha(journal),video_sha256=sha(directory/'external.mp4') if chosen_video else None,action_horizon=10,execute_steps=5,environment_seed=7,reset_protocol='upstream_sequential_per_task')
                write(directory/'episode.json',row);rows.append(row);write(out/'progress.json',dict(status='RUNNING',suite=a.suite,completed=len(rows),expected=500,elapsed_seconds=time.monotonic()-started))
                print(a.suite,task_id,index,'completed',len(rows),flush=True)
        finally:env.close()
    assert len(rows)==500;write(out/'result.json',dict(status='COMPLETE_REVIEW_REQUIRED',episodes=rows,expected=500,elapsed_seconds=time.monotonic()-started,metadata=meta))
if __name__=='__main__':main()
