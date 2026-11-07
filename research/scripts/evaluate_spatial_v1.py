#!/usr/bin/env python3
"""Fixed paired spatial subset; save every episode outcome, errors and video."""
import argparse, collections, importlib.util, json, os, pathlib, time, traceback
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--variant',required=True);a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'configs/formal_plan_20261013.json').read_text())['evaluation']
    from egl_device import select_egl,verify_current_egl
    mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import imageio
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    helper_spec=importlib.util.spec_from_file_location('upstream_libero',ROOT/'repos/openpi/examples/libero/main.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    manifest=json.loads((ROOT/'reports/spatial_full_manifest.json').read_text())
    client=WebsocketClientPolicy('127.0.0.1',a.port)
    result={'status':'RUNNING','variant':a.variant,'suite':'libero_spatial','expected_episodes':plan['episodes_per_policy'],'render_mapping':mapping,'episodes':[]}
    env=None;start=time.monotonic()
    try:
        for task_id in plan['task_ids']:
            task=suite.get_task(task_id);assert task.language==manifest['tasks'][task_id]['language']
            states=suite.get_task_init_states(task_id);assert max(plan['initial_state_indices'])<len(states)
            bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index'])
            for init_idx in plan['initial_state_indices']:
                env_seed=1000+100*task_id+init_idx;policy_seed=2000+100*task_id+init_idx
                np.random.seed(env_seed);env.seed(env_seed);obs=env.reset();obs=env.set_init_state(states[init_idx])
                actual=verify_current_egl(mapping['selected_gpu']['uuid'])
                from OpenGL import GL
                result['gl_renderer']=GL.glGetString(GL.GL_RENDERER).decode();result['actual_egl_render_uuid']=actual
                row={'pair_id':f'task{task_id}_init{init_idx}','task_id':task_id,'language':task.language,'initial_state_index':init_idx,'env_seed':env_seed,'policy_seed':policy_seed,'status':'RUNNING','steps':0,'policy_calls':0}
                result['episodes'].append(row);frames=[];queue=collections.deque();success=False;latencies=[]
                for _ in range(plan['wait_steps']):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
                for step in range(plan['max_steps']):
                    image=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                    assert image.shape==wrist.shape==(256,256,3) and image.std()>1 and wrist.std()>1
                    image=image_tools.convert_to_uint8(image_tools.resize_with_pad(image,224,224));wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224));frames.append(image)
                    state=np.concatenate([obs['robot0_eef_pos'],helper._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]);assert state.shape==(8,) and np.isfinite(state).all()
                    if not queue:
                        observation={'observation/image':image,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language}
                        if row['policy_calls']==0:observation['_policy_seed']=policy_seed
                        tick=time.monotonic();chunk=client.infer(observation)['actions'];latencies.append(time.monotonic()-tick)
                        assert chunk.shape==(10,7) and np.isfinite(chunk).all();queue.extend(chunk[:plan['replan_steps']]);row['policy_calls']+=1
                    action=queue.popleft();assert action.shape==(7,) and np.isfinite(action).all()
                    obs,reward,done,info=env.step(action.tolist());row['steps']=step+1
                    if step%20==0:result['elapsed_seconds']=time.monotonic()-start;write(out/'progress.json',result)
                    if done:success=True;break
                row.update(status='TASK_SUCCESS' if success else 'STEP_LIMIT_REACHED',task_success=success,inference_seconds_sum=sum(latencies),inference_calls=len(latencies))
                video=out/f'{row["pair_id"]}_{row["status"]}.mp4';imageio.mimwrite(str(video),frames,fps=10);row['video']=str(video)
                imageio.imwrite(str(out/f'{row["pair_id"]}_first.png'),frames[0])
                result['elapsed_seconds']=time.monotonic()-start;write(out/'progress.json',result)
                print(json.dumps({'completed':len(result['episodes']),'expected':plan['episodes_per_policy'],**row}),flush=True)
            env.close();env=None
        assert len(result['episodes'])==plan['episodes_per_policy'];result['status']='PASS'
    except Exception as exc:
        result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if result['episodes'] and result['episodes'][-1]['status']=='RUNNING':result['episodes'][-1]['status']='PROGRAM_EXCEPTION'
        raise
    finally:
        if env is not None:env.close()
        result['elapsed_seconds']=time.monotonic()-start;write(out/'result.json',result)
if __name__=='__main__':main()
