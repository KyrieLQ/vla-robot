#!/usr/bin/env python3
"""Single-task render/closed-loop validation with explicit outcome and exception records."""
import argparse, collections, importlib.util, json, os, pathlib, time, traceback

ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--render-only',action='store_true');p.add_argument('--cpu-render',action='store_true')
    p.add_argument('--port',type=int,default=8000);p.add_argument('--episodes',type=int,default=2);p.add_argument('--task-id',type=int,default=0)
    p.add_argument('--output',required=True);a=p.parse_args()
    if a.episodes<1:p.error('episodes must be positive')
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if a.cpu_render:
        os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa';os.environ['CUDA_VISIBLE_DEVICES']=''
        mapping={'backend':'OSMesa CPU','gpu_rendering_verified':False}
    else:
        from egl_device import select_egl
        mapping=select_egl()
        # Simulator needs EGL, not CUDA compute. robosuite 1.4.1 otherwise compares
        # numeric EGL IDs against CUDA UUID strings. Explicit UUID-matched EGL ID remains set.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        mapping['simulator_cuda_visible_devices']=''
    import numpy as np
    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    helper_path=ROOT/'repos/openpi/examples/libero/main.py'
    spec=importlib.util.spec_from_file_location('upstream_libero_example',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    task=suite.get_task(a.task_id);states=suite.get_task_init_states(a.task_id)
    manifest=json.loads((ROOT/'reports/data_manifest.json').read_text())
    if a.task_id==manifest['task_id']:assert task.language==manifest['language']
    result={'status':'RUNNING','suite':'libero_spatial','task_id':a.task_id,'language':task.language,
            'render_only':a.render_only,'render_mapping':mapping,'action_horizon':10,'replan_steps':5,'episodes':[]}
    client=None if a.render_only else WebsocketClientPolicy('127.0.0.1',a.port)
    env=None
    try:
        bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        render_id=-1 if a.cpu_render else mapping['matched_egl_index']
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=render_id)
        from OpenGL import GL
        renderer=GL.glGetString(GL.GL_RENDERER)
        result['gl_renderer']=renderer.decode() if renderer else None
        # Check only our own open device FDs; never inspect other users' processes.
        if not a.cpu_render:
            selected_index=mapping['selected_gpu']['index'];opened=[]
            for fd in pathlib.Path('/proc/self/fd').iterdir():
                try:target=os.readlink(str(fd))
                except OSError:continue
                if target.startswith('/dev/nvidia') and target[len('/dev/nvidia'):].isdigit():opened.append(int(target[len('/dev/nvidia'):]))
            result['own_open_nvidia_device_nodes']=sorted(set(opened))
            if selected_index not in opened:raise RuntimeError('Renderer did not open the selected NVIDIA device')
            # NVIDIA driver enumeration may open all nodes; actual renderer choice is verified by the EGL PCI match.
        for episode in range(a.episodes):
            seed=7+episode;np.random.seed(seed);env.seed(seed);obs=env.reset();obs=env.set_init_state(states[episode])
            if not a.cpu_render:
                from egl_device import verify_current_egl
                result['actual_egl_render_uuid']=verify_current_egl(mapping['selected_gpu']['uuid'])
            row={'episode':episode,'seed':seed,'initial_state_index':episode,'steps':0,'status':'RUNNING'};result['episodes'].append(row)
            frames=[];queue=collections.deque();success=False
            for _ in range(10):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
            for step in range(12 if a.render_only else 220):
                img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                assert img.shape==(256,256,3) and wrist.shape==(256,256,3)
                assert img.std()>1 and wrist.std()>1, 'Rendered frames are blank or constant'
                img=image_tools.convert_to_uint8(image_tools.resize_with_pad(img,224,224))
                wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,224,224));frames.append(img)
                state=np.concatenate([obs['robot0_eef_pos'],helper._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']])
                assert state.shape==(8,) and np.isfinite(state).all()
                if a.render_only:action=np.asarray(helper.LIBERO_DUMMY_ACTION)
                else:
                    if not queue:
                        chunk=client.infer({'observation/image':img,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language})['actions']
                        assert chunk.shape==(10,7) and np.isfinite(chunk).all()
                        queue.extend(chunk[:5])
                    action=queue.popleft()
                assert action.shape==(7,) and np.isfinite(action).all()
                obs,reward,done,info=env.step(action.tolist());row['steps']=step+1
                if done:success=True;break
            row['status']='RENDER_SMOKE_COMPLETED' if a.render_only else ('TASK_SUCCESS' if success else 'STEP_LIMIT_REACHED')
            row['task_success']=success
            video=out/f'episode_{episode:03d}_{row["status"]}.mp4';imageio.mimwrite(str(video),frames,fps=10)
            imageio.imwrite(str(out/f'episode_{episode:03d}_first.png'),frames[0]);row['video']=str(video)
            print(json.dumps(row),flush=True)
        result['status']='PASS'
    except Exception as exc:
        result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if result['episodes'] and result['episodes'][-1]['status']=='RUNNING':result['episodes'][-1]['status']='PROGRAM_EXCEPTION'
        raise
    finally:
        if env is not None:env.close()
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
