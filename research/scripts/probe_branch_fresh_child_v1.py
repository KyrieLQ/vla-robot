"""Finite repeatability probe of one fixed action prefix, with numeric diagnostics."""
import hashlib,importlib.util,json,os,pathlib,time
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
from egl_device import select_egl
mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
spec=importlib.util.spec_from_file_location('helper',ROOT/'repos/openpi/examples/libero/main.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0);task=suite.get_task(0);states=suite.get_task_init_states(0)
path=ROOT/'results/research_diagnosis_20261014_v3/ckpt30k_prefix5/episodes/task0_init5'
with np.load(path/'policy_trace.npz') as x:trace={k:x[k] for k in x.files}
with np.load(path/'oracle.npz') as x:oracle={k:x[k] for k in x.files}
rows=[];first_images=None
for repeat in range(1):
    np.random.seed(1005)
    env=OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index'])
    try:
        env.seed(1002);env.reset();env.set_init_state(states[2])
        for _ in range(10):env.step(helper.LIBERO_DUMMY_ACTION)
        env.seed(1005);env.reset();obs=env.set_init_state(states[5])
        for _ in range(10):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
        errors=[];imgs=[]
        for step in range(21):
            im=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224))
            wr=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224))
            both=np.concatenate([im,wr],axis=1);imgs.append(both.copy())
            h=hashlib.sha256(im.tobytes()+wr.tobytes()).hexdigest()
            errors.append({'step':step,'image_matches':h==trace['image_sha256'][step],
                           'qpos_max':float(np.max(np.abs(env.sim.data.qpos-oracle['qpos'][step]))),
                           'qvel_max':float(np.max(np.abs(env.sim.data.qvel-oracle['qvel'][step]))),
                           'control_max':float(np.max(np.abs(env.sim.data.ctrl-oracle['control'][step]))),
                           'different_pixels_vs_first_repeat':None if first_images is None else int((both!=first_images[step]).sum())})
            if step<20:obs,_,_,_=env.step(trace['action'][step].tolist())
        if first_images is None:first_images=imgs
        rows.append({'repeat':repeat,'errors':errors})
        print(json.dumps({'repeat':repeat,'mismatch_steps':[r['step'] for r in errors if not r['image_matches']],'max_qpos_error':max(r['qpos_max'] for r in errors)}),flush=True)
    finally:env.close()
pathlib.Path(os.environ['BRANCH_PROBE_OUTPUT']).write_text(json.dumps(rows,indent=2)+'\n')
