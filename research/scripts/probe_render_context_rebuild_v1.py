"""Fixed-state graphics-only probe: no sampling of policy outcomes."""
import copy,hashlib,json,os,pathlib,pickle
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
from egl_device import select_egl,verify_current_egl
mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import mujoco
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from robosuite.environments.robot_env import IMAGE_CONVENTION_MAPPING
from robosuite import macros
from robosuite.utils.binding_utils import GLContext
from research_telemetry_v1 import OracleRecorder

source=ROOT/'results/research_diagnosis_20261014_v3/ckpt30k_prefix5/episodes/task7_init44'
with np.load(source/'policy_trace.npz') as f:trace={k:f[k] for k in f.files}
with np.load(source/'oracle.npz') as f:old={k:f[k] for k in f.files}
suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0);task=suite.get_task(7);states=suite.get_task_init_states(7)
conv=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
def render(env):
    arrays=[]
    for name in ('agentview','robot0_eye_in_hand'):
        raw=env.sim.render(camera_name=name,width=256,height=256,depth=False)[::conv]
        arrays.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(raw[::-1,::-1]),224,224)))
    return hashlib.sha256(arrays[0].tobytes()+arrays[1].tobytes()).hexdigest()
def reconstruct_context(env):
    ctx=env.sim._render_context_offscreen
    flags=ctx.scn.flags.copy()
    ctx.con.free();ctx.gl_ctx.free()
    ctx.gl_ctx=GLContext(max_width=640,max_height=480,device_id=ctx.device_id);ctx.gl_ctx.make_current()
    ctx.scn=mujoco.MjvScene(ctx.model._model,maxgeom=1000);ctx.scn.flags[:]=flags
    ctx._set_mujoco_context_and_buffers()
records=[]
for repeat in range(8):
    np.random.seed(1744)
    env=OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index'])
    try:
        for idx in [2,12,22,32,42]:
            env.seed(1700+idx);env.reset();env.set_init_state(states[idx])
            for _ in range(10):env.step([0,0,0,0,0,0,-1])
        env.seed(1744);env.reset();obs=env.set_init_state(states[44])
        for _ in range(10):obs,_,_,_=env.step([0,0,0,0,0,0,-1])
        for step in range(60):obs,_,_,_=env.step(trace['action'][step].tolist())
        rec=OracleRecorder(env);rec.record(audit=True)
        for key in ('qpos','qvel','control','object_position','object_quaternion'):
            np.testing.assert_allclose(rec.rows[0][key],old[key][60],atol=1e-7,rtol=0)
        before=pickle.dumps(env.sim.data._data);rng=np.random.get_state()
        images=[{'attempt':0,'matches':render(env)==trace['image_sha256'][60]}]
        for attempt in range(1,5):
            reconstruct_context(env)
            assert pickle.dumps(env.sim.data._data)==before,'Graphics rebuild changed native MjData'
            after_rng=np.random.get_state();assert np.array_equal(rng[1],after_rng[1]) and rng[2:]==after_rng[2:]
            images.append({'attempt':attempt,'matches':render(env)==trace['image_sha256'][60]})
            assert pickle.dumps(env.sim.data._data)==before,'Read-only rendering changed native MjData'
        row={'repeat':repeat,'images':images,'native_data_unchanged':True,'actual_egl_uuid':verify_current_egl(mapping['selected_gpu']['uuid'])};records.append(row);print(json.dumps(row),flush=True)
    finally:env.close()
(ROOT/'reports/render_context_rebuild_probe.json').write_text(json.dumps(records,indent=2)+'\n')
