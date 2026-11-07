"""Bounded reset-only diagnosis of LIBERO initializer history; no policy calls."""
import hashlib, importlib.util, json, os, pathlib
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
from egl_device import select_egl, verify_current_egl
mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
task=suite.get_task(4);states=suite.get_task_init_states(4)
rows=[]
old=np.asarray(Image.open(ROOT/'results/formal_pipeline_20261013_v2/trained_eval/episodes/task4_init22_first.png'))
for indices in [[22],[2,12,22]]:
    env=OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index'])
    history=[]
    try:
        for idx in indices:
            env.seed(1400+idx);env.reset();obs=env.set_init_state(states[idx])
            for _ in range(10):obs,_,_,_=env.step([0,0,0,0,0,0,-1])
            img=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);img=image_tools.convert_to_uint8(image_tools.resize_with_pad(img,224,224))
            body=env.env.obj_body_id['wooden_cabinet_1']
            history.append({'init_idx':idx,'property_initializer_count':len(env.env.object_property_initializers),'cabinet_pos':env.sim.data.body_xpos[body].tolist(),'first_sha':hashlib.sha256(img.tobytes()).hexdigest()})
        rows.append({'reset_indices':indices,'resets':history,'final_matches_historical':bool(np.array_equal(old,img)),'actual_egl_uuid':verify_current_egl(mapping['selected_gpu']['uuid'])})
    finally:env.close()
path=ROOT/'reports/research_reset_history_probe.json';path.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
