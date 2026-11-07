"""Replay an already-reviewed development trace; time inference but never execute new actions."""
import argparse,hashlib,importlib.util,json,os,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(exist_ok=True)
 from egl_device import select_egl,verify_current_egl
 mapping=select_egl();os.environ['CUDA_VISIBLE_DEVICES']=''
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 from openpi_client.websocket_client_policy import WebsocketClientPolicy
 spec=importlib.util.spec_from_file_location('helper',R/'repos/openpi/examples/libero/main.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 source=R/plan['preflight_positive'];old=json.loads((source/'episode.json').read_text());tr=np.load(source/'policy_trace.npz');oracle=np.load(source/'oracle.npz');assert old['task_success'] and old['canonical_reset_history']==[]
 client=WebsocketClientPolicy('127.0.0.1',a.port);meta=client.get_server_metadata();assert meta['checkpoint_sha256']==plan['checkpoint_sha256'] and meta['epoch']==3
 suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0);task=suite.get_task(old['task_id']);initials=suite.get_task_init_states(old['task_id']);seed=1000+100*old['task_id']+old['initial_state_index'];np.random.seed(seed)
 tick=time.monotonic();env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,render_gpu_device_id=mapping['matched_egl_index']);creation=time.monotonic()-tick
 try:
  tick=time.monotonic();env.seed(seed);env.reset();obs=env.set_init_state(initials[old['initial_state_index']]);verify_current_egl(mapping['selected_gpu']['uuid'])
  for _ in range(10):obs,_,_,_=env.step(helper.LIBERO_DUMMY_ACTION)
  reset_wait=time.monotonic()-tick;np.testing.assert_allclose(env.sim.data.qpos,oracle['qpos'][0],atol=1e-7,rtol=0);np.testing.assert_allclose(env.sim.data.qvel,oracle['qvel'][0],atol=1e-7,rtol=0)
  steps=[];rpc=[];chunks=[];hashes=[];actual=[];max_error=0.;begin=time.monotonic()
  for i,action in enumerate(tr['action']):
   if i%5==0:
    image=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]);state=np.concatenate([obs['robot0_eef_pos'],helper._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']])
    tick=time.monotonic();pred=client.infer({'observation/image':image,'observation/wrist_image':wrist,'observation/state':state,'prompt':task.language});rpc.append(time.monotonic()-tick)
    chunk=np.asarray(pred['actions']);assert chunk.shape==(10,7) and np.isfinite(chunk).all() and np.max(np.abs(chunk))<=1.000001;chunks.append(chunk)
    hashes.append(hashlib.sha256(image.tobytes()+wrist.tobytes()+state.astype('<f4').tobytes()+task.language.encode()).hexdigest())
   tick=time.monotonic();obs,_,done,_=env.step(action.tolist());steps.append(time.monotonic()-tick);actual.append(bool(done));assert bool(done)==bool(env.check_success())
   max_error=max(max_error,float(np.max(np.abs(env.sim.data.qpos-oracle['qpos'][i+1]))));assert max_error<1e-7
  assert actual==tr['done'].tolist() and actual[-1]
  np.savez_compressed(out/'probe.npz',policy_chunks=np.asarray(chunks),input_hashes=np.asarray(hashes),step_seconds=np.asarray(steps),rpc_seconds=np.asarray(rpc))
  result=dict(status='PASS',physical_gpu=mapping['selected_gpu']['index'],source=str(source),steps=len(steps),calls=len(rpc),max_qpos_error=max_error,success_reproduced=True,creation_seconds=creation,reset_wait_seconds=reset_wait,step_mean_seconds=float(np.mean(steps)),step_p95_seconds=float(np.quantile(steps,.95)),rpc_median_seconds=float(np.median(rpc[1:])),rpc_p95_seconds=float(np.quantile(rpc[1:],.95)),loop_elapsed_seconds=time.monotonic()-begin,new_closedloop_samples=0,new_training_updates=0,confirmation_accessed=False,checkpoint_sha256=plan['checkpoint_sha256'],note='Only stored actions executed; newly inferred chunks discarded. Spatial throughput cannot guarantee other-suite runtime.')
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:env.close()
if __name__=='__main__':main()
