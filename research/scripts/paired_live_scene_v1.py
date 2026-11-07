"""Canonical LIBERO history reconstruction and independent action replay."""
import hashlib
import importlib.util
import json
import os
import pathlib
import numpy as np
import imageio
from paired_live_audit_v1 import arrays, check_physical, checked_close, image_hash, sha, NUMERIC_FIELDS
from research_telemetry_v1 import OracleRecorder
from root_observation_snapshot_v1 import read_verified_snapshot, restore_observation

ROOT = pathlib.Path(os.environ['PROJECT_ROOT'])


def write(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n'); temp.replace(path)


class LiveScene:
    def __init__(self, plan, mapping, heartbeat):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        self.OffScreenRenderEnv=OffScreenRenderEnv; self.get_libero_path=get_libero_path
        self.image_tools=image_tools
        spec=importlib.util.spec_from_file_location('upstream_libero', ROOT/'repos/openpi/examples/libero/main.py')
        self.helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.helper)
        self.suite=benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
        self.plan=plan;self.mapping=mapping;self.heartbeat=heartbeat
        self.cache={};self.references={};self.root_images={};self.env=None

    def close(self):
        if self.env is not None:self.env.close();self.env=None

    def robot_state(self, obs):
        state=np.concatenate([obs['robot0_eef_pos'],self.helper._quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']])
        assert state.shape==(8,) and np.isfinite(state).all()
        return state

    def inputs(self, obs):
        if '_root_snapshot_external' in obs:
            external=obs['_root_snapshot_external'];wrist=obs['_root_snapshot_wrist']
        else:
            external=np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
            wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
            assert external.shape==wrist.shape==(256,256,3) and external.std()>1 and wrist.std()>1
            external=self.image_tools.convert_to_uint8(self.image_tools.resize_with_pad(external,224,224))
            wrist=self.image_tools.convert_to_uint8(self.image_tools.resize_with_pad(wrist,224,224))
        return external,wrist,self.robot_state(obs)

    def reference(self, root):
        source=ROOT/self.plan['source_episodes']/root['pair_id']
        assert sha(source/'policy_trace.npz')==root['source_trace_sha256']
        assert sha(source/'oracle.npz')==root['source_oracle_sha256']
        if root['pair_id'] not in self.references:
            self.references[root['pair_id']]=(arrays(source/'policy_trace.npz'),arrays(source/'oracle.npz'),json.loads((source/'episode.json').read_text()))
        return self.references[root['pair_id']]

    def reset_prefix(self, root):
        from egl_device import verify_current_egl
        self.close();task_id=root['task_id'];index=root['initial_state_index']
        if task_id not in self.cache:
            task=self.suite.get_task(task_id);states=self.suite.get_task_init_states(task_id)
            bddl=pathlib.Path(self.get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
            self.cache[task_id]=(task,states,bddl)
        task,states,bddl=self.cache[task_id]
        np.random.seed(root['env_seed'])
        self.env=self.OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,
                                       render_gpu_device_id=self.mapping['matched_egl_index'])
        env=self.env
        for previous in self.plan['reset_protocol']['anchor_indices']:
            if previous>=index:break
            env.seed(1000+100*task_id+previous);env.reset();env.set_init_state(states[previous])
            for _ in range(self.plan['wait_steps']):env.step(self.helper.LIBERO_DUMMY_ACTION)
        np.random.seed(root['env_seed']);env.seed(root['env_seed']);env.reset();obs=env.set_init_state(states[index])
        verify_current_egl(self.mapping['selected_gpu']['uuid'])
        for _ in range(self.plan['wait_steps']):obs,_,_,_=env.step(self.helper.LIBERO_DUMMY_ACTION)
        trace,oracle,_=self.reference(root)
        errors={key:0.0 for key in ('qpos','qvel','control','robot_state')}
        for step in range(root['branch_step']):
            for field,value in [('qpos',env.sim.data.qpos),('qvel',env.sim.data.qvel),('control',env.sim.data.ctrl)]:
                errors[field]=max(errors[field],checked_close(value,oracle[field][step],1e-7))
            errors['robot_state']=max(errors['robot_state'],checked_close(self.robot_state(obs),trace['robot_state'][step],1e-6))
            obs,_,done,_=env.step(trace['action'][step].tolist());assert not done
            if step%20==0:self.heartbeat('prefix_reconstruction',step+1,root['branch_step'])
        return task,obs,errors

    def prepare(self, root, directory):
        task,obs,prefix_errors=self.reset_prefix(root)
        trace,oracle,old_row=self.reference(root);step=root['branch_step']
        recorder=OracleRecorder(self.env);assert not recorder.record(audit=True)
        errors=check_physical(recorder.rows[0],{k:oracle[k][step] for k in NUMERIC_FIELDS})
        external,wrist,state=self.inputs(obs);checked_close(state,trace['robot_state'][step],1e-6)
        raw_hash=image_hash(external,wrist);rid=root['root_id']
        if rid in self.plan['root_observation_snapshots']:
            spec=self.plan['root_observation_snapshots'][rid]
            images=read_verified_snapshot(spec,spec['policy_image_sha256'])
            source=spec['source_directory']
        elif rid in self.root_images:
            images=self.root_images[rid];source='first pre-action measurement in this worker'
        else:
            # Approval covers one new measurement, not repeated attempts until a
            # historical pixel hash matches. Each root belongs to only one worker.
            images=(external.copy(),wrist.copy());source='single new reference acquired before any candidate outcome'
        self.root_images[rid]=(images[0].copy(),images[1].copy())
        obs=restore_observation(obs,state,images,trace['robot_state'][step])
        restored=self.inputs(obs);digest=image_hash(restored[0],restored[1])
        assert digest==image_hash(*images)
        audit={'root_id':rid,'source':source,'root_policy_image_sha256':digest,'historical_root_image_sha256':str(trace['image_sha256'][step]),
               'matches_historical_root':digest==trace['image_sha256'][step],'raw_rerender_sha256':raw_hash,
               'physical_max_errors':errors,'prefix_max_errors':prefix_errors,'single_fixed_reference':True,
               'future_observations_restored':False,'selected_by_outcome':False}
        write(directory/'root_reconstruction.json',audit)
        imageio.imwrite(str(directory/'root_external.png'),restored[0]);imageio.imwrite(str(directory/'root_wrist.png'),restored[1])
        if raw_hash!=digest:
            imageio.imwrite(str(directory/'root_rerender_external.png'),external);imageio.imwrite(str(directory/'root_rerender_wrist.png'),wrist)
        write(directory/'oracle_metadata.json',recorder.meta)
        return task,obs,recorder,old_row,audit

    def replay_physics(self, root, directory, row):
        trace=arrays(directory/'policy_trace.npz');reference=arrays(directory/'oracle.npz')
        _,obs,prefix_errors=self.reset_prefix(root)
        recorder=OracleRecorder(self.env);assert not recorder.record(audit=True)
        max_errors={key:0.0 for key in NUMERIC_FIELDS};state_error=0.0
        for i in range(row['steps']+1):
            errs=check_physical(recorder.rows[-1],{key:reference[key][i] for key in NUMERIC_FIELDS})
            for key in errs:max_errors[key]=max(max_errors[key],errs[key])
            state_error=max(state_error,checked_close(self.robot_state(obs),trace['robot_state'][i],1e-6))
            assert np.array_equal(recorder.rows[-1]['goal_predicate'],reference['goal_predicate'][i])
            if i==row['steps']:break
            obs,_,done,_=self.env.step(trace['action'][i].tolist())
            assert recorder.record(audit=(i==0))==bool(done)==bool(trace['done'][i])
            if i%20==0:self.heartbeat('physics_replay',i+1,row['steps'])
        return {'status':'PASS','states_checked':row['steps']+1,'physical_max_errors':max_errors,
                'robot_state_max_error':state_error,'prefix_max_errors':prefix_errors,'numeric_atol':1e-7,
                'goal_flags_exact':True,'actions_loaded_from_disk':True,'trace_sha256':sha(directory/'policy_trace.npz'),
                'oracle_sha256':sha(directory/'oracle.npz')}
