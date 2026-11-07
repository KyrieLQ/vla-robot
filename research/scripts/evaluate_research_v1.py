#!/usr/bin/env python3
"""Frozen development-only evaluations with privileged labels in separate files."""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import time
import traceback

ROOT = pathlib.Path(os.environ['PROJECT_ROOT'])


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--telemetry', choices=['full', 'off'], default='full')
    a = p.parse_args()
    plan = json.loads(pathlib.Path(a.plan).read_text())
    job = next(j for j in plan['jobs'] if j['name'] == a.job)
    split_path = ROOT / plan['split_manifest']
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == plan['split_sha256']
    split = json.loads(split_path.read_text())
    cases = split['development'] if job['cases'] == 'development' else plan['preflight_cases']
    dev_ids = {c['pair_id'] for c in split['development']}
    assert all(c['pair_id'] in dev_ids for c in cases)
    assert not dev_ids.intersection(c['pair_id'] for c in split['confirmation'])
    out = pathlib.Path(a.output)
    out.mkdir(exist_ok=False)
    from egl_device import select_egl, verify_current_egl
    mapping = select_egl()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from research_telemetry_v1 import OracleRecorder
    spec = importlib.util.spec_from_file_location('upstream_libero', ROOT / 'repos/openpi/examples/libero/main.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    suite = benchmark.get_benchmark_dict()['libero_spatial'](task_order_index=0)
    manifest = json.loads((ROOT / 'reports/spatial_full_manifest.json').read_text())
    client = WebsocketClientPolicy('127.0.0.1', a.port)
    result = {'status': 'RUNNING', 'job': job, 'split': 'development', 'telemetry': a.telemetry,
              'expected_episodes': len(cases), 'render_mapping': mapping, 'episodes': [],
              'primary_max_steps': 220, 'simulation_control_hz': 20, 'video_fps': 10,
              'no_confirmation_cases': True}
    env = None
    last_task = None
    start = time.monotonic()

    def progress():
        elapsed = time.monotonic() - start
        complete = sum(r['status'] in ('TASK_SUCCESS', 'STEP_LIMIT_REACHED') for r in result['episodes'])
        result.update(elapsed_seconds=elapsed, completed_episodes=complete,
                      heartbeat_at=datetime.datetime.now().astimezone().isoformat(),
                      estimated_remaining_seconds=(elapsed / complete * (len(cases) - complete) if complete else None))
        write(out / 'progress.json', result)

    try:
        for case in cases:
            task_id, init_idx = case['task_id'], case['initial_state_index']
            if env is not None:
                env.close()
                env = None
            if task_id != last_task:
                task = suite.get_task(task_id)
                assert task.language == manifest['tasks'][task_id]['language']
                states = suite.get_task_init_states(task_id)
                assert len(states) == 50
                bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
                last_task = task_id
            env_seed, policy_seed = case['env_seed'], case['policy_seed']
            # LIBERO accumulates articulation initializers across hard resets.
            # A canonical per-case reset history avoids dependence on job/subset order
            # while exactly retaining the historical five-anchor scene definitions.
            assert plan['reset_protocol']['mode'] == 'fresh_environment_canonical_anchor_history'
            np.random.seed(env_seed)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=mapping['matched_egl_index'])
            reset_history = [i for i in plan['reset_protocol']['anchor_indices'] if i < init_idx]
            for previous in reset_history:
                env.seed(1000 + 100*task_id + previous)
                env.reset()
                env.set_init_state(states[previous])
                for _ in range(plan['wait_steps']):
                    env.step(helper.LIBERO_DUMMY_ACTION)
            np.random.seed(env_seed)
            env.seed(env_seed)
            env.reset()
            obs = env.set_init_state(states[init_idx])
            result['actual_egl_render_uuid'] = verify_current_egl(mapping['selected_gpu']['uuid'])
            from OpenGL import GL
            result['gl_renderer'] = GL.glGetString(GL.GL_RENDERER).decode()
            ep = out / case['pair_id']
            ep.mkdir(exist_ok=False)
            row = dict(case, language=task.language, status='RUNNING', steps=0, policy_calls=0,
                       episode_dir=str(ep), success_at_220=False, first_success_step=None,
                       canonical_reset_history=reset_history,
                       property_initializer_count=len(env.env.object_property_initializers))
            result['episodes'].append(row)
            episode_start = time.monotonic()
            for _ in range(plan['wait_steps']):
                obs, _, _, _ = env.step(helper.LIBERO_DUMMY_ACTION)
            oracle = OracleRecorder(env) if a.telemetry == 'full' else None
            if oracle is not None:
                initial_done = oracle.record(audit=True)
                assert not initial_done, 'Initial scene already satisfies goal'
                write(ep / 'oracle_metadata.json', oracle.meta)
            queue = collections.deque()
            actions, robot_states, chunks, call_steps, latencies = [], [], [], [], []
            done_values, image_hashes = [], []
            initial_digest = None
            writers = [imageio.get_writer(str(ep / (name + '.mp4')), fps=10, codec='libx264', ffmpeg_params=['-crf', '18'])
                       for name in ('external', 'wrist')]

            def observation(obs):
                image = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
                assert image.shape == wrist.shape == (256, 256, 3) and image.std() > 1 and wrist.std() > 1
                image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
                state = np.concatenate([obs['robot0_eef_pos'], helper._quat2axisangle(obs['robot0_eef_quat'].copy()), obs['robot0_gripper_qpos']])
                assert state.shape == (8,) and np.isfinite(state).all()
                return image, wrist, state

            try:
                for step in range(job['max_steps']):
                    image, wrist, state = observation(obs)
                    if step == 0:
                        initial_digest = hashlib.sha256(image.tobytes()).hexdigest()
                        imageio.imwrite(str(ep / 'first.png'), image)
                    writers[0].append_data(image)
                    writers[1].append_data(wrist)
                    image_hashes.append(hashlib.sha256(image.tobytes() + wrist.tobytes()).hexdigest())
                    robot_states.append(state)
                    if not queue:
                        # The allowlist is deliberate. No oracle labels, task/index ID or future state is sent.
                        request = {'observation/image': image, 'observation/wrist_image': wrist,
                                   'observation/state': state, 'prompt': task.language}
                        if not row['policy_calls']:
                            request['_policy_seed'] = policy_seed
                        tick = time.monotonic()
                        chunk = client.infer(request)['actions']
                        latencies.append(time.monotonic() - tick)
                        assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
                        chunks.append(chunk.copy())
                        call_steps.append(step)
                        queue.extend(chunk[:job['replan_steps']])
                        row['policy_calls'] += 1
                    action = queue.popleft()
                    assert action.shape == (7,) and np.isfinite(action).all()
                    actions.append(action.copy())
                    obs, reward, done, info = env.step(action.tolist())
                    done_values.append(bool(done))
                    if oracle is not None:
                        assert oracle.record(audit=step in (0, 219)) == bool(done), 'Goal telemetry disagrees with original done'
                    row['steps'] = step + 1
                    if step % 20 == 0:
                        progress()
                    if done:
                        row['first_success_step'] = step + 1
                        break
                image, wrist, state = observation(obs)
                # Include final post-action observation; video and sensor state have T+1 entries.
                writers[0].append_data(image)
                writers[1].append_data(wrist)
                robot_states.append(state)
                image_hashes.append(hashlib.sha256(image.tobytes() + wrist.tobytes()).hexdigest())
                imageio.imwrite(str(ep / 'last.png'), image)
            finally:
                for writer in writers:
                    writer.close()
                if actions:
                    np.savez_compressed(str(ep / 'policy_trace.npz'), action=np.asarray(actions),
                                        robot_state=np.asarray(robot_states), action_chunk=np.asarray(chunks),
                                        call_step=np.asarray(call_steps), inference_seconds=np.asarray(latencies),
                                        done=np.asarray(done_values), image_sha256=np.asarray(image_hashes))
                if oracle is not None and oracle.rows:
                    oracle.save(ep / 'oracle.npz')
            first = row['first_success_step']
            row.update(status='TASK_SUCCESS' if first is not None else 'STEP_LIMIT_REACHED',
                       task_success=first is not None, success_at_220=first is not None and first <= 220,
                       initial_image_sha256=initial_digest, inference_seconds_sum=sum(latencies),
                       elapsed_seconds=time.monotonic() - episode_start, video_frames=row['steps'] + 1)
            write(ep / 'episode.json', row)
            progress()
            print(json.dumps({'completed': result['completed_episodes'], 'expected': len(cases), **row}), flush=True)
            # Local guards, not GPU idle polling or chat notifications.
            vfs = os.statvfs(ROOT)
            assert vfs.f_bavail * vfs.f_frsize >= 100 * 2**30, 'Free-space reserve breached'
            if result['completed_episodes'] % 20 == 0:
                from storage_guard import check
                result['storage'] = check(reserve_bytes=256 * 2**20)
                run_root = out.parent.parent
                import subprocess
                used = int(subprocess.check_output(['du', '-s', '-B1', str(run_root)], text=True).split()[0])
                assert used < plan['new_artifact_budget_gib'] * 2**30 - 256 * 2**20, 'Diagnostic artifact budget nearly exhausted'
        assert len(result['episodes']) == len(cases)
        result['status'] = 'PASS'
    except BaseException as exc:
        result.update(status='FAIL', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if result['episodes'] and result['episodes'][-1]['status'] == 'RUNNING':
            result['episodes'][-1]['status'] = 'PROGRAM_EXCEPTION'
        raise
    finally:
        if env is not None:
            env.close()
        progress()
        write(out / 'result.json', result)


if __name__ == '__main__':
    main()
