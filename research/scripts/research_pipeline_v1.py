#!/usr/bin/env python3
"""Finite, frozen diagnosis queue. One GPU worker, no wait-for-GPU loop."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import traceback

ROOT = pathlib.Path(os.environ['PROJECT_ROOT'])


def now():
    return datetime.datetime.now().astimezone().isoformat()


def write(p, d):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    out = pathlib.Path(args.output)
    assert out.is_dir() and not (out / 'pipeline.json').exists()
    plan_path = pathlib.Path(args.plan)
    plan = json.loads(plan_path.read_text())
    assert os.environ['CODEX_THREAD_ID'] == plan['originating_thread']
    assert plan['confirmation_episodes_this_queue'] == 0 and plan['new_gradient_updates_this_queue'] == 0
    assert plan['initial_worker_count'] == 1
    source = out / 'frozen_source'
    source.mkdir(exist_ok=False)
    script_names = ['research_pipeline_v1.py', 'research_eval_job_v1.py', 'evaluate_research_v1.py',
                    'research_telemetry_v1.py', 'analyze_research_diagnosis_v1.py', 'serve_formal_v1.py',
                    'egl_device.py', 'gpu_run.py', 'research_gpu_admission_v1.py', 'storage_guard.py', 'project_env.sh', 'research_supervisor_v1.py']
    paths = [ROOT / 'scripts' / n for n in script_names] + [plan_path, ROOT / plan['split_manifest']]
    paths += [ROOT / p for p in ['repos/openpi/examples/libero/main.py',
              'repos/openpi/src/openpi/training/config.py', 'repos/openpi/src/openpi/policies/policy.py',
              'repos/openpi/src/openpi/policies/policy_config.py',
              'repos/openpi/third_party/libero/libero/libero/envs/env_wrapper.py',
              'repos/openpi/third_party/libero/libero/libero/envs/bddl_base_domain.py',
              'repos/openpi/third_party/libero/libero/libero/envs/problems/libero_tabletop_manipulation.py',
              'repos/openpi/third_party/libero/libero/libero/envs/predicates/base_predicates.py']]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(out / 'source_manifest.json', {'created_at': now(), 'sha256': hashes,
                                       'note': 'New scripts execute from this frozen copy; pinned repo imports are guarded before each stage.'})
    frozen_plan = source / plan_path.relative_to(ROOT)
    scripts = source / 'scripts'
    child_env = os.environ.copy()
    child_env['PYTHONPATH'] = str(scripts) + os.pathsep + child_env.get('PYTHONPATH', '')
    state = {'status': 'RUNNING', 'started_at': now(), 'pid': os.getpid(), 'active_job': None,
             'jobs': [], 'expected_policy_episodes': 1000, 'confirmation_used': 0,
             'selected_device_window_seconds': 0, 'plan': str(frozen_plan)}
    write(out / 'pipeline.json', state)
    start = time.monotonic()
    from storage_guard import check

    def guard():
        for rel, digest in hashes.items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, 'Source changed: ' + rel
        state['storage'] = check(reserve_bytes=512 * 2**20)
        used = int(subprocess.check_output(['du', '-s', '-B1', str(out)], text=True).split()[0])
        assert used + 512 * 2**20 <= plan['new_artifact_budget_gib'] * 2**30

    try:
        for job in plan['jobs']:
            guard()
            record = {'name': job['name'], 'status': 'RUNNING', 'started_at': now()}
            state['jobs'].append(record)
            state['active_job'] = job['name']
            write(out / 'pipeline.json', state)
            command = [str(ROOT / 'envs/openpi/bin/python'), '-u', str(scripts / 'research_gpu_admission_v1.py'),
                       '--gpu', ','.join(map(str, plan['allowed_physical_gpus'])), '--gpus', '1', '--',
                       str(ROOT / 'envs/openpi/bin/python'), '-u', str(scripts / 'research_eval_job_v1.py'),
                       '--plan', str(frozen_plan), '--job', job['name'], '--output', str(out / job['name'])]
            tick = time.monotonic()
            with (out / (job['name'] + '.log')).open('x') as log:
                child = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = child.pid
                write(out / 'pipeline.json', state)
                code = child.wait()
            record.update(returncode=code, elapsed_seconds=time.monotonic() - tick, finished_at=now(),
                          status='COMPLETE' if code == 0 else ('GPU_UNAVAILABLE' if code == 75 else 'FAILED'))
            state['selected_device_window_seconds'] += record['elapsed_seconds'] if (out / job['name']).exists() else 0
            write(out / 'pipeline.json', state)
            if code:
                state['status'] = record['status']
                return code
            if job['name'] == 'preflight':
                command = [str(ROOT / 'envs/openpi/bin/python'), str(scripts / 'analyze_research_diagnosis_v1.py'),
                           '--output', str(out), '--plan', str(frozen_plan), '--preflight-only']
                subprocess.run(command, env=child_env, check=True)
        guard()
        state['active_job'] = 'analysis'
        write(out / 'pipeline.json', state)
        subprocess.run([str(ROOT / 'envs/openpi/bin/python'), str(scripts / 'analyze_research_diagnosis_v1.py'),
                        '--output', str(out), '--plan', str(frozen_plan)], env=child_env, check=True)
        state.update(status='COMPLETE', automatic_statistics='COMPLETE', interpretation_review='pending')
        return 0
    except BaseException as exc:
        state.update(status='FAILED', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        state.update(finished_at=now(), elapsed_seconds=time.monotonic() - start, active_job=None)
        write(out / 'pipeline.json', state)


if __name__ == '__main__':
    raise SystemExit(main())
