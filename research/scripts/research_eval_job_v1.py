#!/usr/bin/env python3
"""Single selected GPU, one owned policy service, finite evaluation client(s)."""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(os.environ['PROJECT_ROOT'])
SCRIPTS = pathlib.Path(__file__).resolve().parent


def write(p, d):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2) + '\n')
    tmp.replace(p)


def cleanup(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan = json.loads(pathlib.Path(a.plan).read_text())
    job = next(j for j in plan['jobs'] if j['name'] == a.job)
    out = pathlib.Path(a.output)
    out.mkdir(exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    checkpoint = ROOT / plan['checkpoint_root'] / str(job['checkpoint_step'])
    assert (checkpoint / 'params').is_dir() and (checkpoint / 'assets').is_dir()
    command = [str(ROOT / 'envs/openpi/bin/python'), '-u', str(SCRIPTS / 'serve_formal_v1.py'),
               '--port', str(port), '--variant', 'trained', '--checkpoint', str(checkpoint)]
    info = {'job': job, 'checkpoint': str(checkpoint), 'listen_host': '127.0.0.1',
            'gpu_selection': os.environ['VLA_GPU_SELECTION'], 'cleanup_completed': False}
    server, client = None, None
    with (out / 'policy.log').open('x') as log:
        try:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            info['server_pid'] = server.pid
            write(out / 'service.json', info)
            ready = False
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError('Policy service exited; inspect policy.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
                        ready = response.status == 200
                except Exception:
                    pass
                if ready:
                    break
                time.sleep(2)
            if not ready:
                raise TimeoutError('Policy startup exceeded 300 seconds')
            modes = ['off', 'full'] if a.job == 'preflight' else ['full']
            for mode in modes:
                episodes = out / ('episodes_' + mode if a.job == 'preflight' else 'episodes')
                client_command = [str(ROOT / 'envs/libero/bin/python'), '-u', str(SCRIPTS / 'evaluate_research_v1.py'),
                                  '--port', str(port), '--output', str(episodes), '--plan', a.plan,
                                  '--job', a.job, '--telemetry', mode]
                client = subprocess.Popen(client_command)
                info['client_pid'] = client.pid
                info['episodes'] = str(episodes)
                write(out / 'service.json', info)
                code = client.wait(timeout=plan['job_timeout_seconds'])
                if code:
                    return code
            return 0
        finally:
            cleanup(client)
            cleanup(server)
            info.update(server_returncode=server.returncode if server else None,
                        client_returncode=client.returncode if client else None, cleanup_completed=True)
            write(out / 'service.json', info)


if __name__ == '__main__':
    raise SystemExit(main())
