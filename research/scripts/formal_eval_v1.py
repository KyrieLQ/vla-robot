#!/usr/bin/env python3
"""Own one local inference service and one finite spatial evaluator."""
import argparse, json, os, pathlib, signal, socket, subprocess, time, urllib.request
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['base','trained'],required=True);p.add_argument('--experiment',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[str(ROOT/'envs/openpi/bin/python'),'-u',str(ROOT/'scripts/serve_formal_v1.py'),'--port',str(port),'--variant',a.variant]
    checkpoint=None
    if a.variant=='trained':
        checkpoint=json.loads((ROOT/'results'/a.experiment/'checkpoint.json').read_text());assert checkpoint['updates']==30000
        command+=['--checkpoint',checkpoint['path']]
    info={'variant':a.variant,'checkpoint':checkpoint,'output':str(out),'listen_host':'127.0.0.1'}
    with (out/'policy.log').open('x') as log:
        # Inherit the stage process group so the outer supervisor can clean up on timeout.
        server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);info['server_pid']=server.pid
        (out/'service.json').write_text(json.dumps(info,indent=2)+'\n')
        try:
            ready=False
            for _ in range(150):
                if server.poll() is not None:raise RuntimeError('Policy server exited; inspect policy.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as response:ready=response.status==200
                except Exception:pass
                if ready:break
                time.sleep(2)
            if not ready:raise TimeoutError('Policy startup exceeded 300 seconds')
            result=subprocess.run([str(ROOT/'envs/libero/bin/python'),'-u',str(ROOT/'scripts/evaluate_spatial_v1.py'),'--port',str(port),'--variant',a.variant,'--output',str(out/'episodes')],timeout=7200)
            return result.returncode
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait()
            info.update(server_returncode=server.returncode,cleanup_completed=True);(out/'service.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
