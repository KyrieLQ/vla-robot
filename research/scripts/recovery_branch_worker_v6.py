#!/usr/bin/env python3
"""Own exactly one inference server and its simulator process."""
import argparse,json,os,pathlib,signal,socket,subprocess,time,urllib.request
ROOT=pathlib.Path(os.environ['PROJECT_ROOT']);SCRIPTS=pathlib.Path(__file__).resolve().parent
def write(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def stop(p):
    if p is not None and p.poll() is None:
        p.terminate()
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:p.kill();p.wait()
def interrupt(signum,frame):raise KeyboardInterrupt('Owned worker interrupted')
def main():
    signal.signal(signal.SIGTERM,interrupt)
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--worker',type=int,required=True);a=parser.parse_args()
    run=pathlib.Path(a.run);plan=json.loads(pathlib.Path(a.plan).read_text());service=run/'services'/f'worker{a.worker}';service.mkdir(parents=True,exist_ok=False)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    info={'worker':a.worker,'gpu_selection':os.environ['VLA_GPU_SELECTION'],'cleanup_completed':False};server=client=None
    try:
        with (service/'policy.log').open('x') as log:
            server=subprocess.Popen([str(ROOT/'envs/openpi/bin/python'),'-u',str(SCRIPTS/'serve_recovery_branches_v1.py'),'--port',str(port),'--checkpoint',str(ROOT/plan['checkpoint'])],stdout=log,stderr=subprocess.STDOUT)
            info['server_pid']=server.pid;write(service/'service.json',info)
            ready=False
            for _ in range(150):
                if server.poll() is not None:raise RuntimeError('Policy service exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as r:ready=r.status==200
                except Exception:pass
                if ready:break
                time.sleep(2)
            if not ready:raise TimeoutError('Policy startup exceeded 300 seconds')
            client=subprocess.Popen([str(ROOT/'envs/libero/bin/python'),'-u',str(SCRIPTS/'evaluate_recovery_branches_v6.py'),'--port',str(port),'--plan',a.plan,'--worker',str(a.worker),'--output',str(run/'workers'/f'worker{a.worker}')])
            info['client_pid']=client.pid;write(service/'service.json',info)
            return client.wait(timeout=plan['job_timeout_seconds'])
    finally:
        stop(client);stop(server);info.update(cleanup_completed=True,server_returncode=server.returncode if server else None,client_returncode=client.returncode if client else None);write(service/'service.json',info)
if __name__=='__main__':raise SystemExit(main())
