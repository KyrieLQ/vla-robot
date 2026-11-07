#!/usr/bin/env python3
"""Own and clean up a loopback policy service and its finite LIBERO client."""
import argparse, datetime, json, os, pathlib, signal, socket, subprocess, time, urllib.request

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--config',default='pi05_libero_lora_3090_smoke')
    p.add_argument('--episodes',type=int,default=2);p.add_argument('--task-id',type=int,default=0);a=p.parse_args()
    root=pathlib.Path(os.environ['PROJECT_ROOT']);ck=pathlib.Path(a.checkpoint).resolve()
    assert ck.is_relative_to(root/'checkpoints') and (ck/'params').is_dir(),'Use an actual project checkpoint'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');log=root/'logs'/('policy_'+tag+'.log')
    result=root/'results'/('closed_loop_'+tag)
    env=os.environ.copy();env['PYTHONPATH']=str(root/'repos/openpi/third_party/libero')
    cmd=[str(root/'envs/openpi/bin/python'),str(root/'repos/openpi/scripts/serve_policy.py'),'--host','127.0.0.1','--port',str(port),
         'policy:checkpoint','--policy.config',a.config,'--policy.dir',str(ck)]
    with log.open('x') as f:
        server=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        info={'pid':server.pid,'port':port,'listen_host':'127.0.0.1','checkpoint':str(ck),'config':a.config,'result_directory':str(result)}
        print('CLOSED_LOOP_ARTIFACTS '+json.dumps({'policy_record':str(root/'logs'/('policy_'+tag+'.json')),'result_directory':str(result)}),flush=True)
        (root/'logs'/('policy_'+tag+'.json')).write_text(json.dumps(info,indent=2)+'\n')
        try:
            ready=False
            for _ in range(120):
                if server.poll() is not None:raise RuntimeError('Policy server exited; inspect '+str(log))
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as r:ready=r.status==200
                except Exception:pass
                if ready:break
                time.sleep(2)
            if not ready:raise TimeoutError('Policy server not ready within bounded startup window')
            client=subprocess.run([str(root/'envs/libero/bin/python'),str(root/'scripts/libero_validate.py'),
                  '--port',str(port),'--episodes',str(a.episodes),'--task-id',str(a.task_id),'--output',str(result)],env=env,timeout=1800)
            return client.returncode
        finally:
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
            info['server_returncode']=server.returncode;info['cleanup_completed']=True
            (root/'logs'/('policy_'+tag+'.json')).write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':raise SystemExit(main())
