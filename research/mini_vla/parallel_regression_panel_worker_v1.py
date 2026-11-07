"""One admitted GPU, one adapted frontend, three heads, complete task shard."""
import argparse,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
from .pipeline_v1 import ROOT,PY,LIBERO,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);ap.add_argument('--release',required=True);a=ap.parse_args()
    pp=Path(a.plan).absolute();p=json.loads(pp.read_text());run=Path(a.run).absolute();release=Path(a.release);children=[];began=time.monotonic()
    assert len(json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text())['selected'])==1
    def interrupted(signum,frame):raise InterruptedError('Panel worker interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    def execute(name,cmd):
        with (run/(name+'.log')).open('a') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);children.append(child)
            if child.wait()!=0:raise RuntimeError(name+' failed; retain journals')
    try:
        execute('environment_import_gate',['/usr/bin/python3','-B',str(ROOT/'scripts/test_paired_eval_environment_v2.py')])
        execute('positive_replay',[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_regression_panel_v1','--plan',str(pp),'--output',str(run/'simulation_preflight'),'--preflight'])
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with (run/'server.log').open('a') as log:
            server=subprocess.Popen([str(PY),'-u','-B','-m','mini_vla.policy_regression_panel_v1','--plan',str(pp),'--port',str(port),'--contract-output',str(run/'online_contract.json')],stdout=log,stderr=subprocess.STDOUT);children.append(server)
        for _ in range(900):
            if server.poll() is not None:raise RuntimeError('Panel server/online gate failed')
            try:
                with urllib.request.urlopen('http://127.0.0.1:%d/healthz'%port,timeout=1) as response:
                    if response.status==200:break
            except Exception:pass
            time.sleep(1)
        else:raise TimeoutError('Panel server gate timeout')
        execute('metadata',[str(PY),'-u','-B','-m','mini_vla.policy_regression_panel_v1','--plan',str(pp),'--port',str(port),'--metadata-output',str(run/'metadata_gate.json')])
        write(run/'ready.json',dict(status='PASS',global_task_ids=p['evaluation']['global_task_ids'],port=port,elapsed_seconds=time.monotonic()-began))
        for _ in range(900):
            if release.exists():break
            if server.poll() is not None:raise RuntimeError('Panel server stopped before release')
            time.sleep(1)
        else:raise TimeoutError('Common release timeout')
        execute('evaluation',[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_regression_panel_v1','--plan',str(pp),'--output',str(run/'episodes'),'--port',str(port)])
        write(run/'status.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-began))
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':main()
