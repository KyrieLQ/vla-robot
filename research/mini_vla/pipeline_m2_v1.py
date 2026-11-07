"""Finite M2 action-chunk pipeline. All descendants stay in the supervisor-owned process group."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'envs/openpi/bin/python';LIBERO=ROOT/'envs/libero/bin/python'

def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True)
    p.add_argument('--mode',choices=['all','prepare','evaluate'],default='all');p.add_argument('--variant');a=p.parse_args()
    plan_path=Path(a.plan).resolve();plan=json.loads(plan_path.read_text());run=Path(a.run).resolve()
    common=['--plan',str(plan_path),'--run',str(run)]
    def command(module,*args):return [str(PY),'-u','-B','-m',module]+list(args)
    def launch(stage,cmd,timeout):
        frozen=json.loads((run/'source_manifest.json').read_text())
        for name,expected in frozen.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Frozen source changed: '+name
        write(run/'pipeline.json',{'status':'RUNNING','stage':stage,'started_monotonic':time.monotonic(),'pid':os.getpid()})
        with (run/(stage+'.log')).open('a') as log:
            child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,cmd)
    def gated(cmd):
        allowed=plan['resources']['allowed_physical_gpus']
        return [str(PY),str(ROOT/'scripts/gpu_run.py'),'--gpu',','.join(map(str,allowed)),'--gpus','1','--']+cmd
    if a.mode=='evaluate':
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        path=run/'training'/'best.pt'
        with (run/(a.variant+'_server.log')).open('a') as log:
            server=subprocess.Popen(command('mini_vla.experiment_m2_v1','serve',*common,'--checkpoint',str(path),'--port',str(port)),stdout=log,stderr=subprocess.STDOUT)
            try:
                ready=False
                for _ in range(60):
                    if server.poll() is not None:raise RuntimeError('Policy server failed')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:%d/healthz'%port,timeout=2) as response:ready=response.status==200
                    except Exception:pass
                    if ready:break
                    time.sleep(1)
                if not ready:raise TimeoutError('Policy startup timeout')
                cmd=[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_m2_v1','--plan',str(plan_path),'--output',str(run/a.variant),'--variant',a.variant,'--port',str(port)]
                return subprocess.run(cmd,timeout=1800).returncode
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:server.wait(timeout=20)
                    except subprocess.TimeoutExpired:server.kill();server.wait()
    try:
        launch('cache',command('mini_vla.experiment_m2_v1','cache',*common),plan['resources']['cache_seconds'])
        if a.mode=='prepare':
            write(run/'pipeline.json',{'status':'CACHE_READY','stage':None});return 0
        launch('simulation_preflight',gated([str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_m2_v1','--plan',str(plan_path),'--output',str(run/'simulation_preflight'),'--variant','preflight']),300)
        launch('train',gated(command('mini_vla.experiment_m2_v1','train',*common)),7200)
        for variant in plan['evaluation']['variants']:
            launch('eval_'+variant,gated(command('mini_vla.pipeline_m2_v1',*common,'--mode','evaluate','--variant',variant)),1800)
        launch('analysis',command('mini_vla.analyze_m2_v1',*common),300)
        write(run/'pipeline.json',{'status':'COMPLETE','stage':None});return 0
    except subprocess.CalledProcessError as exc:
        write(run/'pipeline.json',{'status':'WAITING_FOR_IDLE_GPU' if exc.returncode==75 else 'FAILED','returncode':exc.returncode});return exc.returncode
if __name__=='__main__':raise SystemExit(main())
