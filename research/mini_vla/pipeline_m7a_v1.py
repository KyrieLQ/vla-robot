"""One approved M7a visual-token control, same four-suite data and fixed50 evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from .pipeline_v1 import write,ROOT,PY,LIBERO

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True)
    p.add_argument('--mode',choices=['all','prepare','evaluate'],default='all');p.add_argument('--variant');a=p.parse_args()
    plan_path=Path(a.plan).resolve();plan=json.loads(plan_path.read_text());run=Path(a.run).resolve();start=time.monotonic()
    common=['--plan',str(plan_path),'--run',str(run)]
    def cmd(module,*args):return [str(PY),'-u','-B','-m',module]+list(args)
    def gate(command):return [str(PY),str(ROOT/'scripts/gpu_run.py'),'--gpu',','.join(map(str,plan['resources']['allowed_physical_gpus'])),'--gpus','1','--']+command
    last_completed_gpu_stage=False
    def launch(stage,command,timeout):
        nonlocal last_completed_gpu_stage
        is_gpu_stage=str(ROOT/'scripts/gpu_run.py') in command
        if is_gpu_stage and last_completed_gpu_stage:
            # Let nvidia-smi's recent utilization window age after our own job.
            # The original two-sample plus prelaunch idle gate is unchanged.
            time.sleep(5)
        for name,digest in json.loads((run/'source_manifest.json').read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Frozen source changed: '+name
        write(run/'pipeline.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'pipeline_started_monotonic':start,'stage_started_monotonic':time.monotonic()})
        with (run/(stage+'.log')).open('a') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
        if code:raise subprocess.CalledProcessError(code,command)
        last_completed_gpu_stage=is_gpu_stage
    if a.mode=='evaluate':
        sub=run/a.variant
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with (sub/'server.log').open('a') as log:
            server=subprocess.Popen(cmd('mini_vla.experiment_m7a_v1','serve','--plan',str(plan_path),'--run',str(sub),'--checkpoint',str(sub/'training/best.pt'),'--port',str(port)),stdout=log,stderr=subprocess.STDOUT)
            try:
                ready=False
                for _ in range(60):
                    if server.poll() is not None:raise RuntimeError('Visual policy server exited')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:%d/healthz'%port,timeout=2) as response:ready=response.status==200
                    except Exception:pass
                    if ready:break
                    time.sleep(1)
                if not ready:raise TimeoutError('Visual server startup timeout')
                command=[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_m2_v1','--plan',str(plan_path),'--output',str(sub/'h5'),'--variant','h5','--port',str(port)]
                return subprocess.run(command,timeout=1800).returncode
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:server.wait(timeout=20)
                    except subprocess.TimeoutExpired:server.kill();server.wait()
    from .train_m6_pipeline_v1 import data_sha
    for path,h in plan['input_sha256'].items():assert data_sha(ROOT/path)==h,path
    scope=json.loads((ROOT/plan['scope_approval']).read_text());assert scope['status']=='APPROVED' and scope['thread']==plan['thread']
    accepted=json.loads((ROOT/plan['data_acceptance']).read_text());assert accepted['status']=='PASS'
    for path,h in accepted['cache_all_files_sha256'].items():assert data_sha(ROOT/path)==h,path
    assert plan['evaluation']['episodes_per_variant']==50 and not plan['evaluation']['confirmation_access']
    try:
        launch('cache',cmd('mini_vla.experiment_m7a_v1','cache',*common),120)
        launch('weights',cmd('mini_vla.visual_preflight_v1','weights',*common),300)
        launch('throughput',gate(cmd('mini_vla.preflight_m7a_v1',*common)),180)
        if a.mode=='prepare':write(run/'pipeline.json',{'status':'PREPARED','stage':None});return 0
        launch('simulation_preflight',gate([str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_m2_v1','--plan',str(plan_path),'--output',str(run/'simulation_preflight'),'--variant','preflight']),300)
        for variant in plan['model_variants']:
            sub=run/variant;sub.mkdir(exist_ok=True)
            launch('train_'+variant,gate(cmd('mini_vla.experiment_m7a_v1','train','--plan',str(plan_path),'--run',str(sub),'--variant',variant)),10000)
            launch('eval_'+variant,gate(cmd('mini_vla.pipeline_m7a_v1',*common,'--mode','evaluate','--variant',variant)),1800)
        launch('analysis',cmd('mini_vla.analyze_m7a_v1',*common),300)
        write(run/'pipeline.json',{'status':'COMPLETE','stage':None});return 0
    except subprocess.CalledProcessError as exc:
        state='WAITING_FOR_IDLE_GPU' if exc.returncode==75 else ('BUDGET_INSUFFICIENT' if exc.returncode==78 else 'FAILED')
        write(run/'pipeline.json',{'status':state,'returncode':exc.returncode});return exc.returncode
if __name__=='__main__':raise SystemExit(main())
