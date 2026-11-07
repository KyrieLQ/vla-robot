"""Single admission, two sequential policies, fixed50 each, no new optimization."""
import argparse,json,os,socket,subprocess,time,urllib.request
from pathlib import Path
from .pipeline_v1 import ROOT,PY,LIBERO,write
from .policy_m8_v1 import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();pp=Path(a.plan).resolve();p=json.loads(pp.read_text());run=Path(a.run).resolve();started=time.monotonic()
 assert os.environ.get('VLA_GPU_SELECTION') and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
 for f,h in p['input_sha256'].items():assert sha(ROOT/f)==h,f
 assert json.loads((ROOT/p['training_acceptance']).read_text())['status']=='PASS'
 def progress(stage,**extras):write(run/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-started,**extras))
 def execute(name,cmd):
  with (run/(name+'.log')).open('a') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   progress(name);last_actual=None
   while child.poll() is None:
    time.sleep(5)
    latest=[]
    for arm in ['flow']:
     f=run/arm/'h5/progress.json'
     if f.exists():latest.append(dict(arm=arm,**json.loads(f.read_text())))
    files=list(run.glob('*/h5/progress.json'))+list(run.glob('*/h5/*/trajectory.jsonl'))
    actual=max([f.stat().st_mtime_ns for f in files],default=0)
    if actual!=last_actual:
     progress(name,arms=latest);last_actual=actual
   if child.returncode:raise RuntimeError(name+' exited '+str(child.returncode))
 execute('environment_import_gate',['/usr/bin/python3','-B',str(ROOT/'scripts/test_paired_eval_environment_v2.py')])
 progress('positive_replay')
 execute('positive_replay',[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_flow_noise_control_v1','--plan',str(pp),'--output',str(run/'simulation_preflight'),'--variant','preflight','--arm','flow'])
 for arm in ['flow']:
  directory=run/arm;directory.mkdir(exist_ok=True)
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  server=None
  try:
   progress(arm+'_online_contract')
   with (directory/'server.log').open('a') as log:
    server=subprocess.Popen([str(PY),'-u','-B','-m','mini_vla.policy_flow_noise_control_v1','--plan',str(pp),'--port',str(port),'--contract-output',str(directory/'online_contract.json')],stdout=log,stderr=subprocess.STDOUT)
   for _ in range(360):
    if server.poll() is not None:raise RuntimeError(arm+' server/online check failed')
    try:
     with urllib.request.urlopen('http://127.0.0.1:%d/healthz'%port,timeout=1) as response:
      if response.status==200:break
    except Exception:pass
    time.sleep(1)
   else:raise TimeoutError('Online gate/server startup timeout')
   gate=json.loads((directory/'online_contract.json').read_text());assert gate['status']=='PASS'
   write(directory/'throughput_estimate.json',dict(max_policy_calls=2200,estimated_policy_seconds=gate['inference_seconds_p95']*2200*1.3,basis='fixed120 train input p95, not rollout-derived; time overrun explicitly authorized'))
   execute(arm+'_evaluation',[str(LIBERO),'-u','-B','-m','mini_vla.libero_eval_flow_noise_control_v1','--plan',str(pp),'--output',str(directory/'h5'),'--variant','h5','--arm',arm,'--port',str(port)])
  finally:
   if server is not None and server.poll() is None:
    server.terminate()
    try:server.wait(timeout=15)
    except subprocess.TimeoutExpired:server.kill();server.wait()
 execute('paired_analysis',[str(PY),'-u','-B','-m','mini_vla.analyze_flow_noise_control_v1','--plan',str(pp),'--run',str(run)])
 execute('noise_seam_analysis',[str(PY),'-u','-B','-m','mini_vla.analyze_flow_noise_seams_v1','--plan',str(pp),'--run',str(run)])
 execute('independent_review',[str(PY),'-u','-B',str(ROOT/'scripts/review_flow_noise_dev50_v1.py'),'--plan',str(pp),'--run',str(run)])
 progress('complete',new_episodes=50);write(run/'pipeline.json',dict(status='COMPLETE_REVIEW_REQUIRED',new_episodes=50,new_training_updates=0,elapsed_seconds=time.monotonic()-started))
 return 0
if __name__=='__main__':raise SystemExit(main())
