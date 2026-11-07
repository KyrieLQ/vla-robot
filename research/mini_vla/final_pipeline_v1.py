"""Two complete-suite workers for the frozen final benchmark; supervised cumulative limits."""
import argparse,json,os,socket,subprocess,threading,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .pipeline_v1 import ROOT,PY,LIBERO,write
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();pp=Path(a.plan).resolve();plan=json.loads(pp.read_text());run=Path(a.run).resolve()
 assert plan['confirmation_released_for_frozen_final_only']
 selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text());assert len(selection['selected'])==2;barrier=threading.Barrier(2)
 def worker(i):
  sub=run/f'worker{i}';sub.mkdir();chosen=selection['selected'][i];local=dict(selection,selected=[dict(chosen,logical_index=0)]);write(sub/'gpu_selection.json',local)
  env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');env['CUDA_VISIBLE_DEVICES']=chosen['uuid'];env['VLA_GPU_SELECTION']=str(sub/'gpu_selection.json')
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  with (sub/'server.log').open('x') as log:server=subprocess.Popen([str(PY),'-u','-B','-m','mini_vla.final_policy_v1','--plan',str(pp),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
  try:
   for _ in range(60):
    if server.poll() is not None:raise RuntimeError('Selected baseline server exited')
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=1) as r:
      if r.status==200:break
    except Exception:pass
    time.sleep(1)
   else:raise TimeoutError('Policy startup')
   barrier.wait(timeout=70)
   completed=[]
   for suite in plan['worker_suites'][i]:
    with (sub/(suite+'.log')).open('a') as log:subprocess.run([str(LIBERO),'-u','-B','-m','mini_vla.final_suite_client_v1','--plan',str(pp),'--output',str(sub/suite),'--suite',suite,'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    completed.append(suite)
   return dict(status='COMPLETE',worker=i,suites=completed)
  finally:
   if server.poll() is None:
    server.terminate()
    try:server.wait(timeout=15)
    except subprocess.TimeoutExpired:server.kill();server.wait()
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,[0,1]))
 for r in results:assert r['status']=='COMPLETE'
 subprocess.run([str(PY),'-u','-B','-m','mini_vla.final_analyze_v1','--plan',str(pp),'--run',str(run)],check=True)
if __name__=='__main__':main()
