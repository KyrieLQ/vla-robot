"""Concurrent two-device M6 interface/throughput check without opening final scenarios."""
import argparse,json,os,socket,subprocess,threading,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .pipeline_v1 import ROOT,PY,LIBERO,write
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();pp=Path(a.plan).resolve();plan=json.loads(pp.read_text());run=Path(a.run).resolve()
 selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text());assert len(selection['selected'])==2;barrier=threading.Barrier(2)
 def worker(i):
  sub=run/f'worker{i}';sub.mkdir();chosen=selection['selected'][i];local=dict(selection,selected=[dict(chosen,logical_index=0)]);write(sub/'gpu_selection.json',local)
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=chosen['uuid'];env['VLA_GPU_SELECTION']=str(sub/'gpu_selection.json')
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  with (sub/'server.log').open('x') as log:server=subprocess.Popen([str(PY),'-u','-B','-m','mini_vla.experiment_m4_v1','serve','--plan',str(pp),'--run',str(sub),'--checkpoint',str(ROOT/plan['checkpoint']),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
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
   with (sub/'probe.log').open('x') as log:subprocess.run([str(LIBERO),'-u','-B','-m','mini_vla.final_probe_client_v1','--plan',str(pp),'--output',str(sub),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150)
   return json.loads((sub/'result.json').read_text())
  finally:
   if server.poll() is None:
    server.terminate()
    try:server.wait(timeout=15)
    except subprocess.TimeoutExpired:server.kill();server.wait()
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(worker,[0,1]))
 for r in results:assert r['status']=='PASS'
 with np.load(run/'worker0/probe.npz') as a0,np.load(run/'worker1/probe.npz') as a1:
  same_inputs=bool(np.array_equal(a0['input_hashes'],a1['input_hashes']));error=float(np.max(np.abs(a0['policy_chunks']-a1['policy_chunks'])))
  if same_inputs:assert error<=1e-6
 slow_step=max(r['step_mean_seconds'] for r in results);slow_rpc=max(r['rpc_p95_seconds'] for r in results);reset=max(r['reset_wait_seconds'] for r in results);creation=max(r['creation_seconds'] for r in results)
 # Whole suites remain serial within a worker, preserving upstream reset order.
 projected=[1.5*(n*(slow_step+slow_rpc/5)+1000*reset+20*creation)+600 for n in (370000,290000)]
 ledger=json.loads((ROOT/plan['resources']['ledger']).read_text());used=sum(e['elapsed_seconds'] for e in ledger['attempts']+ledger.get('additional_entries',[]) if e.get('phase',e.get('stage_id'))=='modern_final2000' and 'elapsed_seconds' in e)
 result=dict(status='COMPLETE_REVIEW_REQUIRED',workers=results,cross_gpu_identical_inputs=same_inputs,cross_gpu_action_max_error=error,projected_worker_seconds=projected,projection='Worst permitted action counts,1.5x measured concurrentSpatial cost plus600s;not a guarantee for other suites',remaining_final_stage_before_this_attempt_seconds=10800-used,final2000_started=False,new_closedloop_samples=0,confirmation_accessed=False,requires_next_stage='Independent audit then frozen full executor and budget admission;never treat probe as final benchmark')
 write(run/'analysis.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
