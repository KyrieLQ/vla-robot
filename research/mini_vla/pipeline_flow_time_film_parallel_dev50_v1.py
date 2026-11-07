"""Multi-GPU complete-task evaluation, gate barrier, canonical merge and audit."""
import argparse,copy,json,os,signal,subprocess,time
from pathlib import Path
from .pipeline_v1 import ROOT,PY,write
from .policy_m8_v1 import sha

def partition_tasks(count):
 assert 2<=count<=4
 result=[list(range(i,10,count)) for i in range(count)]
 assert sorted(t for part in result for t in part)==list(range(10))
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();pp=Path(a.plan).resolve();p=json.loads(pp.read_text());run=Path(a.run).resolve();began=time.monotonic()
 for f,h in p['input_sha256'].items():assert sha(ROOT/f)==h,f
 assert json.loads((ROOT/p['training_acceptance']).read_text())['status']=='PASS'
 selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text());gpus=selection['selected'];partitions=partition_tasks(len(gpus));children=[];logs=[];workers=[]
 def progress(stage,**kw):write(run/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-began,**kw))
 def interrupted(signum,frame):raise InterruptedError('Parallel evaluation interrupted')
 signal.signal(signal.SIGTERM,interrupted)
 try:
  progress('parallel_online_gates',workers=len(gpus))
  for i,(gpu,tasks) in enumerate(zip(gpus,partitions)):
   directory=run/'workers'/str(i);directory.mkdir(parents=True,exist_ok=True);workers.append(directory)
   single=copy.deepcopy(selection);single['selected']=[dict(gpu,logical_index=0)];single['actual_count']=1;write(directory/'gpu_selection.json',single)
   shard=copy.deepcopy(p);shard['scientific_plan_sha256']=sha(pp);shard['execution_worker_id']=i;shard['evaluation']['task_ids']=tasks;shard['evaluation']['episodes_per_variant']=5*len(tasks);write(directory/'plan.json',shard)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],VLA_GPU_SELECTION=str(directory/'gpu_selection.json'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
   log=(directory/'worker.log').open('a');logs.append(log)
   children.append(subprocess.Popen([str(PY),'-u','-B','-m','mini_vla.parallel_flow_time_film_eval_worker_v1','--plan',str(directory/'plan.json'),'--run',str(directory),'--release',str(run/'release_evaluation.json')],env=env,stdout=log,stderr=subprocess.STDOUT))
  for _ in range(900):
   if any(child.poll() is not None for child in children):raise RuntimeError('Worker exited before gate release')
   if all((d/'ready.json').exists() for d in workers):break
   time.sleep(1)
  else:raise TimeoutError('Parallel worker gates timeout')
  gates=[];positives=[]
  for d in workers:
   gate=json.loads((d/'flow/online_contract.json').read_text());positive=json.loads((d/'simulation_preflight/result.json').read_text())
   assert gate['status']==positive['status']=='PASS' and gate['samples']==120 and gate['maximum_action_error']<=1e-6
   assert gate['checkpoint_sha256']==p['checkpoint_sha256'] and gate['all_features_bitwise_equal']
   metadata_gate=json.loads((d/'server_metadata_gate.json').read_text());assert metadata_gate['status']=='PASS' and metadata_gate['new_policy_calls']==0
   from .flow_time_film_metadata_v1 import validate_metadata
   validate_metadata(metadata_gate['metadata'],p)
   gates.append(gate);positives.append(positive)
  (run/'flow').mkdir(exist_ok=True);(run/'simulation_preflight').mkdir(exist_ok=True)
  merged_gate=dict(gates[0],parallel_workers=len(workers),all_workers_verified=True,worker_contracts=[str(d/'flow/online_contract.json') for d in workers],maximum_action_error=max(g['maximum_action_error'] for g in gates))
  write(run/'flow/online_contract.json',merged_gate);write(run/'simulation_preflight/result.json',dict(positives[0],all_workers_verified=True,parallel_workers=len(workers)))
  write(run/'parallel_gates.json',dict(status='PASS',workers=len(workers),task_partitions=partitions,online_contracts=gates,positive_replays=positives))
  estimate=2200/len(workers)*max(g['inference_seconds_p95'] for g in gates)*1.5+1200
  throughput=dict(status='PASS' if estimate<p['resources']['operational_timeout_seconds'] else 'OPERATIONAL_ESTIMATE_BLOCKED',
                  expected_max_policy_calls=2200,workers=len(workers),estimated_seconds=estimate,
                  basis='Slowest120-input p95 per worker,1.5 multiplier,plus1200s physical/IO allowance; no new rollout pilot',
                  limitation='Concurrent gate timings measured; full simulation speedup remains to be measured.')
  write(run/'throughput_admission.json',throughput)
  assert throughput['status']=='PASS',throughput
  # No policy sample can start before every worker passes the same contract.
  write(run/'release_evaluation.json',dict(status='RELEASED',scientific_plan_sha256=sha(pp)))
  rollout_started=time.monotonic();previous=None
  while any(child.poll() is None for child in children):
   if any(child.poll() not in (None,0) for child in children):raise RuntimeError('Parallel evaluator failed; retain committed episodes and journals')
   rows=[];signatures=[]
   for i,d in enumerate(workers):
    f=d/'flow/h5/progress.json'
    if f.exists():rows.append(dict(worker=i,**json.loads(f.read_text())))
    for f in (d/'flow/h5').glob('*/trajectory.jsonl'):signatures.append((str(f),f.stat().st_mtime_ns))
   signature=tuple(signatures)
   if signature!=previous:
    progress('parallel_spatial50',workers=rows,completed=sum(x['completed'] for x in rows),expected=50);previous=signature
   time.sleep(3)
  assert all(child.returncode==0 for child in children)
  canonical=run/'flow/h5';canonical.mkdir(exist_ok=True);episodes=[];seen=set();metadata=None;mappings=[]
  for d in workers:
   result=json.loads((d/'flow/h5/result.json').read_text());assert result['status']=='PASS';metadata=result['metadata'];mappings.append(result['mapping'])
   for e in result['episodes']:
    key=e['task_id'],e['initial_state_index'];assert key not in seen and e['plan_sha256']==sha(pp);seen.add(key);episodes.append(e)
    target=d/'flow/h5'/e['pair_id'];link=canonical/e['pair_id']
    if link.is_symlink():assert link.resolve()==target.resolve()
    else:link.symlink_to(target,target_is_directory=True)
  assert seen=={(t,j) for t in range(10) for j in [2,12,22,32,42]}
  episodes.sort(key=lambda e:(e['task_id'],e['initial_state_index']))
  write(canonical/'result.json',dict(status='PASS',variant='h5',expected=50,episodes=episodes,metadata=metadata,mapping=mappings,parallel_workers=len(workers),elapsed_seconds=time.monotonic()-rollout_started))
  def execute(name,cmd):
   progress(name)
   with (run/(name+'.log')).open('a') as log:
    code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
   if code:raise RuntimeError(name+' failed')
  execute('paired_analysis',[str(PY),'-u','-B','-m','mini_vla.analyze_flow_time_film_dev50_v1','--plan',str(pp),'--run',str(run)])
  execute('independent_review',[str(PY),'-u','-B',str(ROOT/'scripts/review_flow_time_film_dev50_v1.py'),'--plan',str(pp),'--run',str(run)])
  write(run/'pipeline.json',dict(status='COMPLETE_REVIEW_REQUIRED',new_episodes=50,new_training_updates=0,parallel_workers=len(workers),elapsed_seconds=time.monotonic()-began));progress('complete',completed=50,expected=50)
 finally:
  for child in children:
   if child.poll() is None:child.terminate()
  for child in children:
   if child.poll() is None:
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:child.kill();child.wait()
  for log in logs:log.close()
 return 0
if __name__=='__main__':raise SystemExit(main())
