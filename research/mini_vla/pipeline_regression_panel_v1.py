"""All worker contracts pass before any of the frozen600 development episodes."""
import argparse,copy,json,os,signal,subprocess,time
from pathlib import Path
from .pipeline_v1 import ROOT,PY,write
from .policy_m8_v1 import sha
from .regression_panel_contract_v1 import partition_tasks,case_key,validate_metadata

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    pp=Path(a.plan).absolute();p=json.loads(pp.read_text());run=Path(a.run).absolute();began=time.monotonic()
    panel=json.loads((ROOT/p['panel']).read_text());selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text())
    gpus=selection['selected'];partitions=partition_tasks(len(gpus));children=[];logs=[];workers=[]
    def progress(stage,**kw):write(run/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-began,**kw))
    def interrupted(signum,frame):raise InterruptedError('Panel parent interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        progress('all_worker_gates',workers=len(gpus))
        for i,(gpu,tasks) in enumerate(zip(gpus,partitions)):
            directory=run/'workers'/str(i);directory.mkdir(parents=True,exist_ok=True);workers.append(directory)
            single=copy.deepcopy(selection);single['selected']=[dict(gpu,logical_index=0)];single['actual_count']=1;write(directory/'gpu_selection.json',single)
            shard=copy.deepcopy(p);shard['scientific_plan_sha256']=sha(pp);shard['execution_worker_id']=i;shard['evaluation']['global_task_ids']=tasks;write(directory/'plan.json',shard)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],VLA_GPU_SELECTION=str(directory/'gpu_selection.json'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=(directory/'worker.log').open('a');logs.append(log)
            children.append(subprocess.Popen([str(PY),'-u','-B','-m','mini_vla.parallel_regression_panel_worker_v1','--plan',str(directory/'plan.json'),'--run',str(directory),'--release',str(run/'release_evaluation.json')],env=env,stdout=log,stderr=subprocess.STDOUT))
        for _ in range(1200):
            if any(c.poll() is not None for c in children):raise RuntimeError('Worker failed before any new policy episode')
            if all((d/'ready.json').exists() for d in workers):break
            time.sleep(1)
        else:raise TimeoutError('Worker gate timeout')
        gates=[];positives=[]
        for directory in workers:
            gate=json.loads((directory/'online_contract.json').read_text());positive=json.loads((directory/'simulation_preflight/result.json').read_text())
            assets=json.loads((directory/'simulation_preflight/asset_contract.json').read_text());metadata=json.loads((directory/'metadata_gate.json').read_text())
            assert gate['status']==positive['status']==assets['status']==metadata['status']=='PASS'
            assert gate['samples_per_arm']==120 and gate['comparisons']==360 and gate['maximum_action_error']<=1e-6 and gate['all_features_bitwise_equal']
            assert positive['steps']==73 and positive['success_verified'] and assets['tasks']==40 and assets['cases']==200
            validate_metadata(gate['metadata'],p);validate_metadata(metadata['metadata'],p);gates.append(gate);positives.append(positive)
        calls=sum(3*panel['max_action_steps'][c['suite']]//5 for c in panel['cases'])
        physical=sum(3*(panel['max_action_steps'][c['suite']]+10*(1+len(c['development_anchor_history']))) for c in panel['cases'])
        slowest=max(v for g in gates for v in g['inference_seconds_p95'].values())
        physics=max(x['elapsed_seconds']/73 for x in positives)
        estimate=(calls*slowest+physical*physics)*1.5/len(workers)+1800
        admission=dict(status='PASS' if estimate<p['resources']['operational_timeout_seconds'] else 'OPERATIONAL_ESTIMATE_BLOCKED',
           workers=len(workers),max_policy_calls=calls,max_physical_steps_including_anchor_waits=physical,
           slowest_policy_p95_seconds=slowest,positive_physical_seconds_per_step=physics,estimated_seconds=estimate,
           scope='600 frozen cases,not2000',limitation='Physics extrapolated from existingSpatialpositive; allsuiteasset contracts passed;1.5 safety multiplier plus1800s I/O/review allowance')
        write(run/'throughput_admission.json',admission);assert admission['status']=='PASS',admission
        write(run/'parallel_gates.json',dict(status='PASS',workers=len(workers),task_partitions=partitions,online_contracts=gates,positive_replays=positives))
        write(run/'release_evaluation.json',dict(status='RELEASED',scientific_plan_sha256=sha(pp),episodes=600))
        last=None
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Panel evaluator failed; preserve all complete episodes and partial queues')
            rows=[];signatures=[]
            for i,d in enumerate(workers):
                f=d/'episodes/progress.json'
                if f.exists():rows.append(dict(worker=i,**json.loads(f.read_text())))
                for f in (d/'episodes').glob('*/*/*/trajectory.jsonl'):signatures.append((str(f),f.stat().st_mtime_ns))
            signature=tuple(signatures)
            if signature!=last:progress('four_suite_dev600',workers=rows,completed=sum(r['completed'] for r in rows),expected=600);last=signature
            time.sleep(3)
        assert all(c.returncode==0 for c in children)
        episodes=[];seen=set()
        for d in workers:
            result=json.loads((d/'episodes/result.json').read_text());assert result['status']=='PASS';validate_metadata(result['metadata'],p)
            for row in result['episodes']:
                key=(row['arm'],row['pair_id']);assert key not in seen and row['plan_sha256']==sha(pp);seen.add(key);episodes.append(row)
                target=d/'episodes'/row['arm']/row['pair_id'];link=run/row['arm']/row['pair_id'];link.parent.mkdir(parents=True,exist_ok=True)
                if link.is_symlink():assert link.resolve()==target.resolve()
                else:link.symlink_to(target,target_is_directory=True)
        assert seen=={(arm,case_key(c)) for arm in p['arms'] for c in panel['cases']}
        write(run/'result.json',dict(status='PASS',episodes=episodes,expected=600,parallel_workers=len(workers),elapsed_seconds=time.monotonic()-began))
        progress('independent_all600_review')
        with (run/'independent_review.log').open('a') as log:
            rc=subprocess.run([str(PY),'-u','-B',str(ROOT/'scripts/review_regression_panel_v1.py'),'--plan',str(pp),'--run',str(run)],stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0,'Independent600 audit failed'
        write(run/'pipeline.json',dict(status='COMPLETE_REVIEW_REQUIRED',new_episodes=600,new_training_updates=0,elapsed_seconds=time.monotonic()-began))
        progress('complete',completed=600,expected=600)
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            if c.poll() is None:
                try:c.wait(timeout=30)
                except subprocess.TimeoutExpired:c.kill();c.wait()
        for log in logs:log.close()

if __name__=='__main__':main()
