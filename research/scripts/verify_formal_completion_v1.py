#!/usr/bin/env python3
"""Post-run CPU artifact audit. No training, simulator launch, or GPU allocation."""
import concurrent.futures, datetime, hashlib, json, math, os, pathlib, statistics, subprocess
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
RUN=ROOT/'results/formal_pipeline_20261013_v2'
EXP=ROOT/'results/formal_spatial_20261013_v1'
def read(p):return json.loads(p.read_text())
def main():
    plan=read(ROOT/'configs/formal_plan_20261013.json');pipeline=read(RUN/'pipeline.json');training=read(EXP/'training_report.json')
    assert pipeline['status']=='SUCCESS' and training['status']=='PASS'
    assert training['finished_updates']==training['optimizer_updates_this_attempt']==plan['steps']==30000
    assert training['start_step']==0 and not training['resume']
    source=read(RUN/'source_manifest.json');source_checks={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in source['sha256'].items()}
    assert all(source_checks.values())
    attempts=sorted(EXP.glob('attempt_*'));assert len(attempts)==1,'Audit any retry lineage explicitly'
    metrics=[json.loads(line) for line in (attempts[0]/'metrics.jsonl').read_text().splitlines()]
    assert [x['update'] for x in metrics]==list(range(1,30001))
    assert all(math.isfinite(row[key]) for row in metrics for key in ['loss','grad_norm','param_norm','update_seconds','next_batch_seconds'])
    initial=read(EXP/'initial_parameters.json');initial_hashes=initial['parameter_sha256'];trainable=set(initial['trainable_paths'])
    frozen=set(initial_hashes)-trainable;checkpoint_hash_audits=[]
    for step in range(1000,30001,1000):
        saved=read(EXP/f'state_{step}.json');hashes=saved['parameter_sha256']
        assert set(hashes)==set(initial_hashes)
        changed={k for k in hashes if hashes[k]!=initial_hashes[k]}
        assert changed<=trainable and all(hashes[k]==initial_hashes[k] for k in frozen)
        assert any('lora' in k for k in changed) and len(saved['optimizer_sha256'])==64
        checkpoint_hash_audits.append({'step':step,'changed_parameter_leaves':len(changed),'changed_lora_leaves':sum('lora' in k for k in changed),'frozen_unchanged':True})
    ck=pathlib.Path(read(EXP/'checkpoint.json')['path']);assert ck.name=='30000'
    retained=sorted(int(p.name) for p in ck.parent.iterdir() if p.is_dir() and p.name.isdigit())
    assert retained==[10000,20000,30000]
    for step in retained:
        for name in ['params','train_state','assets']:assert (ck.parent/str(step)/name).is_dir()
    norm=ROOT/'configs/assets/pi05_libero_lora_3090_train/physical-intelligence/libero/norm_stats.json'
    ck_norm=ck/'assets/physical-intelligence/libero/norm_stats.json';assert read(norm)==read(ck_norm)
    results={k:read(RUN/(k+'_eval')/'episodes/result.json') for k in ['base','trained']}
    expected={(t,i) for t in plan['evaluation']['task_ids'] for i in plan['evaluation']['initial_state_indices']}
    cases={};jobs=[];policies={}
    for variant,result in results.items():
        assert result['status']=='PASS' and len(result['episodes'])==len(expected)==50
        assert result['actual_egl_render_uuid']==result['render_mapping']['selected_gpu']['uuid']
        actual=set();cases[variant]={}
        for row in result['episodes']:
            key=(row['task_id'],row['initial_state_index']);assert key not in actual;actual.add(key)
            assert row['env_seed']==1000+100*key[0]+key[1] and row['policy_seed']==2000+100*key[0]+key[1]
            assert row['pair_id']==f'task{key[0]}_init{key[1]}' and 1<=row['steps']<=220
            assert row['policy_calls']==row['inference_calls']==math.ceil(row['steps']/5)
            assert row['status']==('TASK_SUCCESS' if row['task_success'] else 'STEP_LIMIT_REACHED')
            if not row['task_success']:assert row['steps']==220
            cases[variant][key]=row;jobs.append((variant,row))
        assert actual==expected
        service=read(RUN/(variant+'_eval')/'service.json');assert service['cleanup_completed'] and service['server_returncode']==-15
        policies[variant]={'physical_gpu':result['render_mapping']['selected_gpu']['index'],'episodes':50,'successes':sum(r['task_success'] for r in result['episodes']),
           'environment_steps':sum(r['steps'] for r in result['episodes']),'policy_calls':sum(r['policy_calls'] for r in result['episodes']),'server_cleanup_verified':True}
    def check_video(job):
        variant,row=job;p=pathlib.Path(row['video']);assert p.is_relative_to(RUN)
        data=json.loads(subprocess.check_output(['ffprobe','-v','error','-threads','1','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(p)],text=True,timeout=60))['streams'][0]
        assert data['width']==data['height']==224 and int(data['nb_read_frames'])==row['steps'] and data['r_frame_rate']=='10/1'
        return {'variant':variant,'pair_id':row['pair_id'],'frames':int(data['nb_read_frames']),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:videos=list(pool.map(check_video,jobs))
    per_task=[];paired_counts={'both_fail':0,'base_only':0,'trained_only':0,'both_success':0}
    for key in sorted(expected):
        b=cases['base'][key];t=cases['trained'][key]
        assert b['env_seed']==t['env_seed'] and b['policy_seed']==t['policy_seed']
        paired_counts[['both_fail','trained_only','base_only','both_success'][int(b['task_success'])*2+int(t['task_success'])]]+=1
    for task in range(10):
        b=sum(cases['base'][(task,i)]['task_success'] for i in plan['evaluation']['initial_state_indices'])
        t=sum(cases['trained'][(task,i)]['task_success'] for i in plan['evaluation']['initial_state_indices'])
        per_task.append({'task_id':task,'base_successes':b,'trained_successes':t,'n':5,'difference':(t-b)/5})
    differences=np.asarray([t['difference'] for t in per_task]);rng=np.random.default_rng(plan['analysis'].get('bootstrap_seed',20261013))
    samples=differences[rng.integers(0,10,size=(10000,10))].mean(axis=1);interval=np.quantile(samples,[.025,.975]).tolist()
    auto=read(RUN/'analysis.json');assert policies['base']['successes']==auto['base_successes'] and policies['trained']['successes']==auto['trained_successes']
    np.testing.assert_allclose(interval,auto['task_cluster_bootstrap_95pct_interval'],rtol=0,atol=1e-12)
    compute=sum(row['update_seconds'] for row in metrics);loading=sum(row['next_batch_seconds'] for row in metrics)
    stages={x['stage']:x for x in pipeline['stages']}
    ledger={'training_stage_seconds':stages['train30000']['elapsed_seconds'],'evaluation_stage_seconds':sum(stages[x]['elapsed_seconds'] for x in ['base_eval','trained_eval']),
      'pipeline_seconds':read(RUN/'status.json')['elapsed_seconds'],'training_gpu_window_hours':2*stages['train30000']['elapsed_seconds']/3600,
      'evaluation_gpu_window_hours':sum(stages[x]['elapsed_seconds'] for x in ['base_eval','trained_eval'])/3600,
      'scope':'selected-device wall windows, including CPU work and I/O; not exclusive reservation or measured GPU-active time'}
    ledger['total_gpu_window_hours']=ledger['training_gpu_window_hours']+ledger['evaluation_gpu_window_hours']
    v=os.statvfs(ROOT);used=int(subprocess.check_output(['du','-s','-B1',str(ROOT)],text=True).split()[0]);free=v.f_bavail*v.f_frsize
    assert used<=120*2**30 and free>=100*2**30
    audit={'status':'PASS','reviewed_at':datetime.datetime.now().astimezone().isoformat(),'source_hash_checks':source_checks,'metric_rows':len(metrics),'metric_step_sequence_exact':True,'finite_training_metrics':True,
      'parameter_leaves':len(initial_hashes),'trainable_leaves':len(trainable),'frozen_leaves':len(frozen),'checkpoint_hash_audits':checkpoint_hash_audits,'retained_checkpoint_steps':retained,'training_inference_norm_identical':True,
      'policies':policies,'paired_cases':paired_counts,'per_task':per_task,'paired_difference':float(differences.mean()),'task_cluster_bootstrap_95pct_interval':interval,'video_checks':videos,
      'training_loss':{'first_1000_mean':statistics.mean(x['loss'] for x in metrics[:1000]),'last_1000_mean':statistics.mean(x['loss'] for x in metrics[-1000:])},
      'timing':{'update_seconds_mean':compute/len(metrics),'update_seconds_median':statistics.median(x['update_seconds'] for x in metrics),'total_synchronous_update_seconds':compute,'total_next_batch_seconds':loading,'training_loop_other_seconds':training['training_wall_seconds']-compute-loading},
      'resources':ledger,'storage':{'project_GiB':used/2**30,'filesystem_free_GiB':free/2**30,'budget_GiB':120,'filesystem_reserve_GiB':100},
      'limitations':['one training seed (42), 50 evaluation episodes per policy, 10 task clusters','not a new training-seed replication, full LIBERO benchmark, or demonstration of unseen-task generalization','whole-device memory includes shared occupancy; cannot treat it as model-only peak','formal run did not resume mid-run; optimizer resume was verified in earlier 20+2 smoke']}
    (RUN/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (ROOT/'reports/formal_completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (ROOT/'reports/formal_resource_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['video_checks','source_hash_checks','checkpoint_hash_audits']},indent=2))
if __name__=='__main__':main()
