"""Full new-round coverage, audit bindings, paired effects and repeat-control noise."""
import argparse,datetime,json,os,pathlib,subprocess
import numpy as np
from paired_live_audit_v1 import check_saved_branch,verify_audit_bindings,checked_close,arrays,sha,METHODS
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
CANDIDATES=('fine20','resample5','retreat5_fine20')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def write(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def robust_rescue_ids(roots,rows):
    rescued=set()
    for root in roots:
        a=rows[(root['root_id'],'continue_a')];b=rows[(root['root_id'],'continue_b')]
        if not a['baseline_success'] and not a['task_success'] and not b['task_success']:
            if any(rows[(root['root_id'],m)]['task_success'] for m in CANDIDATES):rescued.add(root['pair_id'])
    return sorted(rescued)

def paired_metric(roots,rows,method,control,draws):
    pairs=[(rows[(root['root_id'],method)],rows[(root['root_id'],control)]) for root in roots]
    wins=sum(a['task_success'] and not b['task_success'] for a,b in pairs)
    losses=sum(b['task_success'] and not a['task_success'] for a,b in pairs)
    sums=np.zeros(10);counts=np.zeros(10,dtype=int)
    for root,(a,b) in zip(roots,pairs):
        sums[root['task_id']]+=int(a['task_success'])-int(b['task_success']);counts[root['task_id']]+=1
    denominator=counts[draws].sum(1);valid=denominator>0
    boot=sums[draws].sum(1)[valid]/denominator[valid]
    return {'n_used':len(roots),'n_expected':len(roots),'control':control,
            'successes':sum(a['task_success'] for a,b in pairs),'wins':int(wins),'losses':int(losses),
            'paired_difference':(wins-losses)/len(roots),'task_cluster_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
            'task_clusters':int((counts>0).sum()),'valid_bootstrap_draws':int(valid.sum()),
            'policy_calls':sum(a['policy_calls'] for a,b in pairs),
            'extra_policy_calls':sum(a['policy_calls']-b['policy_calls'] for a,b in pairs),
            'inference_seconds':sum(a['inference_seconds'] for a,b in pairs),
            'executed_actions':sum(a['steps'] for a,b in pairs),'manual_actions':sum(a['manual_steps'] for a,b in pairs),
            'per_task':[{'task':i,'n':int(counts[i]),'net_wins':int(sums[i])} for i in range(10)]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    run=pathlib.Path(a.run).resolve();plan=json.loads(pathlib.Path(a.plan).read_text())
    assert plan['paired_live_amendment_approved'] and plan['expected_branches']==1345
    manifest=json.loads((ROOT/plan['root_manifest']).read_text());roots=manifest['roots'];rows={}
    for w in range(plan['workers']):
        data=json.loads((run/'workers'/f'worker{w}'/'result.json').read_text());assert data['status']=='PASS'
        for row in data['branches']:
            key=(row['root_id'],row['method']);assert key not in rows;rows[key]=row
    assert len(rows)==1345==len(roots)*5
    expected={(root['root_id'],method) for root in roots for method in METHODS};assert set(rows)==expected
    source={};videos=0;calls=0;physics=0;checked=0;repeat=[];audit_index=[]
    for root in roots:
        group=[rows[(root['root_id'],m)] for m in METHODS]
        assert len({row['root_image_sha256'] for row in group})==1
        if root['pair_id'] not in source:
            d=ROOT/plan['source_episodes']/root['pair_id']
            source[root['pair_id']]=(json.loads((d/'episode.json').read_text()),arrays(d/'oracle.npz'),arrays(d/'policy_trace.npz'))
        old,old_oracle,old_trace=source[root['pair_id']]
        for row in group:
            directory=pathlib.Path(row['output']).resolve();assert directory.is_relative_to(run/'workers')
            stored=json.loads((directory/'result.json').read_text())
            assert stored==row and row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED') and row['verification_status']=='PASS'
            assert row['baseline_success']==old['success_at_220'] and row['module_partition']==root['module_partition']
            trace,inputs,oracle,meta=check_saved_branch(directory,row,root,plan);verify_audit_bindings(directory,row)
            for field in ('qpos','qvel','control','object_position','object_quaternion'):
                checked_close(oracle[field][0],old_oracle[field][root['branch_step']],1e-7)
            checked_close(trace['robot_state'][0],old_trace['robot_state'][root['branch_step']],1e-6)
            for camera in ('external','wrist'):
                result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames',
                                       '-of','json',str(directory/(camera+'.mp4'))],capture_output=True,text=True,check=True,timeout=30)
                stream=json.loads(result.stdout)['streams'][0]
                assert (stream['width'],stream['height'],int(stream['nb_frames']))==(224,224,row['steps']+1);videos+=1
            calls+=row['policy_calls'];physics+=int(row['method'] in ('continue_a','continue_b'));checked+=1
            audit_index.append({'root_id':root['root_id'],'method':row['method'],'result_sha256':sha(directory/'result.json'),
                                'model_audit_sha256':sha(directory/'model_replay.json'),
                                'physics_audit_sha256':sha(directory/'physics_replay.json') if row['method'] in ('continue_a','continue_b') else None})
        control_a,control_b=group[:2]
        tr_a=arrays(pathlib.Path(control_a['output'])/'policy_trace.npz');tr_b=arrays(pathlib.Path(control_b['output'])/'policy_trace.npz')
        length=min(control_a['steps'],control_b['steps'])
        repeat.append({'root_id':root['root_id'],'pair_id':root['pair_id'],'branch_step':root['branch_step'],'task_id':root['task_id'],
                       'a_success':control_a['task_success'],'b_success':control_b['task_success'],
                       'outcome_disagrees':control_a['task_success']!=control_b['task_success'],
                       'steps_a':control_a['steps'],'steps_b':control_b['steps'],
                       'action_max_difference_common_steps':float(np.abs(tr_a['action'][:length]-tr_b['action'][:length]).max()),
                       'a_differs_from_historical_outcome':control_a['task_success']!=old['success_at_220']})
        if checked%25==0:write(run/'analysis_progress.json',{'status':'RUNNING','branches_verified':checked,'expected':1345,'updated_at':now()})
    assert physics==538 and videos==2690
    draws=np.random.default_rng(plan['analysis']['bootstrap_seed']).integers(0,10,size=(plan['analysis']['bootstrap_draws'],10))
    cohorts={}
    for step in (60,120):
        subset=[root for root in roots if root['branch_step']==step]
        cohorts[str(step)]={'roots':len(subset),'primary_control':'continue_a',
                           'vs_a':{m:paired_metric(subset,rows,m,'continue_a',draws) for m in METHODS},
                           'candidate_sensitivity_vs_b':{m:paired_metric(subset,rows,m,'continue_b',draws) for m in CANDIDATES}}
    rescues=robust_rescue_ids(roots,rows)
    original_failures={pair for pair,(old,_,_) in source.items() if not old['success_at_220']}
    ordinary=set();harms=set()
    for root in roots:
        control=rows[(root['root_id'],'continue_a')];alternatives=[rows[(root['root_id'],m)] for m in CANDIDATES]
        if not control['task_success'] and any(r['task_success'] for r in alternatives):ordinary.add(root['pair_id'])
        if control['task_success'] and any(not r['task_success'] for r in alternatives):harms.add(root['pair_id'])
    assert set(rescues)<=original_failures
    partitions={key:sorted({root['pair_id'] for root in roots if root['module_partition']==key}) for key in ('fit','calibration')}
    assert len(partitions['fit'])==140 and len(partitions['calibration'])==60 and not set(partitions['fit'])&set(partitions['calibration'])
    result={'status':'COMPLETE','generated_at':now(),'branches_used':checked,'branches_expected':1345,'roots_used':269,
            'unique_episodes':len(source),'task_clusters':10,'confirmation_used':0,'new_training_updates':0,
            'all_model_input_replays_passed':True,'model_calls_verified':calls,'all_continue_physics_replays_passed':True,
            'physics_controls_verified':physics,'video_metadata_checked':videos,'primary_control':'continue_a',
            'old_outcomes_mixed_into_new_effect_estimate':False,'cohorts':cohorts,'repeat_controls':repeat,
            'repeat_outcome_disagreements':sum(item['outcome_disagrees'] for item in repeat),
            'repeat_disagreement_episodes':sorted({item['pair_id'] for item in repeat if item['outcome_disagrees']}),
            'robust_rescuable_original_failure_episodes':len(rescues),'robust_rescuable_episode_ids':rescues,
            'original_failure_episodes':len(original_failures),'oracle_any_candidate_rescues_vs_a':len(ordinary),
            'episodes_harmed_by_any_candidate_vs_a':len(harms),'module_partitions':partitions,
            'feasibility_gate_passed':len(rescues)>=8,'minimum_distinct_robust_rescues':8,
            'scope':'Exploratory development feasibility; offline any-candidate oracle is not a deployable selector or confirmation result.',
            'next_stage':'Freeze fit/calibration progress/value module protocol' if len(rescues)>=8 else 'Improve recovery candidates without resampling outcomes',
            'noise_scope':'One same-seed A/B repeat per root quantifies observed execution differences; does not establish zero stochastic variance.'}
    write(run/'artifact_audit_index.json',{'branches':audit_index,'generated_at':now()})
    write(run/'ANALYSIS.json',result)
    lines=['# 统一配对恢复分支：完整开发侧结果','',f'1345/1345分支、269/269根、200场景；真实推理重放{calls}次，538/538个继续分支物理重放通过。确认集0，训练更新0。','',
           'A固定为主对照，B为相同配置/种子的运行差异对照；旧轮次结果未混入效果统计。','',
           '| 时刻 | 条件 | 成功/根 | 胜A/负A | 策略调用 |','|---|---|---:|---:|---:|']
    for step,cohort in cohorts.items():
        for method,m in cohort['vs_a'].items():lines.append(f"| {step} | {method} | {m['successes']}/{m['n_used']} | {m['wins']}/{m['losses']} | {m['policy_calls']} |")
    lines+=['',f"A/B成败不一致：{result['repeat_outcome_disagreements']}/269根；详见逐根记录与任务聚类区间。",
            f'保守可救回原失败场景：{len(rescues)}/{len(original_failures)}（要求A、B均失败，至少一个固定候选成功）。',
            f"可行性门槛（至少8个不同场景且全部审核通过）：{result['feasibility_gate_passed']}。",
            '多候选事后上界不代表上线选择器收益。共享场景的根/分支不是独立样本；60/120步分层，区间按10个任务聚类。',
            '本轮不改变模型参数，不使用确认集。后续模块需沿用冻结fit/calibration划分并另存具体训练协议。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(run/'analysis_progress.json',{'status':'COMPLETE','branches_verified':checked,'expected':1345,'updated_at':now()})
    print(json.dumps({'status':'COMPLETE','branches':checked,'robust_rescues':len(rescues),'repeat_disagreements':result['repeat_outcome_disagreements'],'gate':len(rescues)>=8}),flush=True)
if __name__=='__main__':main()
