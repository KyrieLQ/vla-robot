#!/usr/bin/env python3
"""Complete paired-root feasibility analysis, retaining failed and harmful candidates."""
import argparse,datetime,json,os,pathlib,subprocess
import numpy as np
from verify_continuation_v4 import verify_continuation
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    run=pathlib.Path(a.run);plan=json.loads(pathlib.Path(a.plan).read_text());manifest=json.loads((ROOT/plan['root_manifest']).read_text());roots=manifest['roots'];methods=[m['name'] for m in plan['methods']]
    all_rows={};used=0;video_count=0;continuation_audits=[]
    for w in range(plan['workers']):
        data=json.loads((run/'workers'/f'worker{w}'/'result.json').read_text());assert data['status']=='PASS'
        for row in data['branches']:
            key=(row['root_id'],row['method']);assert key not in all_rows;all_rows[key]=row
    assert len(all_rows)==plan['expected_branches']==len(roots)*len(methods)
    for root in roots:
        rows=[all_rows[(root['root_id'],m)] for m in methods];assert len({r['root_image_sha256'] for r in rows})==1
        control=rows[0];assert control['method']=='continue' and control['continuation_verified'] and control['task_success']==control['baseline_success']
        for row in rows:
            ep=pathlib.Path(row['output']);n=row['steps']
            assert row['status'] in ('TASK_SUCCESS','STEP_LIMIT_REACHED')
            assert n+root['branch_step']<=220 and row['baseline_success']==control['baseline_success']
            with np.load(ep/'policy_trace.npz') as tr,np.load(ep/'oracle.npz') as oracle:
                assert tr['action'].shape==(n,7) and tr['robot_state'].shape==(n+1,8)
                assert len(tr['done'])==n and bool(tr['done'].any())==row['task_success']
                assert np.array_equal(oracle['goal_predicate'][1:,0],tr['done'])
                assert len(tr['inference_seconds'])==row['policy_calls']
                for name in ('action','robot_state','action_chunk','inference_seconds'):assert np.isfinite(tr[name]).all()
                if row['method']=='continue':
                    source_path=ROOT/plan['source_episodes']/root['pair_id']
                    with np.load(source_path/'policy_trace.npz') as reference,np.load(source_path/'oracle.npz') as reference_oracle:
                        continuation_audits.append(verify_continuation(row,tr,oracle,reference,reference_oracle,plan))
            for cam in ('external','wrist'):
                v=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(ep/(cam+'.mp4'))],capture_output=True,text=True,check=True,timeout=30)
                s=json.loads(v.stdout)['streams'][0];assert (s['width'],s['height'],int(s['nb_frames']))==(224,224,n+1);video_count+=1
            used+=1
    rng=np.random.default_rng(plan['analysis']['bootstrap_seed']);draws=rng.integers(0,10,size=(plan['analysis']['bootstrap_draws'],10))
    cohorts={};rescued=set();harmed=set()
    for step in (60,120):
        subset=[r for r in roots if r['branch_step']==step];groups={t:[r for r in subset if r['task_id']==t] for t in range(10)}
        cohort={'roots':len(subset),'task_clusters':sum(bool(x) for x in groups.values()),'methods':{}}
        for method in methods:
            rows=[all_rows[(r['root_id'],method)] for r in subset];controls=[all_rows[(r['root_id'],'continue')] for r in subset]
            wins=sum(r['task_success'] and not c['task_success'] for r,c in zip(rows,controls));losses=sum(c['task_success'] and not r['task_success'] for r,c in zip(rows,controls))
            sums=np.array([sum(int(all_rows[(r['root_id'],method)]['task_success'])-int(all_rows[(r['root_id'],'continue')]['task_success']) for r in groups[t]) for t in range(10)])
            ns=np.array([len(groups[t]) for t in range(10)]);denom=ns[draws].sum(1);valid=denom>0
            sample=sums[draws].sum(1)[valid]/denom[valid];ci=np.quantile(sample,[.025,.975])
            cohort['methods'][method]={'n_used':len(rows),'n_expected':len(subset),'successes':sum(r['task_success'] for r in rows),'wins_vs_continue':wins,'losses_vs_continue':losses,
                                       'paired_difference':(wins-losses)/len(rows),'task_cluster_bootstrap_95pct':ci.tolist(),'valid_bootstrap_draws':int(valid.sum()),
                                       'policy_calls':sum(r['policy_calls'] for r in rows),'extra_policy_calls_vs_continue':sum(r['policy_calls']-c['policy_calls'] for r,c in zip(rows,controls)),
                                       'inference_seconds':sum(r['inference_seconds'] for r in rows),'manual_actions':sum(r['manual_steps'] for r in rows),
                                       'suffix_actions':sum(r['steps'] for r in rows),'replay_seconds_excluded_from_policy_cost':sum(r['prefix_replay_seconds'] for r in rows)}
        cohorts[str(step)]=cohort
    for root in roots:
        control=all_rows[(root['root_id'],'continue')]
        alternatives=[all_rows[(root['root_id'],m)] for m in methods[1:]]
        if not control['task_success'] and any(r['task_success'] for r in alternatives):rescued.add(root['pair_id'])
        if control['task_success'] and any(not r['task_success'] for r in alternatives):harmed.add(root['pair_id'])
    result={'status':'COMPLETE','generated_at':datetime.datetime.now().astimezone().isoformat(),'branches_used':used,'branches_expected':plan['expected_branches'],'unique_episodes':len({r['pair_id'] for r in roots}),'roots':len(roots),'confirmation_used':0,
            'all_continue_suffixes_verified':True,'continuation_audits':continuation_audits,
            'actual_policy_input_images_verified':sum(x['consumed_images_checked'] for x in continuation_audits),
            'approved_record_only_image_differences':sum(x['record_only_image_differences'] for x in continuation_audits),
            'video_metadata_checked':video_count,'cohorts':cohorts,
            'oracle_rescuable_baseline_failure_episodes':len(rescued),'oracle_rescuable_episode_ids':sorted(rescued),
            'baseline_success_episodes_harmed_by_any_candidate':len(harmed),
            'feasibility_gate_passed':len(rescued)>=8,'gate_scope':'Candidate feasibility only. Oracle choices are not a deployable success rate; development results do not establish generalization.',
            'next_stage':'Freeze module development on fit/calibration partitions' if len(rescued)>=8 else 'Improve recovery candidates or teacher; do not train a selector over ineffective candidates',
            'noise_scope':'One frozen random stream per method/root; outcomes are samples, not exact expected intervention values.'}
    (run/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 同状态分支可行性实验','',f'完成 {used}/{plan["expected_branches"]} 个分支，{len(roots)} 个根状态，200 个开发场景；确认集 0。','',
           '全部继续分支逐步复现原策略；以下仅为开发侧探索性配对结果。','',
           '| 分支时刻 | 条件 | 成功 / 根状态 | 相对继续净胜 / 净负 | 策略调用 |','|---|---|---:|---:|---:|']
    for step,co in cohorts.items():
        for method,s in co['methods'].items():lines.append(f"| {step} | {method} | {s['successes']}/{s['n_used']} | {s['wins_vs_continue']} / {s['losses_vs_continue']} | {s['policy_calls']} |")
    lines+=['',f'至少一个候选能救回的基线失败场景：{len(rescued)}/30（离线上界，不能作上线选择器成绩）。',
            f'至少一个候选伤害的基线成功场景：{len(harmed)}/170。',f'预注册候选可行性门槛（至少 8 个不同失败场景可救回）：{result["feasibility_gate_passed"]}。',
            '不同根状态和分支共享原 episode；不把它们当独立样本。统计区间、调用成本和复现约束详见 ANALYSIS.json。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'COMPLETE','branches':used,'rescued':len(rescued),'gate':len(rescued)>=8}),flush=True)
if __name__=='__main__':main()
