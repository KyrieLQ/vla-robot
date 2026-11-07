#!/usr/bin/env python3
"""Independent complete-data review. Preserve the original automatic analysis."""
import collections, concurrent.futures, datetime, hashlib, itertools, json, os, pathlib, subprocess
import numpy as np
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
RUN=ROOT/'results/research_diagnosis_20261014_v3'

def main():
    out=RUN/'review';out.mkdir(exist_ok=True)
    old=json.loads((RUN/'ANALYSIS.json').read_text())
    plan=json.loads((ROOT/'configs/research_diagnosis_20261014_v2.json').read_text())
    split=json.loads((ROOT/plan['split_manifest']).read_text())
    wanted={r['pair_id'] for r in split['development']}
    forbidden={r['pair_id'] for r in split['confirmation']}
    manifest=json.loads((RUN/'source_manifest.json').read_text())['sha256']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    tables={};videos=[];trace_hashes={};counts={}
    for job in plan['jobs'][1:]:
        name=job['name'];path=RUN/name/'episodes'
        result=json.loads((path/'result.json').read_text());assert result['status']=='PASS'
        rows={r['pair_id']:r for r in result['episodes']}
        assert len(rows)==len(result['episodes'])==200 and set(rows)==wanted and not(set(rows)&forbidden)
        successes=0
        for pair,row in rows.items():
            ep=path/pair
            with np.load(ep/'policy_trace.npz') as tr,np.load(ep/'oracle.npz') as oracle:
                n=len(tr['action']);assert n==row['steps']
                hit=np.flatnonzero(tr['done'])+1
                first=int(hit[0]) if len(hit) else None
                success=first is not None and first<=220
                assert success==row['success_at_220'] and first==row['first_success_step']
                assert len(oracle['qpos'])==n+1 and np.array_equal(oracle['goal_predicate'][1:,0],tr['done'])
                assert not len(hit) or len(hit)==1 and first==n
                assert n<=job['max_steps'] and (first is not None or n==job['max_steps'])
                successes+=success
                for key in ('qpos','qvel','object_position','object_quaternion','control'):
                    assert np.isfinite(oracle[key]).all()
            trace_hashes[f'{name}/{pair}']=hashlib.sha256((ep/'policy_trace.npz').read_bytes()).hexdigest()
            videos.extend((ep/(c+'.mp4'),row['steps']+1) for c in ('external','wrist'))
        assert successes==old['summaries'][name]['successes_at_220']
        counts[name]={'used':200,'expected':200,'successes':successes}
        tables[name]=rows
    def video_check(item):
        p,expected=item
        result=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                               '-show_entries','stream=width,height,nb_read_frames','-of','json',str(p)],
                              capture_output=True,text=True,check=True,timeout=60)
        s=json.loads(result.stdout)['streams'][0]
        assert (s['width'],s['height'],int(s['nb_read_frames']))==(224,224,expected),str(p)
        return expected
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        decoded_frames=sum(pool.map(video_check,videos))
    base=tables['ckpt30k_prefix5'];fine=tables['ckpt30k_prefix1']
    all_keys=sorted(wanted)
    assert all(r['matches_success'] and r['matches_first_image'] for r in old['old_anchor_rechecks'])
    rng=np.random.default_rng(plan['primary_analysis']['bootstrap_seed'])
    draws=rng.integers(0,10,size=(10000,10))
    contrast_specs=[('30k-20k','ckpt30k_prefix5','ckpt20k_prefix5'),('20k-10k','ckpt20k_prefix5','ckpt10k_prefix5'),('prefix1-prefix5','ckpt30k_prefix1','ckpt30k_prefix5'),('prefix10-prefix5','ckpt30k_prefix10','ckpt30k_prefix5')]
    for label,a,b in contrast_specs:
        means=np.array([np.mean([int(tables[a][k]['success_at_220'])-int(tables[b][k]['success_at_220']) for k in all_keys if base[k]['task_id']==t]) for t in range(10)])
        ci=np.quantile(means[draws].mean(1),[.025,.975])
        np.testing.assert_allclose(ci,old['paired_comparisons'][label]['task_cluster_bootstrap_95pct'],atol=1e-15,rtol=0)
    proxies=json.loads((RUN/'diagnostic_episode_proxies.json').read_text())['episodes']
    categories=collections.Counter()
    for row in proxies:
        if not row['success_at_220']:
            group='no_4cm_target_lift' if not row['target_lift_4cm_ever'] else ('lifted_but_not_within_5cm_xy_of_plate' if not row['target_near_destination_xy_5cm_ever'] else 'near_plate_but_goal_false')
            categories[group]+=1
    # Conditional equal-difficulty null: distribute 30 failures uniformly among
    # 200 slots and ask whether some task gets at least 10 failures. Diagnostic only.
    null_rng=np.random.default_rng(2026101401)
    exceed=0;trials=100000
    for _ in range(trials):
        failure_slots=null_rng.choice(200,size=30,replace=False)
        exceed+=np.bincount(failure_slots//20,minlength=10).max()>=10
    null_probability=exceed/trials
    audit={'status':'PASS','checked_at':datetime.datetime.now().astimezone().isoformat(),'policy_episodes_used':1000,'policy_episodes_expected':1000,'unique_scenarios':200,'task_clusters':10,'training_seeds':1,'confirmation_used':0,
           'counts':counts,'all_source_hashes_unchanged':True,'all_50_old_anchors_match_image_and_outcome':True,'full_video_decodes':len(videos),'decoded_frames':decoded_frames,'trace_checksums':trace_hashes,
           'original_bootstrap_intervals_reproduced':True,'failure_categories_noncausal':dict(categories),
           'prefix1_call_ratio':old['summaries']['ckpt30k_prefix1']['primary_policy_calls']/old['summaries']['ckpt30k_prefix5']['primary_policy_calls'],
           'prefix1_inference_seconds_ratio':old['summaries']['ckpt30k_prefix1']['primary_inference_seconds']/old['summaries']['ckpt30k_prefix5']['primary_inference_seconds'],
           'counterfactual_warning':'27 wins and 24 harms from changing the whole-episode prefix are not same-state recovery results.',
           'episode_policy_oracle_union_successes':sum(base[k]['success_at_220'] or fine[k]['success_at_220'] for k in all_keys),
           'null_model':{'assumption':'Conditional on 30 total failures, 200 slots exchangeable among 10 tasks; not a model of actual heterogeneity','draws':trials,'seed':2026101401,'some_task_at_least_10_failures_probability':null_probability,'mc_standard_error':float((null_probability*(1-null_probability)/trials)**.5),'interpretation':'Task heterogeneity check only; does not identify a causal failure mechanism.'},
           'proxy_caution':'Original static-motion window may overlap pre-contact approach. It fires in 49/170 successes, so it is not a causal label or usable recovery trigger by itself.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['ckpt10k_prefix5','ckpt20k_prefix5','ckpt30k_prefix5','ckpt30k_prefix1','ckpt30k_prefix10']
    labels=['10k / prefix 5','20k / prefix 5','30k / prefix 5','30k / prefix 1','30k / prefix 10']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    sr=[old['summaries'][n]['success_rate_at_220']*100 for n in names]
    colors=['#91a6bd','#91a6bd','#33658a','#e09f3e','#86bbbd']
    axes[0].barh(labels,sr,color=colors);axes[0].set_xlim(0,100);axes[0].set_xlabel('Success at 220 actions (%) | n=200 each')
    for i,v in enumerate(sr):axes[0].text(v+1,i,f'{v:.1f}%',va='center')
    calls=[old['summaries'][n]['primary_policy_calls']/200 for n in names]
    axes[1].barh(labels,calls,color=colors);axes[1].set_xlabel('Mean policy calls per episode (primary budget)')
    fig.suptitle('Development diagnostics: 200 paired scenarios, 10 tasks, 1 VLA seed')
    fig.savefig(out/'diagnosis_success_cost.png',dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['trace_checksums']},indent=2))

if __name__=='__main__':main()
