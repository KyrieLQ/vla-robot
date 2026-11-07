"""Outcome variation versus benefit variation, genuine VLA seed effect, coverage audit."""
import collections,datetime,json,os,pathlib,time
import numpy as np
from paired_live_audit_v1 import arrays,sha
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def read(p):return json.loads(pathlib.Path(p).read_text())
def main():
    start=time.monotonic();run=ROOT/'results/recovery_seed_stability_20261014_v1';assert read(run/'review/acceptance.json')['status']=='PASS';plan=read(ROOT/'configs/recovery_seed_stability_20261014_v1.json');raw=arrays(run/'paired_outcomes.npz');a=raw['continue_a'];r=raw['resample5'];gain=r-a;rows={}
    for worker in (0,1):
        for row in read(run/f'workers/worker{worker}/result.json')['branches']:rows[row['root_id'],row['method']]=row
    root_lookup={(root['pair_id'],root['replicate_index']):root for root in plan['policy_seed_roots']};percase=[];within=[];across=[];inputs_verified=0
    for i,pair in enumerate(plan['selected_pair_ids']):
        chunks={'continue_a':[],'resample5':[]};digests=set();states=[]
        for seed in range(3):
            root=root_lookup[pair,seed]
            for method in chunks:
                directory=pathlib.Path(rows[root['root_id'],method]['output']);trace=arrays(directory/'policy_trace.npz');obs=arrays(directory/'policy_inputs.npz');assert int(obs['call_step'][0])==60;digests.add(str(trace['image_sha256'][0]));chunks[method].append(trace['action_chunk'][0]);states.append(obs['robot_state'][0]);inputs_verified+=1
            within.append(float(np.max(np.abs(chunks['continue_a'][-1]-chunks['resample5'][-1]))))
        assert len(digests)==1
        for s in states:np.testing.assert_allclose(s,states[0],atol=1e-6,rtol=0)
        for method in chunks:
            for j in range(3):
                for k in range(j):across.append(float(np.max(np.abs(chunks[method][j]-chunks[method][k]))))
        percase.append(dict(pair_id=pair,continue_successes=a[i].tolist(),resample_successes=r[i].tolist(),paired_gains=gain[i].tolist(),continue_varies=bool(np.ptp(a[i])),resample_varies=bool(np.ptp(r[i])),paired_gain_varies=bool(np.ptp(gain[i]))))
    historical={}
    for w in (0,1):
        for b in read(ROOT/f'results/recovery_paired_live_20261014_v3/workers/worker{w}/result.json')['branches']:historical[b['root_id'],b['method']]=b
    coverage=[]
    for pair in plan['selected_pair_ids']:
        rid=pair+'_t60';base=historical[rid,'continue_a'];b=historical[rid,'continue_b'];coverage.append(dict(pair_id=pair,original_episode_failed=not base['baseline_success'],old_a_failed=not base['task_success'],old_a_b_failed=not base['task_success'] and not b['task_success'],old_resample_rescued=not base['task_success'] and historical[rid,'resample5']['task_success'],old_any_candidate_rescued=not base['task_success'] and any(historical[rid,m]['task_success'] for m in ('fine20','resample5','retreat5_fine20'))))
    hist_summary={k:sum(v[k] for v in coverage) for k in coverage[0] if k!='pair_id'}
    result=dict(created_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),fit_cases=20,paired_draws=60,task_clusters=10,continue_total_success=int(a.sum()),resample_total_success=int(r.sum()),paired_wins=int((gain>0).sum()),paired_losses=int((gain<0).sum()),both_success=int(((a==1)&(r==1)).sum()),both_failure=int(((a==0)&(r==0)).sum()),continue_roots_varying=int(np.sum(np.ptp(a,axis=1)>0)),resample_roots_varying=int(np.sum(np.ptp(r,axis=1)>0)),paired_gain_roots_varying=int(np.sum(np.ptp(gain,axis=1)>0)),root_input_bindings_checked=inputs_verified,root_inputs_identical_across_methods_and_seeds=True,first_chunk_A_R_comparisons=len(within),first_chunk_A_R_nonzero=int(np.sum(np.array(within)>0)),first_chunk_A_R_median_max_abs=float(np.median(within)),first_chunk_A_R_largest_max_abs=float(max(within)),first_chunk_cross_seed_comparisons=len(across),first_chunk_cross_seed_nonzero=int(np.sum(np.array(across)>0)),first_chunk_cross_seed_median_max_abs=float(np.median(across)),per_case=percase,historical_coverage_scope='Separate coverage description only. These old results do not enter new paired-effect, variance, intervals or gates.',historical_coverage_summary=hist_summary,historical_coverage_cases=coverage,elapsed_seconds=time.monotonic()-start)
    base_calls=sum(rows[root['root_id'],'continue_a']['policy_calls'] for root in root_lookup.values())
    resample_calls=sum(rows[root['root_id'],'resample5']['policy_calls'] for root in root_lookup.values())
    extra=sum(max(0,rows[root['root_id'],'resample5']['policy_calls']-rows[root['root_id'],'continue_a']['policy_calls']) for root in root_lookup.values())
    result.update(resample_live_total_calls_ratio=resample_calls/base_calls,positive_extra_calls=extra,positive_extra_calls_ratio=extra/base_calls,cost_scope='Live branch suffix calls only. No learned selector query/cache overhead credited; no calibration performed.')
    (run/'review/extended_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    fig,axes=plt.subplots(1,3,figsize=(11,7),layout='constrained');cmap=ListedColormap(['#ca5a57','#ededed','#398aa7']);norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
    for ax,matrix,title in zip(axes,[a,r,gain],['Continue outcome','Resample outcome','Paired gain']):
        ax.imshow(matrix,aspect='auto',cmap=cmap if title=='Paired gain' else ListedColormap(['#ededed','#398aa7']),norm=norm if title=='Paired gain' else None,vmin=None if title=='Paired gain' else 0,vmax=None if title=='Paired gain' else 1);ax.set_xticks(range(3),['Seed1','Seed2','Seed3']);ax.set_yticks(range(20),plan['selected_pair_ids'],fontsize=8);ax.set_title(title)
        for i in range(20):
            for j in range(3):ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center',fontsize=8)
    fig.suptitle('20 fixed fit scenes, 3 policy seeds; independent A/B controls excluded from paired totals');fig.savefig(run/'review/paired_seed_outcomes.png',dpi=160);plt.close(fig);print(json.dumps({k:v for k,v in result.items() if k not in ('per_case','historical_coverage_cases')}),flush=True)
if __name__=='__main__':main()
