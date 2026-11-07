"""Episode-level geometry verification, descriptive task intervals and budget evidence."""
import collections,datetime,json,os,pathlib,time
import numpy as np
from paired_live_audit_v1 import arrays
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def read(p):return json.loads(pathlib.Path(p).read_text())
def episode_values(values,rows):
    groups=collections.defaultdict(list)
    for v,r in zip(values,rows):groups[r['pair_id']].append(v)
    return {k:np.mean(v,axis=0) for k,v in groups.items()}
def main():
    start=time.monotonic();run=ROOT/'results/recovery_spatial_progress_20261014_v1';r=read(run/'SPATIAL_RESULTS.json');assert read(run/'review/acceptance.json')['status']=='PASS';rows=read(run/'roots.json');d=arrays(run/'prepared_arrays.npz');p=arrays(run/'oof_predictions.npz');truth=d['geometry'][d['root_indices']];geo={};ep={};rng=np.random.default_rng(2026101409);draws=rng.integers(0,10,(20000,10));intervals={}
    for name in ('spatial','global'):
        err=np.linalg.norm((p['geometry_'+name].mean(0)-truth).reshape(-1,2,3),axis=-1);ep[name]=episode_values(err,rows);mean=np.mean(list(ep[name].values()),0);expected=r['geometry_errors'][name];np.testing.assert_allclose(mean,[expected['target_xyz_mean_error_m'],expected['goal_xyz_mean_error_m']],atol=1e-9,rtol=0);geo[name]=dict(target_error_m=float(mean[0]),goal_error_m=float(mean[1]))
    taskdiff=np.array([np.mean([v-ep['global'][k] for k,v in ep['spatial'].items() if k.startswith(f'task{t}_')],0) for t in range(10)]);dist=taskdiff[draws].mean(1)
    intervals['spatial_minus_global_geometry']={'target_delta_m':float(taskdiff[:,0].mean()),'target_task_interval_m':np.quantile(dist[:,0],[.025,.975]).tolist(),'goal_delta_m':float(taskdiff[:,1].mean()),'goal_task_interval_m':np.quantile(dist[:,1],[.025,.975]).tolist()}
    ref=arrays(ROOT/'results/recovery_fit_audit_20261014_v1/fit_only_oof_predictions.npz');y=d['outcomes'];probabilities={'spatial':p['spatial'].mean(0),'global':p['global'].mean(0),'oracle_geometry':p['oracle_geometry'],'task_time_prior':ref['task_time_prior']};taskloss={}
    for name,prob in probabilities.items():
        label=(2*y[:,0,None]+y[:,1:]).astype(int);nll=-np.log(np.clip(prob[np.arange(186)[:,None],np.arange(3)[None,:],label],1e-8,1)).mean(1);mse=((prob[:,:,1]-prob[:,:,2]-(y[:,1:]-y[:,0,None]))**2).mean(1);per=episode_values(np.stack([nll,mse],1),rows);taskloss[name]=np.array([np.mean([v for k,v in per.items() if k.startswith(f'task{t}_')],0) for t in range(10)])
    for left,right in [('spatial','global'),('spatial','task_time_prior'),('oracle_geometry','task_time_prior')]:
        delta=taskloss[left]-taskloss[right];distribution=delta[draws].mean(1);intervals[left+'_minus_'+right]={'nll_delta':float(delta[:,0].mean()),'gain_mse_delta':float(delta[:,1].mean()),'nll_task_interval':np.quantile(distribution[:,0],[.025,.975]).tolist(),'gain_mse_task_interval':np.quantile(distribution[:,1],[.025,.975]).tolist()}
    # Estimate a prospective repeated-seed queue from actual same-method t60 costs.
    branches=[]
    for w in (0,1):branches+=read(ROOT/f'results/recovery_paired_live_20261014_v3/workers/worker{w}/result.json')['branches']
    rates={}
    for method in ('continue_a','continue_b','resample5'):
        matched=[b for b in branches if b['method']==method and b['branch_step']==60];seconds=np.array([b['elapsed_seconds'] for b in matched]);sizes=[]
        for b in matched:
            # Only flat files inside the already-known completed branch directory.
            sizes.append(sum(x.stat().st_size for x in pathlib.Path(b['output']).iterdir() if x.is_file()))
        rates[method]={'branches':len(matched),'mean_elapsed_seconds':float(seconds.mean()),'p90_elapsed_seconds':float(np.quantile(seconds,.9)),'mean_logical_bytes':float(np.mean(sizes)),'p90_logical_bytes':float(np.quantile(sizes,.9))}
    report=dict(created_at=datetime.datetime.now().astimezone().isoformat(),fit_episodes=140,fit_roots=186,task_clusters=10,geometry_verified=geo,descriptive_task_intervals=intervals,scope='Intervals conditional on the development trajectory; no independent confirmation or multiplicity adjustment. Oracle geometry is not a guaranteed upper bound.',branch_runtime_storage_evidence=rates,elapsed_seconds=time.monotonic()-start)
    (run/'review/statistical_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');ax[0].bar(['Global','Spatial'],[geo['global']['target_error_m']*100,geo['spatial']['target_error_m']*100],color=['#8295a8','#267eaa']);ax[0].set_ylabel('Target position error (cm)');ax[0].set_title('Geometry prediction improved')
    names=['Task/time','Global','Spatial','True geometry'];vals=[r['metrics'][k]['gain_mse'] for k in ('task_time_prior','global_ensemble','spatial_ensemble','oracle_geometry')];ax[1].bar(names,vals,color=['#8295a8','#8295a8','#267eaa','#bababa']);ax[1].set_ylim(0,.19);ax[1].set_ylabel('Paired-gain MSE (lower is better)');ax[1].set_title('Recovery prediction did not pass');ax[1].tick_params(axis='x',labelrotation=15)
    for axis,values in zip(ax,[[geo['global']['target_error_m']*100,geo['spatial']['target_error_m']*100],vals]):
        for j,value in enumerate(values):axis.text(j,value,f'{value:.3f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('140 fit episodes / 186 roots; grouped OOF development only',fontsize=11);fig.savefig(run/'review/geometry_vs_gain.png',dpi=170);plt.close(fig);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
