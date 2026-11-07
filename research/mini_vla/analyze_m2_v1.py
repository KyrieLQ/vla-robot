"""Paired development comparison of M1 K1H1 and M2 K10H1/K10H5."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import sha256
from .experiment_v1 import ROOT,write,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run);base=ROOT/plan['baseline_run']
    assert sha256(base/'review/independent_acceptance.json')==plan['baseline_acceptance_sha256']
    training=json.loads((run/'training/report.json').read_text());assert training['epochs']==50 and training['updates']==18400
    expected={'task%d_init%d'%(t,i) for t in range(10) for i in [2,12,22,32,42]}
    sources={'m1':base/'best','h1':run/'h1','h5':run/'h5'};rows={};traces={};metrics={}
    for variant,directory in sources.items():
        result=json.loads((directory/'result.json').read_text());assert result['status']=='PASS'
        rows[variant]={x['pair_id']:x for x in result['episodes']};assert set(rows[variant])==expected and len(result['episodes'])==50
        traces[variant]={};latencies=[];calls=0;steps=0
        for key,row in rows[variant].items():
            path=directory/key
            assert sha256(path/'trace.npz')==row['trace_sha256'] and sha256(path/'external.mp4')==row['video_sha256']
            with np.load(path/'trace.npz') as z:trace={k:z[k] for k in z.files}
            assert trace['actions'].shape==(row['steps'],7) and np.isfinite(trace['actions']).all()
            assert bool(trace['done'].any())==row['task_success'] and not trace['done'][:-1].any()
            assert row['task_success'] or row['steps']==220
            if variant!='m1':
                h=1 if variant=='h1' else 5
                assert row['checkpoint_sha256']==training['checkpoint_sha256']['best']
                assert row['plan_sha256']==sha256(Path(a.plan))
                np.testing.assert_array_equal(trace['call_steps'],np.arange(0,row['steps'],h))
                executed=trace['action_chunks'][:,:h].reshape(-1,7)[:row['steps']]
                np.testing.assert_array_equal(executed,trace['actions'])
                assert len(trace['call_steps'])==row['policy_calls']==len(trace['latency'])
            else:assert row['policy_calls']==row['steps']
            traces[variant][key]=trace;latencies.extend(trace['latency'][1:].tolist());calls+=row['policy_calls'];steps+=row['steps']
        metrics[variant]={'successes':sum(r['task_success'] for r in rows[variant].values()),'episodes':50,'policy_calls':calls,'executed_steps':steps,
                          'rpc_median_ms':float(np.median(latencies)*1000),'rpc_p95_ms':float(np.quantile(latencies,.95)*1000)}
    comparisons={};image_differences={}
    for name,reference,variant in [('chunk_supervision','m1','h1'),('execution_prefix','h1','h5'),('combined_vs_m1','m1','h5')]:
        delta=[];task_values=[];image_differences[name]=[]
        for key in sorted(expected):
            for field in ['initial_qpos','initial_qvel']:
                np.testing.assert_allclose(traces[reference][key][field],traces[variant][key][field],atol=1e-7,rtol=0)
            if rows[reference][key]['initial_image_sha256']!=rows[variant][key]['initial_image_sha256']:image_differences[name].append(key)
            delta.append(int(rows[variant][key]['task_success'])-int(rows[reference][key]['task_success']))
        for task in range(10):
            keys=[k for k in expected if rows[variant][k]['task_id']==task]
            task_values.append(np.mean([int(rows[variant][k]['task_success'])-int(rows[reference][k]['task_success']) for k in keys]))
        rng=np.random.default_rng(plan['analysis']['bootstrap_seed'])
        bootstrap=rng.choice(task_values,(10000,10),replace=True).mean(axis=1)
        comparisons[name]={'reference':reference,'variant':variant,'wins':delta.count(1),'losses':delta.count(-1),
                           'difference':float(np.mean(delta)),'task_cluster_descriptive_95_interval':np.quantile(bootstrap,[.025,.975]).tolist()}
    per_task=[]
    for task in range(10):
        per_task.append({'task':task,**{v:sum(r['task_success'] for r in rows[v].values() if r['task_id']==task) for v in rows}})
    output={'status':'COMPLETE_REVIEW_REQUIRED','created_at':now(),'metrics':metrics,'training':training,'comparisons':comparisons,
            'per_task':per_task,'paired_initial_physics':'PASS','initial_render_hash_differences':image_differences,
            'executed_action_queue_reconstruction':'EXACT','new_episodes':100,'reused_m1_episodes':50,
            'claim_boundary':'one seed, same 50 development cases; two exploratory component comparisons, not independent confirmation'}
    write(run/'analysis.json',output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((run/'training/history.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([x['epoch'] for x in history],[x['train_mse'] for x in history],label='Train: valid K10 targets')
    axes[0].plot([x['epoch'] for x in history],[x['validation']['normalized_mse'] for x in history],label='Val: first action')
    axes[0].plot([x['epoch'] for x in history],[x['validation']['all_valid_chunk_mse'] for x in history],label='Val: valid K10 targets')
    axes[0].set(xlabel='Epoch',ylabel='Normalized MSE');axes[0].legend(fontsize=8)
    axes[1].bar(['M1 K1H1','M2 K10H1','M2 K10H5'],[metrics[k]['successes']/50 for k in ['m1','h1','h5']])
    axes[1].set(ylabel='Development success rate',ylim=(0,1));fig.tight_layout();fig.savefig(run/'performance.png',dpi=160);plt.close(fig)
    text='# M2动作分块开发评测\n\n|配置|成功/50|策略调用次数|调用中位耗时ms|\n|---|---:|---:|---:|\n'
    for v in ['m1','h1','h5']:
        m=metrics[v];text+='|%s|%d|%d|%.3f|\n'%(v,m['successes'],m['policy_calls'],m['rpc_median_ms'])
    text+='\n![曲线与结果](performance.png)\n\nM1原50集结果复用；M2两个执行前缀使用同一离线选定检查点。区分增加未来动作监督与降低重规划频率，最终归因仍需独立复核和视频检查。\n'
    (run/'REPORT.md').write_text(text);print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
