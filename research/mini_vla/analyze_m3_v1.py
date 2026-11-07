"""Full paired outcomes for preprocessing, visual capacity and pretraining controls."""
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
    variants=plan['model_variants'];sources={'m2':base/'h5',**{v:run/v/'h5' for v in variants}}
    expected={(t,i) for t in range(10) for i in [2,12,22,32,42]};rows={};traces={};metrics={};reports={}
    for variant,directory in sources.items():
        result=json.loads((directory/'result.json').read_text());assert result['status']=='PASS'
        if variant!='m2':
            report=json.loads((run/variant/'training/report.json').read_text());reports[variant]=report
            assert report['epochs']==50 and report['updates']==18400
            assert sha256(run/variant/'training/best.pt')==report['checkpoint_sha256']['best']
        rows[variant]={};traces[variant]={};latency=[];calls=0;steps=0;rpc_total=0.
        for record in result['episodes']:
            key=(record['task_id'],record['initial_state_index']);assert key in expected and key not in rows[variant]
            ep=directory/record['pair_id'];assert sha256(ep/'trace.npz')==record['trace_sha256'] and sha256(ep/'external.mp4')==record['video_sha256']
            if variant!='m2':assert record['checkpoint_sha256']==reports[variant]['checkpoint_sha256']['best'] and record['plan_sha256']==sha256(Path(a.plan))
            with np.load(ep/'trace.npz') as z:trace={k:z[k] for k in z.files}
            assert trace['actions'].shape==(record['steps'],7) and np.isfinite(trace['actions']).all()
            assert record['task_success']==bool(trace['done'].any()) and not trace['done'][:-1].any()
            assert record['task_success'] or record['steps']==220
            np.testing.assert_array_equal(trace['call_steps'],np.arange(0,record['steps'],5))
            np.testing.assert_array_equal(trace['action_chunks'][:,:5].reshape(-1,7)[:record['steps']],trace['actions'])
            assert record['policy_calls']==len(trace['call_steps'])==len(trace['latency'])
            rows[variant][key]=record;traces[variant][key]=trace;latency.extend(trace['latency'][1:]);calls+=record['policy_calls'];steps+=record['steps'];rpc_total+=float(trace['latency'].sum())
        assert set(rows[variant])==expected and len(result['episodes'])==50
        metrics[variant]={'successes':sum(r['task_success'] for r in rows[variant].values()),'episodes':50,'policy_calls':calls,'executed_steps':steps,
                          'rpc_median_ms':float(np.median(latency)*1000),'rpc_seconds_sum':rpc_total}
    contrasts={};render={}
    comparisons=[('normalization','m2','tiny_norm'),('visual_architecture','tiny_norm','resnet_random'),('pretraining','resnet_random','resnet_pretrained')]
    for name,reference,variant in comparisons:
        delta=[];by_task=[];render[name]=[]
        for key in sorted(expected):
            for field in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(traces[reference][key][field],traces[variant][key][field],rtol=0,atol=1e-7)
            if rows[reference][key]['initial_image_sha256']!=rows[variant][key]['initial_image_sha256']:render[name].append(list(key))
            delta.append(int(rows[variant][key]['task_success'])-int(rows[reference][key]['task_success']))
        for task in range(10):by_task.append(np.mean([int(rows[variant][k]['task_success'])-int(rows[reference][k]['task_success']) for k in expected if k[0]==task]))
        rng=np.random.default_rng(20261019);bootstrap=rng.choice(by_task,(10000,10),replace=True).mean(axis=1)
        contrasts[name]={'reference':reference,'variant':variant,'wins':delta.count(1),'losses':delta.count(-1),'difference':float(np.mean(delta)),
                         'task_cluster_descriptive_95_interval':np.quantile(bootstrap,[.025,.975]).tolist()}
    per_task=[{'task':t,**{v:sum(x['task_success'] for k,x in rows[v].items() if k[0]==t) for v in rows}} for t in range(10)]
    result={'status':'COMPLETE_REVIEW_REQUIRED','created_at':now(),'metrics':metrics,'training':reports,'contrasts':contrasts,'per_task':per_task,'initial_render_hash_differences':render,
            'new_episodes':150,'reused_baseline_episodes':50,'checkpoint_selection':'first-action validation MSE, never closed-loop score',
            'claim_boundary':'One seed, known-task 50-case development subset, adaptive component study. Architecture/capacity and pretraining separately compared; no confirmation claim.'}
    write(run/'analysis.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for variant in variants:
        history=json.loads((run/variant/'training/history.json').read_text())
        axes[0].plot([x['epoch'] for x in history],[x['validation']['normalized_mse'] for x in history],label=variant)
    axes[0].set(xlabel='Epoch',ylabel='First-action validation MSE');axes[0].legend(fontsize=8)
    names=['m2']+variants;axes[1].bar(names,[metrics[v]['successes']/50 for v in names]);axes[1].set(ylabel='Success rate',ylim=(0,1));axes[1].tick_params(axis='x',labelrotation=20)
    fig.tight_layout();fig.savefig(run/'performance.png',dpi=160);plt.close(fig)
    text='# M3视觉组件开发比较\n\n|配置|成功/50|策略调用|RPC中位ms|\n|---|---:|---:|---:|\n'
    for v in names:
        m=metrics[v];text+='|%s|%d|%d|%.3f|\n'%(v,m['successes'],m['policy_calls'],m['rpc_median_ms'])
    text+='\n![比较图](performance.png)\n\n三组新模型均使用相同RGB标准化和128分辨率，全部骨干参数参与训练；ImageNet分类头丢弃。完整对照和成本见analysis.json，尚待会话独立审核。\n'
    (run/'REPORT.md').write_text(text);print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
