"""Verify complete paired M1 outcomes, report descriptive statistics and curves."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import sha256
from .experiment_v1 import write,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run)
    training=json.loads((run/'training/report.json').read_text())
    assert training['status']=='PASS' and training['epochs']==plan['training']['epochs']
    expected={'task%d_init%d'%(t,i) for t in plan['evaluation']['task_ids'] for i in plan['evaluation']['initial_state_indices']}
    rows={};traces={};latencies={};hashes={};image_mismatches=[]
    for variant in plan['evaluation']['variants']:
        result=json.loads((run/variant/'result.json').read_text());assert result['status']=='PASS'
        rows[variant]={x['pair_id']:x for x in result['episodes']};assert set(rows[variant])==expected
        assert len(result['episodes'])==len(expected)
        latencies[variant]=[];traces[variant]={}
        for key,row in rows[variant].items():
            directory=run/variant/key;file=directory/'trace.npz'
            assert sha256(file)==row['trace_sha256'] and sha256(directory/'external.mp4')==row['video_sha256']
            assert row['checkpoint_sha256']==training['checkpoint_sha256'][variant]
            with np.load(file) as data:
                trace={k:data[k] for k in data.files}
            assert trace['actions'].shape==(row['steps'],7) and np.isfinite(trace['actions']).all()
            assert row['policy_calls']==row['steps'] and len(trace['done'])==row['steps']
            assert bool(trace['done'].any())==row['task_success'] and not trace['done'][:-1].any()
            assert row['task_success'] or row['steps']==plan['evaluation']['max_steps']
            traces[variant][key]=trace;latencies[variant].extend(trace['latency'][1:].tolist())
            hashes[str(file.relative_to(run))]=sha256(file)
    ordered=sorted(expected);difference=[]
    for key in ordered:
        for field in ['initial_qpos','initial_qvel']:
            np.testing.assert_allclose(traces['initial'][key][field],traces['best'][key][field],atol=1e-7,rtol=0)
        if rows['initial'][key]['initial_image_sha256']!=rows['best'][key]['initial_image_sha256']:image_mismatches.append(key)
        difference.append(int(rows['best'][key]['task_success'])-int(rows['initial'][key]['task_success']))
    task_rows=[]
    for task in plan['evaluation']['task_ids']:
        keys=[k for k in ordered if rows['best'][k]['task_id']==task]
        task_rows.append({'task_id':task,'episodes':len(keys),
                          'initial_successes':sum(rows['initial'][k]['task_success'] for k in keys),
                          'trained_successes':sum(rows['best'][k]['task_success'] for k in keys)})
    task_delta=np.array([(x['trained_successes']-x['initial_successes'])/x['episodes'] for x in task_rows])
    rng=np.random.default_rng(plan['analysis']['bootstrap_seed'])
    boot=rng.choice(task_delta,size=(plan['analysis']['bootstrap_samples'],len(task_delta)),replace=True).mean(axis=1)
    rates={v:sum(r['task_success'] for r in rows[v].values())/len(expected) for v in rows}
    report={'status':'COMPLETE_REVIEW_REQUIRED','created_at':now(),'episodes_per_variant':len(expected),'unique_cases':len(expected),
            'training':training,'success_rates':rates,'per_task':task_rows,'paired_wins':difference.count(1),'paired_losses':difference.count(-1),
            'paired_ties':difference.count(0),'difference':float(np.mean(difference)),
            'task_cluster_descriptive_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'initial_physics_paired':'PASS','initial_render_hash_differences':image_mismatches,
            'rpc_latency_seconds':{v:{'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'calls_after_first':len(x)} for v,x in latencies.items()},
            'source_traces':hashes,'evidence_boundary':'50 development cases, one training seed; descriptive interval; no confirmation or unseen-task claim',
            'next_stage':'Inspect curves and success/failure videos; freeze M2 protocol after independent review; do not alter completed M1 results'}
    write(run/'analysis.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=json.loads((run/'training/history.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([r['epoch'] for r in history],[r['train_mse'] for r in history],label='Train')
    axes[0].plot([r['epoch'] for r in history],[r['validation']['normalized_mse'] for r in history],label='Validation')
    axes[0].set(xlabel='Epoch',ylabel='Normalized action MSE');axes[0].legend()
    positions=np.arange(10)
    axes[1].bar(positions-.18,[r['initial_successes']/r['episodes'] for r in task_rows],.36,label='Initial')
    axes[1].bar(positions+.18,[r['trained_successes']/r['episodes'] for r in task_rows],.36,label='Trained')
    axes[1].set(xlabel='Spatial task',ylabel='Success rate',ylim=(0,1),xticks=positions);axes[1].legend()
    fig.tight_layout();fig.savefig(run/'performance.png',dpi=160);plt.close(fig)
    text='# Mini VLA M1 首轮结果\n\n完整配对评测：每个模型50集、10个已知Spatial任务；单训练种子，开发集结果。\n\n'
    text+='训练前成功率：%.1f%%；训练后：%.1f%%；配对净变化：%+.1f个百分点。\n\n'%(100*rates['initial'],100*rates['best'],100*np.mean(difference))
    text+='救回%d、损失%d、持平%d。训练50个epoch，按离线验证MSE选择第%d轮。\n\n'%(difference.count(1),difference.count(-1),difference.count(0),training['best_epoch'])
    text+='|任务|训练前成功/5|训练后成功/5|\n|---|---:|---:|\n'
    for row in task_rows:text+='|%d|%d|%d|\n'%(row['task_id'],row['initial_successes'],row['trained_successes'])
    text+='\n![学习曲线与任务成功率](performance.png)\n\n仍需会话独立审核与失败视频分析；离线误差不能替代闭环效果，不作为未见任务泛化结论。完整指标见analysis.json。\n'
    (run/'REPORT.md').write_text(text)
    print(json.dumps({'status':report['status'],'success_rates':rates,'paired_wins':difference.count(1),'paired_losses':difference.count(-1)}),flush=True)
if __name__=='__main__':main()
