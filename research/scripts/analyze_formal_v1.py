#!/usr/bin/env python3
"""Recompute complete paired evaluation and training statistics from raw artifacts."""
import argparse, datetime, json, math, os, pathlib, statistics
import numpy as np
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def paired_summary(base,trained):
    assert base['status']==trained['status']=='PASS'
    b={r['pair_id']:r for r in base['episodes']};t={r['pair_id']:r for r in trained['episodes']}
    assert len(b)==len(t)==50 and set(b)==set(t)
    per_task=[]
    for task in range(10):
        ids=[key for key in b if b[key]['task_id']==task];assert len(ids)==5
        for key in ids:
            for field in ['task_id','initial_state_index','env_seed','policy_seed']:assert b[key][field]==t[key][field]
            assert b[key]['status'] in ['TASK_SUCCESS','STEP_LIMIT_REACHED'] and t[key]['status'] in ['TASK_SUCCESS','STEP_LIMIT_REACHED']
        bs=sum(b[k]['task_success'] for k in ids);ts=sum(t[k]['task_success'] for k in ids)
        per_task.append({'task_id':task,'episodes':5,'base_successes':bs,'trained_successes':ts,'paired_difference':(ts-bs)/5})
    differences=np.array([r['paired_difference'] for r in per_task]);rng=np.random.default_rng(20261013)
    boot=differences[rng.integers(0,10,size=(10000,10))].mean(axis=1)
    return {'episodes_per_policy':50,'tasks':10,'per_task':per_task,'base_successes':sum(r['base_successes'] for r in per_task),
      'trained_successes':sum(r['trained_successes'] for r in per_task),'paired_success_difference':float(differences.mean()),
      'task_cluster_bootstrap_95pct_interval':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_seed':20261013,'bootstrap_samples':10000,
      'scope':'Fixed 50/50 spatial evaluation, 10 task clusters, one training seed; interval excludes training-seed uncertainty; not official full benchmark'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--experiment',required=True);a=p.parse_args()
    run=pathlib.Path(a.run_dir);exp=ROOT/'results'/a.experiment
    base=json.loads((run/'base_eval/episodes/result.json').read_text());trained=json.loads((run/'trained_eval/episodes/result.json').read_text())
    summary=paired_summary(base,trained)
    rows={};attempts=[]
    for d in sorted(exp.glob('attempt_*')):
        q=d/'report.json'
        if not q.exists():continue
        report=json.loads(q.read_text());attempts.append({k:report.get(k) for k in ['status','started_at','ended_at','start_step','finished_updates','observed_peak_device_memory_mib']})
        if (d/'metrics.jsonl').exists():
            for line in (d/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line);rows[row['update']]=row
    assert set(rows)==set(range(1,30001)),'Require all 30000 update metrics'
    assert all(math.isfinite(r[k]) for r in rows.values() for k in ['loss','grad_norm','param_norm'])
    training=json.loads((exp/'training_report.json').read_text());assert training['status']=='PASS' and training['finished_updates']==30000 and training['frozen_unchanged']
    checkpoint=json.loads((exp/'checkpoint.json').read_text());assert checkpoint['updates']==30000
    for result in [base,trained]:
        assert result['actual_egl_render_uuid']==result['render_mapping']['selected_gpu']['uuid']
        assert all(pathlib.Path(x['video']).stat().st_size>0 for x in result['episodes'])
    for variant in ['base','trained']:assert json.loads((run/(variant+'_eval')/'service.json').read_text())['cleanup_completed']
    summary.update(status='COMPLETE',experiment=a.experiment,updated_at=datetime.datetime.now().astimezone().isoformat(),
      training_updates=30000,training_attempts=attempts,training_report=str(exp/'training_report.json'),
      mean_update_seconds=statistics.mean(r['update_seconds'] for r in rows.values()),
      first_1000_mean_loss=statistics.mean(rows[k]['loss'] for k in range(1,1001)),last_1000_mean_loss=statistics.mean(rows[k]['loss'] for k in range(29001,30001)),
      positive_control_required=summary['base_successes']==0 and summary['trained_successes']==0,
      no_extra_training_authorized=True)
    (run/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    means=[statistics.mean(rows[k]['loss'] for k in range(start,start+100)) for start in range(1,30001,100)]
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(np.arange(100,30001,100),means);ax.set(xlabel='Optimizer updates',ylabel='Mean loss per 100 updates',title='Spatial LoRA training (seed 42)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(run/'training_loss.png',dpi=160);plt.close(fig)
    table='\n'.join(f"| {x['task_id']} | {x['base_successes']}/5 | {x['trained_successes']}/5 | {x['paired_difference']:+.0%} |" for x in summary['per_task'])
    text=f'''# π0.5 LoRA spatial 首轮正式训练报告

更新：{summary['updated_at']}。状态：训练 30,000 步及两组 50/50 配对评测完成；只覆盖 spatial，训练 seed=42。

使用官方 base 新初始化；432 条完整轨迹、52,970 帧，全局 batch=2、FSDP=2，共 60,000 个训练样本位置（约 1.13 遍数据）。可训练参数 466,957,072，包含视觉编码器，冻结参数校验通过。

| task | 基础模型成功 | 训练模型成功 | 配对差值 |
|---|---:|---:|---:|
{table}

基础模型 {summary['base_successes']}/50，训练模型 {summary['trained_successes']}/50；配对差 {summary['paired_success_difference']:+.1%}，任务聚类 bootstrap 95% 区间 {summary['task_cluster_bootstrap_95pct_interval']}。只有 10 个任务簇和一个训练 seed，不将此区间解释为多训练种子结论。
程序异常和步数上限分开记录，完整视频在 `base_eval/episodes` 与 `trained_eval/episodes`。模型行为成功率与工程运行通过分开判读。

前/后 1000 步平均 loss：{summary['first_1000_mean_loss']:.6f} / {summary['last_1000_mean_loss']:.6f}；同步更新平均 {summary['mean_update_seconds']:.3f} 秒，完整耗时见阶段状态与训练报告。曲线：`training_loss.png`。正式 checkpoint：`{checkpoint['path']}`。

两组均零成功时需要已知有效策略阳性对照：{summary['positive_control_required']}；在该条件下不把阴性结果归因为模型能力或确定的训练失败。任何新增诊断保留用途、来源、体积与预算证据，不改变本轮结果或重新抽样。
未执行其余 LIBERO 套件或完整 benchmark，不与官方全量/全参数训练成绩直接比较；达到 30,000 步不等于收敛。后续增加训练步数需新的明确授权。
'''
    (run/'REPORT.md').write_text(text);(ROOT/'reports/FORMAL_TRAINING_REPORT.md').write_text(text)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
