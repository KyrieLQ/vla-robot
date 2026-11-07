"""Independent complete development50 audit; no simulator/model execution."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/mini_vla_m7a_20261020_run1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def main():
    started=time.monotonic();out=RUN/'review';out.mkdir(exist_ok=True);assert not (out/'independent_acceptance.json').exists()
    pp=ROOT/'configs/mini_vla_m7a_20261020_v1.json';plan=json.loads(pp.read_text())
    for name,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(ROOT/name)==h,name
    for name,h in plan['input_sha256'].items():assert sha(ROOT/name)==h,name
    train=json.loads((RUN/'review/training_acceptance.json').read_text());assert train['status']=='PASS' and train['selected_epoch']==1
    report=json.loads((RUN/'resnet_pretrained/training/report.json').read_text());plan['checkpoint_sha256']=report['checkpoint_sha256']['best']
    expected={(t,i) for t in range(10) for i in [2,12,22,32,42]};dirs={'m6':ROOT/plan['baseline_run']/'resnet_pretrained/h5','m7a':RUN/'resnet_pretrained/h5'}
    rows={};traces={};metrics={};hashes={}
    for v,d in dirs.items():
        results=json.loads((d/'result.json').read_text());assert results['status']=='PASS' and len(results['episodes'])==50
        rows[v]={};traces[v]={};lat=[];calls=steps=0;rpc=0.
        for p in sorted(d.glob('task*/episode.json')):
            e=json.loads(p.read_text());key=(e['task_id'],e['initial_state_index']);assert key in expected and key not in rows[v]
            assert e in results['episodes'];assert e['action_horizon']==10 and e['execute_steps']==5
            for file,field in [('trace.npz','trace_sha256'),('external.mp4','video_sha256')]:
                h=sha(p.parent/file);assert h==e[field];hashes[str((p.parent/file).relative_to(ROOT))]=h
            if v=='m7a':assert e['checkpoint_sha256']==plan['checkpoint_sha256'] and e['plan_sha256']==sha(pp)
            with np.load(p.parent/'trace.npz') as z:trace={k:z[k] for k in z.files}
            assert trace['actions'].shape==(e['steps'],7) and np.isfinite(trace['actions']).all() and np.abs(trace['actions']).max()<=1
            assert bool(trace['done'].any())==e['task_success'] and not trace['done'][:-1].any()
            assert e['task_success'] or e['steps']==220
            assert trace['action_chunks'].shape==(e['policy_calls'],10,7) and np.isfinite(trace['action_chunks']).all()
            assert len(trace['latency'])==e['policy_calls']==(e['steps']+4)//5
            assert np.isfinite(trace['latency']).all() and np.all(trace['latency']>=0)
            np.testing.assert_array_equal(trace['call_steps'],np.arange(0,e['steps'],5))
            for step,action in enumerate(trace['actions']):np.testing.assert_array_equal(action,trace['action_chunks'][step//5,step%5])
            rows[v][key]=e;traces[v][key]=trace;calls+=e['policy_calls'];steps+=e['steps'];lat.extend(trace['latency'][1:].tolist());rpc+=float(trace['latency'].sum())
        assert set(rows[v])==expected
        metrics[v]=dict(successes=sum(e['task_success'] for e in rows[v].values()),episodes=50,calls=calls,steps=steps,rpc_median_ms=float(np.median(lat)*1000),rpc_p95_ms=float(np.quantile(lat,.95)*1000),rpc_total_seconds=rpc)
    changed=[];render=[];wins=[];losses=[];failures=[];physical_error=0.
    for key in sorted(expected):
        for name in ['initial_qpos','initial_qvel']:
            diff=float(np.max(np.abs(traces['m6'][key][name]-traces['m7a'][key][name])));physical_error=max(physical_error,diff);assert diff<=1e-7
        if rows['m6'][key]['initial_image_sha256']!=rows['m7a'][key]['initial_image_sha256']:render.append(key)
        delta=int(rows['m7a'][key]['task_success'])-int(rows['m6'][key]['task_success']);changed.append(delta)
        if delta==1:wins.append(key)
        if delta==-1:losses.append(key)
        if not rows['m7a'][key]['task_success']:failures.append(key)
    per_task=[dict(task=t,m6=sum(e['task_success'] for k,e in rows['m6'].items() if k[0]==t),m7a=sum(e['task_success'] for k,e in rows['m7a'].items() if k[0]==t)) for t in range(10)]
    means=[(x['m7a']-x['m6'])/5 for x in per_task];rng=np.random.default_rng(20261019);boot=np.asarray(means)[rng.integers(0,10,(10000,10))].mean(1)
    paired=dict(wins=len(wins),losses=len(losses),difference=float(np.mean(changed)),task_cluster_descriptive_95_interval=np.quantile(boot,[.025,.975]).tolist(),rescued_cases=wins,regressed_cases=losses)
    original=json.loads((RUN/'analysis.json').read_text());assert paired['wins']==original['paired']['wins'] and paired['losses']==original['paired']['losses'];np.testing.assert_allclose(paired['task_cluster_descriptive_95_interval'],original['paired']['task_cluster_descriptive_95_interval'])
    assert metrics['m7a']['successes']==original['metrics']['spatial_tokens_m7a']['successes']
    chosen=[(v,k) for k in failures for v in ['m6','m7a']]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    video=[]
    for offset in range(0,len(chosen),4):
        page=chosen[offset:offset+4];fig,axes=plt.subplots(len(page),3,figsize=(10,3*len(page)),squeeze=False)
        for i,(v,k) in enumerate(page):
            e=rows[v][k];reader=imageio.get_reader(dirs[v]/e['pair_id']/'external.mp4')
            for ax,step in zip(axes[i],[0,e['steps']//2,e['steps']]):ax.imshow(reader.get_data(step));ax.set_title(f'{v} {k} step{step} success={e["task_success"]}',fontsize=8);ax.axis('off')
            reader.close();video.append(dict(variant=v,case=list(k),success=e['task_success']))
        fig.tight_layout();fig.savefig(out/f'video_contact_sheet_{offset//4+1}.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(['M6 pooled','M7a spatial'],[metrics['m6']['successes']/50,metrics['m7a']['successes']/50]);axes[0].set_ylim(0,1);axes[0].set_ylabel('Spatial development success');x=np.arange(10);axes[1].bar(x-.2,[d['m6'] for d in per_task],.4,label='M6 pooled');axes[1].bar(x+.2,[d['m7a'] for d in per_task],.4,label='M7a spatial');axes[1].set_xticks(x);axes[1].set_ylim(0,5);axes[1].legend();fig.tight_layout();fig.savefig(RUN/'development_comparison.png',dpi=140);plt.close(fig)
    result=dict(status='PASS',new_episodes=50,baseline_episodes=50,selected_epoch=1,selected_checkpoint_sha256=plan['checkpoint_sha256'],metrics=metrics,paired=paired,per_task=per_task,remaining_failure_cases=failures,initial_physics_max_error=physical_error,initial_image_mismatch=render,all_actual_action_queues_exact=True,artifact_hashes=hashes,videos_selected=video,elapsed_seconds=time.monotonic()-started,new_training_updates=0,confirmation_accessed=False,claim_boundary='Adaptive Spatial50, seed42, matched data/updates/normalization. Spatial tokenization and projection capacity jointly change; not parameter-matched and no four-suite success claim.')
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))

if __name__=='__main__':main()
