"""Independent complete M4 audit, CPU only, no new trajectories or optimizer step."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from mini_vla.attention_model_v1 import AttentionMiniVLA,AttentionConfig

R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_m4_20261019_run3';BASE=R/'results/mini_vla_m3_20261019_run3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.monotonic();torch.set_num_threads(4);out=RUN/'review';out.mkdir(exist_ok=True);assert not (out/'independent_acceptance.json').exists()
    plan=R/'configs/mini_vla_m4_20261019_v2.json';cfg=json.loads(plan.read_text());assert sha(BASE/'review/independent_acceptance.json')==cfg['baseline_acceptance_sha256']
    for p,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/p)==h,p
    sub=RUN/'resnet_pretrained';d=sub/'training';rep=json.loads((d/'report.json').read_text());history=json.loads((d/'history.json').read_text())
    assert len(history)==50 and rep['epochs']==50 and rep['updates']==18400
    for i,row in enumerate(history,1):
        assert row['epoch']==i and row['updates']==i*368 and row['train_frames']==46982 and row['valid_target_steps']==452540
        assert row['validation']['frames']==5988 and row['validation']['valid_target_steps']==57720
    assert min(history,key=lambda x:x['validation']['normalized_mse'])['epoch']==rep['best_epoch']==36
    for name,h in rep['checkpoint_sha256'].items():assert sha(d/(name+'.pt'))==h
    initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False)
    m3_initial=torch.load(BASE/'resnet_pretrained/training/initial.pt',map_location='cpu',weights_only=False)
    for k,t in m3_initial['state_dict'].items():
        if k.startswith(('vision.','language.','proprioception.')):torch.testing.assert_close(t,initial['state_dict'][k],atol=0,rtol=0)
    ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False);assert ck['plan_sha256']==sha(plan)
    assert {int(s['step']) for s in ck['optimizer']['state'].values()}=={36*368}
    model=AttentionMiniVLA(AttentionConfig(**ck['model_config']));model.load_state_dict(ck['state_dict']);model.eval()
    assert sum(p.numel() for p in model.parameters())==13663943
    assert ck['preprocessing']==json.loads((R/cfg['preprocessing']).read_text())
    data={p.stem:np.load(p,mmap_mode='r') for p in (R/cfg['cache']/'validation').glob('*.npy')};n=len(data['state']);assert n==5988
    ids=data['episode_id'];ends={int(e):int(np.flatnonzero(ids==e)[-1]) for e in np.unique(ids)}
    y=np.empty((n,10,7));mask=np.zeros((n,10),bool)
    for i in range(n):
        for k in range(10):
            j=min(i+k,ends[int(ids[i])]);assert ids[i]==ids[j];y[i,k]=data['actions'][j,0];mask[i,k]=i+k<=ends[int(ids[i])]
    assert mask.sum()==57720
    sse=chunk=0.
    with torch.no_grad():
        for i in range(0,n,32):
            j=min(i+32,n);im=torch.from_numpy(np.array(data['images'][i:j])).float()/255
            x=dict(image=im[:,0],wrist_image=im[:,1],**{k:torch.from_numpy(np.array(data[k][i:j])) for k in ['state','token_ids','language_mask']})
            pred=model(**x).numpy().astype('float64');errors=(pred-y[i:j])**2;sse+=errors[:,0].sum();chunk+=(errors*mask[i:j,:,None]).sum()
    val=dict(frames=n,valid_target_steps=int(mask.sum()),first_action_mse=float(sse/(n*7)),chunk_mse=float(chunk/(mask.sum()*7)))
    assert abs(val['first_action_mse']-rep['selected_validation']['normalized_mse'])<2e-5
    assert abs(val['chunk_mse']-rep['selected_validation']['all_valid_chunk_mse'])<2e-5
    im=torch.from_numpy(np.array(data['images'][:2])).float()/255
    model(image=im[:,0],wrist_image=im[:,1],**{k:torch.from_numpy(np.array(data[k][:2])) for k in ['state','token_ids','language_mask']}).square().mean().backward()
    gradients={}
    for prefix in ['vision.','language.','proprioception.','encoder.','decoder.','action_queries','action_output.']:
        params=[(k,p) for k,p in model.named_parameters() if k.startswith(prefix)];assert params
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for k,p in params)
        assert any(bool(p.grad.abs().sum()>0) for k,p in params)
        assert any(not torch.equal(initial['state_dict'][k],ck['state_dict'][k]) for k,p in params)
        gradients[prefix]=dict(tensors=len(params),nonzero_gradients=sum(bool(p.grad.abs().sum()>0) for k,p in params))
    dirs={'m3':BASE/'resnet_pretrained/h5','m4':sub/'h5'};rows={};traces={};metrics={};hashes={};expected={(t,i) for t in range(10) for i in [2,12,22,32,42]}
    for v,di in dirs.items():
        result=json.loads((di/'result.json').read_text());assert len(result['episodes'])==50;rows[v]={};traces[v]={};lat=[];calls=steps=0
        for p in sorted(di.glob('task*/episode.json')):
            e=json.loads(p.read_text());key=(e['task_id'],e['initial_state_index']);assert key in expected and key not in rows[v]
            for fname,field in [('trace.npz','trace_sha256'),('external.mp4','video_sha256')]:
                h=sha(p.parent/fname);assert h==e[field];hashes[str((p.parent/fname).relative_to(R))]=h
            if v=='m4':assert e['plan_sha256']==sha(plan) and e['checkpoint_sha256']==rep['checkpoint_sha256']['best']
            with np.load(p.parent/'trace.npz') as z:t={k:z[k] for k in z.files}
            assert t['actions'].shape==(e['steps'],7) and np.isfinite(t['actions']).all() and np.abs(t['actions']).max()<=1
            assert bool(t['done'].any())==e['task_success'] and not t['done'][:-1].any()
            assert e['task_success'] or e['steps']==220
            assert t['action_chunks'].shape==(e['policy_calls'],10,7)
            np.testing.assert_array_equal(t['call_steps'],np.arange(0,e['steps'],5))
            for step in range(e['steps']):np.testing.assert_array_equal(t['actions'][step],t['action_chunks'][step//5,step%5])
            assert len(t['latency'])==e['policy_calls'];lat.extend(t['latency'][1:]);calls+=e['policy_calls'];steps+=e['steps'];rows[v][key]=e;traces[v][key]=t
        assert set(rows[v])==expected
        metrics[v]=dict(successes=sum(e['task_success'] for e in rows[v].values()),episodes=50,calls=calls,steps=steps,rpc_median_ms=float(np.median(lat)*1000))
    changes=[];per_task=[];render=[]
    for key in sorted(expected):
        for f in ['initial_qpos','initial_qvel']:np.testing.assert_allclose(traces['m3'][key][f],traces['m4'][key][f],atol=1e-7,rtol=0)
        if rows['m3'][key]['initial_image_sha256']!=rows['m4'][key]['initial_image_sha256']:render.append(key)
        changes.append(int(rows['m4'][key]['task_success'])-int(rows['m3'][key]['task_success']))
    for task in range(10):per_task.append(dict(task=task,**{v:sum(e['task_success'] for k,e in rows[v].items() if k[0]==task) for v in rows}))
    means=[(t['m4']-t['m3'])/5 for t in per_task];boot=np.random.default_rng(20261019).choice(means,(10000,10),replace=True).mean(1)
    paired=dict(wins=changes.count(1),losses=changes.count(-1),difference=float(np.mean(changes)),interval=np.quantile(boot,[.025,.975]).tolist())
    orig=json.loads((RUN/'analysis.json').read_text());assert orig['paired']['wins']==paired['wins'] and orig['paired']['losses']==paired['losses'];np.testing.assert_allclose(paired['interval'],orig['paired']['task_cluster_descriptive_95_interval'])
    rescue=next(k for k in sorted(expected) if not rows['m3'][k]['task_success'] and rows['m4'][k]['task_success']);harm=next(k for k in sorted(expected) if rows['m3'][k]['task_success'] and not rows['m4'][k]['task_success']);fail=next(k for k in sorted(expected) if not rows['m3'][k]['task_success'] and not rows['m4'][k]['task_success'])
    selected=[('m3',rescue),('m4',rescue),('m3',harm),('m4',harm),('m4',fail)]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,3,figsize=(10,15));videos=[]
    for i,(v,k) in enumerate(selected):
        e=rows[v][k];reader=imageio.get_reader(dirs[v]/e['pair_id']/'external.mp4')
        for ax,t in zip(axes[i],[0,e['steps']//2,e['steps']]):ax.imshow(reader.get_data(t));ax.set_title(f'{v} {k} t={t} success={e["task_success"]}',fontsize=8);ax.axis('off')
        reader.close();videos.append(dict(variant=v,key=list(k),success=e['task_success']))
    fig.tight_layout();fig.savefig(out/'video_contact_sheet.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].plot([x['epoch'] for x in history],[x['train_mse'] for x in history],label='train chunk');axes[0].plot([x['epoch'] for x in history],[x['validation']['normalized_mse'] for x in history],label='val first action');axes[0].legend();axes[0].set_xlabel('epoch');axes[1].bar(['M3','M4'],[.88,.88]);axes[1].set_ylim(0,1);axes[1].set_ylabel('development success');fig.tight_layout();fig.savefig(RUN/'performance.png',dpi=140);plt.close(fig)
    result=dict(status='PASS',validation=val,gradients=gradients,metrics=metrics,paired=paired,per_task=per_task,initial_render_mismatches=render,all_executed_actions_match_saved_chunks=True,matched_M3_visual_language_state_initialization=True,training_updates=18400,episodes_new=50,episodes_baseline=50,artifact_hashes=hashes,videos=videos,elapsed_seconds=time.monotonic()-start,claim_boundary='Single seed adaptiveSpatial50. No measured success gain overM3; not a full benchmark or fullACT reproduction.')
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))

if __name__=='__main__':main()
