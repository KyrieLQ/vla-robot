"""Independent CPU validation and paired outcome review of the completed M3."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from mini_vla.visual_model_v1 import VisualMiniVLA,VisualConfig

R=Path(__file__).resolve().parents[1]; RUN=R/'results/mini_vla_m3_20261019_run3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.monotonic();torch.set_num_threads(4);out=RUN/'review';out.mkdir(exist_ok=True)
    assert not (out/'independent_acceptance.json').exists()
    orig=R/'results/mini_vla_m3_20261019_run2';partial=json.loads((orig/'review/partial_acceptance.json').read_text())
    for p,h in partial['bound_files'].items():assert sha(R/p)==h
    for p,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/p)==h
    plan=R/'configs/mini_vla_m3_20261019_v2.json';cfg=json.loads(plan.read_text());cache=R/cfg['cache']/'validation'
    data={p.stem:np.load(p,mmap_mode='r') for p in cache.glob('*.npy')};n=len(data['state']);assert n==5988
    ids=data['episode_id'];end={int(e):int(np.flatnonzero(ids==e)[-1]) for e in np.unique(ids)}
    target=np.empty((n,10,7));mask=np.zeros((n,10),bool)
    for i in range(n):
        for k in range(10):
            j=min(i+k,end[int(ids[i])]);assert ids[j]==ids[i]
            target[i,k]=data['actions'][j,0];mask[i,k]=i+k<=end[int(ids[i])]
    assert mask.sum()==57720
    vals={};grads={};visual_initial={};nonvisual={}
    for v in cfg['model_variants']:
        d=RUN/v/'training';report=json.loads((d/'report.json').read_text())
        ck=torch.load(d/'best.pt',map_location='cpu',weights_only=False);assert ck['plan_sha256']==sha(plan)
        model=VisualMiniVLA(VisualConfig(**ck['model_config']));model.load_state_dict(ck['state_dict']);model.eval()
        assert sum(p.numel() for p in model.parameters())==cfg['expected_parameters'][v]
        sse=chunk=0.
        with torch.no_grad():
            for i in range(0,n,32):
                j=min(i+32,n);im=torch.from_numpy(np.array(data['images'][i:j])).float()/255
                x=dict(image=im[:,0],wrist_image=im[:,1],**{k:torch.from_numpy(np.array(data[k][i:j])) for k in ['state','token_ids','language_mask']})
                pred=model(**x).numpy().astype('float64');err=(pred-target[i:j])**2;sse+=err[:,0].sum();chunk+=(err*mask[i:j,:,None]).sum()
        vals[v]={'first_action_mse':float(sse/(n*7)),'chunk_mse':float(chunk/(mask.sum()*7)),'frames':n}
        assert abs(vals[v]['first_action_mse']-report['selected_validation']['normalized_mse'])<2e-5
        assert abs(vals[v]['chunk_mse']-report['selected_validation']['all_valid_chunk_mse'])<2e-5
        # CPU-only gradient connectivity probe; no optimizer and no checkpoint mutation.
        model.train();im=torch.from_numpy(np.array(data['images'][:2])).float()/255
        pred=model(image=im[:,0],wrist_image=im[:,1],**{k:torch.from_numpy(np.array(data[k][:2])) for k in ['state','token_ids','language_mask']})
        pred.square().mean().backward();params=[(k,p) for k,p in model.named_parameters() if k.startswith('vision.')]
        assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for k,p in params)
        grads[v]=dict(trainable_visual_parameters=len(params),nonzero_gradient_tensors=sum(bool(p.grad.abs().sum()>0) for k,p in params))
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False)['state_dict']
        nonvisual[v]={k:t for k,t in initial.items() if not k.startswith('vision.')}
        visual_initial[v]={k:t for k,t in initial.items() if k.startswith('vision.')}
        print('validated',v,vals[v],flush=True)
        del model,ck
    for v in cfg['model_variants'][1:]:
        for k,t in nonvisual['tiny_norm'].items():torch.testing.assert_close(t,nonvisual[v][k],atol=0,rtol=0)
    for k,t in visual_initial['resnet_random'].items():
        if k.startswith('vision.projection.'):torch.testing.assert_close(t,visual_initial['resnet_pretrained'][k],atol=0,rtol=0)
    from torchvision.models import resnet18
    net=resnet18(weights=None);weights=R/cfg['pretraining']['path'];net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    expected=torch.nn.Sequential(*list(net.children())[:-2]).state_dict()
    for k,t in expected.items():torch.testing.assert_close(t,visual_initial['resnet_pretrained']['vision.backbone.'+k],atol=0,rtol=0)
    dirs={'m2':R/cfg['baseline_run']/'h5',**{v:RUN/v/'h5' for v in cfg['model_variants']}};rows={};initials={};metrics={};hashes={}
    keys={(t,i) for t in range(10) for i in [2,12,22,32,42]}
    for v,d in dirs.items():
        rows[v]={};initials[v]={};latency=[];calls=steps=0
        for p in sorted(d.glob('task*/episode.json')):
            row=json.loads(p.read_text());key=(row['task_id'],row['initial_state_index']);assert key not in rows[v]
            for fname,field in [('trace.npz','trace_sha256'),('external.mp4','video_sha256')]:
                h=sha(p.parent/fname);assert h==row[field];hashes[str((p.parent/fname).relative_to(R))]=h
            if v!='m2':
                assert row['plan_sha256']==sha(plan)
                assert row['checkpoint_sha256']==json.loads((RUN/v/'training/report.json').read_text())['checkpoint_sha256']['best']
            with np.load(p.parent/'trace.npz') as z:
                assert z['actions'].shape==(row['steps'],7) and np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1
                assert bool(z['done'].any())==row['task_success'] and not z['done'][:-1].any()
                assert row['task_success'] or row['steps']==220
                assert z['action_chunks'].shape==(row['policy_calls'],10,7)
                for step in range(row['steps']):np.testing.assert_array_equal(z['actions'][step],z['action_chunks'][step//5,step%5])
                np.testing.assert_array_equal(z['call_steps'],np.arange(0,row['steps'],5))
                assert len(z['latency'])==row['policy_calls'];latency.extend(z['latency'][1:].tolist())
                initials[v][key]=(z['initial_qpos'].copy(),z['initial_qvel'].copy())
            calls+=row['policy_calls'];steps+=row['steps'];rows[v][key]=row
        assert set(rows[v])==keys
        metrics[v]=dict(successes=sum(x['task_success'] for x in rows[v].values()),episodes=50,calls=calls,steps=steps,rpc_median_ms=float(np.median(latency)*1000))
    pairs={};render={}
    for label,a,b in [('normalization','m2','tiny_norm'),('visual_architecture','tiny_norm','resnet_random'),('pretraining','resnet_random','resnet_pretrained'),('combined_vs_m2','m2','resnet_pretrained')]:
        delta=[];bytask=[];render[label]=[]
        for k in sorted(keys):
            for x,y in zip(initials[a][k],initials[b][k]):np.testing.assert_allclose(x,y,atol=1e-7,rtol=0)
            if rows[a][k]['initial_image_sha256']!=rows[b][k]['initial_image_sha256']:render[label].append(list(k))
            delta.append(int(rows[b][k]['task_success'])-int(rows[a][k]['task_success']))
        for t in range(10):bytask.append(np.mean([int(rows[b][k]['task_success'])-int(rows[a][k]['task_success']) for k in keys if k[0]==t]))
        rng=np.random.default_rng(20261019);boot=np.asarray(bytask)[rng.integers(0,10,(10000,10))].mean(1)
        pairs[label]=dict(wins=delta.count(1),losses=delta.count(-1),difference=float(np.mean(delta)),interval=np.quantile(boot,[.025,.975]).tolist())
    original=json.loads((RUN/'analysis.json').read_text())
    for k,v in metrics.items():assert v['successes']==original['metrics'][k]['successes']
    for k,v in pairs.items():
        if k in original['contrasts']:
            o=original['contrasts'][k];assert v['wins']==o['wins'] and v['losses']==o['losses'];np.testing.assert_allclose(v['interval'],o['task_cluster_descriptive_95_interval'])
    rescue=next(k for k in sorted(keys) if not rows['resnet_random'][k]['task_success'] and rows['resnet_pretrained'][k]['task_success'])
    harm=next(k for k in sorted(keys) if rows['resnet_random'][k]['task_success'] and not rows['resnet_pretrained'][k]['task_success'])
    failure=next(k for k in sorted(keys) if k[0]==7 and not rows['resnet_pretrained'][k]['task_success'])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[('resnet_random',rescue),('resnet_pretrained',rescue),('resnet_random',harm),('resnet_pretrained',harm),('resnet_pretrained',failure)]
    fig,axes=plt.subplots(len(selected),3,figsize=(10,3*len(selected)));video=[]
    for i,(v,k) in enumerate(selected):
        row=rows[v][k];reader=imageio.get_reader(dirs[v]/row['pair_id']/'external.mp4')
        for ax,step in zip(axes[i],[0,row['steps']//2,row['steps']]):
            ax.imshow(reader.get_data(step));ax.set_title(f'{v} {k} t={step} success={row["task_success"]}',fontsize=8);ax.axis('off')
        reader.close();video.append(dict(variant=v,key=list(k),success=row['task_success'],steps=row['steps']))
    fig.tight_layout();fig.savefig(out/'video_contact_sheet.png',dpi=110);plt.close(fig)
    result=dict(status='PASS',validation=vals,gradient_connectivity=grads,metrics=metrics,comparisons=pairs,initial_render_mismatches=render,
                input_output_hashes=hashes,videos_selected=video,training_variants=3,episodes_new=150,episodes_baseline=50,
                pretraining_all_backbone_tensors_exact_at_initialization=True,all_actual_action_queues_exact=True,elapsed_seconds=time.monotonic()-start,
                limits='Single training seed, adaptive50-case development. Images inspected as selected keyframes; not exhaustive failure-cause labels. No final confirmation opened.')
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_output_hashes'},indent=2),flush=True)

if __name__=='__main__':main()
