"""Independent full validation audit of M7a; no training updates or simulation."""
import json,time,hashlib,math
from pathlib import Path
import numpy as np
import torch
from mini_vla.spatial_tokens_model_v1 import SpatialTokensMiniVLA as AttentionMiniVLA,SpatialTokensConfig as AttentionConfig

R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_m7a_20261020_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.monotonic();torch.set_num_threads(4);assert torch.cuda.is_available() and torch.cuda.device_count()==1;device=torch.device('cuda:0');torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;out=RUN/'review';out.mkdir(exist_ok=True);assert not (out/'training_acceptance.json').exists()
    pp=R/'configs/mini_vla_m7a_20261020_v1.json';cfg=json.loads(pp.read_text());report=json.loads((RUN/'resnet_pretrained/training/report.json').read_text());history=json.loads((RUN/'resnet_pretrained/training/history.json').read_text())
    for p,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/p)==h,p
    assert sha(R/cfg['data_acceptance'])==cfg['input_sha256'][cfg['data_acceptance']]
    for p,h in cfg['input_sha256'].items():assert sha(R/p)==h,p
    assert len(history)==50 and report['epochs']==50 and report['updates']==94950
    for i,row in enumerate(history,1):
        assert row['epoch']==i and row['updates']==i*1899 and row['train_frames']==243000 and row['valid_target_steps']==2362230
        assert row['validation']['frames']==30465 and row['validation']['valid_target_steps']==296235 and set(row['validation']['per_task'])=={str(t) for t in range(40)}
    selected=min(history,key=lambda x:x['validation']['normalized_mse']);assert selected['epoch']==report['best_epoch']==1
    pre=json.loads((R/cfg['preprocessing']).read_text());checkpoints={};initial=best=None
    for name in ['initial','best','latest']:
        p=RUN/'resnet_pretrained/training'/(name+'.pt');assert sha(p)==report['checkpoint_sha256'][name];ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['plan_sha256']==sha(pp) and ck['seed']==42 and ck['preprocessing']==pre
        assert ck['epoch']==(0 if name=='initial' else selected['epoch'] if name=='best' else 50)
        assert ck['updates']==ck['epoch']*1899
        if name!='initial':assert {int(x['step']) for x in ck['optimizer']['state'].values()}=={ck['updates']}
        checkpoints[name]={'epoch':ck['epoch'],'updates':ck['updates'],'sha256':sha(p)}
        if name=='initial':initial=ck['state_dict']
        if name=='best':best=ck
    model=AttentionMiniVLA(AttentionConfig(**best['model_config']));model.load_state_dict(best['state_dict']);model.to(device);model.eval();assert sum(p.numel() for p in model.parameters())==11604103
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for prefix in ['vision.','language.','proprioception.','encoder.','decoder.','action_queries','action_output.','grid_projection.','spatial_position']:
        assert any(not torch.equal(initial[k],v) for k,v in best['state_dict'].items() if k.startswith(prefix))
    base_initial=torch.load(R/'results/mini_vla_m6_train_20261020_run1/training/initial.pt',map_location='cpu',weights_only=False)['state_dict']
    shared=[k for k in initial if k in base_initial]
    assert shared and all(torch.equal(initial[k],base_initial[k]) for k in shared), 'shared initialization mismatch'
    del base_initial
    cache=R/cfg['cache']/'validation';data={p.stem:np.load(p,mmap_mode='r') for p in cache.glob('*.npy')};n=len(data['state']);assert n==30465
    episode_ids=data['episode_id'];ends={int(e):int(np.flatnonzero(episode_ids==e)[-1]) for e in np.unique(episode_ids)}
    target=np.empty((n,10,7));mask=np.zeros((n,10),bool)
    for i in range(n):
        end=ends[int(episode_ids[i])]
        for k in range(10):
            j=min(i+k,end);assert episode_ids[j]==episode_ids[i];target[i,k]=data['actions'][j,0];mask[i,k]=i+k<=end
    assert mask.sum()==296235
    im=torch.from_numpy(np.array(data['images'][:8])).float()/255
    probe=dict(image=im[:,0].to(device),wrist_image=im[:,1].to(device),**{k:torch.from_numpy(np.array(data[k][:8])).to(device) for k in ['state','token_ids','language_mask']})
    tokens=model.observation_tokens(**probe);assert tokens.shape==(8,34,128)
    model.zero_grad(set_to_none=True);model(**probe).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    sums=chunk=constant=0.;per_task={t:[0.,0] for t in range(40)}
    with torch.no_grad():
        for i in range(0,n,128):
            j=min(i+128,n);im=torch.from_numpy(np.array(data['images'][i:j])).float()/255
            x=dict(image=im[:,0].to(device),wrist_image=im[:,1].to(device),**{k:torch.from_numpy(np.array(data[k][i:j])).to(device) for k in ['state','token_ids','language_mask']})
            pred=model(**x).cpu().numpy().astype('float64');errors=(pred-target[i:j])**2;sums+=errors[:,0].sum();chunk+=(errors*mask[i:j,:,None]).sum();constant+=(target[i:j,0]**2).sum()
            for task,err in zip(data['task_id'][i:j],errors[:,0].mean(1)):per_task[int(task)][0]+=float(err);per_task[int(task)][1]+=1
            if i%3200==0:
                progress={'checked_frames':j,'expected_frames':n,'elapsed_seconds':time.monotonic()-started};(out/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
    values={'normalized_mse':float(sums/(n*7)),'all_valid_chunk_mse':float(chunk/(mask.sum()*7)),'mean_action_normalized_mse':float(constant/(n*7))}
    for k,v in values.items():assert abs(v-report['selected_validation'][k])<2e-5,(k,v,report['selected_validation'][k])
    differences={str(t):dict(frames=count,recorded_frames=report['selected_validation']['per_task'][str(t)]['frames'],mse=s/count,recorded_mse=report['selected_validation']['per_task'][str(t)]['mse'],absolute_error=abs(s/count-report['selected_validation']['per_task'][str(t)]['mse'])) for t,(s,count) in per_task.items()}
    (out/'gpu_per_task_recalculation.json').write_text(json.dumps(differences,indent=2)+'\n')
    for t,(s,count) in per_task.items():assert count==report['selected_validation']['per_task'][str(t)]['frames'] and abs(s/count-report['selected_validation']['per_task'][str(t)]['mse'])<2e-5
    suites={name:sum(per_task[t][0] for t in range(i*10,i*10+10))/sum(per_task[t][1] for t in range(i*10,i*10+10)) for i,name in enumerate(['spatial','object','goal','long'])}
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot([x['epoch'] for x in history],[x['train_mse'] for x in history],label='train chunkMSE');axes[0].plot([x['epoch'] for x in history],[x['validation']['all_valid_chunk_mse'] for x in history],label='validation chunkMSE');axes[0].legend();axes[0].set_xlabel('epoch')
    axes[1].plot([x['epoch'] for x in history],[x['validation']['normalized_mse'] for x in history],label='checkpoint selection: first action');axes[1].axvline(1,color='gray',linestyle='--');axes[1].legend();axes[1].set_xlabel('epoch');fig.tight_layout();fig.savefig(RUN/'resnet_pretrained/training_review.png',dpi=140);plt.close(fig)
    result=dict(status='PASS',epochs=50,updates=94950,train_frames_per_epoch=243000,validation_frames=n,valid_K10_validation_targets=int(mask.sum()),selected_epoch=1,checkpoints=checkpoints,independent_GPU_validation=values,numerical_backend="FP32 CUDA, batch128 matching training validation; independent numpy float64 target/error aggregation",per_suite_validation_mse=suites,all_per_task_values_matched=True,changed_module_groups=['vision','language','state','encoder','decoder','queries','output','grid_projection','spatial_position'],elapsed_seconds=time.monotonic()-started,simulation_episodes=0,confirmation_accessed=False,claim_boundary='M7a full offline validation and gradient/initialization audit; no new updates, closed-loop, or confirmation access.')
    (out/'training_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
