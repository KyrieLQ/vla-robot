"""Independent CPU audit of the completed four-suite M4 training."""
import json,time,hashlib,math
from pathlib import Path
import numpy as np
import torch
from mini_vla.attention_model_v1 import AttentionMiniVLA,AttentionConfig

R=Path(__file__).resolve().parents[1];RUN=R/'results/mini_vla_m6_train_20261020_run1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.monotonic();torch.set_num_threads(4);out=RUN/'review';out.mkdir(exist_ok=True);assert not (out/'independent_acceptance.json').exists()
    pp=R/'configs/mini_vla_m6_train_20261020_v1.json';cfg=json.loads(pp.read_text());report=json.loads((RUN/'training/report.json').read_text());history=json.loads((RUN/'training/history.json').read_text())
    for p,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(R/p)==h,p
    assert sha(R/cfg['data_acceptance'])==cfg['input_sha256'][cfg['data_acceptance']]
    assert len(history)==50 and report['epochs']==50 and report['updates']==94950
    for i,row in enumerate(history,1):
        assert row['epoch']==i and row['updates']==i*1899 and row['train_frames']==243000 and row['valid_target_steps']==2362230
        assert row['validation']['frames']==30465 and row['validation']['valid_target_steps']==296235 and set(row['validation']['per_task'])=={str(t) for t in range(40)}
    selected=min(history,key=lambda x:x['validation']['normalized_mse']);assert selected['epoch']==report['best_epoch']==3
    pre=json.loads((R/cfg['preprocessing']).read_text());checkpoints={};initial=best=None
    for name in ['initial','best','latest']:
        p=RUN/'training'/(name+'.pt');assert sha(p)==report['checkpoint_sha256'][name];ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['plan_sha256']==sha(pp) and ck['seed']==42 and ck['preprocessing']==pre
        assert ck['epoch']==(0 if name=='initial' else selected['epoch'] if name=='best' else 50)
        assert ck['updates']==ck['epoch']*1899
        if name!='initial':assert {int(x['step']) for x in ck['optimizer']['state'].values()}=={ck['updates']}
        checkpoints[name]={'epoch':ck['epoch'],'updates':ck['updates'],'sha256':sha(p)}
        if name=='initial':initial=ck['state_dict']
        if name=='best':best=ck
    model=AttentionMiniVLA(AttentionConfig(**best['model_config']));model.load_state_dict(best['state_dict']);model.eval();assert sum(p.numel() for p in model.parameters())==13666695
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for prefix in ['vision.','language.','proprioception.','encoder.','decoder.','action_queries','action_output.']:
        assert any(not torch.equal(initial[k],v) for k,v in best['state_dict'].items() if k.startswith(prefix))
    cache=R/cfg['cache']/'validation';data={p.stem:np.load(p,mmap_mode='r') for p in cache.glob('*.npy')};n=len(data['state']);assert n==30465
    episode_ids=data['episode_id'];ends={int(e):int(np.flatnonzero(episode_ids==e)[-1]) for e in np.unique(episode_ids)}
    target=np.empty((n,10,7));mask=np.zeros((n,10),bool)
    for i in range(n):
        end=ends[int(episode_ids[i])]
        for k in range(10):
            j=min(i+k,end);assert episode_ids[j]==episode_ids[i];target[i,k]=data['actions'][j,0];mask[i,k]=i+k<=end
    assert mask.sum()==296235
    sums=chunk=constant=0.;per_task={t:[0.,0] for t in range(40)}
    with torch.no_grad():
        for i in range(0,n,32):
            j=min(i+32,n);im=torch.from_numpy(np.array(data['images'][i:j])).float()/255
            x=dict(image=im[:,0],wrist_image=im[:,1],**{k:torch.from_numpy(np.array(data[k][i:j])) for k in ['state','token_ids','language_mask']})
            pred=model(**x).numpy().astype('float64');errors=(pred-target[i:j])**2;sums+=errors[:,0].sum();chunk+=(errors*mask[i:j,:,None]).sum();constant+=(target[i:j,0]**2).sum()
            for task,err in zip(data['task_id'][i:j],errors[:,0].mean(1)):per_task[int(task)][0]+=float(err);per_task[int(task)][1]+=1
            if i%3200==0:
                progress={'checked_frames':j,'expected_frames':n,'elapsed_seconds':time.monotonic()-started};(out/'progress.json').write_text(json.dumps(progress)+'\n');print(json.dumps(progress),flush=True)
    values={'normalized_mse':float(sums/(n*7)),'all_valid_chunk_mse':float(chunk/(mask.sum()*7)),'mean_action_normalized_mse':float(constant/(n*7))}
    for k,v in values.items():assert abs(v-report['selected_validation'][k])<2e-5,(k,v,report['selected_validation'][k])
    for t,(s,count) in per_task.items():assert count==report['selected_validation']['per_task'][str(t)]['frames'] and abs(s/count-report['selected_validation']['per_task'][str(t)]['mse'])<2e-5
    suites={name:sum(per_task[t][0] for t in range(i*10,i*10+10))/sum(per_task[t][1] for t in range(i*10,i*10+10)) for i,name in enumerate(['spatial','object','goal','long'])}
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].plot([x['epoch'] for x in history],[x['train_mse'] for x in history],label='train chunkMSE');axes[0].plot([x['epoch'] for x in history],[x['validation']['all_valid_chunk_mse'] for x in history],label='validation chunkMSE');axes[0].legend();axes[0].set_xlabel('epoch')
    axes[1].plot([x['epoch'] for x in history],[x['validation']['normalized_mse'] for x in history],label='checkpoint selection: first action');axes[1].axvline(3,color='gray',linestyle='--');axes[1].legend();axes[1].set_xlabel('epoch');fig.tight_layout();fig.savefig(RUN/'training_review.png',dpi=140);plt.close(fig)
    result=dict(status='PASS',epochs=50,updates=94950,train_frames_per_epoch=243000,validation_frames=n,valid_K10_validation_targets=int(mask.sum()),selected_epoch=3,checkpoints=checkpoints,independent_CPU_validation=values,per_suite_validation_mse=suites,all_per_task_values_matched=True,changed_module_groups=['vision','language','state','encoder','decoder','queries','output'],elapsed_seconds=time.monotonic()-started,simulation_episodes=0,confirmation_accessed=False,claim_boundary='Offline four-suite training acceptance only; normalizedMSE is not success rate and is not directly comparable toSpatial-only normalization.')
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
