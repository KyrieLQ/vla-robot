"""Small frozen-feature outcome/cost models; fit gradients never use calibration labels."""
import argparse,collections,json,os,pathlib,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def episode_weights(rows):
    counts=collections.Counter(r['pair_id'] for r in rows)
    return np.asarray([1/counts[r['pair_id']] for r in rows],dtype=np.float32)

def make_inputs(features,rows,fit):
    phi=features['prefix'].astype(np.float64)
    mean=phi[fit].mean(0);std=np.maximum(phi[fit].std(0),1e-5)
    standardized=(phi-mean)/std
    _,_,vectors=np.linalg.svd(standardized[fit],full_matrices=False)
    components=vectors[:32];z=standardized@components.T
    zmean=z[fit].mean(0);zstd=np.maximum(z[fit].std(0),1e-5);z=(z-zmean)/zstd
    index={(r['pair_id'],r['step']):i for i,r in enumerate(rows)}
    previous=np.zeros_like(z);mask=np.zeros((len(rows),1))
    for i,r in enumerate(rows):
        if r['step']==120:previous[i]=z[index[(r['pair_id'],60)]];mask[i]=1
    numeric=np.concatenate([features['robot_history'].reshape(len(rows),40),features['action_history'].reshape(len(rows),28)],axis=1)
    nmean=numeric[fit].mean(0);nstd=np.maximum(numeric[fit].std(0),1e-5);numeric=(numeric-nmean)/nstd
    clock=np.asarray([r['step']/220 for r in rows])[:,None]
    x=np.concatenate([z,previous,(z-previous)*mask,numeric,clock,mask],axis=1).astype(np.float32)
    assert x.shape==(269,166) and np.isfinite(x).all()
    transform={'feature_mean':mean,'feature_std':std,'pca_components':components,'projected_mean':zmean,'projected_std':zstd,'numeric_mean':nmean,'numeric_std':nstd}
    return x,transform

def remove_history(x):
    x=x.copy();x[:,32:96]=0;x[:,96:128]=0;x[:,136:164]=0;x[:,165]=0;return x

class ValueModel(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.trunk=nn.Sequential(nn.Linear(166,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,64),nn.SiLU())
        self.progress=nn.Linear(64,3);self.joint=nn.Linear(67,12);self.value=nn.Linear(67,4);self.failure=nn.Linear(67,1);self.cost=nn.Linear(67,3)
        active={'progress':variant!='paired_no_progress','joint':variant.startswith('paired'),'value':variant=='marginal_value','failure':variant.startswith('paired') or variant=='failure_trigger','cost':True}
        for name,on in active.items():
            for parameter in getattr(self,name).parameters():parameter.requires_grad_(on)
    def forward(self,x):
        h=self.trunk(x);progress=self.progress(h)
        cue=torch.zeros_like(progress) if self.variant=='paired_no_progress' else progress.sigmoid()
        h=torch.cat([h,cue],dim=-1)
        return {'progress':progress,'joint':self.joint(h).reshape(-1,3,4),'value':self.value(h),'failure':self.failure(h).squeeze(-1),'cost':2*self.cost(h).sigmoid()}

def loss_for(out,target,variant,weight):
    y=target['outcomes'];cost=F.smooth_l1_loss(out['cost'],target['cost'],reduction='none').mean(1)
    progress=F.binary_cross_entropy_with_logits(out['progress'],target['progress'],reduction='none').mean(1)
    failure=F.binary_cross_entropy_with_logits(out['failure'],1-y[:,0],reduction='none')
    if variant.startswith('paired'):
        label=(2*y[:,0,None]+y[:,1:]).long()
        primary=F.cross_entropy(out['joint'].reshape(-1,4),label.reshape(-1),reduction='none').reshape(-1,3).mean(1)
        per_root=primary+.2*failure+.2*cost
    elif variant=='marginal_value':per_root=F.binary_cross_entropy_with_logits(out['value'],y,reduction='none').mean(1)+.2*cost
    else:per_root=failure+.2*cost
    if variant!='paired_no_progress':per_root=per_root+.1*progress
    return (per_root*weight).sum()/weight.sum()

def action_metrics(choice,indices,y,calls,rows):
    selected=y[indices,choice];base=y[indices,0];w=episode_weights([rows[i] for i in indices]);delta=selected-base
    extra=np.maximum(0,calls[indices,choice]-calls[indices,0]);denom=np.sum(w*calls[indices,0])
    return {'roots':len(indices),'episodes':len({rows[i]['pair_id'] for i in indices}),'successes':int(selected.sum()),'baseline_successes':int(base.sum()),
            'wins':int((delta>0).sum()),'harms':int((delta<0).sum()),'net_wins':int(delta.sum()),
            'episode_weighted_mean_gain':float(np.sum(w*delta)/w.sum()),'interventions':int((choice>0).sum()),
            'positive_extra_call_ratio':float(np.sum(w*extra)/denom),'policy_calls':int(calls[indices,choice].sum()),'baseline_calls':int(calls[indices,0].sum())}

def choices(scores,threshold):
    if threshold is None:return np.zeros(len(scores),dtype=int)
    priority=np.array([1,0,2]) # resample, fine, retreat on exact ties.
    best=priority[np.argmax(scores[:,priority],axis=1)]
    return np.where(scores[np.arange(len(scores)),best]>threshold,best+1,0)

def calibrate(scores,indices,y,calls,rows,thresholds):
    candidates=[]
    for threshold in thresholds:
        selection=choices(scores[indices],threshold);m=action_metrics(selection,indices,y,calls,rows)
        if m['harms']==0 and m['positive_extra_call_ratio']<=.2+1e-12:
            key=(m['episode_weighted_mean_gain'],-m['positive_extra_call_ratio'],-m['interventions'],float('inf') if threshold is None else threshold)
            candidates.append((key,threshold,m))
    assert candidates
    _,threshold,metrics=max(candidates,key=lambda v:v[0]);return {'threshold':threshold,'metrics':metrics}

def prediction_arrays(model,x):
    model.eval()
    with torch.no_grad():
        out=model(torch.as_tensor(x));joint=out['joint'].softmax(-1)
        if model.variant.startswith('paired'):gain=joint[:,:,1]-joint[:,:,2]
        elif model.variant=='marginal_value':q=out['value'].sigmoid();gain=q[:,1:]-q[:,0,None]
        else:gain=out['failure'].sigmoid()[:,None].expand(-1,3)
        return {'gain':gain.numpy(),'cost':out['cost'].numpy(),'progress':out['progress'].sigmoid().numpy(),'failure':out['failure'].sigmoid().numpy()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--run',required=True);a=parser.parse_args()
    run=pathlib.Path(a.run);plan=json.loads(pathlib.Path(a.plan).read_text());start=time.monotonic()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=json.loads((run/'roots.json').read_text());fit=np.array([i for i,r in enumerate(rows) if r['partition']=='fit']);cal=np.array([i for i,r in enumerate(rows) if r['partition']=='calibration'])
    assert len(fit)==186 and len(cal)==83 and not {rows[i]['pair_id'] for i in fit}&{rows[i]['pair_id'] for i in cal}
    with np.load(run/'features.npz') as f:features={k:f[k] for k in f.files}
    with np.load(run/'targets.npz') as f:targets={k:f[k] for k in f.files}
    x,transform=make_inputs(features,rows,fit);np.savez_compressed(run/'feature_transform.npz',**transform)
    y=targets['outcomes'];calls=targets['calls'];fit_weight=episode_weights([rows[i] for i in fit]);weight=torch.tensor(fit_weight)
    train_target={k:torch.tensor(v[fit],dtype=torch.float32) for k,v in targets.items() if k in ('outcomes','cost','progress')}
    fit_fail=y[fit,0]==0;weighted=fit_weight*fit_fail
    fixed_utility=((y[fit,1:]-y[fit,0,None])-.05*targets['cost'][fit])
    fixed_scores=(weighted[:,None]*fixed_utility).sum(0)/weighted.sum()
    priority=[1,0,2];fixed=max(priority,key=lambda i:(fixed_scores[i],-priority.index(i)))
    outdir=run/'models';outdir.mkdir(exist_ok=False)
    summaries={};all_predictions={};completed=0
    for variant in plan['variants']:
        input_x=remove_history(x) if variant=='paired_no_history' else x
        seed_scores=[];seed_results=[]
        for seed in plan['training']['seeds']:
            torch.manual_seed(seed);np.random.seed(seed);model=ValueModel(variant);active=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(active,lr=.001,weight_decay=.001)
            train_x=torch.tensor(input_x[fit]);model.train();tick=time.monotonic();loss_value=None
            for update in range(plan['training']['updates_per_model']):
                optimizer.zero_grad(set_to_none=True);output=model(train_x);loss=loss_for(output,train_target,variant,weight)
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(active,1.0);optimizer.step();loss_value=float(loss.detach())
                if update%200==0:(run/'fit_progress.json').write_text(json.dumps({'status':'RUNNING','variant':variant,'seed':seed,'update':update,'models_completed':completed,'expected_models':15,'elapsed_seconds':time.monotonic()-start})+'\n')
            prediction=prediction_arrays(model,input_x);score=prediction['gain']-.05*prediction['cost']
            if variant=='failure_trigger':
                other=np.ones(3,dtype=bool);other[fixed]=False;score[:,other]=-1e6
            calibrated=calibrate(score,cal,y,calls,rows,plan['calibration']['thresholds'])
            path=outdir/f'{variant}_seed{seed}.pt';torch.save({'state_dict':model.state_dict(),'variant':variant,'seed':seed,'input_dimension':166,'vla_parameters_in_checkpoint':False},path)
            np.savez_compressed(outdir/f'{variant}_seed{seed}_predictions.npz',**prediction)
            seed_scores.append(score);seed_results.append({'seed':seed,'final_loss':loss_value,'trainable_parameters':sum(p.numel() for p in active),
                'updates':1000,'fit_root_count':186,'fit_episode_count':140,'elapsed_seconds':time.monotonic()-tick,'calibration':calibrated})
            completed+=1;print(json.dumps({'model_completed':completed,'expected':15,'variant':variant,'seed':seed,'fit_loss':loss_value}),flush=True)
        stack=np.stack(seed_scores);ensemble=stack.mean(0)-stack.std(0)
        calibration=calibrate(ensemble,cal,y,calls,rows,plan['calibration']['thresholds'])
        selected=choices(ensemble,calibration['threshold']);np.savez_compressed(outdir/f'{variant}_ensemble.npz',scores=ensemble,choices=selected)
        detail={'individual_seeds':seed_results,'ensemble_calibration':calibration,'fit_at_calibrated_threshold':action_metrics(selected[fit],fit,y,calls,rows),
                'calibration_by_step':{},'fixed_recovery_for_failure_variant':plan['methods'][fixed+1] if variant=='failure_trigger' else None}
        for step in (60,120):
            idx=np.array([i for i in cal if rows[i]['step']==step]);detail['calibration_by_step'][str(step)]=action_metrics(selected[idx],idx,y,calls,rows)
        summaries[variant]=detail
    primary=summaries['paired_progress'];metrics=primary['ensemble_calibration']['metrics']
    positives=sum(item['calibration']['metrics']['net_wins']>=1 for item in primary['individual_seeds'])
    gate=metrics['net_wins']>=1 and metrics['harms']==0 and positives>=2 and metrics['positive_extra_call_ratio']<=.2
    result={'status':'COMPLETE','fit_episodes':140,'fit_roots':186,'calibration_episodes':60,'calibration_roots':83,'module_seeds':[11,23,47],
            'variants':summaries,'models_trained':15,'vla_updates':0,'module_updates_total':15000,'confirmation_used':0,'new_simulation_rollouts':0,
            'fit_only_pca':True,'fit_only_optimization':True,'optimizer_used_calibration_labels':False,
            'primary_offline_learnability_gate_passed':bool(gate),'primary_positive_individual_seeds':positives,
            'elapsed_seconds':time.monotonic()-start,'scope':'Offline root-level development calibration; threshold selection uses calibration outcomes. Neither test performance nor online episode success.',
            'next_action':'Prepare frozen closed-loop development and cache-tap identity protocol' if gate else 'Analyze learnability with fixed data; do not change this run or open confirmation',
            'uncertainty_note':'Across-initialization standard deviation is a heuristic penalty, not a certified error bound. Three seeds are module seeds on one frozen VLA.'}
    (run/'PILOT_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (run/'fit_progress.json').write_text(json.dumps({'status':'COMPLETE','models_completed':15,'elapsed_seconds':time.monotonic()-start})+'\n')
    print(json.dumps({'status':'COMPLETE','primary_calibration':metrics,'gate':bool(gate),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
