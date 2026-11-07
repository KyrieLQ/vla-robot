"""Fit-only grouped generalization audit; no calibration-label scoring or tuning."""
import argparse,json,os,pathlib,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recovery_value_fit_v1 import make_inputs,episode_weights,ValueModel,loss_for
from paired_live_audit_v1 import sha
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])

def metric(probability,y,weight):
    classes=(2*y[:,0,None]+y[:,1:]).astype(int)
    actual=probability[np.arange(len(y))[:,None],np.arange(3)[None,:],classes]
    nll=-np.log(np.clip(actual,1e-8,1)).mean(1)
    gain=probability[:,:,1]-probability[:,:,2];truth=y[:,1:]-y[:,0,None]
    mse=((gain-truth)**2).mean(1)
    return {'joint_nll':float((weight*nll).sum()/weight.sum()),'gain_mse':float((weight*mse).sum()/weight.sum())}

def prior_predict(rows,train,held,y,weights):
    answer=[]
    classes=(2*y[:,0,None]+y[:,1:]).astype(int)
    for index in held:
        group=np.array([i for i in train if rows[i]['task_id']==rows[index]['task_id'] and rows[i]['step']==rows[index]['step']])
        if not len(group):group=np.array([i for i in train if rows[i]['task_id']==rows[index]['task_id']])
        if not len(group):group=train
        counts=np.ones((3,4),dtype=np.float64)
        for i in group:
            for method in range(3):counts[method,classes[i,method]]+=weights[i]
        answer.append(counts/counts.sum(1,keepdims=True))
    return np.stack(answer)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--run',required=True);a=parser.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());run=pathlib.Path(a.run);parent=ROOT/plan['parent_pilot'];start=time.monotonic()
    for name,digest in plan['feature_files_sha256'].items():assert sha(parent/name)==digest
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=json.loads((parent/'roots.json').read_text());fit=np.array([i for i,r in enumerate(rows) if r['partition']=='fit'])
    assert len(fit)==186 and {rows[i]['pair_id'] for i in fit}==set(plan['fold_by_fit_episode'])
    with np.load(parent/'features.npz') as f:features={k:f[k] for k in f.files}
    with np.load(parent/'targets.npz') as f:targets={k:f[k] for k in f.files}
    y=targets['outcomes'];all_weights=np.zeros(len(rows),dtype=np.float32);all_weights[fit]=episode_weights([rows[i] for i in fit])
    locations={index:j for j,index in enumerate(fit)};weights=all_weights[fit]
    predictions={'mlp':np.full((3,len(fit),3,4),np.nan),'task_time_prior':np.full((len(fit),3,4),np.nan)}
    for value in plan['linear']['l2_values']:predictions['linear_'+str(value)]=np.full((len(fit),3,4),np.nan)
    fold_log=[];model_dir=run/'models';model_dir.mkdir()
    for fold in range(5):
        held=np.array([i for i in fit if plan['fold_by_fit_episode'][rows[i]['pair_id']]==fold]);train=np.array([i for i in fit if i not in set(held)])
        assert len({rows[i]['pair_id'] for i in held})==28 and not {rows[i]['pair_id'] for i in held}&{rows[i]['pair_id'] for i in train}
        assert all(rows[i]['partition']=='fit' for i in list(train)+list(held))
        x,transform=make_inputs(features,rows,train);np.savez_compressed(run/f'fold{fold}_transform.npz',**transform)
        xt=torch.tensor(x[train]);xv=torch.tensor(x[held]);weight=torch.tensor(episode_weights([rows[i] for i in train]));target={k:torch.tensor(targets[k][train],dtype=torch.float32) for k in ('outcomes','cost','progress')}
        positions=np.array([locations[i] for i in held]);entries={'fold':fold,'train_episodes':112,'held_episodes':28,'held_roots':len(held),'models':[]}
        for j,seed in enumerate(plan['mlp']['seeds']):
            torch.manual_seed(seed);model=ValueModel('paired_progress');optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
            model.train()
            for _ in range(1000):
                optimizer.zero_grad(set_to_none=True);loss=loss_for(model(xt),target,'paired_progress',weight);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            model.eval()
            with torch.no_grad():prob=model(xv)['joint'].softmax(-1).numpy()
            predictions['mlp'][j,positions]=prob
            torch.save(model.state_dict(),model_dir/f'fold{fold}_mlp_seed{seed}.pt')
            entries['models'].append({'model':'mlp','seed':seed,'fit_loss':float(loss.detach())})
        label=(2*target['outcomes'][:,0,None]+target['outcomes'][:,1:]).long()
        for coefficient in plan['linear']['l2_values']:
            model=nn.Linear(166,12);nn.init.zeros_(model.weight);nn.init.zeros_(model.bias)
            optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=150,tolerance_grad=1e-6,tolerance_change=1e-9,line_search_fn='strong_wolfe')
            evaluations=[0]
            def closure():
                optimizer.zero_grad(set_to_none=True);logits=model(xt).reshape(-1,3,4)
                losses=F.cross_entropy(logits.reshape(-1,4),label.reshape(-1),reduction='none').reshape(-1,3).mean(1)
                objective=(losses*weight).sum()/weight.sum()+.5*coefficient*sum(p.square().sum() for p in model.parameters())
                assert torch.isfinite(objective);objective.backward();evaluations[0]+=1;return objective
            optimizer.step(closure)
            with torch.no_grad():prob=model(xv).reshape(-1,3,4).softmax(-1).numpy()
            predictions['linear_'+str(coefficient)][positions]=prob
            torch.save(model.state_dict(),model_dir/f'fold{fold}_linear_l2{coefficient}.pt')
            entries['models'].append({'model':'linear','l2':coefficient,'objective_evaluations':evaluations[0]})
        predictions['task_time_prior'][positions]=prior_predict(rows,train,held,y,all_weights)
        fold_log.append(entries);(run/'audit_progress.json').write_text(json.dumps({'status':'RUNNING','folds_completed':fold+1,'expected':5,'elapsed_seconds':time.monotonic()-start})+'\n');print(json.dumps({'fold_completed':fold+1,'expected':5}),flush=True)
    assert all(np.isfinite(value).all() for value in predictions.values())
    np.savez_compressed(run/'fit_only_oof_predictions.npz',**predictions,fit_indices=fit)
    report={'status':'COMPLETE','fit_episodes':140,'fit_roots':186,'folds':fold_log,'all_outer_calibration_excluded':True,'confirmation_used':0,'new_rollouts':0,'vla_updates':0,
            'metrics':{'mlp_ensemble':metric(predictions['mlp'].mean(0),y[fit],weights),'task_time_prior':metric(predictions['task_time_prior'],y[fit],weights)},
            'mlp_individual_seeds':[metric(p,y[fit],weights) for p in predictions['mlp']],
            'zero_gain_mse':float(np.sum(weights*((y[fit,1:]-y[fit,0,None])**2).mean(1))/weights.sum()),
            'elapsed_seconds':time.monotonic()-start,'scope':'Fit-only model-capacity diagnosis. Lambda selection uses OOF fit loss; not independent test or online performance.'}
    for coefficient in plan['linear']['l2_values']:report['metrics']['linear_'+str(coefficient)]=metric(predictions['linear_'+str(coefficient)],y[fit],weights)
    chosen=min(plan['linear']['l2_values'],key=lambda value:(report['metrics']['linear_'+str(value)]['joint_nll'],-value));report['fit_selected_l2']=chosen
    report['selected_linear_beats_mlp_and_prior_in_oof_nll']=report['metrics']['linear_'+str(chosen)]['joint_nll']<min(report['metrics']['mlp_ensemble']['joint_nll'],report['metrics']['task_time_prior']['joint_nll'])
    (run/'FIT_AUDIT_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    (run/'audit_progress.json').write_text(json.dumps({'status':'COMPLETE','folds_completed':5,'elapsed_seconds':report['elapsed_seconds']})+'\n');print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':main()
