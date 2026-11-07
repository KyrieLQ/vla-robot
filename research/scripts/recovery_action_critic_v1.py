"""Fit-only proposal-conditioned regularized critic; no simulation or calibration scoring."""
import argparse,collections,hashlib,json,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from recovery_value_fit_v1 import make_inputs,episode_weights
from recovery_fit_audit_v1 import metric
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def archive(p):
    with np.load(p) as f:return {k:f[k] for k in f.files}
def legal_chunk(trace,step):
    assert trace['call_step'][0]==step,'Proposal not available at decision time'
    out=trace['action_chunk'][0].copy();assert out.shape==(10,7) and np.isfinite(out).all();return out
def check_math():
    # Exhaustive finite distributions: conditioning on informative C helps the Bayes
    # squared-error optimum; independent C gives exactly zero improvement.
    t=np.array([-1.,-1.,1.,1.]);independent=np.array([-1.,1.,-1.,1.])
    null=np.array([t[independent==c].mean() for c in independent]);informative=t.copy()
    assert np.mean((t-null)**2)==np.mean(t*t)==1 and np.mean((t-informative)**2)==0
    trace={'call_step':np.array([60,65]),'action_chunk':np.arange(140.).reshape(2,10,7)};before=legal_chunk(trace,60)
    trace['action_chunk'][1]=1e9;assert np.array_equal(before,legal_chunk(trace,60))
    try:legal_chunk(trace,55)
    except AssertionError:pass
    else:raise AssertionError('Future chunk was accepted')
    # Query A and R once: selected A/fine/R reuse one call; retreat reuses neither.
    assert [2-int(m in (0,1,2)) for m in range(4)]==[1,1,1,2]
    return dict(status='PASS',null_conditional_information_improvement=0,informative_positive_control_improvement=1,future_chunk_rejected=True,future_mutation_ignored=True,proposal_call_overhead=[1,1,1,2])
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();plan=load(pathlib.Path(a.plan));run=pathlib.Path(a.run);start=time.monotonic()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);checks=check_math();(run/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
    parent=ROOT/plan['parent_pilot'];audit=ROOT/plan['parent_fit_audit'];paired=ROOT/plan['parent_run']
    for rel,h in plan['input_sha256'].items():assert sha(ROOT/rel)==h
    rows=load(parent/'roots.json');fit=np.array([i for i,r in enumerate(rows) if r['partition']=='fit']);assert len(fit)==186
    features=archive(parent/'features.npz');targets=archive(parent/'targets.npz');y=targets['outcomes'][fit];fitrows=[rows[i] for i in fit];w=episode_weights(fitrows)
    branches={}
    for worker in (0,1):
        for b in load(paired/f'workers/worker{worker}/result.json')['branches']:branches[b['root_id'],b['method']]=b
    proposals=[];bindings=[]
    for r in fitrows:
        chunks=[]
        for method in ('continue_a','fine20','resample5'):
            branch=branches[r['root_id'],method];path=pathlib.Path(branch['output'])/'policy_trace.npz';tr=archive(path)
            assert str(tr['image_sha256'][0])==r['image_sha256'];chunks.append(legal_chunk(tr,r['step']))
            bindings.append(dict(root=r['root_id'],method=method,path=str(path),sha256=sha(path),used_call=0,step=r['step']))
        np.testing.assert_allclose(chunks[0],chunks[1],atol=1e-6,rtol=0)
        proposals.append(np.concatenate([chunks[0].ravel(),chunks[2].ravel(),(chunks[2]-chunks[0]).ravel()]))
    proposals=np.asarray(proposals);assert proposals.shape==(186,210)
    np.savez_compressed(run/'fit_root_proposals.npz',proposals=proposals,fit_indices=fit)
    (run/'proposal_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    models=run/'models';models.mkdir();pred={f'{kind}_{l2}':np.full((186,3,4),np.nan) for kind in plan['representations'] for l2 in plan['l2_values']};folds=plan['fold_by_fit_episode'];foldlogs=[]
    for fold in range(5):
        held=np.array([j for j,r in enumerate(fitrows) if folds[r['pair_id']]==fold]);train=np.array([j for j,r in enumerate(fitrows) if folds[r['pair_id']]!=fold])
        assert len({fitrows[j]['pair_id'] for j in held})==28 and not {fitrows[j]['pair_id'] for j in held}&{fitrows[j]['pair_id'] for j in train}
        state,transform=make_inputs(features,rows,fit[train]);mean=proposals[train].mean(0);std=np.maximum(proposals[train].std(0),1e-5);act=(proposals-mean)/std
        np.savez_compressed(run/f'fold{fold}_transform.npz',**transform,proposal_mean=mean,proposal_std=std)
        xs={'state':state[fit],'state_action':np.concatenate([state[fit],act],1).astype('float32'),'action_only':act.astype('float32')}
        labels=torch.tensor((2*y[train,0,None]+y[train,1:]).astype('int64'));weight=torch.tensor(episode_weights([fitrows[j] for j in train]));entries=[]
        for kind in plan['representations']:
            xt=torch.tensor(xs[kind][train]);xv=torch.tensor(xs[kind][held])
            for coefficient in plan['l2_values']:
                model=torch.nn.Linear(xt.shape[-1],12);torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
                optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=150,tolerance_grad=1e-6,tolerance_change=1e-9,line_search_fn='strong_wolfe');evaluations=[0]
                def closure():
                    optimizer.zero_grad(set_to_none=True);logits=model(xt).reshape(-1,3,4);loss=F.cross_entropy(logits.reshape(-1,4),labels.reshape(-1),reduction='none').reshape(-1,3).mean(1)
                    obj=(loss*weight).sum()/weight.sum()+.5*coefficient*sum(p.square().sum() for p in model.parameters());assert torch.isfinite(obj);obj.backward();evaluations[0]+=1;return obj
                optimizer.step(closure)
                with torch.no_grad():prob=model(xv).reshape(-1,3,4).softmax(-1).numpy()
                pred[f'{kind}_{coefficient}'][held]=prob;torch.save(model.state_dict(),models/f'fold{fold}_{kind}_l2{coefficient}.pt');entries.append(dict(kind=kind,l2=coefficient,objective_evaluations=evaluations[0]))
        foldlogs.append(dict(fold=fold,held_episodes=28,held_roots=len(held),models=entries));(run/'progress.json').write_text(json.dumps(dict(status='RUNNING',folds_completed=fold+1,elapsed_seconds=time.monotonic()-start))+'\n');print(f'fold {fold+1}/5',flush=True)
    prior=archive(audit/'fit_only_oof_predictions.npz');assert np.array_equal(prior['fit_indices'],fit)
    # Matched original representation is a real positive control for reuse hygiene.
    for coefficient in plan['l2_values']:np.testing.assert_allclose(pred[f'state_{coefficient}'],prior[f'linear_{coefficient}'],atol=2e-6,rtol=2e-6)
    metrics={k:metric(v,y,w) for k,v in pred.items()};metrics['task_time_prior']=metric(prior['task_time_prior'],y,w)
    selected={kind:min(plan['l2_values'],key=lambda v:(metrics[f'{kind}_{v}']['joint_nll'],-v)) for kind in plan['representations']}
    sa=metrics[f'state_action_{selected["state_action"]}'];base=metrics[f'state_{selected["state"]}'];pr=metrics['task_time_prior'];gate=all(sa[k]<min(base[k],pr[k]) for k in ('joint_nll','gain_mse'))
    np.savez_compressed(run/'fit_only_predictions.npz',**pred,fit_indices=fit)
    result=dict(status='COMPLETE',fit_episodes=140,fit_roots=186,folds=foldlogs,models_trained=60,metrics=metrics,fit_selected_l2=selected,proposal_signal_preparation_gate=gate,state_control_matches_parent=True,calibration_used=0,confirmation_used=0,new_rollouts=0,vla_updates=0,gpu_jobs=0,elapsed_seconds=time.monotonic()-start,scope='Adaptive fit-only representation study; selected OOF metrics are not an independent test or deployment evidence.',next_action='Prepare separately frozen calibration of proposal critic with full query costs and unchanged zero-harm/stability rules; no automatic deployment' if gate else 'Do not calibrate or deploy this proposal critic; preserve negative result and prepare spatially grounded representation/data protocol within original scope')
    (run/'ACTION_CRITIC_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(run/'progress.json').write_text(json.dumps(dict(status='COMPLETE',folds_completed=5))+'\n');print(json.dumps({k:v for k,v in result.items() if k!='folds'}),flush=True)
if __name__=='__main__':main()
