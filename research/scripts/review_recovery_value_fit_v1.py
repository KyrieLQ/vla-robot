"""Read-only independent numerical review of completed pilot and grouped fit audit."""
import collections, datetime, hashlib, json, os, pathlib, time
import numpy as np
import torch
from recovery_value_fit_v1 import ValueModel
from recovery_value_data_v1 import load_records

ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
P=ROOT/'results/recovery_value_pilot_20261014_v1'
A=ROOT/'results/recovery_fit_audit_20261014_v1'
def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def npz(p):
    with np.load(p) as f:return {k:f[k] for k in f.files}
def same(a,b,tol=2e-6):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
def weights(rows):
    n=collections.Counter(r['pair_id'] for r in rows)
    return np.array([1/n[r['pair_id']] for r in rows])
def inputs(features,rows,train,saved):
    phi=features['prefix'].astype('float64');num=np.concatenate([features['robot_history'].reshape(269,40),features['action_history'].reshape(269,28)],1)
    for prefix,v in [('feature',phi),('numeric',num)]:
        same(saved[prefix+'_mean'],v[train].mean(0),1e-10);same(saved[prefix+'_std'],np.maximum(v[train].std(0),1e-5),1e-10)
    z=(phi-saved['feature_mean'])/saved['feature_std'];v=saved['pca_components']
    same(v@v.T,np.eye(32),1e-10)
    _,singular,_=np.linalg.svd(z[train],full_matrices=False)
    projected=z@v.T
    same((projected[train]**2).sum(0),singular[:32]**2,1e-8)
    same(saved['projected_mean'],projected[train].mean(0),1e-10);same(saved['projected_std'],np.maximum(projected[train].std(0),1e-5),1e-10)
    z=(projected-saved['projected_mean'])/saved['projected_std'];previous=np.zeros_like(z);mask=np.zeros((269,1));index={(r['pair_id'],r['step']):i for i,r in enumerate(rows)}
    for i,r in enumerate(rows):
        if r['step']==120: previous[i]=z[index[(r['pair_id'],60)]];mask[i]=1
    return np.concatenate([z,previous,(z-previous)*mask,(num-saved['numeric_mean'])/saved['numeric_std'],np.array([r['step']/220 for r in rows])[:,None],mask],1).astype('float32')
def independent_calibration(score,indices,rows,y,calls,grid):
    w=weights([rows[i] for i in indices]);best=None
    for t in grid:
        order=[1,0,2];m=np.array(order)[np.argmax(score[indices][:,order],axis=1)];choice=np.zeros(len(indices),int) if t is None else np.where(score[indices,m]>t,m+1,0)
        delta=y[indices,choice]-y[indices,0];extra=np.maximum(calls[indices,choice]-calls[indices,0],0);cost=(w*extra).sum()/(w*calls[indices,0]).sum()
        if np.any(delta<0) or cost>.2+1e-12:continue
        key=((w*delta).sum()/w.sum(),-cost,-int((choice>0).sum()),float('inf') if t is None else t)
        if best is None or key>best[0]:best=(key,t,dict(wins=int((delta>0).sum()),harms=int((delta<0).sum()),net_wins=int(delta.sum()),interventions=int((choice>0).sum()),positive_extra_call_ratio=float(cost)))
    return best[1:]
def metric(prob,y,rows):
    assert prob.shape==(len(rows),3,4) and np.isfinite(prob).all() and (prob>=0).all();same(prob.sum(-1),1)
    # Average roots within each episode first, then average episodes.
    losses=collections.defaultdict(list)
    for i,r in enumerate(rows):
        nll=[];mse=[]
        for m in range(3):
            c=int(2*y[i,0]+y[i,m+1]);nll.append(-np.log(max(float(prob[i,m,c]),1e-8)))
            mse.append((float(prob[i,m,1]-prob[i,m,2])-float(y[i,m+1]-y[i,0]))**2)
        losses[r['pair_id']].append([np.mean(nll),np.mean(mse)])
    values=np.array([np.mean(v,0) for v in losses.values()]).mean(0)
    return dict(joint_nll=float(values[0]),gain_mse=float(values[1]))
def main():
    start=time.monotonic();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for run in (P,A):
        assert read(run/'status.json')['status']=='COMPLETE'
        for path,h in read(run/'source_manifest.json')['sha256'].items():assert digest(run/'frozen_source'/path)==h and digest(ROOT/path)==h
    plan=read(ROOT/'configs/recovery_value_pilot_20261014_v1.json');aplan=read(ROOT/'configs/recovery_fit_audit_20261014_v1.json')
    for k,h in aplan['feature_files_sha256'].items():assert digest(P/k)==h
    rows=read(P/'roots.json');raw=load_records(plan);features=npz(P/'features.npz');targets=npz(P/'targets.npz');y=targets['outcomes'];calls=targets['calls']
    assert features['prefix'].shape==(269,6144) and np.isfinite(features['prefix']).all()
    for i,r in enumerate(raw):
        assert rows[i]['root_id']==r['root_id'] and rows[i]['partition']==r['partition']
        for key,field in [('outcomes','outcomes'),('calls','calls'),('cost','normalized_positive_extra_calls'),('progress','progress_targets')]:same(targets[key][i],r[field])
        same(features['robot_history'][i],r['robot_history']);same(features['action_history'][i],r['action_history'])
    fit=np.array([i for i,r in enumerate(rows) if r['partition']=='fit']);cal=np.array([i for i,r in enumerate(rows) if r['partition']=='calibration'])
    assert len(fit)==186 and len(cal)==83 and len({rows[i]['pair_id'] for i in fit})==140 and len({rows[i]['pair_id'] for i in cal})==60
    assert not {rows[i]['pair_id'] for i in fit}&{rows[i]['pair_id'] for i in cal}
    x=inputs(features,rows,fit,npz(P/'feature_transform.npz'));pilot=read(P/'PILOT_RESULTS.json');checks=[]
    w=weights([rows[i] for i in fit])*(y[fit,0]==0);u=y[fit,1:]-y[fit,0,None]-.05*targets['cost'][fit];fixed_score=(w[:,None]*u).sum(0)/w.sum();fixed=max([1,0,2],key=lambda m:(fixed_score[m],-[1,0,2].index(m)))
    for variant in plan['variants']:
        xx=x.copy()
        if variant=='paired_no_history':xx[:,32:96]=0;xx[:,96:128]=0;xx[:,136:164]=0;xx[:,165]=0
        scores=[]
        for entry in pilot['variants'][variant]['individual_seeds']:
            seed=entry['seed'];saved=torch.load(P/f'models/{variant}_seed{seed}.pt',weights_only=True,map_location='cpu');assert saved['seed']==seed and not saved['vla_parameters_in_checkpoint']
            model=ValueModel(variant);model.load_state_dict(saved['state_dict']);model.eval()
            with torch.no_grad():out=model(torch.tensor(xx));j=out['joint'].softmax(-1)
            gain=(j[:,:,1]-j[:,:,2]).numpy() if variant.startswith('paired') else ((out['value'].sigmoid()[:,1:]-out['value'].sigmoid()[:,0,None]).numpy() if variant=='marginal_value' else out['failure'].sigmoid()[:,None].expand(-1,3).numpy())
            pred=npz(P/f'models/{variant}_seed{seed}_predictions.npz');same(gain,pred['gain']);same(out['cost'].numpy(),pred['cost'])
            score=gain-.05*pred['cost']
            if variant=='failure_trigger':score[:,[m for m in range(3) if m!=fixed]]=-1e6
            threshold,stats=independent_calibration(score,cal,rows,y,calls,plan['calibration']['thresholds']);assert threshold==entry['calibration']['threshold']
            for k,v in stats.items():same(v,entry['calibration']['metrics'][k])
            scores.append(score)
        ensemble=np.mean(scores,0)-np.std(scores,0);same(ensemble,npz(P/f'models/{variant}_ensemble.npz')['scores'])
        threshold,stats=independent_calibration(ensemble,cal,rows,y,calls,plan['calibration']['thresholds']);expected=pilot['variants'][variant]['ensemble_calibration'];assert threshold==expected['threshold']
        for k,v in stats.items():same(v,expected['metrics'][k])
        checks.append(dict(variant=variant,seeds=3,threshold=threshold,**stats))
    assert not pilot['primary_offline_learnability_gate_passed'] and checks[0]['net_wins']==0
    oof=npz(A/'fit_only_oof_predictions.npz');assert np.array_equal(oof['fit_indices'],fit);loc={i:j for j,i in enumerate(fit)};covered=[];fold_checks=[]
    for fold in range(5):
        held=np.array([i for i in fit if aplan['fold_by_fit_episode'][rows[i]['pair_id']]==fold]);train=np.array([i for i in fit if i not in set(held)])
        assert len({rows[i]['pair_id'] for i in held})==28 and len({rows[i]['pair_id'] for i in train})==112
        assert not {rows[i]['pair_id'] for i in train}&{rows[i]['pair_id'] for i in held}
        xx=inputs(features,rows,train,npz(A/f'fold{fold}_transform.npz'));positions=np.array([loc[i] for i in held]);covered.extend(held.tolist())
        for j,seed in enumerate([11,23,47]):
            model=ValueModel('paired_progress');model.load_state_dict(torch.load(A/f'models/fold{fold}_mlp_seed{seed}.pt',weights_only=True,map_location='cpu'));model.eval()
            with torch.no_grad():pr=model(torch.tensor(xx[held]))['joint'].softmax(-1).numpy()
            same(pr,oof['mlp'][j,positions])
        for coefficient in aplan['linear']['l2_values']:
            model=torch.nn.Linear(166,12);model.load_state_dict(torch.load(A/f'models/fold{fold}_linear_l2{coefficient}.pt',weights_only=True,map_location='cpu'))
            with torch.no_grad():pr=model(torch.tensor(xx[held])).reshape(-1,3,4).softmax(-1).numpy()
            same(pr,oof['linear_'+str(coefficient)][positions])
        trainweight=weights([rows[i] for i in train]);class_y=(2*y[:,0,None]+y[:,1:]).astype(int)
        for pos,index in zip(positions,held):
            subset=[j for j,i in enumerate(train) if rows[i]['task_id']==rows[index]['task_id'] and rows[i]['step']==rows[index]['step']]
            if not subset:subset=[j for j,i in enumerate(train) if rows[i]['task_id']==rows[index]['task_id']]
            if not subset:subset=list(range(len(train)))
            counts=np.ones((3,4))
            for j in subset:
                for m in range(3):counts[m,class_y[train[j],m]]+=trainweight[j]
            same(counts/counts.sum(-1,keepdims=True),oof['task_time_prior'][pos])
        fold_checks.append(dict(fold=fold,held_episodes=28,held_roots=len(held),train_episodes=112,models_replayed=7,fit_only_transform=True))
    assert sorted(covered)==fit.tolist() and len(set(covered))==186
    audit=read(A/'FIT_AUDIT_RESULTS.json');recomputed={}
    for name,expected in audit['metrics'].items():
        pr=oof['mlp'].mean(0) if name=='mlp_ensemble' else oof[name];recomputed[name]=metric(pr,y[fit],[rows[i] for i in fit])
        for key,val in recomputed[name].items():same(val,expected[key])
    assert min(aplan['linear']['l2_values'],key=lambda v:(recomputed['linear_'+str(v)]['joint_nll'],-v))==audit['fit_selected_l2']==.1
    for run,details in [(P,{'models_replayed':15,'calibration_recomputed':checks,'primary_gate_passed':False}),(A,{'models_replayed':35,'fold_checks':fold_checks,'independent_metrics':recomputed,'outer_calibration_scored':False})]:
        out=run/'review';out.mkdir(exist_ok=True);report=dict(status='PASS',reviewed_at=datetime.datetime.now().astimezone().isoformat(),roots=269,fit_roots=186,fit_episodes=140,confirmation_used=0,new_rollouts=0,source_hashes_verified=True,review_script_sha256=digest(pathlib.Path(__file__)),elapsed_seconds=time.monotonic()-start,**details)
        (out/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'PASS','pilot_models':15,'oof_models':35,'fit_roots':186,'fit_episodes':140,'calibration_roots':83,'elapsed_seconds':time.monotonic()-start,'metrics':recomputed}),flush=True)
if __name__=='__main__':main()
