"""Episode-grouped spatial geometry supervision and fixed regularized joint critic."""
import argparse,collections,json,os,pathlib,time
import numpy as np
import torch
import torch.nn.functional as F
from recovery_value_fit_v1 import episode_weights
from recovery_fit_audit_v1 import metric
from paired_live_audit_v1 import arrays
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def weighted_stats(x,w):
    w=w/w.sum();mean=np.sum(x*w[:,None],0);std=np.sqrt(np.sum((x-mean)**2*w[:,None],0));return mean,np.maximum(std,1e-5)
def ridge_fit(x,y,w,coefficient=.1):
    x=x.astype('float64');y=y.astype('float64');w=w.astype('float64');w/=w.sum();xm,xs=weighted_stats(x,w);ym,ys=weighted_stats(y,w);xx=(x-xm)/xs;yy=(y-ym)/ys
    gram=xx.T@(w[:,None]*xx);gram.flat[::len(gram)+1]+=coefficient;beta=np.linalg.solve(gram,xx.T@(w[:,None]*yy));assert np.isfinite(beta).all()
    return dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys,beta=beta)
def ridge_predict(x,model):return (((x-model['x_mean'])/model['x_std'])@model['beta'])*model['y_std']+model['y_mean']
def critic_inputs(geometry,rows,data):
    previous=np.zeros_like(geometry);mask=np.zeros((len(rows),1))
    for i,j in enumerate(data['previous_root']):
        if j>=0:previous[i]=geometry[j];mask[i]=1
    return np.concatenate([geometry,previous,(geometry-previous)*mask,mask,data['robot_history'].reshape(len(rows),40),data['action_history'].reshape(len(rows),28),np.array([r['step']/220 for r in rows])[:,None]],1)
def critic_fit(x,y,w):
    xm,xs=weighted_stats(x.astype('float64'),w.astype('float64'));xx=torch.tensor(((x-xm)/xs).astype('float32'));weight=torch.tensor(w.astype('float32'));label=torch.tensor((2*y[:,0,None]+y[:,1:]).astype('int64'));model=torch.nn.Linear(x.shape[1],12);torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=150,tolerance_grad=1e-6,tolerance_change=1e-9,line_search_fn='strong_wolfe');evaluations=[0]
    def closure():
        optimizer.zero_grad(set_to_none=True);logit=model(xx).reshape(-1,3,4);loss=F.cross_entropy(logit.reshape(-1,4),label.reshape(-1),reduction='none').reshape(-1,3).mean(1);obj=(loss*weight).sum()/weight.sum()+.05*sum(p.square().sum() for p in model.parameters());assert torch.isfinite(obj);obj.backward();evaluations[0]+=1;return obj
    optimizer.step(closure)
    return dict(x_mean=xm,x_std=xs,weight=model.weight.detach().numpy(),bias=model.bias.detach().numpy(),objective_evaluations=np.array(evaluations[0]))
def critic_predict(x,model):
    xx=torch.tensor(((x-model['x_mean'])/model['x_std']).astype('float32'))
    with torch.no_grad():return F.linear(xx,torch.tensor(model['weight']),torch.tensor(model['bias'])).reshape(-1,3,4).softmax(-1).numpy()
def geometry_metric(pred,truth,rows):
    w=episode_weights(rows);error=np.linalg.norm((pred-truth).reshape(-1,2,3),axis=-1)
    return dict(target_xyz_mean_error_m=float(np.sum(w*error[:,0])/w.sum()),goal_xyz_mean_error_m=float(np.sum(w*error[:,1])/w.sum()),by_task={str(t):dict(target_xyz_mean_error_m=float(np.average(error[np.array([r['task_id']==t for r in rows]),0],weights=w[np.array([r['task_id']==t for r in rows])])),goal_xyz_mean_error_m=float(np.average(error[np.array([r['task_id']==t for r in rows]),1],weights=w[np.array([r['task_id']==t for r in rows])]))) for t in sorted({r['task_id'] for r in rows})})
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();plan=json.loads(pathlib.Path(a.plan).read_text());run=pathlib.Path(a.run);start=time.monotonic();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    rows=json.loads((run/'roots.json').read_text());samples=json.loads((run/'samples.json').read_text());data=arrays(run/'prepared_arrays.npz');features=arrays(run/'spatial_features.npz');rootidx=data['root_indices'];y=data['outcomes'];folds=plan['fold_by_fit_episode'];assert len(rows)==186 and len(samples)==1809 and all(r['partition']=='fit' for r in rows)
    expected=arrays(ROOT/plan['parent_pilot']/'features.npz')['prefix'][data['fit_indices']];assert np.array_equal(expected,features['verified_root_pooled'])
    pred={kind:np.full((3,186,3,4),np.nan) for kind in ('spatial','global')};pred['oracle_geometry']=np.full((186,3,4),np.nan);geo={kind:np.full((3,186,6),np.nan) for kind in ('spatial','global')};modeldir=run/'models';modeldir.mkdir();foldlog=[]
    for fold in range(5):
        train=np.array([i for i,r in enumerate(rows) if folds[r['pair_id']]!=fold]);held=np.array([i for i,r in enumerate(rows) if folds[r['pair_id']]==fold]);frame_train=np.array([i for i,r in enumerate(samples) if folds[r['pair_id']]!=fold]);trainids={rows[i]['pair_id'] for i in train};heldids={rows[i]['pair_id'] for i in held}
        assert len(trainids)==112 and len(heldids)==28 and not trainids&heldids and {samples[i]['pair_id'] for i in frame_train}==trainids
        weights=episode_weights([samples[i] for i in frame_train]);rootweight=episode_weights([rows[i] for i in train]);models=[]
        for s,seed in enumerate(plan['critic']['seeds']):
            grid=features['grid'][s];inputs={'spatial':grid.reshape(1809,1024),'global':grid.mean(axis=2).reshape(1809,64)}
            for kind,x in inputs.items():
                geomodel=ridge_fit(x[frame_train],data['geometry'][frame_train],weights);np.savez_compressed(modeldir/f'fold{fold}_{kind}_seed{seed}_geometry.npz',**geomodel)
                estimated=ridge_predict(x[rootidx],geomodel);geo[kind][s,held]=estimated[held];critic_x=critic_inputs(estimated,rows,data);assert critic_x.shape==(186,88)
                model=critic_fit(critic_x[train],y[train],rootweight);pred[kind][s,held]=critic_predict(critic_x[held],model);np.savez_compressed(modeldir/f'fold{fold}_{kind}_seed{seed}_critic.npz',**model);models.append(dict(kind=kind,seed=seed,geometry_train_frames=len(frame_train),critic_train_roots=len(train)))
        privileged=data['geometry'][rootidx];oracle_x=critic_inputs(privileged,rows,data);model=critic_fit(oracle_x[train],y[train],rootweight);pred['oracle_geometry'][held]=critic_predict(oracle_x[held],model);np.savez_compressed(modeldir/f'fold{fold}_oracle_geometry_critic.npz',**model)
        foldlog.append(dict(fold=fold,train_episodes=112,held_episodes=28,held_roots=len(held),geometry_training_frames=len(frame_train),models=models));(run/'fit_progress.json').write_text(json.dumps(dict(status='RUNNING',folds_completed=fold+1,expected=5,elapsed_seconds=time.monotonic()-start))+'\n');print(json.dumps(dict(fold_completed=fold+1,expected=5)),flush=True)
    assert all(np.isfinite(v).all() for v in list(pred.values())+list(geo.values()))
    reference=arrays(ROOT/plan['parent_fit_audit']/'fit_only_oof_predictions.npz');assert np.array_equal(reference['fit_indices'],data['fit_indices']);w=episode_weights(rows);metrics={kind+'_ensemble':metric(pred[kind].mean(0),y,w) for kind in ('spatial','global')};metrics['oracle_geometry']=metric(pred['oracle_geometry'],y,w);metrics['state_linear_0.1']=metric(reference['linear_0.1'],y,w);metrics['task_time_prior']=metric(reference['task_time_prior'],y,w)
    individual={kind:[metric(v,y,w) for v in pred[kind]] for kind in ('spatial','global')};geo_metrics={kind:geometry_metric(geo[kind].mean(0),data['geometry'][rootidx],rows) for kind in ('spatial','global')}
    seedwins=sum(individual['spatial'][i]['gain_mse']<individual['global'][i]['gain_mse'] for i in range(3));gate=all(metrics['spatial_ensemble'][k]<min(metrics['global_ensemble'][k],metrics['task_time_prior'][k]) for k in ('joint_nll','gain_mse')) and seedwins>=2
    np.savez_compressed(run/'oof_predictions.npz',**pred,geometry_spatial=geo['spatial'],geometry_global=geo['global'],fit_indices=data['fit_indices']);result=dict(status='COMPLETE',fit_episodes=140,fit_roots=186,geometry_input_occurrences=1809,folds=foldlog,geometry_models=30,critics=35,seeds=plan['critic']['seeds'],metrics=metrics,individual_seed_metrics=individual,geometry_errors=geo_metrics,spatial_seed_mse_wins_vs_global=seedwins,preparation_gate_passed=gate,calibration_used=0,confirmation_used=0,new_rollouts=0,vla_updates=0,elapsed_seconds=time.monotonic()-start,scope='Adaptive grouped fit-only study; no independent test, calibration or deployed success. Oracle geometry is a nondeployable control, not a guaranteed upper bound.',next_action='Prepare separately frozen calibration and cost/cache protocol; original risk gates still required' if gate else 'Preserve negative result; no calibration or deployment. Consolidate representation/label limitations and remaining budget before any new data protocol.')
    (run/'SPATIAL_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(run/'fit_progress.json').write_text(json.dumps(dict(status='COMPLETE',folds_completed=5,elapsed_seconds=result['elapsed_seconds']))+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('folds','geometry_errors')}),flush=True)
if __name__=='__main__':main()
