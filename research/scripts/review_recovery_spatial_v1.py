"""Independent source/normalization/normal-equation/parameter replay acceptance."""
import argparse,collections,datetime,hashlib,json,os,pathlib,time
import numpy as np
from paired_live_audit_v1 import arrays,image_hash,sha
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def read(p):return json.loads(p.read_text())
def near(a,b,tol=2e-6):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
def weight(rows):
    c=collections.Counter(r['pair_id'] for r in rows);return np.array([1/c[r['pair_id']] for r in rows])
def moments(x,w):
    w=w/w.sum();m=(x*w[:,None]).sum(0);return m,np.maximum(np.sqrt(((x-m)**2*w[:,None]).sum(0)),1e-5)
def geom_predict(x,m):return ((x-m['x_mean'])/m['x_std'])@m['beta']*m['y_std']+m['y_mean']
def numeric(geom,rows,d):
    previous=np.zeros_like(geom);mask=np.zeros((186,1))
    for i,p in enumerate(d['previous_root']):
        if p>=0:
            assert rows[p]['step']==60 and rows[i]['step']==120 and rows[p]['pair_id']==rows[i]['pair_id'];previous[i]=geom[p];mask[i]=1
        else:assert rows[i]['step']==60
    return np.concatenate([geom,previous,(geom-previous)*mask,mask,d['robot_history'].reshape(186,40),d['action_history'].reshape(186,28),np.array([r['step']/220 for r in rows])[:,None]],1)
def critic_prediction(x,m):
    z=((x-m['x_mean'])/m['x_std']).astype('float32');logits=(z@m['weight'].T+m['bias']).reshape(-1,3,4);logits-=logits.max(-1,keepdims=True);p=np.exp(logits);return p/p.sum(-1,keepdims=True)
def metrics(p,y,rows):
    assert np.isfinite(p).all() and np.all(p>=0);near(p.sum(-1),1)
    episodes=collections.defaultdict(list)
    for i,r in enumerate(rows):
        nll=[];mse=[]
        for j in range(3):
            nll.append(-np.log(max(float(p[i,j,int(2*y[i,0]+y[i,j+1])]),1e-8)));mse.append((float(p[i,j,1]-p[i,j,2])-float(y[i,j+1]-y[i,0]))**2)
        episodes[r['pair_id']].append([np.mean(nll),np.mean(mse)])
    nll,mse=np.mean([np.mean(a,0) for a in episodes.values()],0);return dict(joint_nll=float(nll),gain_mse=float(mse))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=pathlib.Path(a.run).resolve();start=time.monotonic();status=read(run/'status.json');assert status['status']=='COMPLETE';plan=read(pathlib.Path(status['plan']));result=read(run/'SPATIAL_RESULTS.json')
    for path,h in read(run/'source_manifest.json')['sha256'].items():assert sha(run/'frozen_source'/path)==h
    for path,h in plan['input_sha256'].items():assert sha(ROOT/path)==h
    rows=read(run/'roots.json');samples=read(run/'samples.json');d=arrays(run/'prepared_arrays.npz');f=arrays(run/'spatial_features.npz');oof=arrays(run/'oof_predictions.npz');assert len(samples)==1809 and len(rows)==186 and len({r['pair_id'] for r in rows})==140
    for b in read(run/'source_bindings.json'):
        row=next(r for r in rows if r['root_id']==b['root_id']);directory=pathlib.Path(row['directory'])
        for n,h in b['sha256'].items():assert sha(directory/n)==h
    current=None;record_count=0
    for i,s in enumerate(samples):
        assert s['pair_id'] in plan['fold_by_fit_episode']
        if s['directory']!=current:
            current=s['directory'];directory=pathlib.Path(current);obs=arrays(directory/'policy_inputs.npz');tr=arrays(directory/'policy_trace.npz');oracle=arrays(directory/'oracle.npz');meta=read(directory/'oracle_metadata.json');ids=[meta['object_names'].index(meta[k]) for k in ('target_object','destination_object')]
        j=s['call_index'];k=s['local_physics_index'];assert s['step']==int(obs['call_step'][j])==int(tr['call_step'][j]);source=next(r for r in rows if r['root_id']==s['source_root']);assert k==s['step']-source['step'];assert image_hash(obs['external'][j],obs['wrist'][j])==s['image_sha256']==str(tr['image_sha256'][k]);near(d['geometry'][i],oracle['object_position'][k,ids].reshape(6),1e-10);record_count+=1
    old=arrays(ROOT/plan['parent_pilot']/'features.npz');assert np.array_equal(old['prefix'][d['fit_indices']],f['verified_root_pooled'])
    for j,seed in enumerate(plan['critic']['seeds']):near(f['projection'][j],np.random.default_rng(seed).normal(size=(2048,32)).astype('float32')/np.sqrt(32),1e-10)
    geometry_models=0;critics=0;max_stationarity=0.;covered=[];folds=plan['fold_by_fit_episode'];rootidx=d['root_indices']
    for fold in range(5):
        train=np.array([i for i,r in enumerate(rows) if folds[r['pair_id']]!=fold]);held=np.array([i for i,r in enumerate(rows) if folds[r['pair_id']]==fold]);frames=np.array([i for i,s in enumerate(samples) if folds[s['pair_id']]!=fold]);assert len({rows[i]['pair_id'] for i in held})==28 and not {rows[i]['pair_id'] for i in train}&{rows[i]['pair_id'] for i in held};covered.extend(held.tolist())
        w=weight([samples[i] for i in frames]);w/=w.sum();rw=weight([rows[i] for i in train]);rw/=rw.sum()
        for s,seed in enumerate(plan['critic']['seeds']):
            grid=f['grid'][s]
            for kind,x in [('spatial',grid.reshape(1809,1024)),('global',grid.mean(2).reshape(1809,64))]:
                g=arrays(run/f'models/fold{fold}_{kind}_seed{seed}_geometry.npz');xm,xs=moments(x[frames].astype('float64'),w);ym,ys=moments(d['geometry'][frames],w)
                near(g['x_mean'],xm);near(g['x_std'],xs);near(g['y_mean'],ym);near(g['y_std'],ys)
                xx=(x[frames]-g['x_mean'])/g['x_std'];yy=(d['geometry'][frames]-g['y_mean'])/g['y_std'];gradient=xx.T@(w[:,None]*(xx@g['beta']-yy))+.1*g['beta'];error=float(abs(gradient).max());assert error<1e-6;max_stationarity=max(error,max_stationarity)
                estimated=geom_predict(x[rootidx],g);near(estimated[held],oof['geometry_'+kind][s,held]);geometry_models+=1
                cx=numeric(estimated,rows,d);c=arrays(run/f'models/fold{fold}_{kind}_seed{seed}_critic.npz');m,sd=moments(cx[train],rw);near(c['x_mean'],m);near(c['x_std'],sd);near(critic_prediction(cx[held],c),oof[kind][s,held]);critics+=1
        cx=numeric(d['geometry'][rootidx],rows,d);c=arrays(run/f'models/fold{fold}_oracle_geometry_critic.npz');m,sd=moments(cx[train],rw);near(c['x_mean'],m);near(c['x_std'],sd);near(critic_prediction(cx[held],c),oof['oracle_geometry'][held]);critics+=1
    assert sorted(covered)==list(range(186)) and geometry_models==30 and critics==35
    ref=arrays(ROOT/plan['parent_fit_audit']/'fit_only_oof_predictions.npz');recomputed={}
    choices={'spatial_ensemble':oof['spatial'].mean(0),'global_ensemble':oof['global'].mean(0),'oracle_geometry':oof['oracle_geometry'],'state_linear_0.1':ref['linear_0.1'],'task_time_prior':ref['task_time_prior']}
    for key,pred in choices.items():
        recomputed[key]=metrics(pred,d['outcomes'],rows)
        for k,v in recomputed[key].items():near(v,result['metrics'][key][k])
    seedwins=0
    for s in range(3):
        a=metrics(oof['spatial'][s],d['outcomes'],rows);b=metrics(oof['global'][s],d['outcomes'],rows);seedwins+=a['gain_mse']<b['gain_mse']
    gate=all(recomputed['spatial_ensemble'][k]<min(recomputed['global_ensemble'][k],recomputed['task_time_prior'][k]) for k in ('joint_nll','gain_mse')) and seedwins>=2;assert gate==result['preparation_gate_passed']
    output=run/'review';output.mkdir(exist_ok=True);report=dict(status='PASS',reviewed_at=datetime.datetime.now().astimezone().isoformat(),input_occurrences_verified=record_count,fit_episodes=140,fit_roots=186,geometry_models=geometry_models,critics=critics,geometry_max_stationarity_error=max_stationarity,independent_metrics=recomputed,preparation_gate_passed=bool(gate),calibration_used=0,confirmation_used=0,elapsed_seconds=time.monotonic()-start,review_script_sha256=sha(pathlib.Path(__file__)))
    (output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
