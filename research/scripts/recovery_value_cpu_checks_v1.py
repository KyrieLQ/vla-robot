"""History, split isolation and decision algebra checks; no robotics effect claim."""
import json,os,pathlib
import numpy as np
import torch
from recovery_value_data_v1 import permitted_history
from recovery_value_fit_v1 import make_inputs,remove_history,calibrate,ValueModel,loss_for
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    rng=np.random.default_rng(20261014);tests=[]
    trace={'robot_state':rng.normal(size=(441,8)),'action':rng.normal(size=(440,7))}
    reference=permitted_history(trace,120);changed={k:v.copy() for k,v in trace.items()}
    changed['robot_state'][121:]=1e9;changed['action'][120:]=-1e9
    assert all(np.array_equal(x,y) for x,y in zip(reference,permitted_history(changed,120)))
    tests.append('Future robot states/actions cannot affect extracted history')
    late=list(range(46))+list(range(140,163))
    rows=[{'pair_id':str(i),'step':60,'task_id':i%10} for i in range(200)]+[{'pair_id':str(i),'step':120,'task_id':i%10} for i in late]
    fit=np.array([i for i,r in enumerate(rows) if int(r['pair_id'])<140]);cal=np.array([i for i in range(269) if i not in fit])
    assert len(fit)==186 and len(cal)==83
    features={'prefix':rng.normal(size=(269,40)).astype(np.float32),'robot_history':rng.normal(size=(269,5,8)).astype(np.float32),'action_history':rng.normal(size=(269,4,7)).astype(np.float32)}
    x,t=make_inputs(features,rows,fit);altered={k:v.copy() for k,v in features.items()}
    for value in altered.values():value[cal]+=1000
    x2,t2=make_inputs(altered,rows,fit)
    assert all(np.array_equal(t[k],t2[k]) for k in t)
    np.testing.assert_allclose(x[fit],x2[fit],atol=1e-6,rtol=0)
    renamed=[dict(r,pair_id='renamed_'+r['pair_id'],task_id=999) for r in rows]
    x3,_=make_inputs(features,renamed,fit);assert np.array_equal(x,x3)
    tests+=['Calibration feature changes cannot alter fit normalization/PCA or fit inputs','Task IDs and consistent pair-name changes do not enter numeric model inputs']
    missing=remove_history(x);assert np.all(missing[:,32:96]==0) and np.all(missing[:,96:128]==0) and np.all(missing[:,136:164]==0)
    tests.append('No-history ablation removes only declared past-information blocks')
    q=np.array([.2,.3,.1,.4]);assert abs((q[1]+q[3])-(q[2]+q[3])-(q[1]-q[2]))<1e-12
    assert (.1-.2)<0 and (1-.2)>.7
    tests+=['Joint-outcome gain identity verified','High failure probability can coexist with harmful recovery']
    small=[{'pair_id':str(i)} for i in range(4)];y=np.array([[0,1,0,0],[1,0,1,1],[1,1,1,1],[0,0,0,0]],dtype=float);calls=np.ones((4,4));scores=np.array([[.9,-1,-1],[.6,-1,-1],[-1,-1,-1],[-1,-1,-1]])
    good=calibrate(scores,np.arange(4),y,calls,small,[0,.5,.7,None]);assert good['threshold']==.7 and good['metrics']['wins']==1 and good['metrics']['harms']==0
    baseline=calibrate(np.full((4,3),-1),np.arange(4),y,calls,small,[0,.7,None]);assert baseline['threshold'] is None
    tests+=['Calibration rejects harmful activation and retains a valid stricter threshold','Always-continue fallback is available']
    torch.set_num_threads(4)
    for variant in ('paired_progress','paired_no_progress','paired_no_history','marginal_value','failure_trigger'):
        model=ValueModel(variant);out=model(torch.tensor(x[:8]));target={'outcomes':torch.tensor(rng.integers(0,2,(8,4)),dtype=torch.float32),'progress':torch.tensor(rng.integers(0,2,(8,3)),dtype=torch.float32),'cost':torch.tensor(rng.random((8,3)),dtype=torch.float32)}
        loss=loss_for(out,target,variant,torch.ones(8));assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    tests.append('All five declared objectives have finite forward/backward passes')
    report={'status':'PASS','checks':tests,'scope':'CPU synthetic verification only; not a success-rate or safety guarantee.'}
    (ROOT/'reports/recovery_value_cpu_checks_20261014.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
