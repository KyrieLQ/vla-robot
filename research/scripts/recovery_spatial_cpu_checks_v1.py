"""Positive regression, leakage and temporal-alignment controls before real features."""
import json,os,pathlib
import numpy as np
import torch
from recovery_spatial_fit_v1 import ridge_fit,ridge_predict,critic_fit,critic_predict
from recovery_spatial_data_v1 import aligned_index
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    torch.set_num_threads(4);tests=[];grid=np.zeros((16,2,16,32));coords=np.array([[r/3,c/3,0.,r/3,c/3,0.] for r in range(4) for c in range(4)])
    for i in range(16):grid[i,:,i,0]=1
    x=grid.reshape(16,1024);train=np.tile(x,(2,1));y=np.tile(coords,(2,1));model=ridge_fit(train,y,np.ones(32));error=float(np.sqrt(np.mean((ridge_predict(x,model)-coords)**2)));assert error<.04
    globalx=grid.mean(2).reshape(16,64);globalmodel=ridge_fit(np.tile(globalx,(2,1)),y,np.ones(32));globalerror=float(np.sqrt(np.mean((ridge_predict(globalx,globalmodel)-coords)**2)));assert globalerror>error*3
    tests.append('Same ridge code recovers known spatial positions while global pooling loses the toy positional signal')
    held=x+100;combined=np.concatenate([x,held]);before=ridge_fit(combined[:16],coords,np.ones(16));combined[16:]=-1e8;after=ridge_fit(combined[:16],coords,np.ones(16));assert all(np.array_equal(before[k],after[k]) for k in before);tests.append('Held feature mutation cannot change fitted normalization or geometry coefficients')
    assert aligned_index(75,60,15)==15
    try:aligned_index(75,60,16)
    except AssertionError:pass
    else:raise AssertionError('Shifted physics timestamp accepted')
    tests.append('Shifted future physics target index is rejected')
    label=np.tile(np.arange(4),8);xx=np.eye(4)[label];base=label//2;candidate=label%2;outcomes=np.column_stack([base,candidate,candidate,candidate]).astype('float32');cm=critic_fit(xx,outcomes,np.ones(32));prob=critic_predict(np.eye(4),cm);assert np.array_equal(prob.argmax(-1),np.arange(4)[:,None].repeat(3,1));tests.append('Actual joint critic recovers all four known outcome classes on repeated held inputs')
    d=np.array([-1.,1.,-1.,1.]);h=np.array([0,0,1,1]);conditional=np.array([d[h==i].mean() for i in h]);assert np.mean((d-conditional)**2)==np.mean(d*d);tests.append('Zero-information conditional-mean null gives no reduction in population gain MSE')
    tokens=np.arange(256).reshape(1,256,1);pooled=tokens.reshape(1,4,4,4,4,1).mean((2,4)).reshape(4,4);expected=np.array([[tokens.reshape(16,16)[r:r+4,c:c+4].mean() for c in range(0,16,4)] for r in range(0,16,4)]);assert np.array_equal(pooled,expected);tests.append('Row-major ViT token pooling preserves declared two-dimensional cells')
    report=dict(status='PASS',tests=tests,toy_spatial_rmse=error,toy_global_rmse=globalerror,scope='Engineering and mathematical controls, not evidence of real robotics benefit.')
    (ROOT/'reports/recovery_spatial_cpu_checks_20261014.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
