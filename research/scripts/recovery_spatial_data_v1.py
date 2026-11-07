"""Align existing consumed images with current geometry, never future observations."""
import argparse,json,os,pathlib,time
import numpy as np
from paired_live_audit_v1 import arrays,sha,image_hash,verify_audit_bindings
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def read(path):return json.loads(pathlib.Path(path).read_text())
def aligned_index(call_step,branch_step,index):
    assert index==int(call_step)-int(branch_step),'Geometry timestamp differs from actual input'
    assert index>=0
    return index
def prepare(plan,run):
    start=time.monotonic();parent=ROOT/plan['parent_run'];pilot=ROOT/plan['parent_pilot'];rows=read(pilot/'roots.json');fit=np.array([i for i,r in enumerate(rows) if r['partition']=='fit']);assert len(fit)==186
    for p,h in plan['input_sha256'].items():assert sha(ROOT/p)==h
    branches={}
    for worker in (0,1):
        for b in read(parent/f'workers/worker{worker}/result.json')['branches']:branches[b['root_id'],b['method']]=b
    samples=[];root_indices=[];targets=[];bindings=[];rootrows=[]
    for i in fit:
        r=rows[i];b=branches[r['root_id'],'continue_a'];directory=pathlib.Path(b['output']);verify_audit_bindings(directory,b)
        tr=arrays(directory/'policy_trace.npz');obs=arrays(directory/'policy_inputs.npz');oracle=arrays(directory/'oracle.npz');meta=read(directory/'oracle_metadata.json');io=read(directory/'policy_inputs.json')
        assert io['prompt']==r['prompt'] and np.array_equal(obs['call_step'],tr['call_step']) and obs['call_step'][0]==r['step']
        target=meta['object_names'].index(meta['target_object']);goal=meta['object_names'].index(meta['destination_object']);root_indices.append(len(samples));rootrows.append(r)
        count=len(obs['call_step']) if r['step']==60 else 1
        for j in range(count):
            step=int(obs['call_step'][j]);k=aligned_index(step,r['step'],step-r['step']);h=image_hash(obs['external'][j],obs['wrist'][j]);assert h==str(tr['image_sha256'][k])
            np.testing.assert_allclose(obs['robot_state'][j],tr['robot_state'][k],atol=1e-6,rtol=0)
            xyz=oracle['object_position'][k,[target,goal]].reshape(6);assert np.isfinite(xyz).all()
            samples.append(dict(pair_id=r['pair_id'],task_id=r['task_id'],source_root=r['root_id'],directory=str(directory),call_index=j,step=step,local_physics_index=k,image_sha256=h,prompt=r['prompt']))
            targets.append(xyz)
        bindings.append(dict(root_id=r['root_id'],calls_used=count,sha256={n:sha(directory/n) for n in ('policy_trace.npz','policy_inputs.npz','policy_inputs.json','oracle.npz','oracle_metadata.json')}))
    assert len(samples)==1809 and len({r['pair_id'] for r in samples})==140 and sum(r['step']==120 for r in rootrows)==46
    features=arrays(pilot/'features.npz');t=arrays(pilot/'targets.npz');mapping={r['root_id']:j for j,r in enumerate(rootrows)}
    previous=np.array([mapping[r['pair_id']+'_t60'] if r['step']==120 else -1 for r in rootrows])
    np.savez_compressed(run/'prepared_arrays.npz',geometry=np.asarray(targets),root_indices=np.asarray(root_indices),fit_indices=fit,previous_root=previous,robot_history=features['robot_history'][fit],action_history=features['action_history'][fit],outcomes=t['outcomes'][fit],original_prefix=features['prefix'][fit])
    (run/'samples.json').write_text(json.dumps(samples,indent=2)+'\n');(run/'roots.json').write_text(json.dumps(rootrows,indent=2)+'\n');(run/'source_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    result=dict(status='PASS',input_occurrences=1809,fit_episodes=140,fit_roots=186,base_calls=1763,additional_original_t120=46,calibration_inputs=0,confirmation_inputs=0,elapsed_seconds=time.monotonic()-start)
    (run/'data_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args();prepare(read(a.plan),pathlib.Path(a.run))
