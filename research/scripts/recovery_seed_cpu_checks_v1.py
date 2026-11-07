"""Plan identity, actual RNG wrapper, audit fault injection and seed-matrix controls."""
import collections,json,os,pathlib,tempfile,time
import numpy as np
from paired_live_audit_v1 import sha,arrays,replay_model_inputs,check_saved_branch
from paired_live_cpu_checks_v1 import fixture,ToyPolicy
from root_observation_snapshot_v1 import read_verified_snapshot
from recovery_seed_analyze_v1 import statistics
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    start=time.monotonic();plan=json.loads((ROOT/'configs/recovery_seed_stability_20261014_v1.json').read_text());proposal=json.loads((ROOT/plan['proposal_file']).read_text());approval=json.loads((ROOT/plan['seed_approval_file']).read_text());assert approval['approved'] and sha(ROOT/plan['proposal_file'])==approval['proposal_sha256']==plan['proposal_sha256'];assert sha(ROOT/plan['seed_approval_file'])==plan['seed_approval_sha256'];tests=[]
    for key in ('jobs','policy_seed_roots','selected_pair_ids','policy_replicate_seeds','interpretation'):assert plan[key]==proposal[key]
    assert collections.Counter(j['worker'] for j in plan['jobs'])=={0:65,1:65};assert all(r['module_partition']=='fit' for r in plan['policy_seed_roots']);tests.append('Approved exact20cases60seeded roots130jobs unchanged,65perworker, all fit')
    for root in plan['policy_seed_roots']:
        spec=plan['root_observation_snapshots'][root['root_id']];read_verified_snapshot(spec,spec['policy_image_sha256'])
    tests.append('All60 roots bind preexisting verified same-state lossless observation snapshots')
    with tempfile.TemporaryDirectory(prefix='seed-cpu-',dir=ROOT/'tmp') as td:
        for method in ('continue_a','continue_b','resample5'):
            path=pathlib.Path(td)/method;row,root,p=fixture(path,method);audit=replay_model_inputs(ToyPolicy(),path,row,root,p);assert audit['max_action_chunk_error']==0
            meta=json.loads((path/'policy_inputs.json').read_text());meta['policy_seed']+=1;(path/'policy_inputs.json').write_text(json.dumps(meta))
            try:check_saved_branch(path,row,root,p)
            except AssertionError:pass
            else:raise AssertionError('Wrong seeded metadata accepted')
    tests.append('All3 method input/action cadences pass and wrong seeds rejected')
    import jax
    from serve_recovery_branches_v1 import ReplayPolicy
    assert jax.default_backend()=='cpu'
    class P:
        def __init__(self):self._rng=jax.random.key(0)
        def infer(self,obs):self._rng,draw=jax.random.split(self._rng);return {'actions':np.asarray(jax.random.normal(draw,(10,7)))}
    p=P();wrapped=ReplayPolicy(p);base={'observation/image':np.zeros((224,224,3),dtype='uint8'),'observation/wrist_image':np.zeros((224,224,3),dtype='uint8'),'observation/state':np.zeros(8),'prompt':'synthetic'}
    a=wrapped.infer(dict(base,_policy_seed=2026101401,_policy_rng_advance=12))['actions'];b=wrapped.infer(dict(base,_policy_seed=2026101401,_policy_rng_advance=12))['actions'];c=wrapped.infer(dict(base,_policy_seed=2026101402,_policy_rng_advance=12))['actions'];assert np.array_equal(a,b) and not np.array_equal(a,c)
    fresh=P();fresh._rng=jax.random.key(2026101401)
    for _ in range(12):fresh._rng,_=jax.random.split(fresh._rng)
    assert np.array_equal(a,fresh.infer(base)['actions']);tests.append('Real RNG wrapper repeats identically, new seeds change draws and advance12 is exact on CPU')
    task=np.repeat(np.arange(10),2);a=np.zeros((20,3),int);r=np.zeros_like(a);controls={i:0 for i in range(0,20,2)};assert statistics(a,r,controls,task)['interpretation']=='LOW_OBSERVED_SEED_VARIATION';r[:8,1]=1;assert statistics(a,r,controls,task)['interpretation']=='HIGH_OBSERVED_SEED_VARIATION';controls[0]=1;assert statistics(a,r,controls,task)['interpretation']=='INCONCLUSIVE_IMPLEMENTATION_NOISE';tests.append('Stable null,8varying roots positive, and A/B mismatch gate exercised through same analysis')
    report=dict(status='PASS',created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),checks=tests,elapsed_seconds=time.monotonic()-start,scope='CPU engineering controls; actual twoGPU positive/negative trace replay still required. No experiment outcomes generated.');(ROOT/'reports/recovery_seed_cpu_checks_20261014.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
