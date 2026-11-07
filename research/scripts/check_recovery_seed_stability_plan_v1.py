"""Validate the concrete proposed queue without simulations or outcome-driven selection."""
import collections,hashlib,itertools,json,os,pathlib
import numpy as np
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    p=ROOT/'configs/recovery_seed_stability_20261014_v1_PROPOSED.json';plan=json.loads(p.read_text());source=ROOT/plan['original_root_manifest'];assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['original_root_manifest_sha256'];roots=json.loads(source.read_text())['roots'];fit=[r for r in roots if r['module_partition']=='fit' and r['branch_step']==60];chosen=[]
    for task in range(10):chosen+=sorted([r for r in fit if r['task_id']==task],key=lambda r:hashlib.sha256(('seed_stability_20261014:'+r['pair_id']).encode()).hexdigest())[:2]
    assert [r['pair_id'] for r in chosen]==plan['selected_pair_ids'];assert len(plan['jobs'])==130 and len(plan['policy_seed_roots'])==60;lookup={r['root_id']:r for r in plan['policy_seed_roots']};actual=collections.Counter()
    seeds=set()
    for r in plan['policy_seed_roots']:
        assert r['source_root_id'] in {v['root_id'] for v in chosen};original=next(v for v in chosen if v['root_id']==r['source_root_id']);assert r['source_trace_sha256']==original['source_trace_sha256'] and r['source_oracle_sha256']==original['source_oracle_sha256']
        for name,field in [('continue','policy_seed'),('resample','resample_seed')]:
            value=int(hashlib.sha256(f'{r["replicate_seed"]}:{r["pair_id"]}:{name}'.encode()).hexdigest()[:8],16)%2147483647;assert r[field]==value;seeds.add(value)
    assert len(seeds)==120
    for job in plan['jobs']:
        assert lookup[job['root_id']]['worker']==job['worker'];actual[job['method']]+=1;assert job['pair_id'] in plan['selected_pair_ids']
    assert actual==dict(continue_a=60,resample5=60,continue_b=10);workers=collections.Counter(j['worker'] for j in plan['jobs']);assert workers=={0:65,1:65}
    triples=np.array(list(itertools.product([-1.,0.,1.],repeat=3)));assert abs(triples.var(axis=1,ddof=1).mean()-2/3)<1e-12 and abs(triples.mean(1).var()-2/9)<1e-12
    resource=plan['resources'];assert resource['paired_limit_approved'] is False and resource['prior_paired_active_seconds']+2100<16200 and resource['prior_recovery_active_seconds']+2100<28800
    result=dict(status='PASS',proposal_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fit_episodes=20,task_counts={str(t):sum(v['task_id']==t for v in chosen) for t in range(10)},roots=60,branches=130,workers=dict(workers),methods=dict(actual),seed_values=120,physics_controls=70,no_calibration_or_confirmation=True,math=dict(expected_sample_variance=2/3,variance_of_three_sample_mean=2/9,stable_null_variance=0),authorized_to_launch=False,only_requested_permission='Paired effective window14400 to16200 seconds; other limits unchanged')
    (ROOT/'reports/recovery_seed_stability_plan_checks_20261014.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
