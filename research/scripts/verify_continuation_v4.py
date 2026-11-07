"""Independent audit of consumed images and numerics under the approved scope."""
import numpy as np


def verify_continuation(row, trace, oracle, source, source_oracle, plan):
    assert row['method']=='continue' and row['continuation_verified']
    start=int(row['branch_step']);n=int(row['steps']);end=start+n
    assert end==min(len(source['action']),plan['max_total_steps'])
    assert np.array_equal(trace['call_step'],np.arange(start,end,5))
    assert trace['image_sha256'].shape==(n+1,)
    expected_images=source['image_sha256'][start:end+1]
    assert trace['image_sha256'][0]==expected_images[0]==row['root_image_sha256']
    calls=trace['call_step']-start
    assert np.array_equal(trace['image_sha256'][calls],expected_images[calls]), 'Actual policy-input image mismatch'
    differences=np.flatnonzero(trace['image_sha256']!=expected_images)
    assert not set(differences.tolist())&set(calls.tolist())
    if len(differences):assert plan['unconsumed_suffix_image_amendment_approved'] is True
    records=row.get('suffix_render_differences',[])
    assert len(records)==len(differences)
    assert {int(d['step']) for d in records}=={start+int(i) for i in differences}
    for record in records:
        i=int(record['step'])-start
        assert record['policy_consumes_frame'] is False
        assert bool(record['terminal_record_frame'])==(i==n)
        assert record['actual_hash']==trace['image_sha256'][i]
        assert record['reference_hash']==expected_images[i]
    numeric={}
    for name,actual,reference in (
            ('action',trace['action'],source['action'][start:end]),
            ('robot_state',trace['robot_state'],source['robot_state'][start:end+1]),
            ('action_chunk',trace['action_chunk'],source['action_chunk'][trace['call_step']//5])):
        np.testing.assert_allclose(actual,reference,atol=plan['acceptance']['numeric_atol'],rtol=0,err_msg=name)
        numeric[name]=float(np.max(np.abs(actual-reference)))
    for name in ('qpos','qvel','control','object_position','object_quaternion'):
        actual=oracle[name];reference=source_oracle[name][start:end+1]
        np.testing.assert_allclose(actual,reference,atol=plan['acceptance']['root_simulator_atol'],rtol=0,err_msg=name)
        numeric[name]=float(np.max(np.abs(actual-reference)))
    assert np.array_equal(trace['done'],source['done'][start:end])
    assert np.array_equal(oracle['goal_predicate'][1:,0],trace['done'])
    assert bool(trace['done'].any())==row['task_success']==row['baseline_success']
    hit=np.flatnonzero(trace['done'])
    assert row['first_success_step']==(start+int(hit[0])+1 if len(hit) else None)
    return {'root_id':row['root_id'],'status':'PASS','numeric_max_errors':numeric,
            'consumed_images_checked':len(calls),'record_only_image_differences':len(differences),
            'terminal_image_difference':bool(n in differences)}
