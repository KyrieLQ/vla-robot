"""Separate deployable observation history from privileged supervised targets."""
import json,os,pathlib,math
import numpy as np
from paired_live_audit_v1 import arrays,sha
ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
METHODS=('continue_a','fine20','resample5','retreat5_fine20')
def permitted_history(trace,step):
    states=np.asarray(trace['robot_state'])[step+np.array([-20,-15,-10,-5,0])].copy()
    actions=np.asarray(trace['action'])[step+np.array([-4,-3,-2,-1])].copy()
    assert states.shape==(5,8) and actions.shape==(4,7)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    return states,actions
def load_records(plan):
    parent=ROOT/plan['parent_run'];assert sha(parent/'review/acceptance.json')==plan['parent_acceptance_sha256']
    assert sha(parent/'ANALYSIS.json')==plan['parent_analysis_sha256']
    manifest=ROOT/plan['root_manifest'];assert sha(manifest)==plan['root_manifest_sha256']
    roots=json.loads(manifest.read_text())['roots'];rows={}
    for worker in (0,1):
        for row in json.loads((parent/'workers'/f'worker{worker}'/'result.json').read_text())['branches']:
            rows[(row['root_id'],row['method'])]=row
    assert len(rows)==1345
    references={};records=[]
    for root in roots:
        rid=root['root_id'];directory=pathlib.Path(rows[(rid,'continue_a')]['output'])
        source=ROOT/plan['source_episodes']/root['pair_id']
        if root['pair_id'] not in references:
            assert sha(source/'policy_trace.npz')==root['source_trace_sha256']
            references[root['pair_id']]=(arrays(source/'policy_trace.npz'),arrays(source/'oracle.npz'))
        original,original_oracle=references[root['pair_id']];tr=arrays(directory/'policy_trace.npz')
        np.testing.assert_allclose(tr['robot_state'][0],original['robot_state'][root['branch_step']],atol=1e-6,rtol=0)
        history,actions=permitted_history(original,root['branch_step'])
        metadata=json.loads((directory/'oracle_metadata.json').read_text());oracle=arrays(directory/'oracle.npz')
        target=metadata['object_names'].index(metadata['target_object']);destination=metadata['object_names'].index(metadata['destination_object'])
        xyz=oracle['object_position'][0];initial_z=original_oracle['object_position'][0,target,2]
        cues=[bool(oracle['bilateral_finger_contact'][0,target]),bool(xyz[target,2]-initial_z>=.04),bool(np.linalg.norm(xyz[target,:2]-xyz[destination,:2])<=.05)]
        outcomes=[int(rows[(rid,m)]['task_success']) for m in METHODS]
        calls=[rows[(rid,m)]['policy_calls'] for m in METHODS]
        assert rows[(rid,'continue_b')]['task_success']==bool(outcomes[0])
        records.append({'root_id':rid,'pair_id':root['pair_id'],'task_id':root['task_id'],'step':root['branch_step'],
            'partition':root['module_partition'],'directory':str(directory),'prompt':rows[(rid,'continue_a')]['language'],
            'image_sha256':rows[(rid,'continue_a')]['root_image_sha256'],'state':tr['robot_state'][0].tolist(),
            'robot_history':history.tolist(),'action_history':actions.tolist(),'outcomes':outcomes,'calls':calls,
            'progress_targets':cues,'normalized_positive_extra_calls':np.clip(np.maximum(0,np.array(calls[1:])-calls[0])/math.ceil((220-root['branch_step'])/5),0,2).tolist()})
    fit={r['pair_id'] for r in records if r['partition']=='fit'};cal={r['pair_id'] for r in records if r['partition']=='calibration'}
    assert len(records)==269 and len(fit)==140 and len(cal)==60 and not fit&cal
    assert sum(r['partition']=='fit' for r in records)==186
    return records
