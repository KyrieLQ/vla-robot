"""Independent readers for immutable, lossless policy I/O and executed actions."""
import hashlib
import json
import pathlib
import numpy as np

NUMERIC_FIELDS = ('qpos', 'qvel', 'control', 'object_position', 'object_quaternion')
METHODS = ('continue_a', 'continue_b', 'fine20', 'resample5', 'retreat5_fine20')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def image_hash(external, wrist):
    assert external.shape == wrist.shape == (224, 224, 3)
    assert external.dtype == wrist.dtype == np.uint8
    return hashlib.sha256(external.tobytes() + wrist.tobytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def checked_close(actual, reference, atol):
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=0)
    return float(np.abs(actual-reference).max()) if actual.size else 0.0


def check_physical(actual, reference, atol=1e-7):
    return {key: checked_close(np.asarray(actual[key]), np.asarray(reference[key]), atol)
            for key in NUMERIC_FIELDS}


def check_saved_branch(directory, row, root, plan):
    """Check on-disk inputs, actual execution cadence and independent predicates."""
    directory = pathlib.Path(directory)
    trace = arrays(directory/'policy_trace.npz')
    obs = arrays(directory/'policy_inputs.npz')
    oracle = arrays(directory/'oracle.npz')
    meta = json.loads((directory/'policy_inputs.json').read_text())
    n = int(row['steps']); count = int(row['policy_calls']); start = int(root['branch_step'])
    assert row['root_id'] == root['root_id'] and row['method'] in METHODS
    assert 0 < n <= plan['max_total_steps']-start
    assert trace['action'].shape == (n, 7)
    assert trace['robot_state'].shape == (n+1, 8)
    assert trace['image_sha256'].shape == (n+1,)
    assert trace['done'].shape == (n,)
    assert trace['call_step'].shape == (count,)
    assert trace['action_chunk'].shape == (count, 10, 7)
    assert obs['external'].shape == obs['wrist'].shape == (count, 224, 224, 3)
    assert obs['robot_state'].shape == (count, 8)
    assert np.array_equal(obs['call_step'], trace['call_step'])
    assert all(np.isfinite(trace[key]).all() for key in ('action','action_chunk','robot_state','inference_seconds'))
    assert meta['prompt'] == row['language']
    assert meta['policy_seed'] == (root['resample_seed'] if row['method']=='resample5' else root['policy_seed'])
    assert meta['policy_rng_advance'] == (0 if row['method']=='resample5' else start//5)
    assert trace['image_sha256'][0] == row['root_image_sha256']
    assert np.array_equal(oracle['goal_predicate'][1:, 0], trace['done'])
    assert bool(trace['done'].any()) == row['task_success']
    hit = np.flatnonzero(trace['done'])
    assert not len(hit) or (len(hit)==1 and hit[0]==n-1)
    assert row['first_success_step'] == (start+n if len(hit) else None)
    if not len(hit): assert start+n == plan['max_total_steps']
    for field in NUMERIC_FIELDS:
        assert len(oracle[field]) == n+1 and np.isfinite(oracle[field]).all()
    queue = []; call = 0; consumed = []
    for i in range(n):
        if row['method']=='retreat5_fine20' and i < plan['retreat_steps']:
            expected = np.asarray(plan['retreat_action'])
        else:
            if not queue:
                assert trace['call_step'][call] == start+i
                assert image_hash(obs['external'][call], obs['wrist'][call]) == trace['image_sha256'][i]
                checked_close(obs['robot_state'][call], trace['robot_state'][i], 1e-6)
                fine = ((row['method']=='fine20' and i < plan['fine_steps']) or
                        (row['method']=='retreat5_fine20' and i < plan['retreat_steps']+plan['fine_steps']))
                queue.extend(trace['action_chunk'][call, :1 if fine else 5])
                consumed.append(start+i); call += 1
            expected = queue.pop(0)
        checked_close(trace['action'][i], expected, 1e-6)
    assert call == count
    assert np.array_equal(trace['call_step'], np.asarray(consumed))
    return trace, obs, oracle, meta


def replay_model_inputs(client, directory, row, root, plan, progress=None):
    """Real inference on disk-loaded inputs; no reference predictions in requests."""
    directory = pathlib.Path(directory)
    trace, obs, _, meta = check_saved_branch(directory, row, root, plan)
    max_error = 0.0
    for i, step in enumerate(trace['call_step']):
        request = {'observation/image': obs['external'][i],
                   'observation/wrist_image': obs['wrist'][i],
                   'observation/state': obs['robot_state'][i], 'prompt': meta['prompt']}
        if i == 0:
            request.update(_policy_seed=meta['policy_seed'], _policy_rng_advance=meta['policy_rng_advance'])
        prediction = np.asarray(client.infer(request)['actions'])
        max_error = max(max_error, checked_close(prediction, trace['action_chunk'][i], 1e-6))
        if progress is not None: progress(i+1, len(trace['call_step']))
    return {'status':'PASS','calls_checked':len(trace['call_step']), 'max_action_chunk_error':max_error,
            'numeric_atol':1e-6, 'actual_inputs_loaded_from_disk':True,
            'policy_inputs_sha256':sha(directory/'policy_inputs.npz'),
            'policy_metadata_sha256':sha(directory/'policy_inputs.json'),
            'trace_sha256':sha(directory/'policy_trace.npz'),
            'oracle_sha256':sha(directory/'oracle.npz')}


def verify_audit_bindings(directory, row):
    directory = pathlib.Path(directory)
    audit = json.loads((directory/'model_replay.json').read_text())
    assert audit['status']=='PASS' and audit['calls_checked']==row['policy_calls']
    assert audit['numeric_atol']==1e-6 and audit['max_action_chunk_error']<=1e-6
    for field, name in [('policy_inputs_sha256','policy_inputs.npz'),
                        ('policy_metadata_sha256','policy_inputs.json'),
                        ('trace_sha256','policy_trace.npz'),('oracle_sha256','oracle.npz')]:
        assert audit[field] == sha(directory/name)
    if row['method'] in ('continue_a','continue_b'):
        audit = json.loads((directory/'physics_replay.json').read_text())
        assert audit['status']=='PASS' and audit['states_checked']==row['steps']+1
        assert audit['goal_flags_exact'] and audit['numeric_atol']==1e-7
        assert audit['trace_sha256']==sha(directory/'policy_trace.npz')
        assert audit['oracle_sha256']==sha(directory/'oracle.npz')
        assert all(x<=1e-7 for x in audit['physical_max_errors'].values())
        assert audit['robot_state_max_error']<=1e-6
    return True
