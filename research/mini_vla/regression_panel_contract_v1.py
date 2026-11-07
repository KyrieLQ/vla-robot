"""Server/client identity shared by the training and LIBERO Python environments."""
def expected_metadata(plan):
    return dict(model='adapted_regression_three_arm_dev200_v1', action_horizon=10,
                execute_steps=5, backbone_checkpoint_sha256=plan['backbone_checkpoint_sha256'],
                arms={name: {key: row[key] for key in
                      ['checkpoint_sha256', 'epoch', 'updates', 'decoder_layers', 'parameters']}
                      for name, row in plan['arms'].items()})

def validate_metadata(metadata, plan):
    expected = expected_metadata(plan)
    for key, value in expected.items():
        assert metadata.get(key) == value, 'Policy identity mismatch: ' + key
    return expected

def partition_tasks(workers):
    assert 1 <= workers <= 4
    return [list(range(i, 40, workers)) for i in range(workers)]

def case_key(case):
    return '%s/task%d_init%d' % (case['suite'], case['task_id'], case['initial_state_index'])
