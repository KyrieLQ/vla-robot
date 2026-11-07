"""TimeFiLM server/client identity contract, usable in both Python environments."""
def validate_metadata(metadata, plan, arm='flow'):
    expected = dict(model='fixed_adapted_vlm_flow_time_film', time_conditioning=plan['time_conditioning'],decoder_layers=2,head_parameters=763335,arm=arm,
                    epoch=plan['selected_epoch'], updates=plan['selected_updates'],
                    checkpoint_sha256=plan['arms'][arm]['checkpoint_sha256'],
                    backbone_checkpoint_sha256=plan['backbone_checkpoint_sha256'],
                    action_horizon=10, euler_steps=10,
                    noise_seed_rule='SHA256(20261020,task,init,call_index)')
    for key,value in expected.items():
        assert metadata.get(key)==value, 'Policy metadata mismatch: '+key
    assert plan['decoder_layers']==2 and plan['head_parameters']==763335
    return expected
