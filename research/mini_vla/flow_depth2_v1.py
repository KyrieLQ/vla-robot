"""One extra independent decoder block; every original parameter keeps its seed42 value."""
import copy
from .vlm_action_heads_v1 import FlowHead,HeadConfig

class Depth2FlowHead(FlowHead):
    def __init__(self,config=HeadConfig()):
        super().__init__(config)
        self.decoder.layers.append(copy.deepcopy(self.decoder.layers[0]))
        self.decoder.num_layers=2
        assert len(self.decoder.layers)==2

def verify_common_initial(model,original_state):
    actual=model.state_dict();assert set(original_state).issubset(actual)
    import torch
    assert all(torch.equal(actual[k].cpu(),v.cpu()) for k,v in original_state.items())
    extra=set(actual)-set(original_state)
    expected={k.replace('decoder.layers.0.','decoder.layers.1.') for k in original_state if k.startswith('decoder.layers.0.')}
    assert extra==expected
    for name,p in model.decoder.layers[0].named_parameters():
        other=dict(model.decoder.layers[1].named_parameters())[name]
        assert p.data_ptr()!=other.data_ptr() and torch.equal(p,other)
