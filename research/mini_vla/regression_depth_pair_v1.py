"""Matched regression heads on the same fixed adapted VLM; no Flow objective."""
import copy
import torch
from .vlm_action_heads_v1 import HeadConfig,RegressionHead

class RegressionDepthHead(RegressionHead):
    def __init__(self,config=HeadConfig(),layers=1):
        super().__init__(config)
        if layers not in (1,2):raise ValueError('Only preregistered1/2-layer contrast')
        if layers==2:self.decoder.layers.append(copy.deepcopy(self.decoder.layers[0]))
        self.decoder.num_layers=layers
        self.layers=layers

def verify_initial(one,two,original):
    a=one.state_dict();b=two.state_dict()
    assert set(a)==set(original) and all(torch.equal(a[k].cpu(),v.cpu()) for k,v in original.items())
    assert all(torch.equal(a[k],b[k]) for k in a)
    extra=set(b)-set(a)
    assert extra=={k.replace('decoder.layers.0.','decoder.layers.1.') for k in a if k.startswith('decoder.layers.0.')}
    for key,value in two.decoder.layers[0].named_parameters():
        other=dict(two.decoder.layers[1].named_parameters())[key]
        assert value.data_ptr()!=other.data_ptr() and torch.equal(value,other)
    assert sum(v.numel() for v in one.parameters())==398663
    assert sum(v.numel() for v in two.parameters())==663239
