"""Time-conditioned affine residual after each of two Decoder blocks.

The original query time embedding remains. Zero-initialized per-block scale/shift
starts as exactly the structural control, with independent trainable modulation.
"""
import torch
from torch import nn
from .flow_depth2_v1 import Depth2FlowHead
from .vlm_action_heads_v1 import HeadConfig

class TimeFiLMFlowHead(Depth2FlowHead):
    def __init__(self, config=HeadConfig()):
        super().__init__(config)
        # Isolate construction RNG: training/noise stream matches the control.
        with torch.random.fork_rng(devices=[]):
            self.time_modulation=nn.ModuleList([nn.Linear(config.width,2*config.width) for _ in range(2)])
        for projection in self.time_modulation:
            nn.init.zeros_(projection.weight);nn.init.zeros_(projection.bias)

    def velocity(self, *, memory, memory_padding, noisy_action, time, action_mask=None):
        b=memory.shape[0]
        if noisy_action.shape!=(b,self.config.horizon,self.config.action_dim) or time.shape!=(b,):
            raise ValueError('Noisy action or time shape')
        if action_mask is not None:
            raise ValueError('Tail-consistent inference never receives future action mask')
        phase=time[:,None]*1000.*self.time_frequencies[None]
        embedding=self.time_mlp(torch.cat((phase.sin(),phase.cos()),-1))
        x=self.noisy_action_projection(noisy_action)+self.action_queries[None]+embedding[:,None]
        for layer,modulation in zip(self.decoder.layers,self.time_modulation):
            x=layer(x,memory,memory_key_padding_mask=memory_padding)
            scale,shift=modulation(embedding).chunk(2,dim=-1)
            x=x+x*scale[:,None]+shift[:,None]
        if self.decoder.norm is not None:x=self.decoder.norm(x)
        return self.output(x)

def verify_common_initial(model,original_state):
    actual=model.state_dict()
    assert set(original_state).issubset(actual)
    assert all(torch.equal(actual[k].cpu(),v.cpu()) for k,v in original_state.items())
    extra=set(actual)-set(original_state)
    assert extra=={f'time_modulation.{i}.{name}' for i in range(2) for name in ['weight','bias']}
    assert all(torch.count_nonzero(actual[k])==0 for k in extra)
