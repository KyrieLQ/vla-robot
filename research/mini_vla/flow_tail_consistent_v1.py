"""Training-only tail convention: all K input tokens, valid-only velocity loss.

The existing dataset repeats the terminal action to form a length-K target.
This module changes neither model parameters nor deployment inference. The
future validity mask is consumed only by the training loss, never the Decoder.
"""
from .model import masked_mse

def flow_loss_tail_consistent(model,observations,target,action_mask,noise,time):
    x=(1-time[:,None,None])*noise+time[:,None,None]*target
    expected=target-noise
    velocity=model(**observations,noisy_action=x,time=time)
    return masked_mse(velocity,expected,action_mask)
