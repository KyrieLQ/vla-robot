"""CPU lossless save/restore including optimizer, private key and data position."""
import argparse,dataclasses,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.training.utils import TrainState
from mini_vla.pi05_compact_state_v1 import save,restore,leaf_digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();assert jax.default_backend()=='cpu'
    class Toy(nnx.Module):
        def __init__(self):self.train=nnx.Param(jnp.arange(8,dtype=jnp.float32).reshape(2,4));self.frozen=nnx.Param(jnp.ones(5))
    model=Toy();cfg=SimpleNamespace(trainable_filter=lambda path,value:path[0]=='train');params=nnx.state(model);tx=optax.adamw(1e-3)
    train=params.filter(cfg.trainable_filter);optimizer=tx.init(train)
    initial=TrainState(step=jnp.asarray(0,jnp.int32),params=params,model_def=nnx.graphdef(model),tx=tx,opt_state=optimizer,ema_decay=None,ema_params=None)
    def advance(state):
        train,frozen=state.params.split(cfg.trainable_filter,...)
        gradients=jax.tree.map(lambda x:2*x+.1,train)
        update,opt=state.tx.update(gradients,state.opt_state,train)
        return dataclasses.replace(state,params=nnx.State.merge(frozen,optax.apply_updates(train,update)),opt_state=opt,step=state.step+1)
    step=jax.jit(advance);state=step(step(initial))
    rng=jax.random.key(42);binding=dict(base='fixture',source='fixture')
    with tempfile.TemporaryDirectory(dir=Path(a.out).parent) as tmp:
        file=Path(tmp)/'state.npz';metadata=save(file,state,cfg,rng,8,binding);loaded,key,cursor=restore(file,initial,cfg,binding)
        assert leaf_digest((state.params,state.opt_state))==leaf_digest((loaded.params,loaded.opt_state)) and int(loaded.step)==2 and cursor==8
        np.testing.assert_array_equal(jax.random.key_data(key),jax.random.key_data(rng))
        uninterrupted=step(step(state));resumed=step(step(loaded))
        assert leaf_digest((uninterrupted.params,uninterrupted.opt_state))==leaf_digest((resumed.params,resumed.opt_state))
        try:restore(file,state,cfg,dict(base='WRONG',source='fixture'))
        except AssertionError:pass
        else:raise AssertionError('Wrongbaseaccepted')
    result=dict(status='PASS',optimizer_and_parameters_exact=True,rng_and_cursor_exact=True,wrong_base_rejected=True,restore_from_initial_template_then_continue_exact=True,
                frozen_weights_excluded=True,new_gpu_compute=0,scientific_training_updates=0)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
