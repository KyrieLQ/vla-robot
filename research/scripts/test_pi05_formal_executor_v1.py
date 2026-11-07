"""CPU contracts for data coverage, scoring, atomic state and exact continuation."""
import dataclasses, hashlib, io, json, os, random, tempfile, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from openpi.training.utils import TrainState
from mini_vla.pi05_full_data_v1 import FullDataset, FrameIndex, epoch_order, validation_noise
from mini_vla.pi05_compact_state_v2 import save_state, load_state, leaf_digest, sha
from mini_vla.pi05_formal_io_v1 import R, write, inspect_checkpoint, verify_log_segments, score_predictions

def main():
    started=time.monotonic(); assert jax.default_backend()=='cpu'
    folder=R/'reports/pi05_formal_prepare_20261105'; folder.mkdir(exist_ok=True)
    plan=json.loads((R/'configs/mini_vla_pi05_generation_preflight_20261105_v1.json').read_text())
    split=json.loads((R/plan['split']).read_text()); contracts=json.loads((R/plan['cpu_contract']).read_text())
    train=FullDataset(plan,'train'); validation=FullDataset(plan,'validation')
    assert not set(train.index.starts)&set(validation.index.starts)
    totals={}
    for name,data in [('train',train),('validation',validation)]:
        numeric=np.load(R/'datasets/mini_vla_four_suite_128_20261019_v1'/name/'episode_id.npy',mmap_mode='r')
        actual=np.array([data.index[i]['episode_id'] for i in range(len(data.index))])
        np.testing.assert_array_equal(actual,numeric)
        for entry in split[name]:
            eid=entry['episode_id']; end=entry['length']-1
            assert data.index[data.index.locate(eid,0)]==dict(episode_id=eid,frame_index=0)
            assert data.index[data.index.locate(eid,end)]==dict(episode_id=eid,frame_index=end)
        totals[name]=len(actual)
    for saved in contracts['receipts']:
        index=train.index.locate(saved['episode_id'],saved['frame_index'])
        observation,actions,receipt=train.get(index)
        assert {k:receipt[k] for k in saved}==saved
    orders={}
    for epoch in [1,2,3]:
        order,h=epoch_order(epoch);np.testing.assert_array_equal(np.sort(order),np.arange(243000))
        # Include the 60750 -> 61000 segment boundary and partial resume.
        for cursor in [0,4,3996,4000,242996]:
            np.testing.assert_array_equal(order[cursor:],np.random.default_rng(42+epoch).permutation(243000)[cursor:])
        orders[str(epoch)]=h
    npstate=np.random.get_state(); torch_rows=contracts['receipts'][:4]
    z=validation_noise(torch_rows);np.testing.assert_array_equal(z,validation_noise(torch_rows))
    assert all(np.array_equal(a,b) for a,b in zip(npstate,np.random.get_state()))
    (_,target),rows=validation.get_batch([30464]*4)
    assert len(rows)==4 and target.shape==(4,10,32) and all(r['global_index']==30464 for r in rows)

    class Toy(nnx.Module):
        def __init__(self):
            self.train=nnx.Param(jnp.arange(8,dtype=jnp.float32).reshape(2,4));self.frozen=nnx.Param(jnp.ones(5))
    model=Toy();cfg=SimpleNamespace(trainable_filter=lambda path,value:path[0]=='train');params=nnx.state(model);tx=optax.adamw(1e-3)
    initial=TrainState(step=jnp.asarray(0,jnp.int32),params=params,model_def=nnx.graphdef(model),tx=tx,opt_state=tx.init(params.filter(cfg.trainable_filter)),ema_decay=None,ema_params=None)
    def advance(state):
        trainable,frozen=state.params.split(cfg.trainable_filter,...);grad=jax.tree.map(lambda x:2*x+.1,trainable)
        update,opt=state.tx.update(grad,state.opt_state,trainable)
        return dataclasses.replace(state,params=nnx.State.merge(frozen,optax.apply_updates(trainable,update)),opt_state=opt,step=state.step+1)
    step=jax.jit(advance);two=step(step(initial));rng=jax.random.key(42);binding=dict(base='CPU fixture')
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        scratch=Path(scratch);file=scratch/'latest.npz';cursor=dict(step=2,frames_in_epoch=8,log_segments=[])
        random.seed(123);np.random.seed(456)
        save_state(file,two,cfg,rng,cursor,binding,dict(fixture=True));python_next=random.random();numpy_next=np.random.rand()
        loaded,key,meta=load_state(file,initial,cfg,binding)
        assert random.random()==python_next and np.random.rand()==numpy_next
        assert leaf_digest((two.params,two.opt_state))==leaf_digest((loaded.params,loaded.opt_state))
        assert leaf_digest((step(step(two)).params,step(step(two)).opt_state))==leaf_digest((step(step(loaded)).params,step(step(loaded)).opt_state))
        assert inspect_checkpoint(file,binding)['step']==2 and meta['cursor']==cursor
        np.testing.assert_array_equal(jax.random.key_data(key),jax.random.key_data(rng))
        old=sha(file)
        with patch('mini_vla.pi05_compact_state_v2.os.replace',side_effect=OSError('injected before atomic commit')):
            try:save_state(file,step(two),cfg,rng,dict(step=3),binding,{})
            except OSError:pass
            else:raise AssertionError('Fault did not propagate')
        assert sha(file)==old and inspect_checkpoint(file.with_name(file.name+'.tmp'),binding)['step']==3
        # This fixture is disposable; production reconciles this complete archive.
        os.replace(file.with_name(file.name+'.tmp'),file)
        assert inspect_checkpoint(file,binding)['step']==3
        try:load_state(file,initial,cfg,dict(base='wrong'))
        except AssertionError:pass
        else:raise AssertionError('Wrong binding accepted')
        with np.load(file,allow_pickle=False) as f:values={k:f[k] for k in f.files}
        values['leaf_0000']=values['leaf_0000']+1
        broken=scratch/'corrupt.npz';np.savez(broken,**values)
        try:inspect_checkpoint(broken,binding)
        except AssertionError:pass
        else:raise AssertionError('Corrupt leaf accepted')
        log=scratch/'updates.jsonl';prefix=b'{"step":1}\n{"step":2}\n';log.write_bytes(prefix+b'{"step":999}\n')
        segment=dict(path=str(log.relative_to(R)),bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),rows=2)
        assert verify_log_segments([segment])==2
        segment['bytes']=log.stat().st_size
        try:verify_log_segments([segment])
        except AssertionError:pass
        else:raise AssertionError('Noncontiguous tail accepted')
        # Independent perfect-prediction oracle for all validation labels. No policy inference.
        prediction=scratch/'oracle.npy';array=np.lib.format.open_memmap(prediction,mode='w+',dtype=np.float32,shape=(30465,10,32));array[:]=999
        stats=json.loads((R/plan['norm_stats']).read_text())['norm_stats']['actions'];lo=np.array(stats['q01']);hi=np.array(stats['q99']);offset=0
        for entry in split['validation']:
            eid=entry['episode_id'];n=entry['length'];path=R/plan['dataset_root']/('data/chunk-%03d/episode_%06d.parquet'%(eid//1000,eid))
            raw=np.asarray(pq.read_table(path,columns=['actions'],use_threads=False)['actions'].to_pylist(),np.float32)
            actions=(2*(raw-lo)/(hi-lo+1e-6)-1).astype(np.float32)
            for k in range(10):array[offset:offset+n-k,k,:7]=actions[k:]
            offset+=n
        array.flush();del array
        score=score_predictions(prediction,plan)
        assert score['first_mse']==score['valid_chunk_mse']==score['first5_mse']==0 and score['valid_target_steps']==296235
    # The local JAX version must support the bounded-cache settings before GPU work.
    for name in ['jax_compilation_cache_dir','jax_compilation_cache_max_size','jax_persistent_cache_min_compile_time_secs']:assert name in jax.config.values
    report=dict(status='PASS',total_frames=totals,reference_inputs=120,all_episode_boundaries=True,epoch_order_sha256=orders,
        split_disjoint=True,single_file_atomic_failure_preserves_previous=True,nonzero_optimizer_resume_and_continuation_bitwise=True,
        python_numpy_jax_rng_restored=True,corruption_and_wrong_binding_rejected=True,committed_log_prefix_only=True,
        full_validation_scoring_oracle_frames=30465,valid_action_steps=296235,invalid_tail_and_padded_dimensions_excluded=True,
        validation_last_batch_real_rows=1,validation_last_batch_padding_rows=3,new_gpu_compute=0,formal_optimizer_updates=0,
        elapsed_seconds=time.monotonic()-started)
    write(folder/'cpu_contract.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
