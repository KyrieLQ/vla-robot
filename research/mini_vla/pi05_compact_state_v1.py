"""Small checkpoint = trainable leaves + optimizer + RNG, with immutable-base binding."""
import dataclasses,hashlib,json,os,random
from pathlib import Path
import flax.nnx as nnx
import jax
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def leaf_digest(tree):
    result={}
    for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a=np.asarray(jax.device_get(value));result[jax.tree_util.keystr(path)]=hashlib.sha256(a.tobytes()).hexdigest()
    return result

def save(path,state,cfg,rng,cursor,binding):
    path=Path(path);trainable=state.params.filter(cfg.trainable_filter)
    pairs,definition=jax.tree_util.tree_flatten_with_path((trainable,state.opt_state))
    arrays={};inventory=[]
    for i,(key,value) in enumerate(pairs):
        array=np.asarray(jax.device_get(value));name='leaf_%04d'%i;arrays[name]=array
        inventory.append(dict(name=name,path=jax.tree_util.keystr(key),shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    arrays['rng']=np.asarray(jax.random.key_data(rng));temporary=path.with_suffix('.tmp.npz')
    np.savez(temporary,**arrays)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    temporary.replace(path)
    npstate=np.random.get_state()
    metadata=dict(format='pi05_trainable_optimizer_rng_v1',step=int(state.step),data_cursor=cursor,binding=binding,
                  leaves=inventory,treedef=str(definition),npz_sha256=sha(path),bytes=path.stat().st_size,
                  python_rng=random.getstate(),numpy_rng=[npstate[0],npstate[1].tolist(),npstate[2],npstate[3],npstate[4]],
                  stores_frozen_weights=False,formal_training_checkpoint=False)
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(metadata,indent=2)+'\n');tmp.replace(path.with_suffix('.json'))
    return metadata

def restore(path,template,cfg,binding):
    path=Path(path);metadata=json.loads(path.with_suffix('.json').read_text())
    assert metadata['format']=='pi05_trainable_optimizer_rng_v1' and metadata['binding']==binding and sha(path)==metadata['npz_sha256']
    trainable,frozen=template.params.split(cfg.trainable_filter,...)
    pairs,definition=jax.tree_util.tree_flatten_with_path((trainable,template.opt_state))
    assert len(pairs)==len(metadata['leaves']) and str(definition)==metadata['treedef'];restored=[]
    with np.load(path,allow_pickle=False) as arrays:
        for (key,prototype),entry in zip(pairs,metadata['leaves']):
            array=arrays[entry['name']]
            assert jax.tree_util.keystr(key)==entry['path'] and list(prototype.shape)==entry['shape']==list(array.shape)
            assert str(prototype.dtype)==entry['dtype']==str(array.dtype)
            assert hashlib.sha256(array.tobytes()).hexdigest()==entry['sha256']
            restored.append(jax.device_put(array,prototype.sharding) if hasattr(prototype,'sharding') else jax.device_put(array))
        rng=jax.random.wrap_key_data(jax.device_put(arrays['rng']))
    params,opt_state=jax.tree.unflatten(definition,restored)
    state=dataclasses.replace(template,params=nnx.State.merge(frozen,params),opt_state=opt_state,
                              step=jax.device_put(np.asarray(metadata['step'],dtype=np.int32),template.step.sharding))
    def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
    random.setstate(tuples(metadata['python_rng']));n=metadata['numpy_rng'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    return state,rng,metadata['data_cursor']
