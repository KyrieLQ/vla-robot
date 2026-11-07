"""Atomic single-file state: arrays, integrity manifest, optimizer, RNG and cursor."""
import dataclasses,hashlib,io,json,os,random
from pathlib import Path
import flax.nnx as nnx
import jax
import numpy as np
from .pi05_compact_state_v1 import sha,leaf_digest

def fsync_directory(path):
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def state_payload(state,cfg,rng,cursor,binding,provenance):
    pairs,definition=jax.tree_util.tree_flatten_with_path((state.params.filter(cfg.trainable_filter),state.opt_state))
    arrays={};inventory=[]
    for i,(key,value) in enumerate(pairs):
        array=np.asarray(jax.device_get(value));assert np.isfinite(array).all();name='leaf_%04d'%i;arrays[name]=array
        inventory.append(dict(name=name,path=jax.tree_util.keystr(key),shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    arrays['rng']=np.asarray(jax.random.key_data(rng));npstate=np.random.get_state()
    metadata=dict(format='pi05_single_archive_v2',kind='optimizer_state',step=int(state.step),cursor=cursor,binding=binding,
        provenance=provenance,leaves=inventory,treedef=str(definition),python_rng=random.getstate(),
        numpy_rng=[npstate[0],npstate[1].tolist(),npstate[2],npstate[3],npstate[4]],stores_frozen_weights=False)
    return arrays,metadata

def write_archive(destination,arrays,metadata):
    header=np.frombuffer(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    if hasattr(destination,'write'):
        np.savez(destination,metadata=header,**arrays);destination.seek(0);return
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    if temporary.exists():raise RuntimeError('Preserve and reconcile existing temporary checkpoint: '+str(temporary))
    with temporary.open('xb') as stream:
        np.savez(stream,metadata=header,**arrays);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);fsync_directory(path.parent)

def save_state(path,state,cfg,rng,cursor,binding,provenance):
    arrays,metadata=state_payload(state,cfg,rng,cursor,binding,provenance);write_archive(path,arrays,metadata)
    return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size,step=int(state.step),cursor=cursor)

def inspect_archive(path,binding=None):
    if hasattr(path,'seek'):path.seek(0)
    with np.load(path,allow_pickle=False) as values:
        metadata=json.loads(values['metadata'].tobytes().decode())
        assert metadata['format']=='pi05_single_archive_v2'
        if binding is not None:assert metadata['binding']==binding
        for entry in metadata['leaves']:
            value=values[entry['name']];assert list(value.shape)==entry['shape'] and str(value.dtype)==entry['dtype']
            assert hashlib.sha256(value.tobytes()).hexdigest()==entry['sha256'] and np.isfinite(value).all()
    return metadata

def load_state(path,template,cfg,binding):
    if hasattr(path,'seek'):path.seek(0)
    trainable,frozen=template.params.split(cfg.trainable_filter,...)
    pairs,definition=jax.tree_util.tree_flatten_with_path((trainable,template.opt_state));restored=[]
    with np.load(path,allow_pickle=False) as values:
        meta=json.loads(values['metadata'].tobytes().decode());assert meta['format']=='pi05_single_archive_v2' and meta['kind']=='optimizer_state' and meta['binding']==binding
        assert str(definition)==meta['treedef'] and len(pairs)==len(meta['leaves'])
        for (key,prototype),entry in zip(pairs,meta['leaves']):
            value=values[entry['name']]
            assert jax.tree_util.keystr(key)==entry['path'] and list(prototype.shape)==entry['shape']==list(value.shape) and str(prototype.dtype)==entry['dtype']==str(value.dtype)
            assert hashlib.sha256(value.tobytes()).hexdigest()==entry['sha256'] and np.isfinite(value).all()
            restored.append(jax.device_put(value,prototype.sharding))
        key=jax.random.wrap_key_data(jax.device_put(values['rng']))
    params,opt=jax.tree.unflatten(definition,restored)
    state=dataclasses.replace(template,params=nnx.State.merge(frozen,params),opt_state=opt,step=jax.device_put(np.asarray(meta['step'],np.int32),template.step.sharding))
    def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
    random.setstate(tuples(meta['python_rng']));n=meta['numpy_rng'];np.random.set_state((n[0],np.asarray(n[1],np.uint32),n[2],n[3],n[4]))
    return state,key,meta

def save_parameters(path,state,cfg,binding,point):
    pairs,definition=jax.tree_util.tree_flatten_with_path(state.params.filter(cfg.trainable_filter));arrays={};inventory=[]
    for i,(key,value) in enumerate(pairs):
        array=np.asarray(jax.device_get(value));name='leaf_%04d'%i;arrays[name]=array
        inventory.append(dict(name=name,path=jax.tree_util.keystr(key),shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    meta=dict(format='pi05_single_archive_v2',kind='parameters',step=int(state.step),point=point,binding=binding,leaves=inventory,treedef=str(definition),stores_frozen_weights=False)
    write_archive(path,arrays,meta);return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size,step=int(state.step),point=point)
