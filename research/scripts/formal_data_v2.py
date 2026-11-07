#!/usr/bin/env python3
"""Audit all spatial numeric data and compute upstream-equivalent batch-one stats without decoding every image."""
import datetime, hashlib, json, os, pathlib, time
import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize
from openpi.training import config, data_loader

ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def calculate(manifest,progress=False):
    stats={k:normalize.RunningStats() for k in ['state','actions']};frames=0;started=time.monotonic()
    info=json.loads((ROOT/'reports/source_metadata/meta/info.json').read_text())
    tasks={x['task_index']:x['task'] for x in map(json.loads,(ROOT/'reports/source_metadata/meta/tasks.jsonl').read_text().splitlines())}
    for number,e in enumerate(manifest['episodes']):
        relative=info['data_path'].format(episode_chunk=e['episode_index']//info['chunks_size'],episode_index=e['episode_index'])
        table=pq.read_table(ROOT/'datasets/lerobot/physical-intelligence/libero'/relative,columns=['state','actions','episode_index','frame_index','task_index'])
        n=e['length'];assert table.num_rows==n
        state=np.asarray(table['state'].to_pylist(),dtype=np.float32);actions=np.asarray(table['actions'].to_pylist(),dtype=np.float32)
        assert state.shape==(n,8) and actions.shape==(n,7) and np.isfinite(state).all() and np.isfinite(actions).all()
        assert set(table['episode_index'].to_pylist())=={e['episode_index']}
        assert table['frame_index'].to_pylist()==list(range(n))
        assert all(tasks[t] in e['tasks'] for t in table['task_index'].to_pylist())
        # Same ordered frame batches and terminal repeat padding as the pinned
        # batch_size=1 compute_norm_stats entry. LiberoInputs leaves these numeric
        # fields unchanged; image parsing is irrelevant to state/action stats.
        for i in range(n):
            stats['state'].update(state[i:i+1])
            stats['actions'].update(actions[np.minimum(np.arange(i,i+10),n-1)][None])
        frames+=n
        if progress and (number%10==0 or number+1==len(manifest['episodes'])):
            elapsed=time.monotonic()-started
            print(json.dumps({'stage':'norm_stats','episodes':number+1,'total_episodes':len(manifest['episodes']),'frames':frames,'elapsed_seconds':round(elapsed),'remaining_seconds_estimate':round(elapsed*(manifest['frames']-frames)/frames)}),flush=True)
    assert frames==manifest['frames']
    return {k:s.get_statistics() for k,s in stats.items()}

def main():
    started=time.monotonic()
    smoke=json.loads((ROOT/'reports/data_manifest.json').read_text());got=calculate(smoke)
    reference=normalize.load(ROOT/'configs/assets/pi05_libero_lora_3090_smoke/physical-intelligence/libero')
    for key in got:
        for field in ['mean','std','q01','q99']:
            np.testing.assert_allclose(getattr(got[key],field),getattr(reference[key],field),rtol=1e-6,atol=1e-7)
    manifest=json.loads((ROOT/'reports/spatial_full_manifest.json').read_text())
    output=ROOT/'configs/assets/pi05_libero_lora_3090_train/physical-intelligence/libero'
    stats=calculate(manifest,progress=True)
    if (output/'norm_stats.json').exists():
        assert (output/'norm_stats.json').read_text()==normalize.serialize_json(stats),'Existing formal stats differ; do not overwrite'
    else:normalize.save(output,stats)
    cfg=config.get_config('pi05_libero_lora_3090x2_train');dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    ds=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
    assert len(ds)==manifest['frames']
    raw=ds
    while isinstance(raw,data_loader.TransformedDataset):raw=raw._dataset
    checks=[];offset=0
    for e in manifest['episodes']:
        # Validate upstream subset-index mapping at every trajectory's boundaries.
        for idx in [offset,offset+e['length']-1]:
            query,padding=raw._get_query_indices(idx,e['episode_index'])
            assert all(offset<=q<offset+e['length'] for q in query['actions'])
        offset+=e['length']
    for task in manifest['tasks']:
        indices=[i for i,e in enumerate(manifest['episodes']) if task['language'] in e['tasks']]
        i=indices[len(indices)//2];idx=int(raw.episode_data_index['from'][i]);row=ds[idx]
        assert row['task']==task['language'] and tuple(row['image'].shape)==(3,256,256)
        assert tuple(row['wrist_image'].shape)==(3,256,256)
        assert row['image'].std()>0 and row['wrist_image'].std()>0
        checks.append({'task_id':task['task_id'],'episode':manifest['episodes'][i]['episode_index'],'image_shape':list(row['image'].shape)})
    loader=data_loader.create_data_loader(cfg,num_batches=1);iterator=iter(loader);obs,actions=next(iterator)
    assert obs.state.shape==(2,32) and actions.shape==(2,10,32)
    assert np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(obs.state)).all()
    iterator.close();workers=loader._data_loader.torch_loader._iterator
    if workers is not None:workers._shutdown_workers()
    record={'status':'PASS','completed_at':datetime.datetime.now().astimezone().isoformat(),'elapsed_seconds':time.monotonic()-started,
      'episodes_used':len(manifest['episodes']),'frames_used':len(ds),'tasks':len(checks),'all_numeric_values_finite':True,
      'upstream_subset_boundaries_checked':2*len(manifest['episodes']),'image_checks':checks,'actual_batch_shape':list(actions.shape),
      'norm_stats_path':str(output/'norm_stats.json'),'norm_stats_sha256':hashlib.sha256((output/'norm_stats.json').read_bytes()).hexdigest(),
      'numeric_stats_equivalence':'per-frame upstream RunningStats in identical order; all mean/std/q01/q99 match prior actual-loader smoke stats within rtol1e-6/atol1e-7'}
    (ROOT/'reports/formal_data_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
