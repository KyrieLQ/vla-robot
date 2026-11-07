"""Independent full data/numeric audit with deterministic per-episode image checks."""
import hashlib,json,math,re,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from mini_vla.data import decode_image,encode_instruction
from mini_vla.chunk_model_v1 import ChunkCachedData

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/mini_vla_m6_prepare_20261019_run3'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    started=time.monotonic();out=RUN/'review';out.mkdir(exist_ok=True);assert not (out/'independent_acceptance.json').exists()
    for p,h in json.loads((RUN/'source_manifest.json').read_text()).items():assert sha(ROOT/p)==h,p
    cfg=json.loads((ROOT/'configs/mini_vla_m6_prepare_20261019_v3.json').read_text());split=json.loads((ROOT/cfg['split']).read_text());pre=json.loads((ROOT/cfg['preprocessing']).read_text());base=ROOT/cfg['dataset_root'];cache=ROOT/cfg['cache']
    info=json.loads((base/'meta/info.json').read_text());meta=json.loads((ROOT/cfg['source_metadata']).read_text());assert meta['sha']==cfg['data_revision']
    source_files={x['rfilename']:x for x in meta['siblings'] if x['rfilename'].endswith('.parquet')};assert len(source_files)==1693
    integrity=json.loads((RUN/'data_integrity.json').read_text());assert integrity['status']=='PASS' and len(integrity['files'])==1693
    receipts={x['path']:x for x in integrity['files']};assert len(receipts)==1693
    for i,(name,row) in enumerate(source_files.items()):
        p=base/name;assert p.stat().st_size==row['size'];assert sha(p)==row['lfs']['sha256']==receipts[name]['sha256']
        if i%200==0:print('source_hashes',i+1,'/1693',flush=True)
    train_ids={x['episode_id'] for x in split['train']};val_ids={x['episode_id'] for x in split['validation']}
    assert len(train_ids)==1506 and len(val_ids)==187 and not train_ids&val_ids and train_ids|val_ids==set(range(1693))
    tokens=sorted({w for e in split['train'] for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?",e['instruction'].lower().replace('_',' '))})
    assert pre['vocabulary']=={'<PAD>':0,'<UNK>':1,**{w:i+2 for i,w in enumerate(tokens)}} and len(pre['vocabulary'])==71
    complete=json.loads((cache/'complete.json').read_text());assert complete['binding']==dict(split_sha256=sha(ROOT/cfg['split']),preprocessing_sha256=sha(ROOT/cfg['preprocessing']),image_size=128)
    task_text={e['task_index']:e['task'] for e in map(json.loads,(base/'meta/tasks.jsonl').read_text().splitlines())}
    train_stats={k:[] for k in ['state','actions']};cache_hashes={};summaries={};image_checks=0;max_numeric_error=0.;window_checks=0
    for part in ['train','validation']:
        arrays={p.stem:np.load(p,mmap_mode='r') for p in (cache/part).glob('*.npy')};n=sum(e['length'] for e in split[part]);assert n==len(arrays['state'])
        for name,arr in arrays.items():assert list(arr.shape)==complete['parts'][part]['shapes'][name]
        dataset=ChunkCachedData(cache,part);offset=0;target_steps=0;task_counts={};seen=set()
        for number,e in enumerate(split[part]):
            eid=e['episode_id'];length=e['length'];end=offset+length;gid=SUITES.index(e['suite'])*10+e['task_id'];task_counts[gid]=task_counts.get(gid,0)+1;seen.add(eid)
            p=base/info['data_path'].format(episode_chunk=eid//info['chunks_size'],episode_index=eid)
            table=pq.read_table(p,columns=['state','actions','episode_index','frame_index','task_index','image','wrist_image'],use_threads=False)
            assert table.num_rows==length and set(table['episode_index'].to_pylist())=={eid} and table['frame_index'].to_pylist()==list(range(length))
            assert all(task_text[t]==e['instruction'] for t in table['task_index'].to_pylist())
            assert np.all(arrays['episode_id'][offset:end]==eid) and np.all(arrays['task_id'][offset:end]==gid)
            ids,mask=encode_instruction(e['instruction'],pre['vocabulary']);assert np.all(arrays['token_ids'][offset:end]==ids.numpy()) and np.all(arrays['language_mask'][offset:end]==mask.numpy())
            for key,width in [('state',8),('actions',7)]:
                raw=np.asarray(table[key].to_pylist(),dtype='float32');assert raw.shape==(length,width) and np.isfinite(raw).all()
                if part=='train':train_stats[key].append(raw.astype('float64'))
                stats=pre['normalization'][key];expected=(raw-np.asarray(stats['mean'],dtype='float32'))/np.asarray(stats['scale'],dtype='float32');actual=arrays[key][offset:end] if key=='state' else arrays[key][offset:end,0]
                np.testing.assert_array_equal(actual,expected);max_numeric_error=max(max_numeric_error,float(np.max(np.abs(actual-expected))))
            for local in sorted({0,length//2,length-1}):
                row=table.slice(local,1).to_pylist()[0]
                for camera,key in enumerate(['image','wrist_image']):np.testing.assert_array_equal(arrays['images'][offset+local,camera].astype('float32')/255,decode_image(row[key]).numpy());image_checks+=1
            for local in sorted(set([0,length//2]+list(range(max(0,length-10),length)))):
                idx,valid=dataset.window_indices(offset+local);expected_idx=np.minimum(np.arange(offset+local,offset+local+10),end-1);np.testing.assert_array_equal(idx,expected_idx);np.testing.assert_array_equal(valid,np.arange(local,local+10)<length);assert np.all(arrays['episode_id'][idx]==eid);window_checks+=1
            target_steps+=sum(min(10,length-i) for i in range(length));offset=end
            if number%200==0:print('cache_compare',part,number+1,'/',len(split[part]),flush=True)
        assert seen==(train_ids if part=='train' else val_ids) and set(task_counts)==set(range(40))
        assert offset==n and target_steps==(2362230 if part=='train' else 296235)
        for p in (cache/part).glob('*.npy'):cache_hashes[str(p.relative_to(ROOT))]=sha(p)
        summaries[part]=dict(episodes=len(seen),frames=n,valid_K10_targets=target_steps,task_count=40,episodes_per_global_task=task_counts)
    for key,blocks in train_stats.items():
        raw=np.concatenate(blocks);mean=raw.mean(0);std=raw.std(0);scale=np.where(std<1e-6,1.,std)
        np.testing.assert_allclose(mean,pre['normalization'][key]['mean'],rtol=0,atol=1e-12);np.testing.assert_allclose(scale,pre['normalization'][key]['scale'],rtol=0,atol=1e-12)
    result=dict(status='PASS',source_files=1693,source_revision=meta['sha'],summaries=summaries,vocabulary_size=71,train_validation_episode_disjoint=True,train_only_statistics_recomputed=True,cache_numeric_bitwise_equal_to_source_transform=True,max_numeric_error=max_numeric_error,image_frame_camera_checks=image_checks,image_check_scope='first/middle/last frame of every episode; not an independent pixel re-decode of every frame',window_boundary_checks=window_checks,cache_all_files_sha256=cache_hashes,cache_manifest_sha256=sha(cache/'complete.json'),preprocessing_sha256=sha(ROOT/cfg['preprocessing']),elapsed_seconds=time.monotonic()-started,new_training_updates=0,new_simulation_episodes=0,confirmation_accessed=False)
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cache_all_files_sha256'},indent=2))

if __name__=='__main__':main()
