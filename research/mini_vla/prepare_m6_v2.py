"""Pinned four-suite data acquisition and resumable CPU cache construction."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import urllib.request

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from .data import build_vocabulary, encode_instruction, fit_normalization, decode_image, write_once

ROOT=Path(__file__).resolve().parents[1]
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def download_one(row,base,revision,stop):
    dest=base/row['rfilename'];dest.resolve().relative_to(base.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
    expected=row['lfs']['sha256'];size=row['size'];part=dest.with_suffix('.parquet.part')
    if dest.exists():
        if dest.stat().st_size!=size or sha(dest)!=expected:raise ValueError('Existing data differs; preserve and investigate: '+dest.name)
        return dict(path=row['rfilename'],bytes=size,sha256=expected,downloaded=False)
    for host in ['huggingface.co','hf-mirror.com']:
        for attempt in range(3):
            if stop.is_set():raise RuntimeError('Download cancelled after another file failed')
            try:
                offset=part.stat().st_size if part.exists() else 0
                if offset>size:raise ValueError('Oversized partial file')
                if offset<size:
                    url=f'https://{host}/datasets/physical-intelligence/libero/resolve/{revision}/{row["rfilename"]}?download=true&resume_offset={offset}&request_nonce={time.time_ns()}'
                    headers={'Cache-Control':'no-cache'}
                    if offset:headers['Range']=f'bytes={offset}-'
                    req=urllib.request.Request(url,headers=headers)
                    with urllib.request.urlopen(req,timeout=30) as response:
                        if response.status==206:
                            content_range=response.headers.get('Content-Range','')
                            if not content_range.startswith(f'bytes {offset}-'):raise ValueError('Unexpected resume range')
                            mode='ab' if offset else 'wb'
                        elif response.status==200:mode='wb';offset=0
                        else:raise ValueError('Unexpected HTTP status')
                        with part.open(mode) as stream:
                            while True:
                                if stop.is_set():raise RuntimeError('Download cancelled')
                                block=response.read(2**20)
                                if not block:break
                                offset+=len(block)
                                if offset>size:raise ValueError('Response exceeds pinned byte count')
                                fs=os.statvfs(ROOT)
                                if fs.f_bavail*fs.f_frsize<100*2**30:raise RuntimeError('Filesystem reserve breached')
                                stream.write(block)
                if part.stat().st_size!=size:raise ValueError('Truncated response')
                if sha(part)!=expected:raise ValueError('SHA256 mismatch; preserve file for investigation')
                part.replace(dest)
                return dict(path=row['rfilename'],bytes=size,sha256=expected,downloaded=True,transport_host=host)
            except Exception as exc:
                print(json.dumps(dict(stage='download_retry',file=dest.name,host=host,attempt=attempt+1,error=type(exc).__name__,reason_type=type(getattr(exc,'reason',exc)).__name__,partial_bytes=part.stat().st_size if part.exists() else 0)),flush=True)
                if isinstance(exc,ValueError) and 'SHA256' in str(exc):raise
                if stop.is_set():raise
                if attempt<2:time.sleep(2*(attempt+1))
    raise RuntimeError('Pinned data unavailable after bounded retries: '+dest.name)

def download(cfg,out):
    meta_path=ROOT/cfg['source_metadata'];assert sha(meta_path)==cfg['source_metadata_sha256']
    meta=json.loads(meta_path.read_text());assert meta['sha']==cfg['data_revision']
    files=sorted([x for x in meta['siblings'] if x['rfilename'].endswith('.parquet')],key=lambda x:x['rfilename'])
    assert len(files)==1693
    base=ROOT/cfg['dataset_root'];start=time.monotonic();stop=threading.Event();verified=[]
    initial_bytes=sum((base/x['rfilename']).stat().st_size for x in files if (base/x['rfilename']).exists())
    total=sum(x['size'] for x in files)
    it=iter(files)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(download_one,next(it),base,meta['sha'],stop) for _ in range(2)}
        try:
            while pending:
                done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for future in done:
                    verified.append(future.result())
                    row=next(it,None)
                    if row is not None:pending.add(pool.submit(download_one,row,base,meta['sha'],stop))
                received=0
                for row in files:
                    p=base/row['rfilename'];q=p.with_suffix('.parquet.part')
                    for candidate in [p,q]:
                        try:received+=candidate.stat().st_size
                        except FileNotFoundError:pass
                elapsed=time.monotonic()-start;rate=max(0,received-initial_bytes)/max(elapsed,1)
                status=dict(stage='download',verified_files=len(verified),expected_files=1693,received_bytes=received,total_bytes=total,new_bytes_per_second=rate,elapsed_seconds=elapsed,estimated_remaining_seconds=(total-received)/rate if rate>0 else None)
                write(out/'progress.json',status)
                if done or int(elapsed)%10==0:print(json.dumps(status),flush=True)
        except BaseException:
            stop.set()
            for p in pending:p.cancel()
            raise
    write(out/'data_integrity.json',dict(status='PASS',revision=meta['sha'],files=verified,total_bytes=total,elapsed_seconds=time.monotonic()-start))

def numeric(cfg,out,split):
    base=ROOT/cfg['dataset_root'];tasks={x['task_index']:x['task'] for x in map(json.loads,(base/'meta/tasks.jsonl').read_text().splitlines())}
    train={'state':[],'actions':[]};counts={};started=time.monotonic()
    for part in ['train','validation']:
        frames=0;labels=0
        for number,e in enumerate(split[part]):
            eid=e['episode_id'];p=base/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet'
            table=pq.read_table(p,columns=['state','actions','episode_index','frame_index','task_index'],use_threads=False)
            n=e['length'];assert table.num_rows==n and set(table['episode_index'].to_pylist())=={eid}
            assert table['frame_index'].to_pylist()==list(range(n))
            assert all(tasks[t]==e['instruction'] for t in table['task_index'].to_pylist())
            for key,width in [('state',8),('actions',7)]:
                array=np.asarray(table[key].to_pylist(),dtype=np.float64);assert array.shape==(n,width) and np.isfinite(array).all()
                if part=='train':train[key].append(array)
            frames+=n;labels+=sum(min(10,n-i) for i in range(n))
            if number%50==0:write(out/'progress.json',dict(stage='numeric_audit',partition=part,episodes=number+1,total=len(split[part]),elapsed_seconds=time.monotonic()-started))
        counts[part]=dict(episodes=len(split[part]),frames=frames,valid_K10_target_steps=labels)
    stats=fit_normalization({k:np.concatenate(v) for k,v in train.items()});vocabulary=build_vocabulary(split['train']);assert len(vocabulary)==71
    pre=dict(vocabulary=vocabulary,normalization=stats,image_size=128,language_length=32,fit_partition='train_only',train_frames=243000,source_split_sha256=sha(ROOT/cfg['split']),source_integrity_sha256=sha(out/'data_integrity.json'))
    path=ROOT/cfg['preprocessing']
    # Integrity report runtime varies on resume; source revision/split, not runtime,
    # defines stats. Keep the original integrity provenance if stats already exist.
    if path.exists():
        old=json.loads(path.read_text());pre['source_integrity_sha256']=old['source_integrity_sha256']
    path.parent.mkdir(parents=True,exist_ok=True);write_once(path,pre)
    write(out/'numeric_audit.json',dict(status='PASS',counts=counts,vocabulary_size=len(vocabulary),all_tasks=40,preprocessing_sha256=sha(path),elapsed_seconds=time.monotonic()-started))
    return pre,counts

def cache(cfg,out,split,pre):
    base=ROOT/cfg['dataset_root'];root=ROOT/cfg['cache'];root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    binding=dict(split_sha256=sha(ROOT/cfg['split']),preprocessing_sha256=sha(ROOT/cfg['preprocessing']),image_size=128)
    if (root/'complete.json').exists():
        record=json.loads((root/'complete.json').read_text());assert record['binding']==binding
        for part,spec in record['parts'].items():
            for name,shape in spec['shapes'].items():assert list(np.load(root/part/(name+'.npy'),mmap_mode='r').shape)==shape
        write(out/'cache_reference.json',record);return
    parts={}
    for part in ['train','validation']:
        target=root/part;target.mkdir(exist_ok=True);n=sum(e['length'] for e in split[part])
        specs={'images':((n,2,3,128,128),'uint8'),'state':((n,8),'float32'),'actions':((n,1,7),'float32'),'token_ids':((n,32),'int64'),'language_mask':((n,32),'bool'),'episode_id':((n,),'int64'),'task_id':((n,),'int64')}
        maps={}
        for k,(shape,dtype) in specs.items():
            p=target/(k+'.npy');maps[k]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype=dtype,shape=shape)
            assert maps[k].shape==shape and maps[k].dtype==np.dtype(dtype)
        checkpoint=target/'build_progress.json'
        old=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(binding=binding,episodes_done=0,frames_done=0,camera_exact_checks=0)
        assert old['binding']==binding
        offset=sum(e['length'] for e in split[part][:old['episodes_done']]);assert offset==old['frames_done']
        checks=old['camera_exact_checks']
        for number in range(old['episodes_done'],len(split[part])):
            e=split[part][number];eid=e['episode_id'];p=base/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet'
            rows=pq.read_table(p,columns=['image','wrist_image','state','actions','episode_index','frame_index'],use_threads=False).to_pylist();assert len(rows)==e['length']
            ids,mask=encode_instruction(e['instruction'],pre['vocabulary']);global_task=SUITES.index(e['suite'])*10+e['task_id']
            for local,row in enumerate(rows):
                assert row['episode_index']==eid and row['frame_index']==local;i=offset+local
                for camera,key in enumerate(['image','wrist_image']):
                    with Image.open(io.BytesIO(row[key]['bytes'])) as im:rgb=np.asarray(im.convert('RGB').resize((128,128),Image.Resampling.BILINEAR));maps['images'][i,camera]=rgb.transpose(2,0,1)
                    if local==len(rows)//2:
                        np.testing.assert_array_equal(maps['images'][i,camera].astype('float32')/255,decode_image(row[key]).numpy());checks+=1
                for key in ['state','actions']:
                    st=pre['normalization'][key];value=(np.asarray(row[key],dtype='float32')-np.asarray(st['mean'],dtype='float32'))/np.asarray(st['scale'],dtype='float32');maps[key][i]=value if key=='state' else value[None]
                maps['token_ids'][i]=ids.numpy();maps['language_mask'][i]=mask.numpy();maps['episode_id'][i]=eid;maps['task_id'][i]=global_task
            offset+=len(rows)
            for value in maps.values():value.flush()
            write(checkpoint,dict(binding=binding,episodes_done=number+1,frames_done=offset,camera_exact_checks=checks))
            elapsed=time.monotonic()-started;progress=dict(stage='cache',partition=part,episodes=number+1,total_episodes=len(split[part]),frames=offset,total_frames=n,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(n-offset)/max(1,offset-old['frames_done']))
            write(out/'progress.json',progress)
            if number%10==0 or number+1==len(split[part]):print(json.dumps(progress),flush=True)
        assert offset==n and checks==2*len(split[part]);parts[part]=dict(frames=n,episodes=len(split[part]),shapes={k:list(x[0]) for k,x in specs.items()},camera_exact_checks=checks)
        del maps
    record=dict(status='PASS',binding=binding,parts=parts,bytes=sum(p.stat().st_size for p in root.glob('*/*.npy')),elapsed_seconds=time.monotonic()-started,global_task_id='suite_order_index*10+local_task_id')
    assert record['bytes']<26*2**30
    write(root/'complete.json',record);write(out/'cache_reference.json',record)

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.plan).read_text());out=Path(a.run)
    for path,h in cfg['input_sha256'].items():assert sha(ROOT/path)==h,path
    split=json.loads((ROOT/cfg['split']).read_text());assert len(split['train'])==1506 and len(split['validation'])==187
    download(cfg,out);pre,counts=numeric(cfg,out,split);cache(cfg,out,split,pre)
    write(out/'prepare_result.json',dict(status='PASS',train=counts['train'],validation=counts['validation'],preprocessing=cfg['preprocessing'],cache=cfg['cache'],confirmation_accessed=False,new_training_updates=0,new_simulation_episodes=0))

if __name__=='__main__':main()
