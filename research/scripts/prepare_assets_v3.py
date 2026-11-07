#!/usr/bin/env python3
"""Download immutable official assets only, with bounded retries and checksum verification."""
import argparse, ast, base64, concurrent.futures, hashlib, json, os, pathlib, subprocess, time, urllib.parse, urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--full-spatial',action='store_true');a=ap.parse_args()
    report=ROOT/'reports'
    source=json.loads((report/'hf_dataset_metadata.json').read_text());revision=source['sha']
    info=json.loads((report/'source_metadata/meta/info.json').read_text())
    task_file=ROOT/'repos/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py'
    tree=ast.parse(task_file.read_text());mapping=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in node.targets):mapping=ast.literal_eval(node.value)
    assert mapping is not None
    names=mapping['libero_spatial'];language=names[0].replace('_',' ')
    episodes=[json.loads(s) for s in (report/'source_metadata/meta/episodes.jsonl').read_text().splitlines()]
    matching=[e for e in episodes if e['tasks']==[language]]
    assert len(matching)>=3, ('No exact task-language mapping',language)
    selected=([e for e in episodes if any(n.replace('_',' ') in e['tasks'] for n in names)] if a.full_spatial else [matching[i] for i in [0,(len(matching)-1)//2,len(matching)-1]])
    task_rows=[json.loads(s) for s in (report/'source_metadata/meta/tasks.jsonl').read_text().splitlines()]
    task_index=next(t['task_index'] for t in task_rows if t['task']==language)
    manifest={'suite':'libero_spatial','task_id':0,'task_order_index':0,'task_name':names[0],'language':language,
              'source_task_index':task_index,'source_repo':'physical-intelligence/libero','source_revision':revision,
              'codebase_version':info['codebase_version'],'all_matched_episode_count':len(matching),
              'selection':'all spatial tasks' if a.full_spatial else 'first, middle, last complete episode of task 0 ordered by original episode_index',
              'episodes':selected,'episode_ids':[e['episode_index'] for e in selected],
              'frames':sum(e['length'] for e in selected),'camera_storage':'embedded parquet images; no video files',
              'eval_seeds':[7,8],'eval_initial_state_indices':[0,1],
              'performance_claim':'deployment smoke only, not benchmark or generalization evidence'}
    if a.full_spatial:
        task_details=[{'task_id':i,'task_name':n,'language':n.replace('_',' ')} for i,n in enumerate(names)]
        language_ids={t['language']:t['task_id'] for t in task_details}
        manifest.update(task_id=None,task_name=None,language=None,source_task_index=None,
                        task_ids=list(range(len(names))),tasks=task_details,
                        episode_records=[dict(e,task_ids=[language_ids[t] for t in e['tasks']]) for e in selected])
    manifest_path=report/('spatial_full_manifest.json' if a.full_spatial else 'data_manifest.json')
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==manifest,'Frozen manifest differs'
    else:manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    want={info['data_path'].format(episode_chunk=e['episode_index']//info['chunks_size'],episode_index=e['episode_index']) for e in selected}
    files=[]
    for x in source['siblings']:
        n=x['rfilename']
        if n.startswith('meta/') or n in want:
            files.append({'url':f'https://huggingface.co/datasets/physical-intelligence/libero/resolve/{revision}/{n}',
                          'path':str(ROOT/'datasets/lerobot/physical-intelligence/libero'/n),'size':x['size'],
                          'sha256':x.get('lfs',{}).get('sha256')})
    base_objects=json.loads((report/'base_objects.json').read_text())
    for x in base_objects:
        files.append({'url':'https://storage.googleapis.com/openpi-assets/'+urllib.parse.quote(x['name'],safe='/')+'?generation='+x['generation'],
                      'path':str(ROOT/'cache/openpi/openpi-assets'/x['name']),'size':int(x['size']),'md5':x.get('md5Hash')})
    with urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/big_vision/o/paligemma_tokenizer.model',timeout=30) as r:tok=json.load(r)
    files.append({'url':'https://storage.googleapis.com/big_vision/paligemma_tokenizer.model?generation='+tok['generation'],
                  'path':str(ROOT/'cache/openpi/big_vision/paligemma_tokenizer.model'),'size':int(tok['size']),'md5':tok.get('md5Hash')})
    usage=int(subprocess.check_output(['du','-s','-B1',str(ROOT)],text=True).split()[0])
    remain=sum(max(0,x['size']-(pathlib.Path(x['path']).stat().st_size if pathlib.Path(x['path']).exists() else 0)) for x in files)
    free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
    budget={'existing_bytes':usage,'download_remaining_bytes':remain,'checkpoint_and_install_reserve_bytes':60*2**30,
            'budget_bytes':120*2**30,'filesystem_free_bytes':free,'minimum_free_bytes':100*2**30}
    (report/'download_plan.json').write_text(json.dumps({'budget':budget,'files':files},indent=2)+'\n')
    assert usage+remain+60*2**30<=120*2**30, 'Project budget would be exceeded'
    assert free-remain-60*2**30>=100*2**30, 'Filesystem reserve would be breached'
    print('frozen task',language,'episodes',manifest['episode_ids'],'frames',manifest['frames'],flush=True)
    print('download GiB',remain/2**30,'reserve GiB',60,flush=True)
    def download(x):
        dest=pathlib.Path(x['path']);dest.parent.mkdir(parents=True,exist_ok=True)
        part=dest.with_name(dest.name+'.part')
        if not dest.exists():
            if x['size']==0:part.touch(exist_ok=True)
            # Preserve partial downloads and verify Content-Range before appending.
            for attempt in range(3):
                try:
                    offset=part.stat().st_size if part.exists() else 0
                    if offset==x['size']:break
                    headers={'Range':f'bytes={offset}-'} if offset else {}
                    # A direct official GCS route was independently verified after proxy TLS failures.
                    opener=(urllib.request.build_opener(urllib.request.ProxyHandler({}))
                            if 'storage.googleapis.com/' in x['url'] else urllib.request.build_opener())
                    with opener.open(urllib.request.Request(x['url'],headers=headers),timeout=60) as src:
                        if offset and src.status==206:
                            assert src.headers['Content-Range'].startswith(f'bytes {offset}-')
                            mode='ab'
                        elif src.status==200:mode='wb'
                        else:raise RuntimeError(f'Unexpected HTTP status {src.status}')
                        with part.open(mode) as out:
                            while True:
                                chunk=src.read(1024**2)
                                if not chunk:break
                                out.write(chunk)
                    break
                except Exception as exc:
                    print('download attempt',attempt+1,'failed',dest.name,type(exc).__name__,flush=True)
                    if attempt==2:raise
                    time.sleep(2*(attempt+1))
            assert part.stat().st_size==x['size'],('size mismatch',str(part))
            target=part
        else:target=dest
        assert target.stat().st_size==x['size'],('size mismatch',str(target))
        if x.get('sha256') or x.get('md5'):
            h=hashlib.sha256() if x.get('sha256') else hashlib.md5()
            with target.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
            actual=h.hexdigest() if x.get('sha256') else base64.b64encode(h.digest()).decode()
            assert actual==(x.get('sha256') or x['md5']),('checksum mismatch',str(target))
        if target==part:part.replace(dest)
        return str(dest.relative_to(ROOT))
    def received():
        n=0
        for x in files:
            p=pathlib.Path(x['path']);part=p.with_name(p.name+'.part')
            for f in [p,part]:
                try:n+=f.stat().st_size
                except FileNotFoundError:pass
        return n
    start=time.monotonic();initial=received();total=sum(x['size'] for x in files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(download,x):x for x in files};pending=set(fs);errors=[]
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_EXCEPTION)
            for f in done:
                try:f.result()
                except Exception as exc:
                    errors.append({'file':fs[f]['path'],'error':str(exc)})
                    print('file failure',errors[-1],flush=True)
            n=received();elapsed=time.monotonic()-start;rate=(n-initial)/max(elapsed,1)
            print('progress %.3f/%.3f GiB %.2f MiB/s remaining_seconds=%s files_pending=%d'%(n/2**30,total/2**30,rate/2**20,round((total-n)/rate) if rate>0 else 'unknown',len(pending)),flush=True)
    if errors:raise RuntimeError(str(errors))
    (report/'assets_integrity.json').write_text(json.dumps({'status':'PASS','file_count':len(files),'total_bytes':total,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
if __name__=='__main__':main()
