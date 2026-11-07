#!/usr/bin/env python3
"""Fetch only the pinned spatial demonstration files; preserve smoke artifacts."""
import ast, concurrent.futures, datetime, hashlib, json, os, pathlib, time, urllib.request
from storage_guard import check

ROOT=pathlib.Path(os.environ['PROJECT_ROOT'])
def main():
    source=json.loads((ROOT/'reports/hf_dataset_metadata.json').read_text())
    info=json.loads((ROOT/'reports/source_metadata/meta/info.json').read_text())
    tree=ast.parse((ROOT/'repos/openpi/third_party/libero/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='libero_task_map' for x in n.targets))
    tasks=[{'task_id':i,'task_name':n,'language':n.replace('_',' ')} for i,n in enumerate(mapping['libero_spatial'])]
    task_ids={t['language']:t['task_id'] for t in tasks}
    episodes=[json.loads(s) for s in (ROOT/'reports/source_metadata/meta/episodes.jsonl').read_text().splitlines()]
    selected=sorted([e for e in episodes if len(e['tasks'])==1 and e['tasks'][0] in task_ids],key=lambda e:e['episode_index'])
    assert len(selected)==432 and sum(e['length'] for e in selected)==52970
    manifest={'suite':'libero_spatial','task_order_index':0,'source_repo':'physical-intelligence/libero',
      'source_revision':source['sha'],'codebase_version':info['codebase_version'],'selection':'all available complete trajectories for all 10 exact language-matched spatial tasks',
      'tasks':tasks,'task_ids':list(range(10)),'episodes':selected,'episode_ids':[e['episode_index'] for e in selected],
      'frames':sum(e['length'] for e in selected),'camera_storage':'embedded parquet images',
      'performance_claim':'one training seed; paired 50-episode spatial evaluation; not full LIBERO benchmark',
      'episode_records':[dict(e,task_ids=[task_ids[e['tasks'][0]]]) for e in selected]}
    m=ROOT/'reports/spatial_full_manifest.json'
    if m.exists():assert json.loads(m.read_text())==manifest,'Do not alter a frozen manifest'
    else:m.write_text(json.dumps(manifest,indent=2)+'\n')
    want={info['data_path'].format(episode_chunk=e['episode_index']//info['chunks_size'],episode_index=e['episode_index']) for e in selected}
    files=[x for x in source['siblings'] if x['rfilename'] in want];assert len(files)==432
    base=ROOT/'datasets/lerobot/physical-intelligence/libero'
    remaining=sum(x['size'] for x in files if not (base/x['rfilename']).exists())
    storage=check(remaining+40*2**30)
    plan={'revision':source['sha'],'files':files,'download_remaining_bytes':remaining,'checkpoint_reserve_bytes':40*2**30,'storage':storage}
    (ROOT/'reports/spatial_download_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def download(x):
        p=base/x['rfilename'];p.parent.mkdir(parents=True,exist_ok=True);part=p.with_suffix(p.suffix+'.part')
        expected=x['lfs']['sha256']
        if not p.exists():
            url=f"https://huggingface.co/datasets/physical-intelligence/libero/resolve/{source['sha']}/{x['rfilename']}"
            for attempt in range(3):
                try:
                    offset=part.stat().st_size if part.exists() else 0
                    if offset==x['size']:break
                    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
                    with urllib.request.urlopen(req,timeout=60) as src:
                        if offset and src.status==206:
                            assert src.headers['Content-Range'].startswith(f'bytes {offset}-');mode='ab'
                        elif src.status==200:mode='wb'
                        else:raise RuntimeError(f'Unexpected HTTP status {src.status}')
                        with part.open(mode) as f:
                            while chunk:=src.read(2**20):f.write(chunk)
                    assert part.stat().st_size==x['size'];break
                except Exception as exc:
                    print('DOWNLOAD_RETRY',p.name,attempt+1,type(exc).__name__,flush=True)
                    if attempt==2:raise
                    time.sleep(2*(attempt+1))
        target=p if p.exists() else part
        assert target.stat().st_size==x['size']
        h=hashlib.sha256()
        with target.open('rb') as f:
            while chunk:=f.read(8*2**20):h.update(chunk)
        assert h.hexdigest()==expected,('checksum mismatch',p.name)
        if target==part:part.replace(p)
        return x['size']
    start=time.monotonic();done_bytes=0;done_count=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(download,x) for x in files}
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=10,return_when=concurrent.futures.FIRST_EXCEPTION)
            for f in done:done_bytes+=f.result();done_count+=1
            elapsed=time.monotonic()-start
            print(json.dumps({'stage':'download','verified_files':done_count,'total_files':len(files),'verified_GiB':done_bytes/2**30,'elapsed_seconds':round(elapsed),'remaining_seconds_estimate':round(elapsed*(len(files)-done_count)/done_count) if done_count else None}),flush=True)
    (ROOT/'reports/spatial_assets_integrity.json').write_text(json.dumps({'status':'PASS','file_count':len(files),'total_bytes':done_bytes,'elapsed_seconds':time.monotonic()-start,'completed_at':datetime.datetime.now().astimezone().isoformat()},indent=2)+'\n')
if __name__=='__main__':main()
