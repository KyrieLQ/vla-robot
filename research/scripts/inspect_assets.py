#!/usr/bin/env python3
import json, os, pathlib, urllib.parse, urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
def get(url):
    with urllib.request.urlopen(url,timeout=45) as r:return json.load(r)

def main():
    frozen=[ROOT/'reports/base_objects.json',ROOT/'reports/hf_dataset_metadata.json']
    frozen += [ROOT/'reports/source_metadata/meta'/n for n in ['info.json','tasks.jsonl','episodes.jsonl']]
    if all(p.is_file() for p in frozen):
        print('Reusing frozen object generations and dataset revision; existing source manifests are not overwritten.')
        return
    items=[];token=None
    while True:
        q={'prefix':'checkpoints/pi05_base/','fields':'items(name,size,md5Hash,generation),nextPageToken','maxResults':'1000'}
        if token:q['pageToken']=token
        d=get('https://storage.googleapis.com/storage/v1/b/openpi-assets/o?'+urllib.parse.urlencode(q))
        items.extend(d.get('items',[]));token=d.get('nextPageToken')
        if not token:break
    (ROOT/'reports/base_objects.json').write_text(json.dumps(items,indent=2)+'\n')
    print('pi05_base objects',len(items),'GiB',sum(int(x['size']) for x in items)/2**30,flush=True)
    info=get('https://huggingface.co/api/datasets/physical-intelligence/libero/revision/a4336d589d589045d1c56423ffdf3b88a0e19b1f?blobs=true')
    (ROOT/'reports/hf_dataset_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    print('dataset_revision',info['sha'],'files',len(info['siblings']),'GiB',sum(x.get('size',0) for x in info['siblings'])/2**30,flush=True)
    for name in ['meta/info.json','meta/tasks.jsonl','meta/episodes.jsonl']:
        url='https://huggingface.co/datasets/physical-intelligence/libero/resolve/'+info['sha']+'/'+name
        with urllib.request.urlopen(url,timeout=45) as r:content=r.read()
        out=ROOT/'reports/source_metadata'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(content)
    meta=json.loads((ROOT/'reports/source_metadata/meta/info.json').read_text())
    print('dataset_info',json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
