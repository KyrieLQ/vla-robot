"""CPU-only durable records, archive inspection and complete validation scoring."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def digest_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def inspect_checkpoint(path,binding):
    with np.load(path,allow_pickle=False) as values:
        m=json.loads(values['metadata'].tobytes().decode());assert m['format']=='pi05_single_archive_v2' and m['binding']==binding
        for e in m['leaves']:
            x=values[e['name']];assert list(x.shape)==e['shape'] and str(x.dtype)==e['dtype'] and np.isfinite(x).all()
            assert hashlib.sha256(x.tobytes()).hexdigest()==e['sha256']
        if m['kind']=='optimizer_state':
            counts=[int(values[e['name']]) for e in m['leaves'] if e['path'].endswith('.count')]
            assert counts and all(c==m['step'] for c in counts)
    return m
def verify_log_segments(segments):
    expected_step=1
    for s in segments:
        path=R/s['path'];h=hashlib.sha256();seen=0;used=0
        with path.open('rb') as f:
            while used<s['bytes']:
                line=f.readline();assert line and used+len(line)<=s['bytes'];used+=len(line);h.update(line);row=json.loads(line)
                assert row['step']==expected_step;expected_step+=1;seen+=1
        assert seen==s['rows'] and h.hexdigest()==s['sha256']
    return expected_step-1

def score_predictions(file,plan):
    pred=np.load(file,mmap_mode='r');assert pred.shape==(30465,10,32) and pred.dtype==np.float32 and np.isfinite(pred).all()
    entries=json.loads((R/plan['split']).read_text())['validation'];norm=json.loads((R/plan['norm_stats']).read_text())['norm_stats']['actions'];q01=np.asarray(norm['q01']);q99=np.asarray(norm['q99'])
    first=chunk=first5=0.;valid=valid5=frames=tail_frames=0;tail=non_tail=0.;dims=np.zeros(7);per_suite={};per_episode=[]
    for e in entries:
        eid=e['episode_id'];path=R/plan['dataset_root']/('data/chunk-%03d/episode_%06d.parquet'%(eid//1000,eid))
        table=pq.read_table(path,columns=['actions','episode_index','frame_index'],use_threads=False).to_pydict();raw=np.asarray(table['actions'],np.float32);n=e['length']
        assert raw.shape==(n,7) and set(table['episode_index'])=={eid};np.testing.assert_array_equal(table['frame_index'],np.arange(n))
        target=((raw-q01)/(q99-q01+1e-6)*2-1).astype(np.float32)
        offsets=np.arange(n)[:,None]+np.arange(10)[None];mask=offsets<n;targets=target[np.minimum(offsets,n-1)]
        error=np.square(pred[frames:frames+n,:,:7].astype(np.float64)-targets.astype(np.float64));a=float(error[:,0].sum());b=float(error[mask].sum());tailmask=~mask[:,-1]
        first+=a;chunk+=b;valid+=int(mask.sum());first5+=float(error[:,:5][mask[:,:5]].sum());valid5+=int(mask[:,:5].sum());dims+=error[:,0].sum(0)
        tail+=float(error[tailmask,0].sum());non_tail+=float(error[~tailmask,0].sum());tail_frames+=int(tailmask.sum())
        s=per_suite.setdefault(e['suite'],dict(frames=0,first_squared_sum=0.));s['frames']+=n;s['first_squared_sum']+=a
        per_episode.append(dict(episode_id=eid,frames=n,first_mse=a/(n*7)));frames+=n
    assert frames==30465 and valid==296235 and len(entries)==187
    return dict(frames=frames,episodes=187,valid_target_steps=valid,first_mse=first/(frames*7),valid_chunk_mse=chunk/(valid*7),first5_mse=first5/(valid5*7),first_per_dimension=(dims/frames).tolist(),tail_frames=tail_frames,tail_first_mse=tail/(tail_frames*7),non_tail_first_mse=non_tail/((frames-tail_frames)*7),by_suite={k:dict(frames=v['frames'],first_mse=v['first_squared_sum']/(v['frames']*7)) for k,v in per_suite.items()},per_episode=per_episode)
