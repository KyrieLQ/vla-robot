"""One independent batch1 VLM worker; raw BF16 transfer over a private pipe."""
import argparse,hashlib,json,os,struct,sys,time,traceback
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*2**20),b''):h.update(chunk)
    return h.hexdigest()
def bits_sha(a):return hashlib.sha256(memoryview(a).cast('B')).hexdigest()
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def send(stream,header,array=None):
    def all_bytes(value):
        view=memoryview(value).cast('B')
        while len(view):
            written=stream.write(view)
            if not written:raise EOFError('Broken result pipe')
            view=view[written:]
    header=dict(header,nbytes=0 if array is None else array.nbytes)
    data=json.dumps(header,separators=(',',':')).encode();all_bytes(struct.pack('!Q',len(data)));all_bytes(data)
    if array is not None:all_bytes(array)
    stream.flush()

class Context:
    def __init__(self,p):
        self.p=p
        split=json.loads((R/p['split']).read_text())
        self.meta={e['episode_id']:e for part in ['train','validation'] for e in split[part]}
        self.layout={e['episode_id']:{**e,'part':part} for part in ['train','validation'] for e in json.loads((R/p['cache']/part/'index.json').read_text())}
        self.source={x['path']:x['sha256'] for x in json.loads((R/p['source_integrity']).read_text())['files']}
        self.reference={x['episode_id']:x for x in map(json.loads,(R/p['reference_adapted_receipts']).read_text().splitlines())}
        assert len(self.layout)==len(self.reference)==1693
    def episode(self,front,eid,progress=None):
        import pyarrow.parquet as pq
        from .vlm_frontend_v1 import decode_record,bits_bf16
        e=self.layout[eid];meta=self.meta[eid];n=e['end']-e['start'];assert n==meta['length']
        rel=f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet';path=R/self.p['dataset_root']/rel
        assert sha(path)==self.source[rel]
        rows=pq.read_table(path,columns=['image','wrist_image','episode_index','frame_index'],use_threads=False).to_pylist();assert len(rows)==n
        hashes=json.loads((R/self.p['cache']/e['directory']/'input_hashes.json').read_text());values=np.empty((n,e['tokens'],960),np.uint16);digest=hashlib.sha256()
        for j,row in enumerate(rows):
            assert row['episode_index']==eid and row['frame_index']==j
            feature=front.encode([decode_record(row,meta['instruction'])])[0];bits=feature['bits']
            assert bits.shape==(e['tokens'],960)
            expected=hashes[j]['input_sha256'] if isinstance(hashes[j],dict) else hashes[j]
            assert feature['input_sha256']==expected;values[j]=bits;digest.update(expected.encode())
            if progress is not None and (j+1)%100==0:progress(j+1,n)
        assert np.isfinite(bits_bf16(values).float().numpy()).all()
        receipt=dict(part=e['part'],episode_id=eid,frames=n,shape=list(values.shape),dtype='uint16 rawBF16',feature_sha256=bits_sha(values),source_sha256=self.source[rel],input_sequence_sha256=digest.hexdigest())
        assert receipt==self.reference[eid],('Not bitwise identical to accepted adapted features',eid)
        values.flags.writeable=False
        return values,receipt

def load_front(p):
    import torch
    from .policy_paired_v1 import AdaptedFrontend
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.device_count()==1
    file=R/p['backbone_checkpoint'];assert sha(file)==p['backbone_checkpoint_sha256']
    ck=torch.load(file,map_location='cpu',weights_only=False);front=AdaptedFrontend(R/p['model_directory'],ck['adapters'])
    assert not any(x.requires_grad for x in front.model.parameters())
    return front

def gate(p,front):
    from .paired_continuation_v1 import RawData
    started=time.monotonic();data=RawData(p,'train');index={e['episode_id']:e for e in data.index}
    samples=json.loads((R/p['sample_manifest']).read_text())['samples'];ref=R/p['reference_adapted_gate_run'];rows=json.loads((ref/'frontend_gate.json').read_text())['rows']
    for s,g in zip(samples,rows):
        assert (s['episode_id'],s['frame_index'])==(g['episode_id'],g['frame_index'])
        item=data[index[s['episode_id']]['start']+s['frame_index']];f=ref/'gate'/g['file'];assert sha(f)==g['file_sha256'];expected=np.load(f)
        one=front.encode([item['record']])[0];two=front.encode([item['record']])[0]
        assert one['input_sha256']==g['input_sha256'] and np.array_equal(one['bits'],expected) and np.array_equal(two['bits'],expected)
    seconds=time.monotonic()-started
    return dict(status='PASS',samples=120,repeat_features_bitwise_equal=True,all_saved_adapted_gate_features_equal=True,loading=front.loading,elapsed_seconds=seconds,encode_frames=240,frames_per_second=240/seconds,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);ap.add_argument('--worker-id',type=int,required=True);ap.add_argument('--response-fd',type=int);ap.add_argument('--gate-only',action='store_true');a=ap.parse_args()
    p=json.loads(Path(a.plan).read_text());out=Path(a.run)/'workers'/str(a.worker_id);out.mkdir(parents=True,exist_ok=True)
    stream=os.fdopen(a.response_fd,'wb',buffering=0) if a.response_fd is not None else None
    try:
        front=load_front(p);result=gate(p,front);write(out/'gate.json',result)
        if a.gate_only:return
        assert stream is not None;ctx=Context(p);send(stream,dict(kind='ready',gate=result))
        for line in sys.stdin:
            request=json.loads(line)
            if request['kind']=='stop':break
            assert request['kind']=='episode';eid=int(request['episode_id']);began=time.monotonic()
            def progress(done,total):write(out/'progress.json',dict(stage='extracting',episode_id=eid,frames_in_episode=done,total_in_episode=total))
            values,receipt=ctx.episode(front,eid,progress)
            send(stream,dict(kind='episode',receipt=receipt,elapsed_seconds=time.monotonic()-began),values)
            write(out/'progress.json',dict(stage='episode_committed_to_pipe',episode_id=eid))
        write(out/'status.json',dict(status='COMPLETE',pid=os.getpid()))
    except BaseException:
        error=traceback.format_exc();write(out/'status.json',dict(status='FAILED',error=error,pid=os.getpid()))
        if stream is not None:
            try:send(stream,dict(kind='error',error=error))
            except Exception:pass
        raise
    finally:
        if stream is not None:stream.close()
if __name__=='__main__':main()
