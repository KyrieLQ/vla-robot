"""Independent episode extraction on up to4 GPUs, one bounded pipe per worker."""
import concurrent.futures,hashlib,json,os,select,struct,subprocess,threading,time
from pathlib import Path
import numpy as np
from .flow_feature_worker_v1 import Context,write,bits_sha,sha,R

def partitions(entries,count):
    bins=[[] for _ in range(count)];cost=[0]*count
    for e in sorted(entries,key=lambda e:(-(e['end']-e['start'])*e['tokens'],e['episode_id'])):
        k=min(range(count),key=lambda j:(cost[j],j));bins[k].append(e['episode_id']);cost[k]+=(e['end']-e['start'])*e['tokens']
    assert len({eid for part in bins for eid in part})==len(entries)
    return bins

def exact_read(stream,count,timeout=1200):
    data=bytearray(count);view=memoryview(data);offset=0;deadline=time.monotonic()+timeout
    while offset<count:
        remaining=deadline-time.monotonic()
        if remaining<=0 or not select.select([stream.fileno()],[],[],remaining)[0]:raise TimeoutError('Worker result pipe stalled')
        n=stream.readinto(view[offset:])
        if not n:raise EOFError('Worker result pipe closed')
        offset+=n
    return data

def receive(stream,max_bytes=2**30):
    size=struct.unpack('!Q',exact_read(stream,8))[0];assert size<2**20
    header=json.loads(exact_read(stream,size));n=header['nbytes'];assert 0<=n<=max_bytes
    payload=exact_read(stream,n) if n else None
    if header['kind']=='error':raise RuntimeError(header['error'])
    return header,payload

def process_rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    except FileNotFoundError:pass
    return 0

def extract(p,run,front,gate_rows):
    import torch
    from flow_ram_resources_v1 import guard,host_available,current_rss
    started=time.monotonic();run=Path(run);selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text());gpus=selection['selected'];assert 2<=len(gpus)<=4
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(g['uuid'] for g in gpus)
    assert torch.cuda.current_device()==0
    context=Context(p);entries=list(context.layout.values());parts=partitions(entries,len(gpus));order={e['episode_id']:i for i,e in enumerate(entries)}
    tokens={};receipts={};lock=threading.Lock();abort=threading.Event();barrier=threading.Barrier(len(gpus));children=[];logs=[];connections=[];state={'frames':0,'bytes':0};previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    gate_map={(x['episode_id'],x['frame_index']):x for x in gate_rows};gate_seen=set()
    def check_resources():
        guard(p['resources']);aggregate=current_rss()+sum(process_rss(child.pid) for child in children)
        if aggregate>p['resources']['max_process_rss_bytes']:raise RuntimeError('AGGREGATE_PAYLOAD_RSS_LIMIT_REACHED')
        return aggregate
    def commit(worker,eid,value,receipt):
        assert receipt==context.reference[eid] and bits_sha(value)==receipt['feature_sha256']
        with lock:
            if abort.is_set():raise RuntimeError('Parallel extraction cancelled')
            assert eid not in tokens;value.flags.writeable=False;tokens[eid]=value;receipts[eid]=receipt
            for identity,row in gate_map.items():
                if identity[0]==eid:assert bits_sha(value[identity[1]])==row['feature_sha256'];gate_seen.add(identity)
            state['frames']+=len(value);state['bytes']+=value.nbytes;elapsed=time.monotonic()-started;rss=check_resources()
            with (run/'parallel_receipts_journal.jsonl').open('a') as f:f.write(json.dumps(dict(worker=worker,receipt=receipt))+'\n');f.flush()
            write(run/'progress.json',dict(stage='parallel_adapted_RAM_extraction',frames=state['frames'],total_frames=273465,episodes=len(tokens),total_episodes=1693,workers=len(gpus),ram_feature_bytes=state['bytes'],rss_bytes=rss,host_available_bytes=host_available(),elapsed_seconds=elapsed,estimated_remaining_seconds=(273465-state['frames'])*elapsed/state['frames']))
    def drive_parent():
        barrier.wait(timeout=600)
        for eid in parts[0]:
            if abort.is_set():raise RuntimeError('Parallel extraction cancelled')
            def progress(done,total):
                if abort.is_set():raise RuntimeError('Parallel extraction cancelled')
                write(run/'workers/0/progress.json',dict(episode_id=eid,frames=done,total=total))
            value,receipt=context.episode(front,eid,progress);commit(0,eid,value,receipt)
    def drive_child(i,child,stream):
        header,_=receive(stream);assert header['kind']=='ready' and header['gate']['status']=='PASS'
        assert header['gate']['cuda_visible_devices']==gpus[i]['uuid']
        barrier.wait(timeout=600)
        for eid in parts[i]:
            if abort.is_set():raise RuntimeError('Parallel extraction cancelled')
            child.stdin.write((json.dumps(dict(kind='episode',episode_id=eid))+'\n').encode());child.stdin.flush()
            e=context.layout[eid];expected=(e['end']-e['start'])*e['tokens']*960*2
            header,payload=receive(stream,max_bytes=expected);assert header['kind']=='episode' and header['nbytes']==expected
            receipt=header['receipt'];assert receipt['episode_id']==eid
            value=np.frombuffer(payload,dtype=np.uint16).reshape(receipt['shape']);commit(i,eid,value,receipt)
        child.stdin.write(b'{"kind":"stop"}\n');child.stdin.flush();child.stdin.close()
        assert child.wait(timeout=60)==0
    executor=None
    try:
        (run/'workers/0').mkdir(parents=True,exist_ok=True)
        write(run/'progress.json',dict(stage='parallel_frontend_gates',workers=len(gpus)))
        write(run/'parallel_assignment.json',dict(gpus=gpus,episode_ids=parts,rule='Deterministic longest-frame-token-cost-first assignment; canonical receipt order restored after completion'))
        for i in range(1,len(gpus)):
            directory=run/'workers'/str(i);directory.mkdir(parents=True,exist_ok=True)
            readfd,writefd=os.pipe();stream=os.fdopen(readfd,'rb',buffering=0);connections.append(stream)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[i]['uuid'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            log=(directory/'worker.log').open('a');logs.append(log)
            cmd=[str(R/'envs/openpi/bin/python'),'-u','-B','-m','mini_vla.flow_feature_worker_v1','--plan',str(run/'plan.json'),'--run',str(run),'--worker-id',str(i),'--response-fd',str(writefd)]
            try:child=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,pass_fds=(writefd,))
            finally:os.close(writefd)
            children.append(child)
        executor=concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus))
        futures=[executor.submit(drive_parent)]+[executor.submit(drive_child,i,child,stream) for i,(child,stream) in enumerate(zip(children,connections),1)]
        for future in concurrent.futures.as_completed(futures):future.result()
        assert len(tokens)==1693 and state['frames']==273465 and state['bytes']==p['resources']['ram_cache_bytes'] and len(gate_seen)==120
        canonical=[receipts[eid] for eid in sorted(receipts,key=order.get)]
        for row in canonical:assert bits_sha(tokens[row['episode_id']])==row['feature_sha256']
        path=run/'cache_receipts.jsonl';temp=path.with_suffix('.tmp');temp.write_text(''.join(json.dumps(x)+'\n' for x in canonical));temp.replace(path)
        assert sha(path)==sha(R/p['reference_adapted_receipts'])
        write(run/'ram_cache_acceptance.json',dict(status='PASS',frames=273465,episodes=1693,bytes=state['bytes'],all_source_hashes_verified=True,all_original_input_hashes_verified=True,allRAM_receipts_second_pass_verified=True,fixed120_gate_matches=True,all1693_features_equal_original_adapted_receipts=True,parallel_workers=len(gpus),elapsed_seconds=time.monotonic()-started,cache_storage='processRAM_only;not persistent;old disk cache untouched',frontend_checkpoint_sha256=p['backbone_checkpoint_sha256'],receipt_sha256=sha(path)))
        return tokens
    except BaseException:
        abort.set()
        barrier.abort()
        for child in children:
            if child.poll() is None:child.terminate()
        raise
    finally:
        for child in children:
            if child.poll() is None:
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
        for stream in connections:stream.close()
        for log in logs:log.close()
        torch.set_num_threads(previous_threads)
