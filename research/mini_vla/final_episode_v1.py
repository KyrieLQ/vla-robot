"""Testable final episode loop; action journal survives interruptions and retains outcomes."""
from collections import deque
import hashlib,json,os,time
from pathlib import Path
import numpy as np
def load_journal(path):
    records=[]
    with Path(path).open('rb') as f:
        for raw in f:
            if not raw.endswith(b'\n'):break
            records.append(json.loads(raw))
    return records
def run_episode(env,obs,infer,observation,max_steps,journal,writer=None):
    queue=deque();steps=0;calls=0;success=False
    def emit(row,terminal=False):
        journal.write(json.dumps(row,separators=(',',':'))+'\n');journal.flush()
        if terminal:os.fsync(journal.fileno())
    for step in range(max_steps):
        external,wrist,state,prompt=observation(obs)
        if writer is not None:writer.append_data(external)
        if not queue:
            tick=time.monotonic();result=infer({'observation/image':external,'observation/wrist_image':wrist,'observation/state':state,'prompt':prompt});latency=time.monotonic()-tick
            chunk=np.asarray(result['actions']);assert chunk.shape==(10,7) and np.isfinite(chunk).all() and np.max(np.abs(chunk))<=1.000001
            ih=hashlib.sha256(external.tobytes()+wrist.tobytes()+np.asarray(state,dtype='<f4').tobytes()+prompt.encode()).hexdigest()
            emit(dict(kind='call',step=step,actions=chunk.tolist(),input_sha256=ih,latency=latency,server_ms=result['server_timing']['infer_ms']));queue.extend(chunk[:5]);calls+=1
        action=queue.popleft();obs,_,done,_=env.step(action.tolist());assert bool(done)==bool(env.check_success());steps+=1;success=bool(done)
        emit(dict(kind='step',step=step,action=action.tolist(),state=np.asarray(state).tolist(),done=success,qpos=env.sim.data.qpos.tolist(),qvel=env.sim.data.qvel.tolist()),terminal=success or steps==max_steps)
        if success:break
    if writer is not None:writer.append_data(observation(obs)[0])
    return dict(task_success=success,steps=steps,policy_calls=calls)
def validate_records(records,limit):
    calls=[r for r in records if r['kind']=='call'];steps=[r for r in records if r['kind']=='step']
    assert calls and steps
    assert [r['step'] for r in steps]==list(range(len(steps)))
    assert [r['step'] for r in calls]==list(range(0,len(steps),5))
    for i,s in enumerate(steps):
        np.testing.assert_array_equal(np.asarray(s['action']),np.asarray(calls[i//5]['actions'][i%5]))
        assert not s['done'] or i==len(steps)-1
    assert steps[-1]['done'] or len(steps)==limit
    return dict(task_success=bool(steps[-1]['done']),steps=len(steps),policy_calls=len(calls))
