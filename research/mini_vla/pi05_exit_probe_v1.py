"""Bounded process-exit diagnosis. No policy inference, data samples or updates."""
import argparse,atexit,faulthandler,json,os,resource,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);ap.add_argument('--child',choices=['jax_natural','checkpoint_natural','checkpoint_committed_exit']);a=ap.parse_args()
    p=json.loads(Path(a.plan).read_text());run=Path(a.run).absolute();run.mkdir(exist_ok=True);started=time.monotonic()
    if a.child:
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        faulthandler.enable(all_threads=True)
        import jax
        import jax.numpy as jnp
        import numpy as np
        assert jax.default_backend()=='gpu' and jax.device_count()==4
        params=None
        if a.child!='jax_natural':
            import torch
            from openpi.models import model
            params=model.restore_params(R/p['official_base'],restore_type=np.ndarray)
            assert sum(x.size for x in jax.tree.leaves(params))>3_000_000_000
        mesh=jax.make_mesh((4,),('data',));layout=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('data'))
        x=jax.device_put(np.arange(64,dtype=np.float32).reshape(4,16),layout)
        result=jax.jit(lambda value:value*value+1,in_shardings=layout,out_shardings=layout)(x);jax.block_until_ready(result)
        np.testing.assert_array_equal(np.asarray(jax.device_get(result)),np.arange(64,dtype=np.float32).reshape(4,16)**2+1)
        report=dict(status='PASS_BODY',case=a.child,devices=4,read_base=params is not None,policy_inferences=0,optimizer_updates=0,dataset_inputs=0,elapsed_seconds=time.monotonic()-started)
        write(run/'body.json',report);print(json.dumps(report),flush=True)
        atexit.register(lambda:print('PYTHON_ATEXIT_REACHED',flush=True))
        if a.child=='checkpoint_committed_exit':
            # All device work finished and the only result file is fsynced. No
            # asynchronous checkpoints, data-loader children or pending writes.
            sys.stdout.flush();sys.stderr.flush();os._exit(0)
        return
    outcomes=[]
    for case in ['jax_natural','checkpoint_natural','checkpoint_committed_exit']:
        d=run/case;d.mkdir();write(run/'progress.json',dict(stage='bounded_exit_diagnosis',case=case,elapsed_seconds=time.monotonic()-started))
        with (d/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u','-B','-m','mini_vla.pi05_exit_probe_v1','--plan',a.plan,'--run',str(d),'--child',case],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONFAULTHANDLER='1'))
            try:code=child.wait(timeout=240)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=30);code=child.returncode
        result=dict(case=case,pid=child.pid,returncode=code,signal_number=-code if code<0 else None,body_committed=(d/'body.json').exists())
        write(d/'exit.json',result);outcomes.append(result)
    assert outcomes[-1]['body_committed'] and outcomes[-1]['returncode']==0
    write(run/'report.json',dict(status='PASS_DIAGNOSIS_COMPLETE',cases=outcomes,new_optimizer_updates=0,new_policy_inferences=0,new_dataset_inputs=0,elapsed_seconds=time.monotonic()-started,limitations='Smallbackend/base-loadinglifecycle test;doesnotreplayorreplacecompletedVLAupdates. Committedexitavoidsinterpreterfinalizersonlyafterdurableallpassedbody.'))
    write(run/'progress.json',dict(stage='complete',elapsed_seconds=time.monotonic()-started))
if __name__=='__main__':main()
