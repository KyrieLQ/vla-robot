#!/usr/bin/env python3
import json, os, pathlib, subprocess, time
import jax
import jax.numpy as jnp

root=pathlib.Path(os.environ['PROJECT_ROOT']);selected=json.loads(pathlib.Path(os.environ['VLA_GPU_SELECTION']).read_text())['selected']
assert jax.default_backend()=='gpu' and len(jax.devices())==len(selected)
start=time.monotonic();sums=[]
for device in jax.devices():
    x=jax.device_put(jnp.arange(256,dtype=jnp.float32).reshape(16,16),device)
    y=jax.jit(lambda z:z@z.T)(x);jax.block_until_ready(y)
    assert bool(jnp.isfinite(y).all());sums.append(float(y.sum()))
# A small sharded reduction also checks the two-device execution path.
collective_sum=None
if len(selected)>1:
    import numpy as np
    mesh=jax.make_mesh((len(selected),),('devices',))
    split=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec('devices'))
    array=jax.device_put(np.arange(32,dtype=np.float32),split)
    reduced=jax.jit(lambda v:jnp.sum(v),in_shardings=split)(array)
    jax.block_until_ready(reduced);collective_sum=float(reduced);assert collective_sum==496.0
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
own=[line.split(',')[0].strip() for line in apps.splitlines() if len(line.split(','))==2 and line.split(',')[1].strip()==str(os.getpid())]
assert set(own)==set(g['uuid'] for g in selected)
report={'status':'PASS','backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],
        'own_process_gpu_uuids':own,'seconds_including_first_compile':time.monotonic()-start,'per_device_sums':sums,'sharded_reduction_sum':collective_sum}
(root/'reports/gpu_probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
