"""Prefer four GPUs, compare two on the identical global batch, never start training."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from .pipeline_v1 import ROOT,PY,write
from .pi05_compact_state_v1 import sha
from .preflight_exit_record_v1 import record_exit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);a=ap.parse_args();pp=Path(a.plan).absolute();p=json.loads(pp.read_text());run=Path(a.run).absolute();began=time.monotonic();child=None
    selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text());gpus=selection['selected'];assert len(gpus)>=2
    primary=4 if len(gpus)>=4 else 2;modes=[primary]+([2] if primary==4 else [])
    def interrupted(signum,frame):raise InterruptedError('pi05preflight interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for mode_index,count in enumerate(modes):
            directory=run/('gpu%d'%count);directory.mkdir();env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in gpus[:count]),JAX_ENABLE_COMPILATION_CACHE='false',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),CUDA_CACHE_MAXSIZE='67108864',TRITON_CACHE_DIR=str(ROOT/'cache/triton'))
            write(directory/'gpu_binding.json',dict(selected=gpus[:count],actual_count=count,original_selection=os.environ['VLA_GPU_SELECTION']))
            cmd=[str(PY),'-u','-B','-m','mini_vla.pi05_preflight_worker_v1','--plan',str(pp),'--out',str(directory)]
            if mode_index:cmd+=['--reference',str(run/('gpu%d'%primary))]
            with (directory/'worker.log').open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);last=None
                while child.poll() is None:
                    source=directory/'progress.json'
                    if source.exists():
                        stamp=source.stat().st_mtime_ns
                        if stamp!=last:write(run/'progress.json',dict(stage='pi05_preflight',current_devices=count,worker=json.loads(source.read_text()),elapsed_seconds=time.monotonic()-began));last=stamp
                    else:write(run/'progress.json',dict(stage='pi05_preflight_worker_startup',current_devices=count,elapsed_seconds=time.monotonic()-began))
                    time.sleep(3)
                exit_record=record_exit(directory,child.returncode)
                if child.returncode!=0:raise RuntimeError('pi05preflight gpu%d returncode=%d handling=%s; preservecompleteartifacts,no automaticrepeat'%(count,child.returncode,exit_record['handling']))
        reports={str(n):json.loads((run/('gpu%d'%n)/'report.json').read_text()) for n in modes}
        assert reports[str(primary)]['compact_checkpoint_restore_bitwise'] is True
        chosen=primary;speedup=None
        if primary==4:
            speedup=reports['2']['mean_update_seconds']/reports['4']['mean_update_seconds']
            equivalent=reports['2']['multigpu_numeric_comparison']['status']=='PASS'
            if equivalent and speedup<1.05:chosen=2
        estimates={}
        for n,report in reports.items():
            estimates[n]={str(epochs):dict(frames=243000*epochs,updates=(243000//4)*epochs,
                    compute_hours=report['mean_update_seconds']*(243000//4)*epochs/3600) for epochs in [1,3,5]}
        result=dict(status='COMPLETE_REVIEW_REQUIRED',actual_gpu_modes=modes,selected_engineering_mode=chosen,
                    four_vs_two_speedup=speedup,hardware_reports=reports,full_data_training_cost_projections=estimates,
                    formal_training_started=False,new_simulation_episodes=0,disposable_optimizer_updates=sum(r['disposable_updates'] for r in reports.values()),
                    limitation='Small fixed-input engineering throughput;formalvalidation/IO/coverage andstorage mustbefrozen separately. No performanceclaim or automatictraining.',elapsed_seconds=time.monotonic()-began)
        write(run/'report.json',result);write(run/'progress.json',dict(stage='complete',elapsed_seconds=time.monotonic()-began))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':main()
