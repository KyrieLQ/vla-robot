"""Long-lived CPU controller; short GPU workers commit exact training segments."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
from .pi05_formal_io_v1 import R,write,sha,inspect_checkpoint,verify_log_segments,score_predictions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);ap.add_argument('--mode',default='all');a=ap.parse_args()
    pp=Path(a.plan).absolute();p=json.loads(pp.read_text());attempt=Path(a.run).absolute();root=R/p['experiment_dir'];binding=p['checkpoint_binding'];started=time.monotonic();child=None
    selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text());selected=selection['selected'];assert len(selected)==4
    expected_uuids=[g['uuid'] for g in selected];root.mkdir(exist_ok=True);(root/'training').mkdir(exist_ok=True);(root/'jobs').mkdir(exist_ok=True)
    experiment=root/'experiment.json'
    if experiment.exists():assert json.loads(experiment.read_text())['binding']==binding
    else:write(experiment,dict(binding=binding,scientific_plan=p['scientific_plan'],created_by_attempt=str(attempt.relative_to(R))))
    def progress(stage,**kw):
        value=dict(stage=stage,elapsed_seconds=time.monotonic()-started,experiment_dir=p['experiment_dir'],**kw);write(attempt/'progress.json',value);write(root/'progress.json',value)
    def stopped(signum,frame):
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
        raise InterruptedError('Controller interrupted; preserve all committed artifacts')
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    def check_source_stats():
        for file,row in json.loads((attempt/'source_stats.json').read_text()).items():
            stat=(R/file).stat();assert [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]==row,'Source identity changed: '+file
    def ready():
        from gpu_run import sample
        while True:
            check_source_stats();
            from .pi05_numa_admission_v1 import audit_selected
            audit_selected(selected)
            first={g['uuid']:g for g in sample()};time.sleep(3);second={g['uuid']:g for g in sample()}
            threshold=p['resources']['minimum_free_gpu_mib']
            if all(u in first and u in second and min(first[u]['free_mib'],second[u]['free_mib'])>=threshold for u in expected_uuids):
                latest={g['uuid']:g for g in sample()}
                if all(latest[u]['free_mib']>=threshold for u in expected_uuids):return
            progress('waiting_for_bound_four_gpu_memory',required_free_mib=threshold,free_mib={u:second.get(u,{}).get('free_mib') for u in expected_uuids})
            time.sleep(300)
    def reconcile():
        latest=root/'training/latest.npz';temp=latest.with_name(latest.name+'.tmp')
        if temp.exists():
            metadata=inspect_checkpoint(temp,binding);assert metadata['kind']=='optimizer_state' and verify_log_segments(metadata['cursor']['log_segments'])==metadata['step']
            if latest.exists():assert metadata['step']>=inspect_checkpoint(latest,binding)['step']
            receipt=dict(status='PROMOTED_COMPLETE_PENDING_ARCHIVE',step=metadata['step'],sha256=sha(temp),previous_latest_sha256=sha(latest) if latest.exists() else None)
            os.replace(temp,latest);write(root/'checkpoint_recovery.json',receipt)
        metadata=inspect_checkpoint(latest,binding) if latest.exists() else None
        if metadata:
            assert verify_log_segments(metadata['cursor']['log_segments'])==metadata['step'] and 0<=metadata['step']<=182250
            for epoch in [1,2,3]:
                adapter=root/'training'/('epoch%d.npz'%epoch);pending=adapter.with_name(adapter.name+'.tmp')
                if pending.exists():
                    candidate=inspect_checkpoint(pending,binding);assert candidate['step']==epoch*60750==metadata['step']
                    train=[x['sha256'] for x in metadata['leaves'] if x['path'].startswith('[0]')];assert train==[x['sha256'] for x in candidate['leaves']]
                    os.replace(pending,adapter)
        return metadata
    def active_seconds():
        return sum(json.loads(f.read_text())['elapsed_seconds'] for f in (root/'jobs').glob('*/exit.json'))
    def execute(mode,**kwargs):
        nonlocal child
        if active_seconds()>p['resources']['active_work_limit_seconds']:raise RuntimeError('ACTIVE_WORK_LIMIT_REQUIRES_REVIEW_NO_SHORTENING')
        ready();number=len(list((root/'jobs').iterdir()));directory=root/'jobs'/('job_%05d_%s'%(number,mode));directory.mkdir();job=dict(mode=mode,attempt=str(attempt.relative_to(R)),**kwargs);write(directory/'job.json',job)
        tick=time.monotonic();env=dict(os.environ,JAX_ENABLE_COMPILATION_CACHE='true',PYTHONFAULTHANDLER='1',CUDA_CACHE_PATH=str(R/'cache/cuda'),CUDA_CACHE_MAXSIZE='67108864',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        progress('worker_starting',mode=mode,job=str(directory.relative_to(R)),committed_step=kwargs.get('expected_step',0))
        with (directory/'worker.log').open('xb') as log:
            child=subprocess.Popen([str(R/'envs/openpi/bin/python'),'-u','-B','-m','mini_vla.pi05_numa_admission_v1','--plan',str(pp),'--run',str(root),'--job',str(directory/'job.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
            previous=None
            while child.poll() is None:
                file=directory/'progress.json'
                if file.exists() and file.stat().st_mtime_ns!=previous:
                    previous=file.stat().st_mtime_ns;progress('formal_worker',mode=mode,worker=json.loads(file.read_text()),committed_step=kwargs.get('expected_step',0),job=str(directory.relative_to(R)))
                if time.monotonic()-tick>p['resources']['worker_timeout_seconds']:raise TimeoutError('Boundedworker timedout;preservepartial')
                time.sleep(5)
        code=child.returncode;record=dict(returncode=code,signal_number=-code if code<0 else None,pid=child.pid,elapsed_seconds=time.monotonic()-tick,body_report_present=(directory/'report.json').exists());write(directory/'exit.json',record)
        if code!=0:raise RuntimeError('Workerexit=%d; inspectcompletecommitsbeforeanyresume: %s'%(code,directory))
        report=json.loads((directory/'report.json').read_text());assert report['status'].startswith('PASS_')
        if mode in ['gate','train']:reconcile()
        return report,record
    def validation(point,step):
        folder=root/'validation'/point;complete=folder/'complete.json'
        if complete.exists():
            result=json.loads(complete.read_text());assert result['step']==step and result['binding']==binding and sha(folder/'predictions.npy')==result['prediction_sha256'];return result
        while True:
            file=folder/'progress.json';done=json.loads(file.read_text())['committed_rows'] if file.exists() else 0
            if done==30465:
                final=folder/'predictions.npy';partial=folder/'predictions.partial.npy'
                if not final.exists():
                    q=json.loads(file.read_text());array=np.load(partial,mmap_mode='r')
                    for c in q['chunks']:assert hashlib.sha256(array[c['start']:c['end']].tobytes()).hexdigest()==c['sha256']
                    del array;os.replace(partial,final)
                break
            execute('validate',point=point,expected_step=step,start_row=done)
        progress('independent_full_validation_scoring',point=point,step=step)
        result=dict(status='PASS',point=point,step=step,binding=binding,prediction_sha256=sha(folder/'predictions.npy'),metrics=score_predictions(folder/'predictions.npy',p));write(complete,result);return result
    try:
        state=reconcile()
        if not (root/'gate.json').exists():
            assert state is None or state['step']==0
            report,exit_record=execute('gate');assert exit_record['returncode']==0 and report['full_model_output_bitwise'] and report['nonzero_optimizer_codec_bitwise'];write(root/'gate.json',dict(status='PASS',body=report,exit=exit_record));state=reconcile()
        if a.mode=='gate':
            write(attempt/'report.json',dict(status='PASS_FULL_EXECUTOR_GATE',experiment_dir=p['experiment_dir'],formal_optimizer_updates=0,new_simulations=0));progress('gate_complete');return
        assert state is not None
        history=[]
        if state['step']==0:history.append(validation('initial',0))
        else:history.append(validation('initial',0))
        for epoch in [1,2,3]:
            end=epoch*60750;state=reconcile();current=state['step']
            while current<end:
                next_stop=min(end,((current//1000)+1)*1000)
                report,_=execute('train',expected_step=current,end_step=next_stop);assert report['step']==next_stop
                current=reconcile()['step'];assert current==next_stop
            if current==end:
                if not (root/'training'/('epoch%d.npz'%epoch)).exists() or not (root/'training'/('epoch%d_frozen_check.json'%epoch)).exists():execute('finalize',expected_step=end)
                history.append(validation('epoch%d'%epoch,end))
            else:history.append(validation('epoch%d'%epoch,end))
            write(root/'validation_history.json',history)
        final=reconcile();assert final['step']==182250 and verify_log_segments(final['cursor']['log_segments'])==182250 and len(history)==4
        best=min(history,key=lambda x:(x['metrics']['first_mse'],x['step']))
        selected=dict(status='FROZEN_BY_FULL_VALIDATION',point=best['point'],step=best['step'],first_mse=best['metrics']['first_mse'],checkpoint='official_initial_recipe' if best['step']==0 else str((root/'training'/('epoch%d.npz'%(best['step']//60750))).relative_to(R)),binding=binding)
        write(root/'best_selection.json',selected)
        result=dict(status='COMPLETE_TRAINING_REVIEW_REQUIRED',updates=182250,epochs=3,frame_presentations=729000,validation_frames_per_point=30465,validation_points=4,selected=selected,log_segments=final['cursor']['log_segments'],experiment_dir=p['experiment_dir'],new_simulations=0,elapsed_seconds=time.monotonic()-started,active_worker_seconds=active_seconds(),next='Independentaudit andfrozenonlinecontract thenauthorizeddev200;notfinal2000')
        write(root/'report.json',result);write(attempt/'report.json',result);progress('complete',committed_step=182250)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=180)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':main()
