"""Audit committed preflight artifacts without repeating any GPU updates."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import jax
R=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run).absolute();p=json.loads((run/'plan.json').read_text());out=run/'review';out.mkdir(exist_ok=True);began=time.monotonic()
    assert jax.default_backend()=='cpu'
    for file,digest in p['input_sha256'].items():
        assert sha(R/file)==digest,file
        frozen=run/'frozen_source'/file
        if frozen.exists():assert sha(frozen)==digest
    raw_status=json.loads((run/'status.json').read_text());assert raw_status['status']=='FAILED'
    contract=json.loads((R/p['cpu_contract']).read_text());expected=[contract['receipts'][i] for i in p['probe_sample_indices']]
    summaries={};all_reports={};initials={};metric_tables={}
    for n in [4,2]:
        folder=run/('gpu%d'%n);report=json.loads((folder/'report.json').read_text());metrics=json.loads((folder/'metrics.json').read_text());progress=json.loads((folder/'progress.json').read_text());initial=json.loads((folder/'initial_parameters.json').read_text())
        assert report['status']=='PASS' and progress['stage']=='complete' and report['devices']==n and report['global_batch']==4
        assert report['first40_training_inputs']==expected and len(expected)==40 and len({(e['suite'],e['task_id']) for e in expected})==40
        assert report['all_frozen_parameters_unchanged'] and report['language_and_action_lora_updated'] and report['trainable_parameters']==52153376
        assert report['formal_training_updates']==report['new_simulation_episodes']==0
        assert len(metrics)==10 and [row['step'] for row in metrics]==list(range(1,11))
        assert all(np.isfinite(list(row.values())).all() and row['grad_norm']>0 for row in metrics)
        seconds=np.asarray([row['compute_seconds']+row['input_seconds'] for row in metrics[2:]])
        assert abs(seconds.mean()-report['mean_update_seconds'])<1e-12 and abs(np.quantile(seconds,.95)-report['p95_update_seconds'])<1e-12
        summaries[str(n)]=dict(devices=n,global_batch=4,mean_update_seconds=float(seconds.mean()),p95_update_seconds=float(np.quantile(seconds,.95)),frames_per_second=4/float(seconds.mean()),
          allocator_peak_active_gib=max(m['peak_bytes_in_use'] for m in report['device_memory_stats'])/2**30,
          allocator_peak_pool_gib=max(m['peak_pool_bytes'] for m in report['device_memory_stats'])/2**30,
          initialization_seconds=report['initialization_seconds'],compile_seconds=report['compile_seconds'],
          complete_worker_body_elapsed_seconds=report['elapsed_seconds'],disposable_updates=report['disposable_updates'])
        all_reports[str(n)]=report;initials[str(n)]=initial;metric_tables[str(n)]=metrics
    assert initials['4']==initials['2'] and all_reports['4']['compact_checkpoint_restore_bitwise'] is True
    snapshot=run/'gpu4/probe_step2.npz';metadata=json.loads(snapshot.with_suffix('.json').read_text())
    assert sha(snapshot)==metadata['npz_sha256'] and snapshot.stat().st_size==metadata['bytes']==625862288
    assert metadata['step']==2 and metadata['data_cursor']==8 and not metadata['stores_frozen_weights'] and not metadata['formal_training_checkpoint']
    assert metadata['binding']['plan_sha256']==sha(run/'plan.json') and metadata['binding']['base_manifest_sha256']==p['official_base_manifest_sha256'] and metadata['binding']['samples_sha256']==p['probe_sample_order_sha256']
    count=0;optimizer_counters=[]
    with np.load(snapshot,allow_pickle=False) as values:
        assert set(values.files)=={row['name'] for row in metadata['leaves']}|{'rng'} and len(metadata['leaves'])==86
        for entry in metadata['leaves']:
            value=values[entry['name']];assert list(value.shape)==entry['shape'] and str(value.dtype)==entry['dtype']
            assert sha_bytes(value)==entry['sha256'] and np.isfinite(value).all() and '/img/' not in entry['path'] and "['img']" not in entry['path']
            if entry['path'].startswith('[0]'):count+=value.size
            if entry['path'].endswith('.count'):optimizer_counters.append(int(value))
        np.testing.assert_array_equal(values['rng'],np.asarray(jax.random.key_data(jax.random.split(jax.random.key(42))[0])))
    assert count==52153376 and optimizer_counters==[2,2]
    reference=run/'gpu4/reference_trainable_step10.npz';reference_meta=json.loads((run/'gpu4/reference_parameter_manifest.json').read_text());assert sha(reference)==reference_meta['sha256'] and reference_meta['updates']==10
    changed=[];keys=list(initials['4']['trainable_parameters']);size=0
    with np.load(reference,allow_pickle=False) as values:
        assert len(values.files)==len(keys)==reference_meta['leaves']==28
        for i,key in enumerate(keys):
            array=values['p%04d'%i];size+=array.size;assert array.dtype==np.float32 and np.isfinite(array).all()
            if sha_bytes(array)!=initials['4']['trainable_parameters'][key]:changed.append(key)
    assert size==52153376 and len(changed)==28 and any('lora' in k and '_1' in k for k in changed) and any('lora' in k and '_1' not in k for k in changed)
    loss=max(abs(x['loss']-y['loss'])/max(abs(y['loss']),1e-8) for x,y in zip(metric_tables['2'],metric_tables['4']))
    grad=max(abs(x['grad_norm']-y['grad_norm'])/max(abs(y['grad_norm']),1e-8) for x,y in zip(metric_tables['2'],metric_tables['4']))
    numeric=all_reports['2']['multigpu_numeric_comparison'];assert abs(loss-numeric['loss_relative_error'])<1e-12 and abs(grad-numeric['gradient_norm_relative_error'])<1e-12
    gate=p['probe'];rejected=loss>gate['loss_relative_tolerance'] or grad>gate['gradient_norm_relative_tolerance']
    assert rejected and numeric['status']=='NUMERICAL_COMPARISON_NOT_ADMITTED'
    updates=sum(row['disposable_updates'] for row in summaries.values());assert updates==p['probe']['maximum_disposable_optimizer_steps']==28
    projections={str(ep):dict(frames=243000*ep,optimizer_updates=60750*ep,training_compute_io_hours=summaries['4']['mean_update_seconds']*60750*ep/3600) for ep in [1,3,5]}
    allocated=int(subprocess.check_output(['du','-s','-B1',str(R)],text=True).split()[0]);ledger=json.loads((R/p['resources']['ledger']).read_text());retired=sum(row['allocated_bytes'] for row in ledger['modern']['preexisting_asset_retirements']);charged=allocated-ledger['modern']['baseline_project_bytes']+retired;fs=os.statvfs(R)
    resources=dict(project_allocated_bytes=allocated,filesystem_available_bytes=fs.f_bavail*fs.f_frsize,modern_charged_bytes=charged,modern_remaining_bytes=ledger['modern']['artifact_cap_bytes']-charged,
                   preflight_sequence_bytes=allocated-p['resources']['sequence_baseline_project_bytes'],preflight_cap_bytes=p['resources']['sequence_artifact_cap_bytes'],
                   compact_snapshot_bytes=metadata['bytes'],trainable_reference_bytes=reference.stat().st_size,atomic_latest_and_best_minimum_bytes=2*metadata['bytes']+reference.stat().st_size)
    assert resources['preflight_sequence_bytes']<resources['preflight_cap_bytes'] and resources['filesystem_available_bytes']>=ledger['filesystem_reserve_bytes'] and resources['modern_remaining_bytes']>0
    result=dict(status='PASS_ARTIFACT_AUDIT_FOUR_GPU_RETAINED',raw_parent_status_preserved='FAILED',worker_artifacts_complete=[4,2],
      pipeline_failure=dict(observation='Parent saw nonzeroGPU2exitaftercomplete report/progress/finalstdout;exactcode notpersisted',root_cause='UNRESOLVED_PROCESS_EXIT_AFTER_COMPLETE_ARTIFACTS',numeric_gate_rejection_is_distinct=True),
      selected_engineering_gpu_count=4,two_gpu_rejected=True,numeric_thresholds_unchanged=True,numeric_comparison=numeric,
      independent_loss_relative_error=loss,independent_gradient_norm_relative_error=grad,
      initial_parameters_bitwise_equal=True,compact_npz_all86leaves_rehashed=True,compact_optimizer_counters=optimizer_counters,
      compact_rng_and_cursor_verified=True,primary_restore_then_continue_bitwise=True,changed_trainable_leaves=28,
      mode_summaries=summaries,projections=projections,resources=resources,disposable_updates_used=updates,disposable_updates_remaining=0,
      fresh_gpu_updates=0,fresh_simulation_episodes=0,formal_training_started=False,elapsed_seconds=time.monotonic()-began,
      limitations=['Fullfrozenparameterinvariance/restoredoptimizeridentity verifiedbyexecutedsourceassertionsandretainedreports,not anotherGPUrerun.',
                  'TwoGPUparameterdifference cannot be freshly recomputed because its finalweights were not persisted;lossandgradnorm rejection independentlyrecomputed.',
                  'Eighttimedupdatespermode;full243000frameI/O andEuler10completevalidationthroughputnotyetmeasured.',
                  'Allocatoractive/poolpeaksare not wholedeviceNVMLpeaks;18GiBfreeadmissionmustnotbereducedbasedonactivebytesalone.'])
    write(out/'independent_acceptance.json',result);write(out/'recovered_pipeline_report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['limitations','pipeline_failure']}))
def sha_bytes(array):return hashlib.sha256(array.tobytes()).hexdigest()
if __name__=='__main__':main()
