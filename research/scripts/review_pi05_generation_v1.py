"""CPU verification of completed inference-only timing artifacts, no new forwards."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run).absolute();p=json.loads((run/'plan.json').read_text());started=time.monotonic();out=run/'review';out.mkdir(exist_ok=True)
    for name,h in p['input_sha256'].items():assert sha(R/name)==h,name
    report=json.loads((run/'report.json').read_text());rows=json.loads((run/'input_receipts.json').read_text());cpu=json.loads((R/p['cpu_contract']).read_text());metrics=json.loads((run/'metrics.json').read_text());pred=np.load(run/'training_input_generation.npy')
    assert report['status']=='PASS' and report['devices']==report['batch']==4 and report['training_inputs']==120 and report['training_input_presentations']==124
    assert report['optimizer_updates']==report['new_validation_frames']==report['new_simulation_episodes']==0 and report['all_parameters_unchanged'] and report['repeat_first4_bitwise']
    assert len(rows)==120 and pred.shape==(120,10,32) and pred.dtype==np.float32 and np.isfinite(pred).all()
    for row,expected in zip(rows,cpu['receipts']):
        assert {k:row[k] for k in expected}==expected
        seed=int.from_bytes(hashlib.sha256(('20261105:%d:%d'%(row['episode_id'],row['frame_index'])).encode()).digest()[:8],'little')%(2**63-1)
        noise=torch.randn((10,32),generator=torch.Generator().manual_seed(seed),dtype=torch.float32).numpy()
        assert seed==row['noise_seed'] and hashlib.sha256(noise.tobytes()).hexdigest()==row['noise_sha256']
    assert len(metrics)==30 and [r['first_input'] for r in metrics]==list(range(0,120,4))
    values=np.asarray([r['compute_seconds']+r['input_seconds'] for r in metrics[2:]]);mean=float(values.mean());assert abs(mean-report['mean_batch_compute_io_seconds'])<1e-12
    validation=mean*((30465+3)//4);assert abs(validation-report['projected_full30465_validation_seconds'])<1e-9
    wrapper=R/'reports/gpu_shared4_selection_20261105_025832.json';exit_record=json.loads(wrapper.read_text());assert exit_record['returncode']==-11 and json.loads((run/'status.json').read_text())['returncode']==245
    train=json.loads((R/p['completed_preflight']/'review/independent_acceptance.json').read_text());training=train['projections']['3']['training_compute_io_hours']
    result=dict(status='PASS_ARTIFACTS_WITH_EXIT_SIGSEGV',raw_failure_preserved=True,wrapper_child_returncode=-11,wrapper_encoded_returncode=245,
      complete_report_before_exit=True,source_hashes_verified=len(p['input_sha256']),training_inputs_used=120,expected_inputs=120,private_CPUfloat32_noise_verified=True,finite_generation_shape=list(pred.shape),
      repeat_and_parameter_identity_evidence='Executedhashandarrayequalassertionsinboundsourceandcommittedreport;notanewGPUforward',
      mean_batch_compute_io_seconds=mean,projected_one_full_validation_hours=validation/3600,
      projected_three_epoch_training_hours=training,projected_initial_plus_three_epoch_validation_hours=4*validation/3600,
      projected_training_plus_validation_hours=training+4*validation/3600,
      optimizer_updates=0,new_simulations=0,new_gpu_work_by_audit=0,elapsed_seconds=time.monotonic()-started,
      next='Inspectboundedexitdiagnosis;formalstorageapprovalandcompleteexecutorcontractsremainpending',
      limitations='Timingfrom120traininginputs,notvalidationquality;full-dataI/O/checkpointoverheadandclosedloopevaluationexcludedfromprojection.')
    (out/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
