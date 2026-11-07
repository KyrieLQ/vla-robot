"""Independent CPU reconstruction of fixed input identity and resource arithmetic."""
import hashlib,json,time
from pathlib import Path
import pyarrow.parquet as pq
from mini_vla.vlm_frontend_v1 import decode_record,prompt_for
from mini_vla.preflight_m8_v2 import make_processor
R=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 t=time.monotonic();run=R/'results/mini_vla_m6_lora_preflight_20261027_run1';p=json.loads((run/'plan.json').read_text());checks={}
 for f,h in p['input_sha256'].items():
  assert sha(R/f)==h and sha(run/'frozen_source'/f)==h,f
 checks['all_live_and_frozen_inputs_match']=len(p['input_sha256'])
 manifest=json.loads((R/p['model_manifest']).read_text())
 for f,m in manifest['files'].items():assert sha(R/p['model_directory']/f)==m['sha256']
 checks['base_weight_and_processor_files_verified']=len(manifest['files'])
 split=json.loads((R/p['split']).read_text());train={x['episode_id']:x for x in split['train']};val={x['episode_id'] for x in split['validation']};assert not set(train)&val
 samples=json.loads((R/p['sample_manifest']).read_text())['samples'];parity=json.loads((run/'parity.json').read_text());gate=json.loads((R/p['cache']/'gate/complete.json').read_text())
 proc=make_processor(R/p['model_directory']);rows=[]
 for s,a,g in zip(samples,parity['input_hashes'],gate['comparisons'],strict=True):
  eid=s['episode_id'];f=s['frame_index'];assert eid in train and eid not in val
  ep=R/p['dataset_root']/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet'
  row=pq.read_table(ep,columns=['image','wrist_image','episode_index','frame_index'],use_threads=False).slice(f,1).to_pylist()[0]
  assert (row['episode_index'],row['frame_index'])==(eid,f)
  rec=decode_record(row,train[eid]['instruction']);digest=hashlib.sha256(b''.join(im.tobytes() for im in rec['images'])+prompt_for(proc,rec['instruction']).encode()).hexdigest()
  assert digest==a['sha256']==g['input_sha256'] and (a['episode_id'],a['frame_index'])==(eid,f)
  rows.append({'episode_id':eid,'frame_index':f,'sha256':digest})
 assert len(rows)==120 and len({(train[s['episode_id']]['suite'],train[s['episode_id']]['task_id']) for s in samples})==40
 checks['independently_reconstructed_raw_rgb_prompt_hashes']=120
 assert parity['zero_adapter_hidden_bits_exact'] and parity['max_action_error']==0 and parity['cache_relative_L2_max']==0
 result=json.loads((run/'result.json').read_text());recomputed={}
 for arm,m in result['measurements'].items():
  values=m['raw_step_seconds'];assert len(values)==12
  avg=sum(values[3:])/9/4;assert abs(avg-m['seconds_per_example'])<1e-12
  recomputed[arm]=avg
 assert set(result['measurements']['lora']['updated_groups'])=={'vision','text'}
 estimate=1.25*(243000*recomputed['lora']+2*30465*result['validation_forward_seconds_per_example']+2100)
 assert abs(estimate-result['estimated_one_gpu_lora_one_epoch_plus_two_validations_and_control_eval_seconds'])<1e-6
 assert result['retained_training_updates']==0 and result['new_simulation_episodes']==0
 ledger=json.loads((R/'results/mini_vla_m6_20261019_ledger.json').read_text());used=sum(x.get('elapsed_seconds',0) for x in ledger['attempts']+ledger.get('additional_entries',[]))
 output=dict(status='PASS_WITH_MEASUREMENT_LIMITATIONS',checks=checks,input_hashes=rows,raw_training_seconds_per_example=recomputed,projection_seconds=estimate,old_total_remaining_seconds=ledger['limit_seconds']-used,
  limitations=['GPU parity and gradient/update assertions audited from frozen executed source plus terminal records; GPU outputs and full base before/after hashes were not saved, and GPU was not rerun.', 'Training timings use first48 of fixed120 already decoded inputs; random parquet IO and two-GPU scaling were not benchmarked.', 'Forward timing mean has no persisted per-sample timings; independently reproducible arithmetic does not independently validate those timings.', 'Peak allocated is PyTorch tensor allocation, not total NVML memory; no full effective-batch128 timing yet.', 'No retained formal training or new closed-loop performance.'],elapsed_seconds=time.monotonic()-t)
 (run/'review/independent_acceptance.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='input_hashes'},ensure_ascii=False))
if __name__=='__main__':main()
