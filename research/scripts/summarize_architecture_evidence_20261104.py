"""Read-only recombination of audited validation and development evidence; no GPU."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rel='results/mini_vla_m6_paired_20261028_run1/review/full_validation_recomputed_v2_suite_task.json';d=json.loads((R/rel).read_text());out={}
for arm in ['frozen','lora']:
 v=d['arms'][arm]['endpoint'];assert v['frames']==30465;rows=v['per_task'];assert len(rows)==40
 out[arm]={}
 for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
  vals=[x for k,x in rows.items() if k.split(':')[0]==suite];assert len(vals)==10;n=sum(x['frames'] for x in vals)
  out[arm][suite]=dict(frames=n,first_mse=sum(x['mse']*x['frames'] for x in vals)/n)
 assert sum(v['frames'] for v in out[arm].values())==30465
report=dict(status='PASS',source=rel,source_sha256=sha(R/rel),validation_frames_used=30465,validation_frames_total=30465,suites=out,pi05_sample_exposures=dict(old_steps=30000,batch=2,positions=60000,old_spatial_frames=52970,equivalent_spatial_passes=60000/52970,four_suite_train_frames=243000,equivalent_four_suite_passes_if_unchanged=60000/243000),new_training_updates=0,new_rollouts=0,claim_boundary='Saved-validation rescoring is not closed-loop success evidence. One epoch LoRA endpoint vs matched continued frozen endpoint; no inferred 2000 score.')
m=json.loads((R/'reports/spatial_full_manifest.json').read_text());split=json.loads((R/'configs/mini_vla_four_suite_demo_split_20261019_v1.json').read_text());old=set(m['episode_ids']);validation={e['episode_id'] for e in split['validation']};overlap=sorted(old&validation);assert len(overlap)==48
report['pi05_old_training_new_validation_overlap']=dict(old_training_episodes=len(old),new_validation_episodes=len(validation),overlap=len(overlap),overlapping_episode_ids=overlap,old_manifest_sha256=sha(R/'reports/spatial_full_manifest.json'),split_sha256=sha(R/'configs/mini_vla_four_suite_demo_split_20261019_v1.json'),implication='Start new matched four-suite study from officialbase; oldpi05ckpt30k already trained on48 planned validation demonstrations.')
p=R/'reports/architecture_optimization_evidence_20261104.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
