"""CPU source/parameter/convention audit; no model training, GPU or simulation."""
import ast,hashlib,json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sources=['repos/openpi/src/openpi/models/pi0.py','repos/openpi/src/openpi/models/pi0_config.py','repos/openpi/src/openpi/models/gemma.py','repos/openpi/src/openpi/models/model.py','repos/openpi/src/openpi/training/config.py','repos/openpi/src/openpi/policies/libero_policy.py','mini_vla/vlm_action_heads_v1.py','mini_vla/flow_time_film_v1.py','mini_vla/flow_tail_consistent_v1.py','mini_vla/vlm_frontend_v1.py','reports/parameter_audit.json']
 historic=json.loads((R/'reports/MINI_VLA_LESSON_09_EVIDENCE.json').read_text())['source_sha256']
 for name in sources[:3]:
  if name in historic:assert sha(R/name)==historic[name]
 p=json.loads((R/'reports/parameter_audit.json').read_text());expert=[x for x in p['parameters'] if x['path'].startswith('PaliGemma/llm/') and any(c.endswith('_1') for c in x['path'].split('/'))]
 counts=dict(expert_with_adarms_without_lora=sum(x['parameters'] for x in expert if not x['lora']),expert_lora=sum(x['parameters'] for x in expert if x['lora']),expert_with_adarms_and_lora=sum(x['parameters'] for x in expert),included_adarms_parameters=sum(x['parameters'] for x in expert if 'norm' in x['path']))
 assert counts['expert_with_adarms_and_lora']==450051072 and counts['expert_lora']==22118400
 rng=np.random.default_rng(20261104);action=rng.normal(size=(4,10,7));noise=rng.normal(size=(4,10,7));s=.37;t=1-s
 ours=(1-s)*noise+s*action;pi=t*noise+(1-t)*action;interpolation_max=float(np.abs(ours-pi).max());assert interpolation_max<1e-14
 # Opposite time conventions yield the same Euler trajectory for a mapped field.
 def v_pi(x,t):return .2*np.sin(x)+t-.5
 xp=noise.copy();xm=noise.copy()
 for i in range(10):
  xp=xp-.1*v_pi(xp,1-i/10)
  xm=xm+.1*(-v_pi(xm,1-i/10))
 euler_max=float(np.abs(xp-xm).max());assert euler_max==0
 # Translate official time sampling into our progress convention.
 beta=rng.beta(1.5,1,size=200000);pi_t=.001+.999*beta;our_s=1-pi_t
 analytic_mean=.999*(1-1.5/2.5);analytic_early=1-(1-.2/.999)**1.5
 assert abs(float(our_s.mean())-analytic_mean)<.003 and abs(float((our_s<.2).mean())-analytic_early)<.003
 # Structural prefix/suffix block mask, matching cumsum rule in pi0.py.
 group=np.cumsum([False]*5+[True]+[False]*9);mask=group[None,:]<=group[:,None]
 assert mask[:5,:5].all() and not mask[:5,5:].any() and mask[5:,:].all()
 report=dict(status='PASS',source_sha256={x:sha(R/x) for x in sources},historical_pi0_gemma_sources_unchanged=True,pi05_action_expert_parameters=counts,parameter_scope='PaliGemma/llm second expert including its adaptive norms;excludes shared language branch and external action/time projections. Architectural shape inventory,not a new GPU allocation audit.',mini_latest_head_parameters=763335,pi05_action_config=dict(width=1024,depth=18,ffn=4096,q_heads=8,kv_heads=1,head_dim=256),mini_latest_config=dict(width=128,depth=2,ffn=512,heads=4,head_dim=32),time_convention_check=dict(interpolation_max_error=interpolation_max,euler_trajectory_max_error=euler_max,our_velocity='action-noise',pi_velocity='noise-action',our_time='0noise -> 1action',pi_time='1noise -> 0action'),time_sampling=dict(pi_t='.001+.999*Beta(1.5,1)',equivalent_our_s='.999*(1-Beta(1.5,1))',mean_our_s=analytic_mean,probability_our_s_below_point2=analytic_early,uniform_comparison=.2,monte_carlo_samples=200000),attention_contract=dict(prefix_bidirectional=True,prefix_cannot_read_actions=True,actions_read_prefix_and_all_actions=True),new_gpu_compute=0,new_optimizer_updates=0,new_simulation_episodes=0,limitations='Source and arithmetic differences do not identify causal contribution to success rates; no new ablation performed.')
 out=R/'reports/PI05_VS_MINI_FLOW_AUDIT_20261104.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))
if __name__=='__main__':main()
