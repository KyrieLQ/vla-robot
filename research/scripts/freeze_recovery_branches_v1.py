#!/usr/bin/env python3
"""Freeze roots, interventions, RNG streams and decision gates before branch outcomes."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def dump(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False);f.write('\n')
def main():
    source=ROOT/'results/research_diagnosis_20261014_v3/ckpt30k_prefix5/episodes'
    diag=ROOT/'configs/research_diagnosis_20261014_v2.json'
    oldplan=json.loads(diag.read_text());splitpath=ROOT/oldplan['split_manifest'];split=json.loads(splitpath.read_text())
    rows={r['pair_id']:r for r in json.loads((source/'result.json').read_text())['episodes']}
    dev=split['development'];assert len(dev)==200 and set(rows)=={c['pair_id'] for c in dev}
    roots=[];fit=[];cal=[]
    for t in range(10):
        taskcases=[c for c in dev if c['task_id']==t]
        ranked=sorted(taskcases,key=lambda c:hashlib.sha256(('module-split-20261014:'+c['pair_id']).encode()).hexdigest())
        fit.extend(c['pair_id'] for c in ranked[:14]);cal.extend(c['pair_id'] for c in ranked[14:])
    for i,case in enumerate(dev):
        row=rows[case['pair_id']]
        for step in (60,120):
            if row['steps']<=step:continue
            root_id=f'{case["pair_id"]}_t{step}'
            roots.append(dict(case,root_id=root_id,branch_step=step,worker=i%2,
                              resample_seed=int.from_bytes(hashlib.sha256(('resample-20261014:'+root_id).encode()).digest()[:4],'little'),
                              module_partition='fit' if case['pair_id'] in fit else 'calibration',
                              source_trace_sha256=hashlib.sha256((source/case['pair_id']/'policy_trace.npz').read_bytes()).hexdigest(),
                              source_oracle_sha256=hashlib.sha256((source/case['pair_id']/'oracle.npz').read_bytes()).hexdigest()))
    assert len(roots)==269 and collections_count(roots,60)==200 and collections_count(roots,120)==69
    stamp=datetime.datetime.now().astimezone().isoformat()
    manifest=ROOT/'configs/recovery_branch_roots_20261014_v1.json'
    dump(manifest,{'created_at':stamp,'roots':roots,'module_fit_cases':fit,'module_calibration_cases':cal,
                   'scope':'Development data only. Calibration is for module development, not final confirmation. Same episode roots remain in same worker and split.'})
    plan={'version':1,'created_at':stamp,'phase':'recovery_branch_feasibility','originating_thread':oldplan['originating_thread'],
          'authorization':'User approved starting the 1–2 month VLA improvement project and automatic follow-through within resource limits.',
          'source_diagnosis':str(diag.relative_to(ROOT)),'source_diagnosis_sha256':hashlib.sha256(diag.read_bytes()).hexdigest(),
          'source_episodes':str(source.relative_to(ROOT)),'root_manifest':str(manifest.relative_to(ROOT)),'root_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
          'development_split':oldplan['split_manifest'],'development_split_sha256':oldplan['split_sha256'],
          'checkpoint':oldplan['checkpoint_root']+'/30000','checkpoint_updates':30000,'new_gradient_updates':0,
          'unique_scenarios':200,'roots':269,'branches_per_root':4,'expected_branches':1076,'confirmation_cases_used':0,
          'root_rule':'Fixed t=60 for all 200 episodes; t=120 additionally if the baseline is still active at that time (69 cases). No future success filtering.',
          'methods':[
              {'name':'continue','rule':'Resume original RNG at number of previous calls; execute prefix5 to the same total 220-step cap.'},
              {'name':'fine20','rule':'Same RNG at the root; prefix1 for exactly 20 actual actions, then return to prefix5.'},
              {'name':'resample5','rule':'Restart the policy noise stream using the frozen independent root seed, retain prefix5 for the remainder.'},
              {'name':'retreat5_fine20','rule':'Five legal OSC commands [0,0,0.3,0,0,0,-1] (upwards with open gripper), then 20 prefix1 actions and return to prefix5. All actions count toward 220.'}],
          'method_order':'continue first as the per-root integrity control, then fine20,resample5,retreat5_fine20; each branch starts from an independently replayed identical root.',
          'prefix_steps':5,'action_horizon':10,'denoising_steps':10,'max_total_steps':220,'fine_steps':20,'retreat_steps':5,'retreat_action':[0,0,0.3,0,0,0,-1],
          'reset_protocol':oldplan['reset_protocol'],'wait_steps':10,
          'state_reconstruction':'Canonical initialization plus exact replay of all original actions to the root, rebuilding native controller/internal history. No claim that qpos/qvel alone restore state. No rollout rewind after a branch begins.',
          'rng_reconstruction':'Policy.infer splits its JAX key once per call. Rebuild original key from episode seed and advance the carried key root_step/5 times; first continue prediction must match the archived chunk.',
          'acceptance':{'image_hashes':'exact before every replayed action and at root; full continue suffix also exact','numeric_atol':1e-6,'numeric_rtol':0,'root_simulator_atol':1e-7,'continue_final_outcome_and_steps':'exact at the 220-action cap','fail_policy':'Preserve evidence and stop affected collection; never relax gates or choose a different sample because of a mismatch.'},
          'preflight_roots':[{'pair_id':'task0_init2','branch_step':60},{'pair_id':'task4_init22','branch_step':120}],
          'preflight_protocol':'Each GPU independently replays and regenerates both a known successful and unsuccessful continue suffix; all four must pass before collecting intervention outcomes.',
          'workers':2,'maximum_simultaneous_gpus':2,'allowed_physical_gpus':[0,1,2,3],
          'queue_timeout_seconds':28800,'job_timeout_seconds':25200,'project_budget_gib':120,'filesystem_reserve_gib':100,'new_artifact_budget_gib':4,
          'periodic_chat_notifications':False,'persistent_idle_monitor':False,
          'analysis':{'units':'269 paired roots nested in 200 episodes and 10 tasks; early/late cohorts reported separately','bootstrap_seed':2026101402,'bootstrap_draws':10000,
                      'metrics':['paired successes and losses vs continue','rescued baseline-failure episodes','harmed baseline-success episodes','additional policy calls and inference seconds','actual steps consumed','per-task counts'],
                      'intervals':'Exploratory task-cluster bootstrap; no multiplicity-adjusted or confirmation claim. Roots and branches are not independent episodes.',
                      'oracle':'Any-candidate success is only an offline feasibility upper bound, never a deployed selector score.',
                      'feasibility_gate':'If at least 8 distinct baseline-failure episodes have a successful alternative and every continue control passes, proceed to frozen-encoder progress/value module development with fixed fit/calibration split. Otherwise improve the recovery candidates/teacher before training a selector; no outcome-driven repeated branch sampling.',
                      'noise_limit':'One fixed stochastic stream per method/root; labels are sampled outcomes, not exact conditional expected values.'},
          'self_review':['Full-episode prefix1 wins do not imply locally recoverable failures; this experiment tests that gap.',
                         'Fixed upward/open retreat can drop a held object or collide; it is a bounded simulator candidate, not a safe real-robot controller.',
                         'A failure classifier cannot help when no recovery candidate works. The oracle feasibility gate precedes module training.',
                         'Preserve successful-case controls to measure false intervention harm; never collect only failed trajectories.',
                         'The 60/120 roots are convenient development probes, not a learned trigger or proof of optimal timing.',
                         'All geometric/contact labels stay evaluator-side; no object IDs, task IDs or future labels enter policy inputs.'],
          'references':['https://mujoco.readthedocs.io/en/3.2.3/programming/simulation.html#state-and-control','https://docs.jax.dev/en/latest/random-numbers.html'],
          'next_stage_requires':'Complete integrity and feasibility review, then freeze module architecture, training targets, seeds, costs and held-out calibration rules before training.'}
    path=ROOT/'configs/recovery_branch_plan_20261014_v1.json';dump(path,plan)
    print(json.dumps({'plan':str(path),'roots':len(roots),'branches':len(roots)*4,'workers':[sum(r['worker']==w for r in roots) for w in range(2)],'fit_cases':len(fit),'calibration_cases':len(cal)}))
def collections_count(rows,step):return sum(r['branch_step']==step for r in rows)
if __name__=='__main__':main()
