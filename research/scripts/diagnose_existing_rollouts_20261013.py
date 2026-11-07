#!/usr/bin/env python3
"""CPU-only exploratory summaries and unaltered video-frame storyboards."""
import json, math, pathlib, statistics
import av
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[1]
RUN=ROOT/'results/formal_pipeline_20261013_v2'
OUT=ROOT/'reports/research_diagnosis_20261013'
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [c-h,c+h]
def conditional_low_task_probability():
    coef=[1]+[0]*8
    for _ in range(10):
        new=[0]*9
        for i,x in enumerate(coef):
            for failures in range(3):
                if i+failures<=8:new[i+failures]+=x*math.comb(5,failures)
        coef=new
    return 1-coef[8]/math.comb(50,8)
def main():
    OUT.mkdir(exist_ok=True)
    result=json.loads((RUN/'trained_eval/episodes/result.json').read_text());rows=result['episodes']
    failed=[r for r in rows if not r['task_success']];successful=[r for r in rows if r['task_success']]
    for group in range(2):
        fig,axes=plt.subplots(4,8,figsize=(16,9.3))
        for ri,row in enumerate(failed[group*4:(group+1)*4]):
            indices=np.unique(np.round(np.linspace(0,row['steps']-1,8)).astype(int));wanted=set(indices);frames={}
            with av.open(row['video']) as c:
                for i,f in enumerate(c.decode(video=0)):
                    if i in wanted:frames[i]=f.to_ndarray(format='rgb24')
            for col,idx in enumerate(indices):
                ax=axes[ri,col];ax.imshow(frames[idx]);ax.set_title(f'{row["pair_id"]} | step {idx+1}',fontsize=9);ax.axis('off')
        fig.suptitle('Existing failed rollouts: uniform frame samples; no new trials',fontsize=13)
        fig.tight_layout();fig.savefig(OUT/f'failures_{group+1}.png',dpi=130);plt.close(fig)
    success_ids=['task1_init2','task3_init22','task4_init2','task5_init2']
    fig,axes=plt.subplots(4,8,figsize=(16,9.3))
    for ri,pair_id in enumerate(success_ids):
        row=next(r for r in rows if r['pair_id']==pair_id);indices=np.unique(np.round(np.linspace(0,row['steps']-1,8)).astype(int));wanted=set(indices);frames={}
        with av.open(row['video']) as c:
            for i,f in enumerate(c.decode(video=0)):
                if i in wanted:frames[i]=f.to_ndarray(format='rgb24')
        for col,idx in enumerate(indices):
            ax=axes[ri,col];ax.imshow(frames[idx]);ax.set_title(f'{pair_id} | step {idx+1}',fontsize=9);ax.axis('off')
    fig.suptitle('Existing successful examples: illustrative, not additional evaluation',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'success_comparators.png',dpi=130);plt.close(fig)
    q=sum(math.comb(5,k)*.84**k*.16**(5-k) for k in range(3))
    summary={'scope':'Existing 50 planned episodes; exploratory after the original experiment; no new training or simulation',
      'n':50,'successes':42,'failures':8,'iid_wilson_95pct_reference_only':wilson(42,50),
      'per_task_wilson_reference_only':{str(k):wilson(k,5) for k in [2,3,4,5]},
      'important_interval_caveat':'iid reference ignores task clustering, fixed rather than random initial states, and training-seed variation; do not substitute it for the preregistered interval',
      'null_model':{'assumption':'10 equally capable tasks with independent n=5 Bernoulli trials, p=.84; illustrative only',
         'probability_any_task_at_most_2_of_5':1-(1-q)**10,'reason':'Post hoc lowest task score alone is weak evidence of a specific structural defect'},
      'conditional_null_model':{'assumption':'Exactly 8 failures exchangeably assigned among 50 cases in ten groups of five; no task difference. Illustrative check, not proof of exchangeability.',
         'probability_any_task_at_most_2_successes_given_42_of_50_total':conditional_low_task_probability(),
         'calculation':'1 - coefficient(x^8, (1+5x+10x^2)^10) / C(50,8)'},
      'success_steps':{'median':statistics.median(r['steps'] for r in successful),'min':min(r['steps'] for r in successful),'max':max(r['steps'] for r in successful)},
      'failed_cases':[{'pair_id':r['pair_id'],'steps':r['steps'],'task_id':r['task_id']} for r in failed],
      'resolution_per_episode_pp':2,'current_training_seed_count':1,'upstream_default_trials_per_task':50,
      'planned_spatial_standard_trials':500,'frozen_known_initial_states':[2,12,22,32,42],
      'available_raw_telemetry':list(rows[0]),'missing_telemetry':['per-step executed actions and full predicted chunks','per-step wrist image','gripper width, grasp/contact state and target pose','per-step model features and repeat/no-progress indicators']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
