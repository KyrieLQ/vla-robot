import csv,json,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];run=R/'results/mini_vla_m7_fixed_lora_flow_dev50_20261030_run1';out=run/'review';started=time.monotonic();d=json.loads((out/'independent_acceptance.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(13,3.8));names=['Original Flow','Adapted VLM + Flow','Adapted VLM + regression'];roles=['old_flow','flow','regression'];colors=['#8898aa','#e58b2b','#277eb3'];rates=[d['metrics'][a]['success_rate']*100 for a in roles]
ax[0].bar(range(3),rates,color=colors)
for i,(a,v) in enumerate(zip(roles,rates)):ax[0].text(i,v+1,f"{d['metrics'][a]['successes']}/50 ({v:.0f}%)",ha='center')
ax[0].set_xticks(range(3),['Old Flow','New Flow','Regression']);ax[0].set(ylim=(0,105),ylabel='Success rate (%)',title='Fixed Spatial50 development')
for role,label,color in zip(roles,names,colors):ax[1].plot([x['task_id'] for x in d['per_task']],[x[role] for x in d['per_task']],'o-',label=label,color=color)
ax[1].set(xlabel='Task',ylabel='Successes out of 5',ylim=(-.2,5.8),title='Task-level outcomes');ax[1].legend(fontsize=7)
ax[2].bar(['vs old Flow','vs regression'],[7,2],color='#4e9a6c',label='Rescued cases');ax[2].bar(['vs old Flow','vs regression'],[-4,-27],color='#c65b51',label='New failures');ax[2].axhline(0,color='black',linewidth=.8);ax[2].set(ylabel='Number of cases',title='Paired changes for new Flow');ax[2].legend(fontsize=8)
fig.suptitle('One seed, adaptive development cases; head training histories differ.',fontsize=11);fig.tight_layout();fig.savefig(out/'fixed_lora_flow_results.png',dpi=170);fig.savefig(out/'fixed_lora_flow_results.svg');plt.close(fig)
with (out/'per_task.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(d['per_task'][0]));w.writeheader();w.writerows(d['per_task'])
analysis=json.loads((run/'analysis.json').read_text());old={(x['task_id'],x['initial_state_index']):x['left'] for x in analysis['paired_vs_old_flow']['pairs']}
with (out/'paired_cases.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['task_id','init','regression_success','old_flow_success','new_flow_success'])
 for x in analysis['paired_vs_regression']['pairs']:w.writerow([x['task_id'],x['initial_state_index'],int(x['left']),int(old[x['task_id'],x['initial_state_index']]),int(x['right'])])
(out/'figure_build.json').write_text(json.dumps(dict(status='PASS',elapsed_seconds=time.monotonic()-started,new_training_updates=0,new_rollouts=0))+'\n')
