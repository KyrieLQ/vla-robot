import csv,json,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];run=R/'results/mini_vla_m6_paired_dev50_20261029_run3';out=run/'review';started=time.monotonic();d=json.loads((out/'independent_acceptance.json').read_text());training=json.loads((R/'results/mini_vla_m6_paired_20261028_run1/review/independent_acceptance.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,3.8));names=['Original','Head-only +1 epoch','LoRA +1 epoch'];colors=['#8797aa','#237db1','#e98a2d'];values=[78,88,92]
axes[0].bar(range(3),values,color=colors)
for i,(v,n) in enumerate(zip(values,[39,44,46])):axes[0].text(i,v+1,f'{n}/50 ({v}%)',ha='center')
axes[0].set_xticks(range(3),names,rotation=12);axes[0].set(ylim=(0,105),ylabel='Success rate (%)',title='Fixed Spatial50 development')
for i,(arm,color) in enumerate(zip(['original','frozen','lora'],colors)):
 axes[1].plot([x['task_id'] for x in d['per_task']],[x[arm] for x in d['per_task']],'o-',label=names[i],color=color)
axes[1].set(xlabel='Task index',ylabel='Successes out of 5',ylim=(-.2,5.8),title='Paired task counts');axes[1].legend(fontsize=8)
values=[training['arms'][a]['wall_seconds']/3600 for a in ['frozen','lora']];axes[2].bar(['Head-only','LoRA'],values,color=colors[1:])
for i,v in enumerate(values):axes[2].text(i,v+.3,f'{v:.2f} h',ha='center')
axes[2].set(ylim=(0,27),ylabel='Wall time (hours)',title='Continuation + validation cost')
fig.suptitle('One seed; 50 paired development cases. LoRA: 2 rescues, 0 new failures vs head-only.',fontsize=11);fig.tight_layout();fig.savefig(out/'paired_results.png',dpi=170);fig.savefig(out/'paired_results.svg');plt.close(fig)
with (out/'per_task.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(d['per_task'][0]));w.writeheader();w.writerows(d['per_task'])
recorded=json.loads((run/'analysis.json').read_text());original={(x['task_id'],x['initial_state_index']):x['left'] for x in recorded['versus_original']['lora']['pairs']}
with (out/'paired_cases.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['task_id','initial_state_index','original_success','frozen_continuation_success','lora_continuation_success','lora_minus_frozen'])
 for p in recorded['paired']['pairs']:w.writerow([p['task_id'],p['initial_state_index'],int(original[p['task_id'],p['initial_state_index']]),int(p['left']),int(p['right']),p['difference']])
(out/'figure_build.json').write_text(json.dumps(dict(status='PASS',elapsed_seconds=time.monotonic()-started,source='independent_acceptance.json and frozen training acceptance',new_rollouts=0,new_training=0))+'\n')
