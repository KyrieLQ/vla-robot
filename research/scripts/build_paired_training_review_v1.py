import json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];run=R/'results/mini_vla_m6_paired_20261028_run1';out=run/'review';started=time.monotonic();audit=json.loads((out/'independent_acceptance.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,3.5))
for arm,color in [('frozen','#2278b5'),('lora','#ef8b2c')]:
 rows=[json.loads(x) for x in (run/arm/'updates.jsonl').read_text().splitlines()];loss=np.array([x['loss'] for x in rows]);smooth=np.convolve(loss,np.ones(50)/50,mode='valid');axes[0].plot(np.arange(50,len(loss)+1),smooth,label=arm,color=color)
 m=audit['arms'][arm];axes[1].plot([0,1],[m['initial_mse'],m['endpoint_mse']],'o-',label=arm,color=color)
axes[0].set(xlabel='Optimizer update',ylabel='Mean of last 50 batch losses',title='Training loss (descriptive)');axes[0].legend()
axes[1].set(xlabel='Additional full epochs',ylabel='Full validation first-action MSE',title='Offline validation');axes[1].set_xticks([0,1]);axes[1].set_ylim(0,.36);axes[1].legend()
axes[2].bar(['frozen','lora'],[audit['arms'][a]['wall_seconds']/3600 for a in ['frozen','lora']],color=['#2278b5','#ef8b2c']);axes[2].set(ylabel='Hours',title='Training + validation wall time')
fig.tight_layout();fig.savefig(out/'training_validation_cost.png',dpi=160);plt.close(fig)
(out/'figure_build.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started,source='updates.jsonl and independent_acceptance.json',new_training=0,new_rollouts=0))+'\n')
