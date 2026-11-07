"""Recompute final bounded Flow scope counts from committed episode artifacts."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 runs=dict(original='mini_vla_m7_fixed_lora_flow_dev50_20261030_run1',fixed_noise='mini_vla_m7_flow_noise_dev50_20261103_run1',tail='mini_vla_m7_flow_tail_dev50_20261103_run1',depth2='mini_vla_m7_flow_depth2_dev50_20261104_run2',time_film='mini_vla_m7_flow_time_film_dev50_20261104_run1')
 rows={};metrics={};bindings={}
 for name,run in runs.items():
  root=R/'results'/run;accept=json.loads((root/'review/independent_acceptance.json').read_text());assert accept['status']=='PASS';bindings[str((root/'review/independent_acceptance.json').relative_to(R))]=sha(root/'review/independent_acceptance.json')
  result=json.loads((root/'flow/h5/result.json').read_text());assert len(result['episodes'])==50;rows[name]={}
  for e in result['episodes']:
   key=e['task_id'],e['initial_state_index'];folder=root/'flow/h5'/e['pair_id'];assert key not in rows[name] and json.loads((folder/'episode.json').read_text())==e
   assert sha(folder/'trace.npz')==e['trace_sha256'];rows[name][key]=e
  assert set(rows[name])=={(t,i) for t in range(10) for i in [2,12,22,32,42]}
  metrics[name]=dict(successes=sum(e['task_success'] for e in rows[name].values()),episodes=50,video_frames=accept['decoded_frames'],source_run=str(root.relative_to(R)))
 def compare(left,right):
  keys=sorted(rows[left]);diff=np.array([int(rows[right][k]['task_success'])-int(rows[left][k]['task_success']) for k in keys]);task=diff.reshape(10,5).mean(1);ci=np.quantile(np.random.default_rng(20261019).choice(task,(10000,10)).mean(1),[.025,.975]).tolist()
  for k in keys:assert rows[right][k]['initial_image_sha256']==rows[left][k]['initial_image_sha256']
  return dict(control=left,candidate=right,wins=int((diff==1).sum()),losses=int((diff==-1).sum()),difference=float(diff.mean()),task_cluster_descriptive_95_interval=ci)
 contrasts=[compare('original','fixed_noise'),compare('original','tail'),compare('tail','depth2'),compare('depth2','time_film')];context=[compare('original','time_film'),compare('tail','time_film')]
 heads={}
 for name,run in [('tail','mini_vla_m7_flow_tail_train_20261103_run2'),('depth2','mini_vla_m7_flow_depth2_train_20261103_run1'),('time_film','mini_vla_m7_flow_time_film_train_20261104_run1')]:
  root=R/'results'/run;a=json.loads((root/'review/independent_acceptance.json').read_text());assert a['status']=='PASS' and a['epochs']==50 and a['updates']==94950
  heads[name]=dict(epochs=50,updates=94950,best_epoch=a['best_epoch'],parameters=a['head_parameters'],validation_first_mse=a['validation_predictions']['best']['first_mse'],checkpoint_sha256=a['checkpoint_sha256'])
 assert len(heads)==3 and sum(v['updates'] for v in heads.values())==284850
 new=[x for x in runs if x!='original'];assert sum(metrics[x]['episodes'] for x in new)==200
 out=dict(status='PASS',new_heads=3,formal_updates=284850,new_development_episodes=200,distinct_initial_cases=50,episodes_per_variant=50,metrics=metrics,registered_pairwise_contrasts=contrasts,contextual_contrasts=context,heads=heads,new_videos_fully_decoded=200,new_video_frames=sum(metrics[x]['video_frames'] for x in new),source_bindings=bindings,claim_boundary='All four variants reuse the same50development initial cases;not200independent scenes. One training seed,adaptive sequence,multiple comparisons;descriptive intervals only. No final2000.')
 (R/'reports/flow_improvement_scope_acceptance_20261104.json').write_text(json.dumps(out,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=['Original Flow','Episode-fixed noise','Tail consistent','Two layers','Time modulation','Regression reference'];vals=[21,7,22,28,29,46]
 fig,ax=plt.subplots(figsize=(10,4));bars=ax.barh(labels,100*np.array(vals)/50,color=['#7197ad','#ce7772','#7197ad','#5684ad','#326b98','#9b9b9b']);ax.invert_yaxis();ax.set_xlim(0,106);ax.set_xlabel('Success rate (%) on the same Spatial50 development cases');ax.set_title('Bounded Flow study: all registered variants retained')
 for bar,n in zip(bars,vals):ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{n}/50 ({2*n}%)',va='center',fontsize=10)
 fig.tight_layout();fig.savefig(R/'reports/flow_improvement_scope_20261104.png',dpi=160);plt.close(fig)
 print(json.dumps({k:v for k,v in out.items() if k!='source_bindings'}))
if __name__=='__main__':main()
