"""Full saved validation diagnostics; original checkpoint selection remains unchanged."""
import json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();run=R/'results/mini_vla_m7_flow_tail_train_20261103_run2';base=R/'results/mini_vla_m7_fixed_lora_flow_20261029_run1';p=json.loads((run/'plan.json').read_text());out=run/'review';out.mkdir(exist_ok=True)
 index=json.loads((R/p['cache']/'validation/index.json').read_text());actions=np.load(R/p['numeric_cache']/'validation/actions.npy',mmap_mode='r')[:,0];n=len(actions);target=np.empty((n,10,7),np.float32);mask=np.zeros((n,10),bool);suites=np.empty(n,object)
 for e in index:
  ix=np.arange(e['start'],e['end'])[:,None]+np.arange(10);target[e['start']:e['end']]=actions[np.minimum(ix,e['end']-1)];mask[e['start']:e['end']]=ix<e['end'];suites[e['start']:e['end']]=e['suite']
 assert n==30465 and mask.sum()==296235
 norm=json.loads((R/p['preprocessing']).read_text())['normalization']['actions'];scale=np.array(norm['scale']);mean=np.array(norm['mean']);truth=target*scale+mean
 old_initial=np.load(base/'training/initial_predictions.npy');new_initial=np.load(run/'training/initial_predictions.npy');initial_max=float(np.abs(old_initial-new_initial).max());np.testing.assert_allclose(old_initial,new_initial,atol=1e-6,rtol=0)
 results={}
 for name,path in [('original_flow',base/'training/best_predictions.npy'),('tail_consistent',run/'training/best_predictions.npy')]:
  pred=np.load(path);assert pred.shape==(30465,10,7) and np.isfinite(pred).all();err=(pred.astype(np.float64)-target)**2;raw=pred*scale+mean
  results[name]=dict(first_mse=float(err[:,0].mean()),first5_valid_mse=float(err[:,:5][mask[:,:5]].mean()),all_valid_chunk_mse=float(err[mask].mean()),first_per_dim_mse=err[:,0].mean(0).tolist(),tail_first_mse=float(err[~mask.all(1),0].mean()),non_tail_first_mse=float(err[mask.all(1),0].mean()),first_gripper_sign_accuracy=float(((raw[:,0,6]>=0)==(truth[:,0,6]>=0)).mean()),per_suite_first_mse={s:float(err[suites==s,0].mean()) for s in sorted(set(suites))})
 report=dict(status='PASS',frames=n,tail_frames=int((~mask.all(1)).sum()),valid_target_steps=int(mask.sum()),initial_prediction_bitwise_equal=bool(np.array_equal(old_initial,new_initial)),initial_prediction_max_difference=initial_max,models=results,checkpoint_selection_unchanged=True,new_simulation_episodes=0,new_training_updates=0,elapsed_seconds=time.monotonic()-start)
 (out/'full_validation_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,3.8))
 for name,path in [('Original Flow',base),('Tail input consistent',run)]:
  h=json.loads((path/'training/history.json').read_text());x=[v['epoch'] for v in h]
  axes[0].plot(x,[v['validation']['normalized_mse'] for v in h],label=name)
  axes[1].plot(x,[v['train_velocity_mse'] for v in h],label=name)
 axes[0].set_title('Full validation generated first-action MSE');axes[1].set_title('Training velocity MSE (not selection metric)')
 for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
 fig.tight_layout();fig.savefig(out/'training_validation_comparison.png',dpi=160);plt.close(fig)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
