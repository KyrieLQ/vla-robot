"""All-validation diagnostics and learning curves; does not change checkpoint selection."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run).absolute()
    p=json.loads((run/'plan.json').read_text());out=run/'review';accept=json.loads((out/'independent_acceptance.json').read_text());assert accept['status']=='PASS'
    truth=np.load(R/p['numeric_cache']/'validation/actions.npy')[:,0]
    index=json.loads((R/p['cache']/'validation/index.json').read_text())
    split=json.loads((R/p['split']).read_text());meta={e['episode_id']:e for e in split['validation']}
    target=np.zeros((30465,10,7),np.float64);mask=np.zeros((30465,10),bool);suite=np.empty(30465,dtype='U20')
    for e in index:
        for k in range(10):
            starts=np.arange(e['start'],e['end']);indices=starts+k;mask[starts,k]=indices<e['end'];target[starts,k]=truth[np.minimum(indices,e['end']-1)]
        suite[e['start']:e['end']]=meta[e['episode_id']]['suite']
    assert mask.sum()==296235 and len(index)==187
    paths={name:run/name/'best_predictions.npy' for name in ['depth1','depth2']}
    paths['existing']=R/'results/mini_vla_m6_paired_20261028_run1/lora/endpoint_predictions.npy'
    rows={}
    for name,path in paths.items():
        pred=np.load(path);assert pred.shape==target.shape and np.isfinite(pred).all();err=np.square(pred.astype(np.float64)-target)
        tail=~mask[:,-1]
        rows[name]=dict(frames=30465,valid_steps=int(mask.sum()),first_mse=float(err[:,0].mean()),
          chunk_mse=float(err[mask].mean()),first5_mse=float(err[:,:5][mask[:,:5]].mean()),
          first_per_dimension=err[:,0].mean(0).tolist(),valid_chunk_per_dimension=err[mask].mean(0).tolist(),
          tail_first_mse=float(err[tail,0].mean()),tail_frames=int(tail.sum()),non_tail_first_mse=float(err[~tail,0].mean()),
          by_suite={s:dict(frames=int((suite==s).sum()),first_mse=float(err[suite==s,0].mean()),chunk_mse=float(err[suite==s][mask[suite==s]].mean())) for s in sorted(set(suite))})
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['depth1','depth2']:
        history=json.loads((run/name/'history.json').read_text());x=[r['epoch'] for r in history]
        axes[0].plot(x,[r['train_mse'] for r in history],label=name)
        axes[1].plot(x,[r['validation']['normalized_mse'] for r in history],label=name)
        best=accept['arms'][name]['best_epoch'];axes[1].scatter([best],[rows[name]['first_mse']])
    axes[0].set_title('Valid-action training MSE');axes[1].set_title('Full-validation first-action MSE')
    axes[1].axhline(rows['existing']['first_mse'],ls='--',color='gray',label='Existing LoRA endpoint (different history)')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    result=dict(status='PASS',predictions_used=30465,predictions_expected=30465,validation_episodes=187,models=rows,
                checkpoint_selection_changed=False,new_training_updates=0,new_simulation_episodes=0)
    (out/'validation_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
