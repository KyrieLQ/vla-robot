"""Independent full validation rescoring from saved predictions and numeric target arrays."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run)
 plan=json.loads((run/'plan.json').read_text());index=json.loads((R/plan['cache']/'validation/index.json').read_text());target=np.load(R/plan['numeric_cache']/'validation/actions.npy',mmap_mode='r');output={}
 for arm in ['frozen','lora']:
  output[arm]={}
  for stage in ['initial','endpoint']:
   f=run/arm/(stage+'_predictions.npy');values=np.load(f,mmap_mode='r');metric=json.loads((run/arm/(stage+'_validation.json')).read_text());assert values.shape==(30465,10,7) and np.isfinite(values).all()
   assert hashlib.sha256(f.read_bytes()).hexdigest()==metric['predictions_sha256'];sums=[0.,0.];count=0;tasks={}
   for e in index:
    per=tasks.setdefault(str(e['task_id']),[0.,0]);start,end=e['start'],e['end']
    for i in range(start,end):
     n=min(10,end-i);err=(values[i,:n].astype(np.float64)-target[i:i+n,0])**2
     sums[0]+=err[0].sum();sums[1]+=err.sum();count+=n;per[0]+=err[0].sum()/7;per[1]+=1
   assert count==296235
   first=sums[0]/(30465*7);chunk=sums[1]/(count*7)
   assert abs(first-metric['normalized_mse'])<1e-10 and abs(chunk-metric['all_valid_chunk_mse'])<1e-10
   output[arm][stage]=dict(first_mse=first,chunk_mse=chunk,frames=30465,valid_target_steps=count,per_task={k:dict(mse=s/n,frames=n) for k,(s,n) in tasks.items()})
  selected='endpoint.pt' if output[arm]['endpoint']['first_mse']<output[arm]['initial']['first_mse'] else 'initial.pt'
  assert json.loads((run/arm/'complete.json').read_text())['selected']==selected;output[arm]['selected']=selected
 dest=run/'review';dest.mkdir(exist_ok=True);(dest/'full_validation_recomputed.json').write_text(json.dumps(dict(status='PASS',arms=output),indent=2)+'\n')
if __name__=='__main__':main()
