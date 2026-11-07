"""Durable policy-call/physical-step journal; outcomes never drive retries."""
import json,os
from pathlib import Path
import numpy as np

def append(path,row):
 with Path(path).open('a') as f:
  f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def load(path):
 calls={};steps=[]
 if not Path(path).exists():return calls,steps
 for raw in Path(path).read_text().splitlines():
  row=json.loads(raw)
  if row['kind']=='call':
   s=row['step'];assert s%5==0 and s not in calls and s==len(steps)
   assert np.asarray(row['actions']).shape==(10,7) and np.asarray(row['normalized_actions']).shape==(10,7)
   calls[s]=row
  elif row['kind']=='step':
   s=row['step'];assert s==len(steps) and (s//5)*5 in calls
   np.testing.assert_array_equal(row['action'],calls[(s//5)*5]['actions'][s%5]);assert not steps or not steps[-1]['done']
   steps.append(row)
  else:raise ValueError('Unknown journal row')
 return calls,steps
