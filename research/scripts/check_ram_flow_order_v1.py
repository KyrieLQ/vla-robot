import json,time,torch
from pathlib import Path
from torch.utils.data import DataLoader,TensorDataset

def main():
 start=time.monotonic();dataset=TensorDataset(torch.arange(243000));outputs=[]
 for workers in [0,2]:
  kw=dict(batch_size=128,shuffle=True,num_workers=workers,generator=torch.Generator().manual_seed(43))
  if workers:kw['multiprocessing_context']='spawn'
  loader=DataLoader(dataset,**kw);values=[]
  for i,batch in enumerate(loader):
   values.append(batch[0])
   if i==7:break
  outputs.append(torch.cat(values))
 assert torch.equal(*outputs)
 result=dict(status='PASS',matched_first_epoch_first1024_indices=True,workers_compared=[0,2],seed=43,elapsed_seconds=time.monotonic()-start)
 Path('reports/ram_flow_order_contract_20261029.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
