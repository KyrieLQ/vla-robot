"""CPU-only full-token RAM layout, immutable storage, masks, seeds and resource checks."""
import ast,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.ram_token_data_v1 import RamTokenData
from mini_vla.vlm_cached_data_v2 import collate_cached
from mini_vla.vlm_action_heads_v1 import FlowHead,HeadConfig,flow_loss
from mini_vla.train_vlm_flow_v1 import seed,fixed_noise
import flow_ram_resources_v1 as resource
R=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();torch.set_num_threads(2);checks={}
 with tempfile.TemporaryDirectory(dir=R/'tmp') as tmp:
  root=Path(tmp);cache=root/'layout';numeric=root/'numeric';(cache/'train').mkdir(parents=True);(numeric/'train').mkdir(parents=True)
  (cache/'complete.json').write_text(json.dumps({'status':'PASS','parts':{'train':{'frames':5}}}))
  index=[dict(episode_id=10,start=0,end=2,tokens=3,directory='missing1'),dict(episode_id=11,start=2,end=5,tokens=4,directory='missing2')];(cache/'train/index.json').write_text(json.dumps(index))
  arrays={'state':np.zeros((5,8),np.float32),'actions':np.arange(35,dtype=np.float32).reshape(5,1,7),'episode_id':np.array([10,10,11,11,11]),'task_id':np.array([0,0,1,1,1])}
  for k,v in arrays.items():np.save(numeric/'train'/k,v)
  ram={10:torch.randn(2,3,960).to(torch.bfloat16).view(torch.uint16).numpy(),11:torch.randn(3,4,960).to(torch.bfloat16).view(torch.uint16).numpy()}
  for a in ram.values():a.flags.writeable=False
  ds=RamTokenData(cache,numeric,'train',ram);s=ds[1];assert s['action_mask'].sum()==1 and s['episode_id']==10 and torch.equal(s['hidden'].view(torch.uint16),torch.from_numpy(ram[10][1].copy()))
  assert np.array_equal(s['actions'][0],arrays['actions'][1,0]);s['hidden'].zero_();assert ram[10][1].any()
  batch=collate_cached([ds[0],ds[2]]);assert batch['hidden'].shape==(2,4,960) and not batch['hidden_mask'][0,3]
  checks['RAM_only_no_disk_hidden_reads_lossless_copy_tail_padding']=True
  a=fixed_noise(batch);b=fixed_noise(dict(episode_id=batch['episode_id'].flip(0),frame_index=batch['frame_index'].flip(0)));assert torch.equal(a,b.flip(0));checks['original_validation_noise_identity_independent_of_order']=True
  seed(42);head=FlowHead();before=head.output.weight.detach().clone();opt=torch.optim.AdamW(head.parameters(),lr=3e-4);obs={k:batch[k] for k in ['hidden','hidden_mask','state']};loss=flow_loss(head,obs,batch['actions'],batch['action_mask'],a,torch.tensor([.2,.7]));loss.backward();opt.step();assert not torch.equal(before,head.output.weight);head.eval();out=head.sample(**obs,noise=a);assert out.shape==(2,10,7) and torch.isfinite(out).all();checks['actual_flow_head_backward_and10step_generation']=True
 seed(42);head=FlowHead();ck=torch.load(R/'results/mini_vla_m9_flow_train_20261021_run1/training/initial.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,ck['state_dict'][k]) for k,v in head.state_dict().items());assert sum(x.numel() for x in head.parameters())==432711;checks['same_random_initialization_as_originalFlow_no_regression_head_transfer']=True
 limits=dict(ram_cache_bytes=84665278080,ram_overhead_bytes=8*2**30,host_memory_reserve_bytes=128*2**30,max_process_rss_bytes=100*2**30)
 host=resource.host_available;group=resource.cgroup_remaining
 try:
  resource.host_available=lambda:240*2**30;resource.cgroup_remaining=lambda:1000*2**30;assert resource.admission(limits)['status']=='RAM_READY'
  resource.host_available=lambda:200*2**30;assert resource.admission(limits)['status']=='WAITING_FOR_HOST_MEMORY'
  resource.host_available=lambda:300*2**30;resource.cgroup_remaining=lambda:40*2**30;assert resource.admission(limits)['status']=='WAITING_FOR_HOST_MEMORY'
 finally:resource.host_available=host;resource.cgroup_remaining=group
 checks['RAM_admission_respects_host_reserve_and_cgroup']=True
 for f in ['mini_vla/train_adapted_flow_ram_v1.py','mini_vla/ram_token_data_v1.py','scripts/flow_ram_resources_v1.py']:ast.parse((R/f).read_text())
 result=dict(status='PASS',checks=checks,elapsed_seconds=time.monotonic()-start,current_RAM_admission=resource.admission(limits),new_gpu_compute=0,discarded_CPU_head_updates=1,new_simulation_episodes=0)
 (R/'reports/adapted_flow_ram_cpu_contract_20261029.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
