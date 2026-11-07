"""Actual120 feature/label fixture, initialization and optimizer isolation tests."""
import ast,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch
from mini_vla.regression_depth_pair_v1 import RegressionDepthHead,verify_initial
from mini_vla.vlm_action_heads_v1 import RegressionHead,HeadConfig
from mini_vla.vlm_cached_data_v2 import collate_cached
from mini_vla.model import masked_mse
from mini_vla.train_regression_depth_pair_v1 import optimize_batch
R=Path(__file__).resolve().parents[1]
def main():
 tick=time.monotonic();torch.set_num_threads(2);old=torch.load(R/'results/mini_vla_m8_regression_train_20261021_run1/training/initial.pt',map_location='cpu',weights_only=False);cfg=HeadConfig(**old['model_config'])
 torch.manual_seed(42);one=RegressionDepthHead(cfg,1);torch.manual_seed(42);two=RegressionDepthHead(cfg,2);verify_initial(one,two,old['state_dict']);reference=RegressionHead(cfg);reference.load_state_dict(old['state_dict'])
 run=R/'results/mini_vla_m7_fixed_lora_flow_20261029_run1';p=json.loads((run/'plan.json').read_text());samples=json.loads((R/p['sample_manifest']).read_text())['samples'];rows=json.loads((run/'frontend_gate.json').read_text())['rows'];index={e['episode_id']:e for e in json.loads((R/p['cache']/'train/index.json').read_text())};actions=np.load(R/p['numeric_cache']/'train/actions.npy',mmap_mode='r');states=np.load(R/p['numeric_cache']/'train/state.npy',mmap_mode='r');task_ids=np.load(R/p['numeric_cache']/'train/task_id.npy',mmap_mode='r');examples=[]
 for s,row in zip(samples,rows):
  assert (s['episode_id'],s['frame_index'])==(row['episode_id'],row['frame_index']);path=run/'gate'/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['file_sha256'];h=torch.from_numpy(np.load(path)).view(torch.bfloat16);e=index[s['episode_id']];i=e['start']+s['frame_index'];ids=i+np.arange(10);mask=ids<e['end'];target=np.array(actions[np.minimum(ids,e['end']-1),0]);examples.append(dict(hidden=h,hidden_mask=torch.ones(len(h),dtype=torch.bool),actions=torch.from_numpy(target),action_mask=torch.from_numpy(mask),state=torch.from_numpy(np.array(states[i])),episode_id=s['episode_id'],frame_index=s['frame_index'],task_id=int(task_ids[i])))
 assert len(examples)==120 and len({e['task_id'] for e in examples})==40
 batch=collate_cached(examples);obs={k:batch[k] for k in ['hidden','hidden_mask','state']}
 with torch.no_grad():torch.testing.assert_close(one(**obs),reference(**obs),atol=0,rtol=0)
 pred=two(**obs);loss=masked_mse(pred,batch['actions'],batch['action_mask']);loss.backward();norms=[]
 for layer in two.decoder.layers:
  g=[p.grad for p in layer.parameters()];assert all(v is not None and torch.isfinite(v).all() for v in g);norms.append(float(sum(v.square().sum() for v in g).sqrt()));assert norms[-1]>0
 models={'depth1':one,'depth2':two};opts={n:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4) for n,m in models.items()};small=collate_cached(examples[:8])
 for _ in range(3):optimize_batch(models,opts,dict(depth1='cpu',depth2='cpu'),small)
 bundle=dict(states={n:m.state_dict() for n,m in models.items()},optimizers={n:o.state_dict() for n,o in opts.items()},updates=3);stream=io.BytesIO();torch.save(bundle,stream);stream.seek(0);loaded=torch.load(stream,weights_only=False);assert loaded['updates']==3
 for n,layers in [('depth1',1),('depth2',2)]:
  restored=RegressionDepthHead(cfg,layers);restored.load_state_dict(loaded['states'][n]);opt=torch.optim.AdamW(restored.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(loaded['optimizers'][n]);assert all(int(v['step'])==3 for v in opt.state.values())
  with torch.no_grad():torch.testing.assert_close(restored(**{k:small[k] for k in obs}),models[n](**{k:small[k] for k in obs}),atol=0,rtol=0)
 for f in ['mini_vla/train_regression_depth_pair_v1.py','mini_vla/adapted_feature_parallel_v2.py']:ast.parse((R/f).read_text())
 report=dict(status='PASS',samples=120,tasks=40,tail_samples=sum(not bool(e['action_mask'].all()) for e in examples),parameters={'depth1':398663,'depth2':663239},shared_initial_exact=True,depth1_reference_forward_exact=True,depth2_layer_gradient_norms=norms,paired_checkpoint_optimizer_roundtrip_exact=True,discarded_cpu_optimizer_updates=6,formal_training_updates=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-tick,remaining_runtime_checks='GPU1vs2 numerical trajectory and throughput;fourGPU rawfrontend gates andfull1693receipts')
 (R/'reports/regression_depth_pair_cpu_contract_20261104.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
