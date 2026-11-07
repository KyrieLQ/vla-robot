"""CPU-only real-cache contract for four suites and the71-word M4 model."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from mini_vla.spatial_tokens_model_v1 import SpatialTokensMiniVLA as AttentionMiniVLA,SpatialTokensConfig as AttentionConfig
from mini_vla.chunk_model_v1 import ChunkCachedData
from mini_vla.experiment_m3_v1 import inputs
from mini_vla.model import masked_mse

R=Path(__file__).resolve().parents[1]
def main():
    start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(42)
    pre=json.loads((R/'configs/mini_vla_m6_preprocessing_20261019_v1.json').read_text());assert len(pre['vocabulary'])==71
    data=ChunkCachedData(R/'datasets/mini_vla_four_suite_128_20261019_v1','train');assert len(data)==243000
    ids=data.arrays['task_id'];selected=[]
    for suite in range(4):
        valid=np.flatnonzero((ids>=suite*10)&(ids<suite*10+10));selected.extend([int(valid[0]),int(valid[-1])])
    batch=default_collate([data[i] for i in selected]);x=inputs(batch,torch.device('cpu'))
    assert set(x)=={'image','wrist_image','state','token_ids','language_mask'}
    model=AttentionMiniVLA(AttentionConfig(71,horizon=10),initialize_pretrained=True,pretrained_path=R/'cache/torch/hub/checkpoints/resnet18-f37072fd.pth');model.eval()
    pred=model(**x);assert pred.shape==(8,10,7) and torch.isfinite(pred).all()
    assert sum(p.numel() for p in model.parameters())==11604103
    loss=masked_mse(pred,batch['actions'],batch['action_mask']);assert torch.isfinite(loss);loss.backward()
    for name in ['vision.backbone.0.weight','language.weight','encoder.layers.0.self_attn.in_proj_weight','decoder.layers.0.multihead_attn.in_proj_weight','action_queries']:
        g=dict(model.named_parameters())[name].grad;assert g is not None and torch.isfinite(g).all() and bool(g.abs().sum()>0)
    for row,idx in zip(batch['action_mask'],selected):
        indices,mask=data.window_indices(idx);assert np.all(data.arrays['episode_id'][indices]==data.arrays['episode_id'][idx]);np.testing.assert_array_equal(row.numpy(),mask)
    import tempfile,io
    assert tempfile.gettempdir().startswith(str(R/'tmp'))
    buf=io.BytesIO();torch.save({'config':model.configuration(),'state':model.state_dict()},buf);buf.seek(0);roundtrip=torch.load(buf,weights_only=False);restored=AttentionMiniVLA(AttentionConfig(**roundtrip['config']));restored.load_state_dict(roundtrip['state']);restored.eval()
    with torch.no_grad():torch.testing.assert_close(model(**x),restored(**x),rtol=0,atol=0)
    from torch.utils.data import DataLoader
    loader=DataLoader(data,batch_size=8,num_workers=2);iterator=iter(loader);worker_batch=next(iterator);assert worker_batch['images'].shape==(8,2,3,128,128);iterator._shutdown_workers()
    result=dict(status='PASS',serialization_roundtrip_exact=True,two_worker_IPC_pass=True,temporary_directory_on_data_disk=True,real_cpu_batch=8,suites_covered=4,vocabulary_size=71,parameters=11604103,output_shape=list(pred.shape),forward_backward_finite=True,optimizer_updates=0,new_simulation_episodes=0,elapsed_seconds=time.monotonic()-start)
    p=R/'reports/mini_vla_m7a_training_contract_20261020.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
