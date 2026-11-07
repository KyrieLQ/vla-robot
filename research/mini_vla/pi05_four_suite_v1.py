"""Prospective clean-base pi05 adaptation recipe and raw-parquet observation path.

This module does not register or launch formal training. The engineering probe
freezes SigLIP, tunes existing Gemma LoRA and action/time projections, and uses
the native pi05 loss. State-token behavior remains the official LIBERO setting.
"""
import dataclasses,hashlib,io,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import flax.nnx as nnx
import jax
from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies.libero_policy import LiberoInputs
from openpi.shared import nnx_utils,normalize
from openpi.training import config

R=Path(__file__).resolve().parents[1]
def make_config(batch=4,devices=4):
    old=config.get_config('pi05_libero_lora_3090_train')
    freeze=nnx.Any(old.model.get_freeze_filter(),nnx_utils.PathRegex('.*PaliGemma/img.*'))
    return dataclasses.replace(old,name='pi05_four_suite_frozen_vision_lora_probe_v1',freeze_filter=freeze,
        batch_size=batch,fsdp_devices=devices,num_workers=0,ema_decay=None,wandb_enabled=False,
        policy_metadata=dict(scope='engineering_probe_only',formal_training=False),
        num_train_steps=4,save_interval=4,keep_period=None)

class RawTrainingInputs:
    def __init__(self,plan):
        self.plan=plan;split=json.loads((R/plan['split']).read_text())
        self.episodes={e['episode_id']:e for e in split['train']}
        assert len(self.episodes)==1506 and not set(self.episodes)&{e['episode_id'] for e in split['validation']}
        self.samples=json.loads((R/plan['sample_manifest']).read_text())['samples']
        assert len(self.samples)==120 and all(s['episode_id'] in self.episodes for s in self.samples)
        self.stats=normalize.deserialize_json((R/plan['norm_stats']).read_text())
        self.model=make_config().model
        assert self.model.pi05 and not self.model.discrete_state_input and self.model.action_horizon==10
        self.transform=transforms.compose([LiberoInputs(model_type=self.model.model_type),
            transforms.Normalize(self.stats,use_quantiles=True),*config.ModelTransformFactory()(self.model).inputs])
        self.table_cache={}

    def get(self,index):
        sample=self.samples[index%len(self.samples)];eid=sample['episode_id'];frame=sample['frame_index'];meta=self.episodes[eid]
        source=R/self.plan['dataset_root']/('data/chunk-%03d/episode_%06d.parquet'%(eid//1000,eid))
        # Bounded one-episode cache; no Arrow dataset materialization or RGB array rebuild.
        if eid not in self.table_cache:
            self.table_cache={eid:pq.read_table(source,columns=['image','wrist_image','state','actions','episode_index','frame_index'],use_threads=False).to_pylist()}
        table=self.table_cache[eid];row=table[frame];assert len(table)==meta['length'] and row['episode_index']==eid and row['frame_index']==frame
        images=[np.asarray(Image.open(io.BytesIO(row[k]['bytes'])).convert('RGB')) for k in ['image','wrist_image']]
        assert all(x.shape==(256,256,3) for x in images)
        actions=np.asarray([table[min(frame+k,len(table)-1)]['actions'] for k in range(10)],dtype=np.float32)
        state=np.asarray(row['state'],dtype=np.float32)
        raw=dict(**{'observation/image':images[0],'observation/wrist_image':images[1],'observation/state':state},actions=actions,prompt=meta['instruction'])
        transformed=self.transform(raw)
        transformed['image_mask']={k:np.asarray(v,dtype=bool) for k,v in transformed['image_mask'].items()}
        observation=model_lib.Observation.from_dict(transformed)
        observation=jax.tree.map(lambda x:np.asarray(x,dtype=np.float32) if np.issubdtype(np.asarray(x).dtype,np.floating)
                                 else np.asarray(x,dtype=np.int32) if np.issubdtype(np.asarray(x).dtype,np.integer)
                                 else np.asarray(x),observation)
        assert observation.state.shape==(32,) and transformed['actions'].shape==(10,32)
        assert observation.tokenized_prompt.shape==(200,) and not bool(observation.image_masks['right_wrist_0_rgb'])
        assert np.count_nonzero(transformed['actions'][:,7:])==0 and np.count_nonzero(observation.state[8:])==0
        q=self.stats['actions'];expected=(actions-q.q01)/(q.q99-q.q01+1e-6)*2-1
        np.testing.assert_allclose(transformed['actions'][:,:7],expected,atol=1e-7,rtol=0)
        restored=transforms.Unnormalize({'actions':self.stats['actions']},use_quantiles=True)(dict(actions=transformed['actions']))['actions'][:,:7]
        np.testing.assert_allclose(restored,actions,atol=2e-6,rtol=0)
        receipt=dict(episode_id=eid,frame_index=frame,suite=meta['suite'],task_id=meta['task_id'],
                     raw_input_sha256=hashlib.sha256(images[0].tobytes()+images[1].tobytes()+state.tobytes()+meta['instruction'].encode()).hexdigest(),
                     action_window_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),valid_target_steps=min(10,len(table)-frame))
        model_actions=np.asarray(transformed['actions'],dtype=np.float32)
        h=hashlib.sha256()
        for leaf in jax.tree.leaves((observation,model_actions)):h.update(np.asarray(leaf).tobytes())
        receipt['model_input_sha256']=h.hexdigest()
        return observation,model_actions,receipt

    def batch(self,start,size):
        data=[self.get(start+i) for i in range(size)]
        batch=jax.tree.map(lambda *xs:np.stack(xs),*[item[:2] for item in data])
        return batch,[item[2] for item in data]
