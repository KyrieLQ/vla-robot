#!/usr/bin/env python3
import json, os, pathlib
import numpy as np
from openpi.training import config, data_loader
from openpi.models.tokenizer import PaligemmaTokenizer

def main():
    root=pathlib.Path(os.environ['PROJECT_ROOT']);m=json.loads((root/'reports/data_manifest.json').read_text())
    cfg=config.get_config('pi05_libero_lora_3090_smoke');dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    ds=data_loader.ManifestSubsetLeRobotDataset(dc.repo_id,root=dc.local_dataset_root,revision=dc.dataset_revision,
           episodes=m['episode_ids'],delta_timestamps={'actions':[t/10 for t in range(cfg.model.action_horizon)]},video_backend='pyav')
    assert len(ds)==m['frames'];checks=[]
    for i,e in enumerate(m['episodes']):
        lo=int(ds.episode_data_index['from'][i]);hi=int(ds.episode_data_index['to'][i]);assert hi-lo==e['length']
        for idx in [lo,(lo+hi-1)//2,hi-1]:
            row=ds[idx];assert int(row['episode_index'])==e['episode_index']
            assert row['task']==m['language'];assert int(row['task_index'])==m['source_task_index']
            query,padding=ds._get_query_indices(idx,e['episode_index'])
            assert all(lo<=q<hi for q in query['actions'])
            expected=np.stack([ds.hf_dataset[q]['actions'].numpy() for q in query['actions']])
            np.testing.assert_array_equal(row['actions'].numpy(),expected)
            if idx==hi-1:
                assert row['actions_is_pad'].tolist()==[False]+[True]*9
                np.testing.assert_array_equal(row['actions'].numpy(),np.repeat(expected[:1],10,axis=0))
            checks.append({'original_episode':e['episode_index'],'local_frame':idx,'queries':query['actions'],
                           'padding':row['actions_is_pad'].tolist()})
    # Check every state/action value in the actual selected dataset, not only sampled frames.
    raw_state=np.stack([x.numpy() for x in ds.hf_dataset['state']]);raw_actions=np.stack([x.numpy() for x in ds.hf_dataset['actions']])
    assert np.isfinite(raw_state).all() and np.isfinite(raw_actions).all()
    sample=ds[0];assert tuple(sample['image'].shape)==(3,256,256);assert tuple(sample['wrist_image'].shape)==(3,256,256)
    tok=PaligemmaTokenizer(cfg.model.max_token_len);tokens,mask=tok.tokenize(m['language'])
    assert tokens.shape==(200,) and mask.any()
    processed=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
    transformed=data_loader.transform_dataset(processed,dc,skip_norm_stats=True)[0]
    assert transformed['actions'].shape==(10,32) and transformed['state'].shape==(32,)
    result={'status':'PASS','frames_used':len(ds),'frames_expected':m['frames'],'episodes':m['episode_ids'],
            'task_mapping':'exact LIBERO task-map language equality, suite task_order_index=0',
            'image_shape':[3,256,256],'raw_state_shape':list(raw_state.shape),'raw_actions_shape':list(raw_actions.shape),
            'state_min':raw_state.min(axis=0).tolist(),'state_max':raw_state.max(axis=0).tolist(),
            'action_min':raw_actions.min(axis=0).tolist(),'action_max':raw_actions.max(axis=0).tolist(),
            'tokenized_prompt_shape':list(tokens.shape),'valid_tokens':int(mask.sum()),'boundary_checks':checks,
            'extra_delta_transform':False,'norm_stats':'computed separately from selected trajectories',
            'action_semantics':'LIBERO OSC_POSE normalized relative controller commands (6) + gripper command (1); not raw meters/radians',
            'padding_semantics':'repeat terminal action with LeRobot actions_is_pad; upstream openpi repack does not consume this mask'}
    (root/'reports/data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
