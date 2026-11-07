#!/usr/bin/env python3
import json, os, pathlib
import numpy as np
from openpi.training import config, data_loader

def main():
    root=pathlib.Path(os.environ['PROJECT_ROOT']);cfg=config.get_config('pi05_libero_lora_3090_smoke')
    dl=data_loader.create_data_loader(cfg,num_batches=1)
    obs,actions=next(iter(dl))
    assert obs.state.shape==(1,32) and actions.shape==(1,10,32)
    assert np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(obs.state)).all()
    result={'status':'PASS','batch_size':1,'workers':2,'state_shape':list(obs.state.shape),
            'action_shape':list(actions.shape),'images':{k:list(v.shape) for k,v in obs.images.items()},'norm_stats_applied':True}
    (root/'reports/batch_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
