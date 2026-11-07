#!/usr/bin/env python3
"""Validate real base weights against the full abstract model on CPU, without a training step."""
import importlib.util, json, os, pathlib, time
import flax.nnx as nnx
import flax.traverse_util as tu
import jax
import numpy as np
from openpi.training import config

root=pathlib.Path(os.environ['PROJECT_ROOT']);cfg=config.get_config('pi05_libero_lora_3090_smoke')
spec=importlib.util.spec_from_file_location('upstream_train',root/'repos/openpi/scripts/train.py')
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
start=time.monotonic();shape=jax.eval_shape(lambda k:nnx.state(cfg.model.create(k)),jax.random.key(cfg.seed))
weights=upstream._load_weights_and_validate(cfg.weight_loader,shape.to_pure_dict())
loaded=tu.flatten_dict(weights,sep='/');expected=tu.flatten_dict(shape.to_pure_dict(),sep='/')
missing=set(expected)-set(loaded);assert missing and all('lora' in k for k in missing)
for key,arr in loaded.items():
    assert arr.shape==expected[key].shape and arr.dtype==expected[key].dtype
    assert np.isfinite(arr).all(),key
result={'status':'PASS','scope':'actual base weight restoration and full shape/dtype/finite check on CPU; GPU initialization not tested',
        'base_source':'gs://openpi-assets/checkpoints/pi05_base/params','loaded_parameter_leaves':len(loaded),
        'loaded_parameters':sum(a.size for a in loaded.values()),'allowed_new_lora_leaves':sorted(missing),
        'missing_non_lora_leaves':[],'elapsed_seconds':time.monotonic()-start}
(root/'reports/base_weight_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
