"""Restore the already-measured policy observation at a verified branch root.

This is part of root-state reconstruction: no pixel editing, no future frames,
no outcome selection, and no changes to the simulation or RNG state.
"""
import hashlib,json,pathlib
import numpy as np
from PIL import Image

def image_hash(external,wrist):
    return hashlib.sha256(external.tobytes()+wrist.tobytes()).hexdigest()

def read_verified_snapshot(spec,expected_hash):
    arrays=[]
    for camera in ('external','wrist'):
        path=pathlib.Path(spec[camera+'_path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec[camera+'_file_sha256']
        a=np.asarray(Image.open(path).convert('RGB')).copy()
        assert a.shape==(224,224,3) and a.dtype==np.uint8
        arrays.append(a)
    assert image_hash(*arrays)==expected_hash==spec['policy_image_sha256']
    return tuple(arrays)

def restore_observation(obs,current_state,images,reference_state):
    # Robot observation is measured from the reconstructed simulator, not replaced.
    np.testing.assert_allclose(current_state,reference_state,atol=1e-6,rtol=0)
    restored=dict(obs)
    restored['_root_snapshot_external']=images[0].copy()
    restored['_root_snapshot_wrist']=images[1].copy()
    return restored
