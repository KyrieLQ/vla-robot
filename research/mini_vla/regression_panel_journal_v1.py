"""Durable JSONL with bit-preserving float64 physical-state payloads.

qpos/qvel are stored as little-endian float64 base64, without rounding or dropping
any value. Policy calls/actions remain ordinary JSON. Reader returns the same
qpos/qvel lists as the original journal API for exact physical prefix replay.
"""
import base64
import numpy as np
from .paired_eval_journal_v1 import append as raw_append, load as raw_load

def pack(row):
    row=dict(row)
    if row['kind']=='step':
        for field in ['qpos','qvel']:
            if field in row:
                value=np.asarray(row.pop(field),dtype='<f8')
                assert value.ndim==1 and np.isfinite(value).all()
                row[field+'_float64_le_b64']=base64.b64encode(value.tobytes()).decode('ascii')
    return row

def unpack(row):
    row=dict(row)
    for field in ['qpos','qvel']:
        key=field+'_float64_le_b64'
        if key in row:
            data=base64.b64decode(row.pop(key),validate=True)
            assert len(data)%8==0
            value=np.frombuffer(data,dtype='<f8');assert np.isfinite(value).all()
            row[field]=value.tolist()
    return row

def append(path,row):raw_append(path,pack(row))

def load(path):
    calls,steps=raw_load(path)
    return calls,[unpack(row) for row in steps]
