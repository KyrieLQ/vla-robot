"""Runtime-only extraction parallelism; original single-GPU head training is unchanged."""
import json,os
from pathlib import Path

def main():
    selection=json.loads(Path(os.environ['VLA_GPU_SELECTION']).read_text());selected=selection['selected'];assert 2<=len(selected)<=4
    # Set before importing torch or constructing any CUDA model. Child workers use their own UUID.
    os.environ['CUDA_VISIBLE_DEVICES']=selected[0]['uuid']
    from . import train_flow_tail_consistent_ram_v1 as training
    from .flow_parallel_extract_v1 import extract
    training.extract=extract
    return training.main()
if __name__=='__main__':raise SystemExit(main())
