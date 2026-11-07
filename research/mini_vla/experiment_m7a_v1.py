"""Reuse the frozen training/evaluation IO; substitute only the M4 model factory."""
from . import experiment_m3_v1 as engine
from .spatial_tokens_model_v1 import SpatialTokensMiniVLA,SpatialTokensConfig

def main():
    engine.MiniVLA=SpatialTokensMiniVLA
    engine.ModelConfig=SpatialTokensConfig
    engine.main()

if __name__=='__main__':main()
