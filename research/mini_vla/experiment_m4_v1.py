"""Reuse the frozen training/evaluation IO; substitute only the M4 model factory."""
from . import experiment_m3_v1 as engine
from .attention_model_v1 import AttentionMiniVLA,AttentionConfig

def main():
    engine.MiniVLA=AttentionMiniVLA
    engine.ModelConfig=AttentionConfig
    engine.main()

if __name__=='__main__':main()
