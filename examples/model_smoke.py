"""Run the actual M1-M4 implementations with synthetic observations on CPU."""
from pathlib import Path
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import torch
from mini_vla.model import MiniVLA, ModelConfig
from mini_vla.chunk_model_v1 import ChunkMiniVLA
from mini_vla.visual_model_v1 import VisualConfig, VisualMiniVLA
from mini_vla.attention_model_v1 import AttentionConfig, AttentionMiniVLA


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    inputs = {
        "image": torch.rand(2, 3, 128, 128),
        "wrist_image": torch.rand(2, 3, 128, 128),
        "state": torch.zeros(2, 8),
        "token_ids": torch.randint(1, 32, (2, 8)),
        "language_mask": torch.ones(2, 8, dtype=torch.bool),
    }
    models = [
        ("CNN / single action", MiniVLA(ModelConfig(vocabulary_size=32)), 1),
        ("CNN / action chunk", ChunkMiniVLA(ModelConfig(vocabulary_size=32, horizon=10)), 10),
        ("ResNet18 / action chunk", VisualMiniVLA(VisualConfig(
            vocabulary_size=32, horizon=10, vision_variant="resnet_random")), 10),
        ("ResNet18 + Transformer", AttentionMiniVLA(AttentionConfig(
            vocabulary_size=32, horizon=10), initialize_pretrained=False), 10),
    ]
    with torch.inference_mode():
        for label, model, horizon in models:
            output = model.cpu().eval()(**inputs)
            assert output.shape == (2, horizon, 7), (label, tuple(output.shape))
            assert torch.isfinite(output).all(), label
            parameters = sum(p.numel() for p in model.parameters())
            print(f"PASS {label}: shape={list(output.shape)}, parameters={parameters:,}")
    print("Synthetic forward passes only; all models use random initial weights.")


if __name__ == "__main__":
    main()
