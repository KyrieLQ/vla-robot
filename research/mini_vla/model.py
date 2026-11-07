"""The first policy: two images + text + proprioception -> one action."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int
    image_size: int = 128
    state_dim: int = 8
    action_dim: int = 7
    horizon: int = 1


class MiniVLA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.horizon != 1:
            raise ValueError('M1 implements one action; action chunks are the M2 extension')
        self.config = config
        # The two cameras share weights; concatenation preserves camera identity.
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
        )
        self.language = nn.Embedding(config.vocabulary_size, 64, padding_idx=0)
        self.proprioception = nn.Sequential(nn.Linear(config.state_dim, 64), nn.ReLU(),
                                           nn.Linear(64, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(256 * 2 + 64 + 64, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                        nn.Linear(128, config.action_dim))

    def forward(self, *, image, wrist_image, state, token_ids, language_mask):
        batch = image.shape[0]
        expected = (batch, 3, self.config.image_size, self.config.image_size)
        if tuple(image.shape) != expected or tuple(wrist_image.shape) != expected:
            raise ValueError('Expected two matching BCHW image tensors')
        if tuple(state.shape) != (batch, self.config.state_dim):
            raise ValueError('Incorrect proprioception shape')
        if (token_ids.ndim != 2 or token_ids.shape[0] != batch or
            token_ids.shape != language_mask.shape or not language_mask.any(dim=1).all()):
            raise ValueError('Expected nonempty masked token sequences')
        images = torch.cat([image, wrist_image], dim=0)
        visual = self.vision(images)
        external, wrist = visual[:batch], visual[batch:]
        embedding = self.language(token_ids)
        mask = language_mask.unsqueeze(-1).to(embedding.dtype)
        text = (embedding * mask).sum(dim=1) / mask.sum(dim=1)
        robot = self.proprioception(state)
        fused = self.fusion(torch.cat([external, wrist, text, robot], dim=-1))
        return self.action_head(fused).unsqueeze(1)  # [B, 1, 7], normalized units

    @torch.no_grad()
    def predict(self, *, image, wrist_image, state, token_ids, language_mask):
        return self(image=image, wrist_image=wrist_image, state=state,
                    token_ids=token_ids, language_mask=language_mask)

    def configuration(self):
        return asdict(self.config)


def masked_mse(prediction, target, action_mask):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('Expected matching [B, K, action_dim] tensors')
    if action_mask.shape != prediction.shape[:2] or action_mask.dtype != torch.bool:
        raise ValueError('Expected boolean [B, K] mask')
    if not action_mask.any():
        raise ValueError('No valid action targets')
    valid = action_mask.unsqueeze(-1)
    # Mask before squaring: even NaN values in padded targets cannot leak in.
    residual = torch.where(valid, prediction - target, 0.0)
    return residual.square().sum() / (action_mask.sum() * prediction.shape[-1])
