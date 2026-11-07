"""M8 regression and M9 conditional flow heads; pretrained frontend remains frozen."""
from dataclasses import dataclass
import hashlib
import math
import torch
from torch import nn
from .model import masked_mse


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int = 960
    state_dim: int = 8
    width: int = 128
    attention_heads: int = 4
    feedforward: int = 512
    horizon: int = 10
    action_dim: int = 7
    euler_steps: int = 10


def example_seed(*identity):
    raw = ":".join(str(x) for x in (20261020, *identity)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63-1)


class RegressionHead(nn.Module):
    def __init__(self, config=HeadConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.observation_projection = nn.Linear(config.hidden_size, d)
        self.state_encoder = nn.Sequential(nn.Linear(config.state_dim, 64), nn.ReLU(), nn.Linear(64, d))
        self.action_queries = nn.Parameter(torch.empty(config.horizon, d))
        nn.init.normal_(self.action_queries, std=.02)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, config.attention_heads, config.feedforward, dropout=0, activation="relu", batch_first=True
        ), 1)
        self.output = nn.Linear(d, config.action_dim)

    def memory(self, *, hidden, hidden_mask, state):
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Expected full pretrained token sequence")
        if hidden_mask.shape != hidden.shape[:2] or hidden_mask.dtype != torch.bool:
            raise ValueError("Expected boolean valid-token mask")
        if not hidden_mask.any(1).all() or state.shape != (hidden.shape[0], self.config.state_dim):
            raise ValueError("Invalid observation shape")
        # Sanitize masked cache padding before projection: even NaNs cannot enter attention.
        hidden = torch.where(hidden_mask[..., None], hidden.float(), 0.)
        memory = torch.cat((self.observation_projection(hidden), self.state_encoder(state.float())[:, None]), 1)
        padding = torch.cat((~hidden_mask, torch.zeros(len(state), 1, dtype=torch.bool, device=state.device)), 1)
        return memory, padding

    def forward(self, *, hidden, hidden_mask, state):
        memory, padding = self.memory(hidden=hidden, hidden_mask=hidden_mask, state=state)
        query = self.action_queries[None].expand(len(state), -1, -1)
        return self.output(self.decoder(query, memory, memory_key_padding_mask=padding))


class FlowHead(RegressionHead):
    def __init__(self, config=HeadConfig()):
        super().__init__(config)
        self.noisy_action_projection = nn.Linear(config.action_dim, config.width)
        self.time_mlp = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(),
                                      nn.Linear(config.width, config.width))
        self.register_buffer("time_frequencies", torch.exp(
            -math.log(10000.) * torch.arange(config.width//2).float() / (config.width//2)
        ), persistent=True)

    def velocity(self, *, memory, memory_padding, noisy_action, time, action_mask=None):
        b = memory.shape[0]
        if noisy_action.shape != (b, self.config.horizon, self.config.action_dim) or time.shape != (b,):
            raise ValueError("Noisy action or time shape")
        target_padding = None
        if action_mask is not None:
            if action_mask.dtype != torch.bool or action_mask.shape != noisy_action.shape[:2] or not action_mask.any(1).all():
                raise ValueError("Action training padding mask")
            noisy_action = torch.where(action_mask[..., None], noisy_action, 0.)
            target_padding = ~action_mask
        phase = time[:, None] * 1000. * self.time_frequencies[None]
        time_embedding = self.time_mlp(torch.cat((phase.sin(), phase.cos()), -1))
        queries = self.noisy_action_projection(noisy_action) + self.action_queries[None] + time_embedding[:, None]
        return self.output(self.decoder(queries, memory, tgt_key_padding_mask=target_padding,
                                        memory_key_padding_mask=memory_padding))

    def forward(self, *, hidden, hidden_mask, state, noisy_action, time, action_mask=None):
        memory, padding = self.memory(hidden=hidden, hidden_mask=hidden_mask, state=state)
        return self.velocity(memory=memory, memory_padding=padding, noisy_action=noisy_action,
                             time=time, action_mask=action_mask)

    @torch.no_grad()
    def sample(self, *, hidden, hidden_mask, state, noise):
        memory, padding = self.memory(hidden=hidden, hidden_mask=hidden_mask, state=state)
        x = noise.clone()
        steps = self.config.euler_steps
        for i in range(steps):
            t = torch.full((len(state),), i/steps, dtype=x.dtype, device=x.device)
            x = x + self.velocity(memory=memory, memory_padding=padding, noisy_action=x, time=t) / steps
        return x


def flow_loss(model, observations, target, action_mask, noise, time):
    """Expert action appears only in the training noisy action and velocity target."""
    valid = action_mask[..., None]
    action = torch.where(valid, target, 0.)
    noise = torch.where(valid, noise, 0.)
    x = (1-time[:, None, None])*noise + time[:, None, None]*action
    expected = action-noise
    velocity = model(**observations, noisy_action=x, time=time, action_mask=action_mask)
    return masked_mse(velocity, expected, action_mask)
