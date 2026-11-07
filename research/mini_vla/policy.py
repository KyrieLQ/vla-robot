"""CPU-testable deployment adapter; it does not start a simulator or server."""
import numpy as np
import torch

from .data import denormalize, encode_instruction, image_tensor, normalize


class MiniVLAPolicy:
    def __init__(self, model, vocabulary, normalization, language_length=32):
        self.model = model.eval()
        self.vocabulary, self.normalization = vocabulary, normalization
        self.language_length = language_length

    def infer(self, observation):
        """Input RGB must already use the demonstration orientation."""
        ids, mask = encode_instruction(observation['prompt'], self.vocabulary, self.language_length)
        size = self.model.config.image_size
        inputs = {
            'image': image_tensor(observation['observation/image'], size),
            'wrist_image': image_tensor(observation['observation/wrist_image'], size),
            'state': normalize(observation['observation/state'], self.normalization['state']),
            'token_ids': ids, 'language_mask': mask,
        }
        device = next(self.model.parameters()).device
        inputs = {key: value.unsqueeze(0).to(device) for key, value in inputs.items()}
        normalized = self.model.predict(**inputs)[0]
        commands = denormalize(normalized, self.normalization['actions'])
        if not torch.isfinite(commands).all():
            raise ValueError('Nonfinite controller commands')
        return {'actions': commands.clamp(-1, 1).cpu().numpy().astype(np.float32)}
