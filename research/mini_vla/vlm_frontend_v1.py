"""Frozen pretrained frontend shared by offline caching and online policies."""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .preflight_m8_v2 import make_processor


def decode_record(row, instruction):
    images = []
    for key in ("image", "wrist_image"):
        with Image.open(io.BytesIO(row[key]["bytes"])) as im:
            images.append(im.convert("RGB").copy())
    if any(im.size != (256, 256) for im in images):
        raise ValueError("Expected oriented original256 RGB; never upscale old128 cache")
    return {"images": images, "instruction": instruction}


def records_from_observation(image, wrist_image, instruction):
    """Callers must apply the established LIBERO rotation before this function."""
    images = []
    for rgb in (image, wrist_image):
        if rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
            raise ValueError("Expected two oriented256 HWC uint8 images")
        images.append(Image.fromarray(rgb))
    return {"images": images, "instruction": instruction}


def prompt_for(processor, instruction):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "External camera:"}, {"type": "image"},
        {"type": "text", "text": "Wrist camera:"}, {"type": "image"},
        {"type": "text", "text": instruction},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def bf16_bits(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError("BF16 required; refusing silent precision conversion")
    return tensor.detach().cpu().contiguous().view(torch.uint16).numpy().copy()


def bits_bf16(array):
    if array.dtype != np.uint16:
        raise ValueError("Cache stores raw uint16 BF16 bits")
    return torch.from_numpy(np.array(array, copy=True)).view(torch.bfloat16)


class FrozenFrontend:
    def __init__(self, model_path):
        from transformers import AutoModelForImageTextToText
        self.processor = make_processor(Path(model_path))
        self.model, loading = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True,
        )
        if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(str(loading))
        self.model.cuda().eval().requires_grad_(False)
        assert not any(p.requires_grad for p in self.model.parameters())
        self.loading = loading

    @torch.inference_mode()
    def encode(self, records):
        text = [prompt_for(self.processor, r["instruction"]) for r in records]
        cpu = self.processor(text=text, images=[r["images"] for r in records],
                             return_tensors="pt", padding=True)
        b = len(records)
        assert cpu["pixel_values"].shape == (b, 2, 3, 512, 512)
        assert ((cpu["input_ids"] == 49190).sum(1) == 128).all()
        inputs = {k: v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda()
                  for k, v in cpu.items()}
        hidden = self.model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        assert hidden.dtype == torch.bfloat16 and hidden.shape[-1] == 960
        assert torch.isfinite(hidden).all()
        result = []
        for i, r in enumerate(records):
            valid = cpu["attention_mask"][i].bool()
            result.append({
                "bits": bf16_bits(hidden[i, valid.to(hidden.device)]),
                "token_ids": cpu["input_ids"][i, valid].numpy().astype(np.int32),
                "input_sha256": hashlib.sha256(
                    b"".join(im.tobytes() for im in r["images"]) + text[i].encode()
                ).hexdigest(),
            })
        return result
