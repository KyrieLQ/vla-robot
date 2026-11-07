"""Opt-in differentiable SmolVLM frontend. Existing frozen frontend is unchanged."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .preflight_m8_v2 import make_processor
from .vlm_frontend_v1 import prompt_for


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise ValueError('LoRA requires a linear base and positive rank')
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.scale = rank, alpha/rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        # Train adapters in FP32; preserve the baseline BF16 activation interface.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return result + delta.to(result.dtype)


def inject_attention_lora(model, rank=8, alpha=8):
    model.requires_grad_(False)
    targets=[]
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, nn.Linear) or '.self_attn.' not in name:
            continue
        if not ('vision_model.' in name or 'text_model.' in name):
            continue
        if name.rsplit('.',1)[-1] not in ('q_proj','k_proj','v_proj','o_proj','out_proj'):
            continue
        parent, key=name.rsplit('.',1)
        model.get_submodule(parent)._modules[key]=LoRALinear(layer,rank,alpha)
        targets.append(name)
    if not targets:
        raise ValueError('No exact vision/text attention targets found')
    return targets


def adapter_parameters(model):
    return {n:p for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))}


class DifferentiableFrontend(nn.Module):
    def __init__(self, model_path, rank=8, alpha=8, checkpointing=True, device='cuda'):
        super().__init__()
        from transformers import AutoModelForImageTextToText
        self.processor=make_processor(Path(model_path))
        self.model, loading=AutoModelForImageTextToText.from_pretrained(
            model_path,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError(str(loading))
        self.targets=inject_attention_lora(self.model,rank,alpha)
        if len(self.targets)!=176 or sum(p.numel() for p in adapter_parameters(self.model).values())!=2228224:
            raise ValueError('Pinned rank8 target layout changed')
        self.model.to(device)
        if checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.loading=loading

    def inputs(self, records):
        cpu=self.processor(text=[prompt_for(self.processor,r['instruction']) for r in records],
                           images=[r['images'] for r in records],return_tensors='pt',padding=True)
        if tuple(cpu['pixel_values'].shape)!=(len(records),2,3,512,512):
            raise ValueError('Input resolution/camera contract changed')
        if not ((cpu['input_ids']==49190).sum(1)==128).all():
            raise ValueError('Image token contract changed')
        dev=next(self.model.parameters()).device
        return {k:v.to(dev,dtype=torch.bfloat16) if v.is_floating_point() else v.to(dev) for k,v in cpu.items()}

    def forward(self, inputs):
        # Deliberately NOT inference_mode/no_grad: action loss must reach vision/text LoRA.
        h=self.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state
        if h.shape[-1]!=960 or h.dtype!=torch.bfloat16:
            raise ValueError('Hidden interface changed')
        return h,inputs['attention_mask'].bool()

    def enable_adapter_training(self, enabled):
        for p in adapter_parameters(self.model).values():p.requires_grad_(enabled)

