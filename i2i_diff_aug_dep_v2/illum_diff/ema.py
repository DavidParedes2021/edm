"""Simple exponential-moving-average wrapper over a torch model."""

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        ema_params = dict(self.ema_model.named_parameters())
        for name, p in model.named_parameters():
            if name in ema_params:
                ema_params[name].data.mul_(self.decay).add_(
                    p.data, alpha=1.0 - self.decay
                )
        ema_buffers = dict(self.ema_model.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].data.copy_(b.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)

    def to(self, device):
        self.ema_model.to(device)
        return self
