"""
Atomic checkpoint manager. Keeps only `checkpoint-last` and `checkpoint-best`
on disk. Writes go to a *.tmp directory and are renamed in-place so a crash
mid-write never leaves a corrupt checkpoint.
"""
from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch


class CheckpointManager:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.base_dir / 'checkpoint-last'
        self.best_path = self.base_dir / 'checkpoint-best'
        self.metric_path = self.base_dir / 'best-metric.txt'

    # ---- save ----
    def save(self, payload: dict, metric: Optional[float] = None,
             metric_lower_better: bool = True):
        """
        Always overwrites checkpoint-last. If metric is provided and improves,
        also updates checkpoint-best.
        """
        self._atomic_save(payload, self.last_path)
        if metric is None:
            return
        prev = self._read_best_metric()
        improved = (
            prev is None
            or (metric_lower_better and metric < prev)
            or ((not metric_lower_better) and metric > prev)
        )
        if improved:
            self._atomic_save(payload, self.best_path)
            self.metric_path.write_text(f"{metric:.8f}\n")

    def _atomic_save(self, payload: dict, dst: Path):
        # Write everything to a sibling .tmp dir, then atomically rename.
        tmp_dir = Path(tempfile.mkdtemp(
            prefix=dst.name + '.tmp.', dir=self.base_dir
        ))
        try:
            torch.save(payload, tmp_dir / 'state.pt')
            # Replace the destination directory atomically
            if dst.exists():
                shutil.rmtree(dst)
            tmp_dir.rename(dst)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def _read_best_metric(self) -> Optional[float]:
        if not self.metric_path.exists():
            return None
        try:
            return float(self.metric_path.read_text().strip())
        except Exception:
            return None

    # ---- load ----
    def load(self, which: str = 'last', map_location: str = 'cpu') -> Optional[dict]:
        path = {'last': self.last_path, 'best': self.best_path}[which]
        f = path / 'state.pt'
        if not f.exists():
            return None
        return torch.load(f, map_location=map_location)


# ---- EMA helper -------------------------------------------------------------
class EMA:
    """Standard exponential moving average over model parameters."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, device):
        self.shadow = {k: v.to(device) for k, v in sd.items()}

    def apply_to(self, model: torch.nn.Module):
        """Copy EMA weights into model in-place. Returns the previous state."""
        prev = {k: v.detach().clone() for k, v in model.state_dict().items()}
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)
        return prev

    @staticmethod
    def restore(model: torch.nn.Module, prev: dict):
        msd = model.state_dict()
        for k, v in prev.items():
            msd[k].copy_(v)
