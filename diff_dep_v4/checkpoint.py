"""Atomic checkpoint manager. Persists exactly two files:

    - checkpoint-last.pt   (always overwritten)
    - checkpoint-best.pt   (only when explicitly flagged is_best=True)

Atomic write semantics: torch.save -> os.replace, so a crash during save
will never leave a half-written checkpoint on disk.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    LAST_NAME = "checkpoint-last.pt"
    BEST_NAME = "checkpoint-best.pt"

    def __init__(self, ckpt_dir: str | Path) -> None:
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.dir / self.LAST_NAME

    @property
    def best_path(self) -> Path:
        return self.dir / self.BEST_NAME

    def _atomic_save(self, state: dict[str, Any], target: Path) -> None:
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, target)

    def save(self, state: dict[str, Any], is_best: bool = False) -> None:
        self._atomic_save(state, self.last_path)
        if is_best:
            # Best is a copy of last to guarantee identical bytes
            shutil.copyfile(self.last_path, self.best_path)

    def load_last(self, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
        if self.last_path.exists():
            return torch.load(self.last_path, map_location=map_location)
        return None

    def load_best(self, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
        if self.best_path.exists():
            return torch.load(self.best_path, map_location=map_location)
        return None
