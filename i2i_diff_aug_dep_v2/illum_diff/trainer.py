"""Training loop: DDPM noise prediction on the L* channel, mask-weighted MSE,
   CFG mask dropout, EMA, AMP, best+latest checkpointing, periodic sampling."""

import os
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from .dataset import ArtifactInpaintingDataset, NormalSampleDataset
from .ema import EMA
from .model import build_unet
from .sampler import generate_samples


# AMP API moved from torch.cuda.amp to torch.amp in PyTorch 2.x. Pick whichever
# the installed torch supports so we work on torch 1.11 (the originally-pinned
# version) AND on newer torchs without firing FutureWarnings.
def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)            # torch >= 2.x
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)               # torch <= 1.x


def _autocast_ctx(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)              # torch >= 2.x
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)                 # torch <= 1.x


def _patch_gradient_checkpointing(unet) -> None:
    """Wrap each down/up/mid block forward with torch.utils.checkpoint so the
    activations of large blocks are recomputed in backward instead of stored.

    diffusers 0.14 UNet2DModel does not implement enable_gradient_checkpointing(),
    so we monkey-patch. Block forwards return tuples; the wrapper preserves them.
    """
    import torch.utils.checkpoint as _ckpt

    def _wrap(module):
        original_forward = module.forward

        def fwd(*args, **kwargs):
            def run(*inner):
                return original_forward(*inner, **kwargs)
            try:
                return _ckpt.checkpoint(run, *args, use_reentrant=False)
            except TypeError:
                return _ckpt.checkpoint(run, *args)

        module.forward = fwd

    for blk in list(getattr(unet, "down_blocks", [])):
        _wrap(blk)
    if getattr(unet, "mid_block", None) is not None:
        _wrap(unet.mid_block)
    for blk in list(getattr(unet, "up_blocks", [])):
        _wrap(blk)


def _resolve_data_dir(cfg: dict) -> str:
    artifact = cfg["model"]["artifact"]
    if artifact == "overexposure":
        return cfg["data"]["overexposed_dir"]
    if artifact == "underexposure":
        return cfg["data"]["underexposed_dir"]
    raise ValueError(f"unknown artifact: {artifact}")


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        torch.manual_seed(int(cfg["train"]["seed"]))
        np.random.seed(int(cfg["train"]["seed"]))

        want_cuda = str(cfg["train"]["device"]).startswith("cuda")
        self.device = torch.device("cuda") if (want_cuda and torch.cuda.is_available()) else torch.device("cpu")
        if want_cuda and self.device.type == "cpu":
            print("[warn] CUDA requested but not available; falling back to CPU.")

        # ----- Datasets -----
        artifact = cfg["model"]["artifact"]
        data_dir = _resolve_data_dir(cfg)

        self.dataset = ArtifactInpaintingDataset(
            img_dir=data_dir,
            image_size=int(cfg["data"]["image_size"]),
            artifact=artifact,
            mask_cfg=cfg["mask"],
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["data"]["num_workers"]),
            drop_last=True,
            pin_memory=(self.device.type == "cuda"),
        )
        # For periodic visualization on real normal frames.
        self.normal_dataset = NormalSampleDataset(
            img_dir=cfg["data"]["normal_dir"],
            image_size=int(cfg["data"]["image_size"]),
            artifact=artifact,
            mask_cfg=cfg["mask"],
            limit=int(cfg["sample"]["num_samples"]),
        )

        # ----- Model + EMA -----
        self.model = build_unet(
            image_size=int(cfg["data"]["image_size"]),
            in_channels=int(cfg["model"]["in_channels"]),
            out_channels=int(cfg["model"]["out_channels"]),
            base_channels=int(cfg["model"]["base_channels"]),
            channel_mults=tuple(cfg["model"]["channel_mults"]),
            num_attn_blocks_from_bottom=int(cfg["model"]["num_attn_blocks_from_bottom"]),
        ).to(self.device)

        # Memory knobs (best-effort: silently skip if the installed diffusers
        # version doesn't expose the API).
        attn_slice = cfg["train"].get("attention_slice", None)
        if attn_slice is not None and hasattr(self.model, "set_attention_slice"):
            try:
                self.model.set_attention_slice(attn_slice)
                print(f"[trainer] attention slicing -> {attn_slice}")
            except Exception as e:
                print(f"[warn] set_attention_slice({attn_slice}) failed: {e}")

        if bool(cfg["train"].get("gradient_checkpointing", False)):
            if hasattr(self.model, "enable_gradient_checkpointing"):
                try:
                    self.model.enable_gradient_checkpointing()
                    print("[trainer] gradient checkpointing enabled (diffusers builtin)")
                except Exception as e:
                    print(f"[warn] enable_gradient_checkpointing failed: {e}")
            else:
                # diffusers 0.14 UNet2DModel has no builtin -- patch each block.
                _patch_gradient_checkpointing(self.model)
                print("[trainer] gradient checkpointing enabled (manual block-wrap)")

        ema_device = self.device if bool(cfg["train"].get("ema_on_gpu", True)) else torch.device("cpu")
        self.ema = EMA(self.model, decay=float(cfg["train"]["ema_decay"])).to(ema_device)
        self.ema_device = ema_device

        # ----- Diffusion scheduler -----
        self.scheduler = DDPMScheduler(
            num_train_timesteps=int(cfg["train"]["num_train_timesteps"]),
            beta_schedule=str(cfg["train"]["beta_schedule"]),
            prediction_type=str(cfg["train"]["prediction_type"]),
        )

        # ----- Optimizer + AMP -----
        self.optim = torch.optim.AdamW(self.model.parameters(),
                                       lr=float(cfg["train"]["lr"]))
        self.amp = bool(cfg["train"]["amp"]) and (self.device.type == "cuda")
        self.scaler = _make_grad_scaler(enabled=self.amp)

        # ----- Logging / checkpointing -----
        self.use_wandb = bool(cfg["wandb"]["enabled"])
        self.wandb = None
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=cfg["wandb"]["project"],
                           name=cfg["wandb"].get("run_name"),
                           config=cfg)
                self.wandb = wandb
            except Exception as e:
                print(f"[warn] wandb disabled: {e}")
                self.wandb = None

        os.makedirs(cfg["paths"]["ckpt_dir"], exist_ok=True)
        os.makedirs(cfg["paths"]["samples_dir"], exist_ok=True)

        self.grad_accum = max(1, int(cfg["train"].get("grad_accum", 1)))
        self.global_step = 0
        self.best_loss = float("inf")
        self._micro_step = 0    # micro-batch counter for gradient accumulation

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[trainer] CUDA mem: free={free_b/2**30:.2f} GiB / total={total_b/2**30:.2f} GiB")
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[trainer] UNet params: {n_params/1e6:.1f} M")

    # ------------------------------------------------------------------ step

    def _forward_loss(self, batch):
        L     = batch["L"].to(self.device, non_blocking=True)      # (B,1,H,W) [-1,1]
        mask  = batch["mask"].to(self.device, non_blocking=True)   # (B,1,H,W) [0,1]
        d     = batch["depth"].to(self.device, non_blocking=True)  # (B,1,H,W) [0,1]
        valid = batch["valid"].to(self.device, non_blocking=True)  # (B,1,H,W) {0,1}

        B = L.shape[0]
        # CFG mask dropout: drop the conditional info (mask + L_known) on a fraction
        # of the batch so the same network can later be queried unconditionally.
        drop = (torch.rand(B, device=self.device)
                < float(self.cfg["train"]["mask_dropout_prob"])).float().view(B, 1, 1, 1)
        cond_mask = mask * (1.0 - drop)
        L_known   = (1.0 - cond_mask) * L

        t = torch.randint(
            0,
            int(self.scheduler.config.num_train_timesteps),
            (B,), device=self.device, dtype=torch.long,
        )
        noise = torch.randn_like(L)
        L_noisy = self.scheduler.add_noise(L, noise, t)

        net_in = torch.cat([L_noisy, L_known, cond_mask, d], dim=1)  # (B,4,H,W)

        with _autocast_ctx(enabled=self.amp):
            pred = self.model(net_in, t).sample                       # (B,1,H,W) eps
            w_in  = float(self.cfg["train"]["mask_loss_weight_inside"])
            w_out = float(self.cfg["train"]["mask_loss_weight_outside"])
            weight_map = (mask * w_in + (1.0 - mask) * w_out) * valid
            sq = (pred - noise) ** 2
            loss = (sq * weight_map).sum() / (weight_map.sum() + 1e-8)
        return loss

    def _train_step(self, batch) -> (float, bool):
        """One micro-batch step. Returns (loss, optimizer_stepped).

        Optimizer + EMA only fire once every `grad_accum` micro-batches so the
        effective batch is `batch_size * grad_accum` while peak VRAM stays at
        the per-micro-batch level.
        """
        if self._micro_step == 0:
            self.optim.zero_grad(set_to_none=True)
        loss = self._forward_loss(batch) / float(self.grad_accum)
        self.scaler.scale(loss).backward()
        self._micro_step += 1
        stepped = False
        if self._micro_step >= self.grad_accum:
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           float(self.cfg["train"]["grad_clip"]))
            self.scaler.step(self.optim)
            self.scaler.update()
            self.ema.update(self.model)
            self._micro_step = 0
            stepped = True
        return float(loss.detach().item()) * float(self.grad_accum), stepped

    # ------------------------------------------------------------------ loop

    def train(self):
        steps = int(self.cfg["train"]["num_steps"])
        log_every    = int(self.cfg["train"]["log_every"])
        sample_every = int(self.cfg["train"]["sample_every"])
        save_every   = int(self.cfg["train"]["save_every"])

        loader_iter = iter(self.loader)
        running = []
        t0 = time.time()
        print(f"[trainer] device={self.device} | artifact={self.cfg['model']['artifact']} "
              f"| #train images={len(self.dataset)} | image_size={self.cfg['data']['image_size']} "
              f"| batch={self.cfg['train']['batch_size']} | grad_accum={self.grad_accum} "
              f"| effective_batch={int(self.cfg['train']['batch_size']) * self.grad_accum}")

        while self.global_step < steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                batch = next(loader_iter)

            loss, stepped = self._train_step(batch)
            if not stepped:
                continue                # accumulating; do not advance global_step yet
            running.append(loss)
            self.global_step += 1

            if self.global_step % log_every == 0:
                avg = float(np.mean(running[-log_every:]))
                elapsed = time.time() - t0
                rate = self.global_step / max(elapsed, 1e-6)
                print(f"[step {self.global_step}/{steps}] loss={avg:.4f} "
                      f"| {rate:.2f} it/s | {elapsed:.0f}s")
                if self.wandb is not None:
                    self.wandb.log({"loss": avg, "step": self.global_step,
                                    "iters_per_s": rate})

            if self.global_step % save_every == 0:
                avg = float(np.mean(running[-log_every:])) if running else float(loss)
                self._save_ckpt(latest=True)
                if avg < self.best_loss:
                    self.best_loss = avg
                    self._save_ckpt(latest=False)

            if self.global_step % sample_every == 0:
                self._sample_and_log()

        # End-of-run snapshot. Always commit a best checkpoint at the end so
        # that short runs (num_steps < save_every) still leave a usable best.pt.
        self._save_ckpt(latest=True)
        if running:
            avg = float(np.mean(running[-log_every:]))
            if avg < self.best_loss:
                self.best_loss = avg
                self._save_ckpt(latest=False)
        self._sample_and_log()
        if self.best_loss == float("inf"):
            print("[trainer] done. (no best.pt yet -- run more than save_every steps to record one)")
        else:
            print(f"[trainer] done. best_loss={self.best_loss:.4f}")

    # ------------------------------------------------------------------ I/O

    def _ckpt_path(self, latest: bool) -> str:
        name = "latest.pt" if latest else "best.pt"
        artifact = self.cfg["model"]["artifact"]
        return os.path.join(self.cfg["paths"]["ckpt_dir"], f"{artifact}_{name}")

    def _save_ckpt(self, latest: bool):
        path = self._ckpt_path(latest=latest)
        torch.save({
            "model":       self.model.state_dict(),
            "ema":         self.ema.state_dict(),
            "optim":       self.optim.state_dict(),
            "global_step": self.global_step,
            "best_loss":   self.best_loss,
            "config":      self.cfg,
        }, path)
        kind = "latest" if latest else "best"
        print(f"[ckpt] saved {kind} -> {path}")

    def _sample_and_log(self, max_samples: Optional[int] = None):
        out_dir = os.path.join(self.cfg["paths"]["samples_dir"],
                               f"{self.cfg['model']['artifact']}_step_{self.global_step:06d}")
        # If EMA lives on CPU, temporarily move it to the sampling device.
        ema_was = next(self.ema.ema_model.parameters()).device
        if ema_was != self.device:
            self.ema.ema_model.to(self.device)
        sampling_ok = True
        try:
            generate_samples(
                cfg=self.cfg,
                model=self.ema.ema_model,
                device=self.device,
                normal_dataset=self.normal_dataset,
                out_dir=out_dir,
                max_samples=max_samples,
            )
        except Exception as e:
            print(f"[warn] sampling failed at step {self.global_step}: {e}")
            sampling_ok = False
        finally:
            if ema_was != self.device:
                self.ema.ema_model.to(ema_was)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        if not sampling_ok:
            return
        if self.wandb is not None:
            try:
                files = sorted(os.listdir(out_dir))
                imgs = []
                for f in files:
                    if f.endswith((".png", ".jpg")):
                        imgs.append(self.wandb.Image(os.path.join(out_dir, f), caption=f))
                if imgs:
                    self.wandb.log({"samples": imgs[:24], "step": self.global_step})
            except Exception as e:
                print(f"[warn] wandb sample log failed: {e}")
