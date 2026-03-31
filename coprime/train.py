"""Pre-training and fine-tuning loops for CoPRIME.

- ``pretrain``: Contrastive pre-training on LibriSpeech960 with all auxiliary losses.
- ``finetune``: Contrastive fine-tuning on MOSEI (loads a pre-trained checkpoint).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from coprime.config import CoPRIMEConfig, TrainingConfig
from coprime.losses import CoPRIMELoss
from coprime.model import CoPRIMEModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    """Linear warmup -> cosine annealing schedule."""
    warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def _save_checkpoint(
    model: CoPRIMEModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logger.info("Checkpoint saved to %s", path)


def _load_checkpoint(
    path: str,
    model: CoPRIMEModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d, step %d)", path, ckpt["epoch"], ckpt["step"])
    return ckpt["epoch"], ckpt["step"]


# ---------------------------------------------------------------------------
# Training step (shared by pretrain and finetune)
# ---------------------------------------------------------------------------

def _train_one_step(
    model: CoPRIMEModel,
    criterion: CoPRIMELoss,
    batch: dict,
    device: str,
    ema_decay: float,
) -> dict:
    """Run one forward + backward pass and update EMA prototypes."""
    waveform = batch["waveform"].to(device)
    token_ids = batch["token_ids"].to(device)
    text_lengths = batch.get("text_lengths")
    if text_lengths is not None:
        text_lengths = text_lengths.to(device)

    outputs = model(waveform, token_ids, text_lengths)

    loss_dict = criterion(
        z_audio=outputs["z_audio"],
        z_text=outputs["z_text"],
        routing_infos=outputs["routing_infos"],
        token_embeddings=outputs["token_embeddings"],
        modality_mask=outputs["modality_mask"],
        expert_prototypes=getattr(model, "expert_prototypes",
                                  torch.empty(0, device=device)).clone(),
    )

    # Update expert prototypes via EMA (only if MoE layers exist)
    if model.config.num_experts > 0 and len(outputs["routing_infos"]) > 0:
        avg_gw = sum(ri.gating_weights for ri in outputs["routing_infos"])
        avg_gw = avg_gw / len(outputs["routing_infos"])
        token_emb_flat = outputs["token_embeddings"].reshape(-1, model.config.hidden_dim)
        mask_flat = outputs["modality_mask"].reshape(-1)
        model.update_prototypes(token_emb_flat, avg_gw, mask_flat, ema_decay)

    return loss_dict


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def pretrain(
    model: CoPRIMEModel,
    train_loader: DataLoader,
    config: TrainingConfig,
    device: str = "cuda",
):
    """Contrastive pre-training on LibriSpeech960.

    Args:
        model: CoPRIME model instance.
        train_loader: DataLoader yielding dicts with keys
                      ``waveform``, ``token_ids``, ``text_lengths``.
        config: Training hyper-parameters.
        device: ``"cuda"`` or ``"cpu"``.
    """
    model = model.to(device)
    model.train()

    criterion = CoPRIMELoss(
        lambda_elbo=config.lambda_elbo,
        lambda_aux=config.lambda_aux,
        temperature=config.temperature,
        sigma_sq=config.sigma_sq,
        num_modalities=model.config.num_modalities,
    )

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = _build_scheduler(optimizer, config.warmup_steps, total_steps)

    global_step = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss_dict = _train_one_step(model, criterion, batch, device, config.ema_decay)
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        logger.info("[Pretrain] epoch %d done  avg_loss %.4f  time %.1fs", epoch, avg_loss, elapsed)

        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(
                config.checkpoint_dir, f"pretrain_epoch{epoch + 1}.pt"
            )
            _save_checkpoint(model, optimizer, epoch, global_step, ckpt_path)

    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune(
    model: CoPRIMEModel,
    train_loader: DataLoader,
    config: TrainingConfig,
    checkpoint_path: str,
    device: str = "cuda",
):
    """Fine-tune a pre-trained CoPRIME model on MOSEI.

    Same contrastive + auxiliary loss objective but loaded from a
    pre-trained checkpoint.

    Args:
        model: CoPRIME model instance (architecture must match checkpoint).
        train_loader: DataLoader yielding dicts with keys
                      ``waveform``, ``token_ids``, ``text_lengths``.
        config: Training hyper-parameters (may use lower lr).
        checkpoint_path: Path to a pre-trained checkpoint.
        device: ``"cuda"`` or ``"cpu"``.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    _load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    model.train()

    criterion = CoPRIMELoss(
        lambda_elbo=config.lambda_elbo,
        lambda_aux=config.lambda_aux,
        temperature=config.temperature,
        sigma_sq=config.sigma_sq,
        num_modalities=model.config.num_modalities,
    )

    total_steps = len(train_loader) * config.epochs
    scheduler = _build_scheduler(optimizer, config.warmup_steps, total_steps)

    global_step = 0
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss_dict = _train_one_step(model, criterion, batch, device, config.ema_decay)
            loss = loss_dict["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        logger.info("[Finetune] epoch %d done  avg_loss %.4f  time %.1fs", epoch, avg_loss, elapsed)

        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(
                config.checkpoint_dir, f"finetune_epoch{epoch + 1}.pt"
            )
            _save_checkpoint(model, optimizer, epoch, global_step, ckpt_path)

    return model
