#!/usr/bin/env python3
"""CoPRIME - CLI entry point.

Usage examples::

    # Pre-train on LibriSpeech960
    python main.py pretrain --model base --librispeech-root data/librispeech

    # Fine-tune on MOSEI
    python main.py finetune --model base --checkpoint checkpoints/pretrain_epoch50.pt \\
        --mosei-root data/mosei

    # Evaluate (0-shot + 10-shot) on IEMOCAP
    python main.py evaluate --model base --checkpoint checkpoints/finetune_epoch50.pt \\
        --dataset iemocap --iemocap-root data/iemocap --n-shots 10
"""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from coprime.config import (
    CoPRIMEConfig,
    TrainingConfig,
    coprime_base_32,
    coprime_large_32,
    dense_base_32,
)
from coprime.data import (
    AudioTextCollator,
    IEMOCAPDataset,
    LibriSpeechDataset,
    MOSEIDataset,
    get_tokenizer,
    train_tokenizer_from_librispeech,
)
from coprime.evaluate import few_shot_evaluate, zero_shot_evaluate
from coprime.model import CoPRIMEModel
from coprime.train import finetune, pretrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "base": coprime_base_32,
    "large": coprime_large_32,
    "dense": dense_base_32,
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_pretrain(args):
    _set_seed(args.seed)
    config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lambda_elbo=args.lambda_elbo,
        lambda_aux=args.lambda_aux,
        checkpoint_dir=args.checkpoint_dir,
        librispeech_root=args.librispeech_root,
        tokenizer_model=args.tokenizer_model,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )

    model_cfg: CoPRIMEConfig = MODEL_CONFIGS[args.model]()
    model = CoPRIMEModel(model_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s  params: %s", args.model, f"{total_params:,}")

    dataset = LibriSpeechDataset(
        root=config.librispeech_root,
        tokenizer_model=config.tokenizer_model,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=AudioTextCollator(max_text_len=model_cfg.max_text_len),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    pretrain(model, loader, config, device=config.device)


def cmd_finetune(args):
    _set_seed(args.seed)
    config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lambda_elbo=args.lambda_elbo,
        lambda_aux=args.lambda_aux,
        checkpoint_dir=args.checkpoint_dir,
        mosei_root=args.mosei_root,
        tokenizer_model=args.tokenizer_model,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )

    model_cfg: CoPRIMEConfig = MODEL_CONFIGS[args.model]()
    model = CoPRIMEModel(model_cfg)

    dataset = MOSEIDataset(
        root=config.mosei_root,
        tokenizer_model=config.tokenizer_model,
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=AudioTextCollator(max_text_len=model_cfg.max_text_len),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    finetune(model, loader, config, checkpoint_path=args.checkpoint, device=config.device)


def cmd_train_tokenizer(args):
    _set_seed(args.seed)
    logger.info("Training tokenizer from LibriSpeech at %s ...", args.librispeech_root)
    sp = train_tokenizer_from_librispeech(
        librispeech_root=args.librispeech_root,
        librispeech_url=args.librispeech_url,
        tokenizer_model=args.tokenizer_model,
        vocab_size=args.vocab_size,
    )
    logger.info("Tokenizer ready - vocab size: %d", sp.get_piece_size())


def cmd_evaluate(args):
    _set_seed(args.seed)
    config = TrainingConfig(
        tokenizer_model=args.tokenizer_model,
        device=args.device,
        seed=args.seed,
    )
    device = config.device

    model_cfg: CoPRIMEConfig = MODEL_CONFIGS[args.model]()
    model = CoPRIMEModel(model_cfg).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    tokenizer = get_tokenizer(config.tokenizer_model)

    # Select dataset
    if args.dataset == "mosei":
        dataset = MOSEIDataset(
            root=args.mosei_root,
            tokenizer_model=config.tokenizer_model,
            split="test",
        )
        class_names = MOSEIDataset.EMOTION_CLASSES
        label_key = "emotion_label"
    elif args.dataset == "iemocap":
        dataset = IEMOCAPDataset(
            root=args.iemocap_root,
            tokenizer_model=config.tokenizer_model,
            split="test",
        )
        class_names = IEMOCAPDataset.EMOTION_CLASSES
        label_key = "emotion_label"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    eval_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=AudioTextCollator(max_text_len=model_cfg.max_text_len),
        num_workers=args.num_workers,
    )

    # 0-shot
    logger.info("=== Zero-shot evaluation on %s ===", args.dataset.upper())
    zero_results = zero_shot_evaluate(
        model, eval_loader, class_names, tokenizer,
        device=device, label_key=label_key,
    )

    # N-shot
    if args.n_shots > 0:
        logger.info("=== %d-shot evaluation on %s ===", args.n_shots, args.dataset.upper())
        few_results = few_shot_evaluate(
            model, dataset, n_shots=args.n_shots,
            device=device, label_key=label_key,
        )

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CoPRIME - Contrastive Probabilistic Routing MoE")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Shared arguments ---
    def add_common(p):
        p.add_argument("--model", choices=["base", "large", "dense"], default="base")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--tokenizer-model", default="data/tokenizer.model")
        p.add_argument("--num-workers", type=int, default=4)

    # --- train-tokenizer ---
    p_tok = subparsers.add_parser("train-tokenizer", help="Train SentencePiece tokenizer from LibriSpeech")
    add_common(p_tok)
    p_tok.add_argument("--librispeech-root", default="data/librispeech")
    p_tok.add_argument("--librispeech-url", default="train-clean-100",
                       help="LibriSpeech subset (e.g. train-clean-100, train-clean-360)")
    p_tok.add_argument("--vocab-size", type=int, default=21800)
    p_tok.set_defaults(func=cmd_train_tokenizer)

    # --- pretrain ---
    p_pre = subparsers.add_parser("pretrain", help="Pre-train on LibriSpeech960")
    add_common(p_pre)
    p_pre.add_argument("--librispeech-root", default="data/librispeech")
    p_pre.add_argument("--epochs", type=int, default=50)
    p_pre.add_argument("--batch-size", type=int, default=256)
    p_pre.add_argument("--lr", type=float, default=1e-4)
    p_pre.add_argument("--lambda-elbo", type=float, default=0.0004)
    p_pre.add_argument("--lambda-aux", type=float, default=0.04)
    p_pre.add_argument("--checkpoint-dir", default="checkpoints")
    p_pre.set_defaults(func=cmd_pretrain)

    # --- finetune ---
    p_ft = subparsers.add_parser("finetune", help="Fine-tune on MOSEI")
    add_common(p_ft)
    p_ft.add_argument("--mosei-root", default="data/mosei")
    p_ft.add_argument("--checkpoint", required=True, help="Path to pre-trained checkpoint")
    p_ft.add_argument("--epochs", type=int, default=20)
    p_ft.add_argument("--batch-size", type=int, default=64)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--lambda-elbo", type=float, default=0.0004)
    p_ft.add_argument("--lambda-aux", type=float, default=0.04)
    p_ft.add_argument("--checkpoint-dir", default="checkpoints")
    p_ft.set_defaults(func=cmd_finetune)

    # --- evaluate ---
    p_ev = subparsers.add_parser("evaluate", help="Evaluate on MOSEI or IEMOCAP")
    add_common(p_ev)
    p_ev.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p_ev.add_argument("--dataset", choices=["mosei", "iemocap"], required=True)
    p_ev.add_argument("--mosei-root", default="data/mosei")
    p_ev.add_argument("--iemocap-root", default="data/iemocap/IEMOCAP_full_release")
    p_ev.add_argument("--n-shots", type=int, default=10)
    p_ev.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
