"""Configuration dataclasses and factory functions for CoPRIME models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoPRIMEConfig:
    """Model architecture configuration.

    The Dense baseline is obtained by setting ``num_moe_blocks=0`` 
    (all blocks become dense Transformer encoder layers).
    """

    hidden_dim: int = 512
    mlp_dim: int = 2048
    num_heads: int = 8
    num_transformer_blocks: int = 3
    num_moe_blocks: int = 6
    num_experts: int = 8
    top_k: int = 2
    patch_size: int = 32
    vocab_size: int = 32000
    max_text_len: int = 128
    num_modalities: int = 2  # audio, text
    dropout: float = 0.1

    # Audio spectrogram parameters
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512

    @property
    def total_blocks(self) -> int:
        return self.num_transformer_blocks + self.num_moe_blocks


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def coprime_base_32() -> CoPRIMEConfig:
    """CoPRIME-B/32: Base model with patch size 32."""
    return CoPRIMEConfig(
        hidden_dim=512,
        mlp_dim=2048,
        num_heads=8,
        num_transformer_blocks=3,
        num_moe_blocks=6,
        num_experts=8,
        top_k=2,
        patch_size=32,
    )


def coprime_large_32() -> CoPRIMEConfig:
    """CoPRIME-L/32: Large model with patch size 32."""
    return CoPRIMEConfig(
        hidden_dim=768,
        mlp_dim=3072,
        num_heads=8,
        num_transformer_blocks=6,
        num_moe_blocks=12,
        num_experts=16,
        top_k=4,
        patch_size=32,
    )


def dense_base_32() -> CoPRIMEConfig:
    """Dense-B/32: Dense baseline (no MoE blocks)."""
    return CoPRIMEConfig(
        hidden_dim=512,
        mlp_dim=2048,
        num_heads=8,
        num_transformer_blocks=9,  # all blocks are dense
        num_moe_blocks=0,
        num_experts=0,
        top_k=0,
        patch_size=32,
    )


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 1
    warmup_steps: int = 2000

    # Loss weights
    lambda_elbo: float = 0.0004
    lambda_aux: float = 0.04
    temperature: float = 0.07

    # ELBO parameters
    ema_decay: float = 0.999
    sigma_sq: float = 1.0

    # Checkpointing / logging
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50
    save_interval: int = 1  # save every N epochs

    # Data
    librispeech_root: str = "data/librispeech"
    mosei_root: str = "data/mosei"
    iemocap_root: str = "data/iemocap/IEMOCAP_full_release"
    tokenizer_model: str = "data/tokenizer.model"
    num_workers: int = 4

    # Few-shot evaluation
    n_shots: int = 10

    # Device
    device: str = "cuda"
    seed: int = 42
