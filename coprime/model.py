"""CoPRIME model architecture.

Implements the full CoPRIME (Contrastive Probabilistic Routing for
IMbalanced tokens with ELBO-regularized mixture of experts) model from
this implementation, including:

- SpectrogramPatchEmbedding: ViT-style audio tokenization
- TextEmbedding: Learned token + position embeddings
- TopKRouter: Sparse gating with noisy load estimation
- ExpertMLP / MoELayer: Sparse mixture-of-experts feed-forward
- TransformerBlock / MoEBlock: Pre-norm encoder blocks
- CoPRIMEModel: Full end-to-end model with EMA expert prototypes
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from coprime.config import CoPRIMEConfig


# ---------------------------------------------------------------------------
# Routing info container
# ---------------------------------------------------------------------------

@dataclass
class RoutingInfo:
    """Per-layer routing information collected for loss computation."""
    gating_weights: torch.Tensor   # (n_tokens, E) softmax probs
    raw_logits: torch.Tensor       # (n_tokens, E) pre-softmax logits
    load_probs: torch.Tensor       # (n_tokens, E) smooth selection probs
    expert_indices: torch.Tensor   # (n_tokens, K) selected expert ids


# ===========================================================================
# Embeddings
# ===========================================================================

class SpectrogramPatchEmbedding(nn.Module):
    """Convert raw waveform -> mel-spectrogram -> ViT-style patch embeddings.

    The spectrogram is divided into non-overlapping PxP patches, each
    flattened and linearly projected into the model dimension D.
    """

    def __init__(self, config: CoPRIMEConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_dim

        # Mel-spectrogram transform
        self.mel_spec = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # Patch projection: flatten(P*P) -> D
        patch_dim = config.patch_size * config.patch_size
        self.projection = nn.Linear(patch_dim, config.hidden_dim)

        # Learnable positional embeddings (pre-allocated, grown dynamically)
        initial_patches = (config.n_mels // config.patch_size) * 10  # reasonable default
        self.pos_embed = nn.Parameter(
            torch.randn(initial_patches, config.hidden_dim) * 0.02
        )

    def _ensure_pos_embed(self, num_patches: int, device: torch.device):
        """Grow positional embeddings if input exceeds current allocation."""
        if self.pos_embed.size(0) < num_patches:
            self.pos_embed = nn.Parameter(
                torch.randn(num_patches, self.hidden_dim, device=device) * 0.02
            )

    def _patchify(self, spec: torch.Tensor) -> torch.Tensor:
        """Divide spectrogram into non-overlapping PxP patches.

        Args:
            spec: (B, 1, H, W) spectrogram (H=n_mels, W=time frames).

        Returns:
            patches: (B, num_patches, P*P)
        """
        P = self.patch_size
        B, _, H, W = spec.shape

        # Pad to make H and W divisible by P
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h > 0 or pad_w > 0:
            spec = F.pad(spec, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        # Unfold into patches
        patches = spec.unfold(2, P, P).unfold(3, P, P)  # (B, 1, nH, nW, P, P)
        patches = patches.contiguous().view(B, -1, P * P)  # (B, num_patches, P*P)
        return patches

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, num_samples) raw audio waveform at ``sample_rate``.

        Returns:
            embeddings: (B, num_patches, D)
        """
        # Compute mel-spectrogram: (B, n_mels, T)
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)
        spec = spec.unsqueeze(1)  # (B, 1, n_mels, T)

        # Patchify and project
        patches = self._patchify(spec)  # (B, N, P*P)
        embeddings = self.projection(patches)  # (B, N, D)

        # Add positional embeddings
        N = embeddings.size(1)
        self._ensure_pos_embed(N, embeddings.device)
        embeddings = embeddings + self.pos_embed[:N]

        return embeddings


class TextEmbedding(nn.Module):
    """Lookup embedding + learned positional embedding for text tokens."""

    def __init__(self, config: CoPRIMEConfig):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.max_text_len, config.hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, S) integer token indices.

        Returns:
            embeddings: (B, S, D)
        """
        B, S = token_ids.shape
        positions = torch.arange(S, device=token_ids.device).unsqueeze(0)
        return self.token_embed(token_ids) + self.pos_embed(positions)


# ===========================================================================
# Router
# ===========================================================================

class TopKRouter(nn.Module):
    """Sparse top-K routing with noisy gating for load-loss computation.

    For each token x the router computes:
        g(x) = softmax(W_g · x)          - gating weights
    and selects the top-K experts.

    For the *load loss*, a smooth selection probability is
    estimated via noisy gating:
        p_e(x) = 1 - Φ((τ_K(x) - (Wx)_e) / σ)
    where σ = 1/E and τ_K is the K-th largest logit.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.sigma = 1.0 / num_experts

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, RoutingInfo]:
        """
        Args:
            x: (n, D) token embeddings (batch of tokens, flattened across B
               and sequence length).

        Returns:
            top_k_weights: (n, K) gating weights for selected experts
                           (re-normalized so they sum to 1).
            top_k_indices: (n, K) expert indices.
            routing_info:  Full routing metadata for loss computation.
        """
        logits = self.gate(x)  # (n, E)
        gating_weights = F.softmax(logits, dim=-1)  # (n, E)

        # Top-K selection
        top_k_weights, top_k_indices = torch.topk(
            gating_weights, self.top_k, dim=-1
        )  # (n, K)
        # Re-normalize selected weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # --- Smooth load probabilities for load loss ---
        # Add noise to logits for smooth approximation
        noise = torch.randn_like(logits) * self.sigma
        noisy_logits = logits + noise  # (n, E)

        # K-th largest noisy logit per token (threshold)
        tau_k, _ = torch.topk(noisy_logits, self.top_k, dim=-1)  # (n, K)
        tau_k = tau_k[:, -1:]  # (n, 1) - the K-th largest value

        # p_e(x) = 1 - Φ((τ_K - logit_e) / σ)
        # Use the *clean* logits for the CDF argument
        z_val = (tau_k - logits) / self.sigma  # (n, E)
        load_probs = 1.0 - 0.5 * (1.0 + torch.erf(z_val / math.sqrt(2.0)))  # (n, E)

        routing_info = RoutingInfo(
            gating_weights=gating_weights,
            raw_logits=logits,
            load_probs=load_probs,
            expert_indices=top_k_indices,
        )
        return top_k_weights, top_k_indices, routing_info


# ===========================================================================
# Experts & MoE layer
# ===========================================================================

class ExpertMLP(nn.Module):
    """Single expert: two-layer MLP with GELU activation."""

    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts layer.

    Routes each token to the top-K experts and computes a weighted sum
    of their outputs.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_dim, mlp_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, RoutingInfo]:
        """
        Args:
            x: (B, S, D) token embeddings.

        Returns:
            output: (B, S, D) MoE output (same shape as input).
            routing_info: Routing metadata for this layer.
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (n, D) where n = B * S

        top_k_weights, top_k_indices, routing_info = self.router(x_flat)
        # top_k_weights: (n, K), top_k_indices: (n, K)

        # Dispatch tokens to experts and accumulate weighted outputs
        output = torch.zeros_like(x_flat)  # (n, D)

        for e_idx in range(self.num_experts):
            # Mask: which (token, slot) pairs selected this expert
            mask = top_k_indices == e_idx  # (n, K) bool
            if not mask.any():
                continue
            # Gather the corresponding weights
            weights = (top_k_weights * mask.float()).sum(dim=-1)  # (n,)
            active = weights > 0  # (n,) bool - tokens that use this expert
            if not active.any():
                continue
            expert_input = x_flat[active]  # (n_active, D)
            expert_output = self.experts[e_idx](expert_input)  # (n_active, D)
            output[active] += weights[active].unsqueeze(-1) * expert_output

        output = output.view(B, S, D)
        return output, routing_info


# ===========================================================================
# Transformer / MoE encoder blocks
# ===========================================================================

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (LayerNorm -> Attn -> residual ->
    LayerNorm -> FFN -> residual)."""

    def __init__(self, hidden_dim: int, mlp_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.drop(h)
        return x


class MoEBlock(nn.Module):
    """Pre-norm encoder block with MoE replacing the FFN.

    LayerNorm -> Attn -> residual -> LayerNorm -> MoELayer -> residual.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.moe = MoELayer(hidden_dim, mlp_dim, num_experts, top_k)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, RoutingInfo]:
        # Self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        # MoE FFN
        h = self.norm2(x)
        h, routing_info = self.moe(h)
        x = x + self.drop(h)
        return x, routing_info


# ===========================================================================
# Full CoPRIME model
# ===========================================================================

class CoPRIMEModel(nn.Module):
    """CoPRIME: Contrastive Probabilistic Routing for IMbalanced tokens
    with ELBO-regularized mixture of experts.

    Architecture:
        1. Modality-specific embeddings (audio patches / text tokens)
        2. Modality flag appended to each token
        3. Concatenated tokens -> shared dense Transformer blocks
        4. Shared MoE blocks (with top-K routing)
        5. Average-pool per modality -> linear projection heads
        6. Returns (z_audio, z_text) for contrastive loss, plus routing info

    The model also maintains per-expert, per-modality prototype centroids
    (μ_e^(m)) updated via EMA for the ELBO loss.
    """

    def __init__(self, config: CoPRIMEConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim

        # --- Modality-specific embeddings ---
        self.audio_embed = SpectrogramPatchEmbedding(config)
        self.text_embed = TextEmbedding(config)

        # Learned modality type embedding (audio=0, text=1)
        self.modality_embed = nn.Embedding(config.num_modalities, D)

        # --- Shared encoder ---
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(D, config.mlp_dim, config.num_heads, config.dropout)
            for _ in range(config.num_transformer_blocks)
        ])
        self.moe_blocks = nn.ModuleList([
            MoEBlock(D, config.mlp_dim, config.num_heads,
                     config.num_experts, config.top_k, config.dropout)
            for _ in range(config.num_moe_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(D)

        # --- Projection heads ---
        self.audio_proj = nn.Linear(D, D, bias=False)
        self.text_proj = nn.Linear(D, D, bias=False)

        # --- Expert prototypes for ELBO loss ---
        # μ_e^(m): (num_modalities, num_experts, D)
        # Updated via EMA during training; stored as buffers.
        if config.num_experts > 0:
            self.register_buffer(
                "expert_prototypes",
                torch.zeros(config.num_modalities, config.num_experts, D),
            )
            self.register_buffer(
                "prototype_counts",
                torch.zeros(config.num_modalities, config.num_experts),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    # EMA prototype update
    # -----------------------------------------------------------------

    @torch.no_grad()
    def update_prototypes(
        self,
        token_embeddings: torch.Tensor,
        gating_weights: torch.Tensor,
        modality_mask: torch.Tensor,
        ema_decay: float = 0.999,
    ):
        """Update expert prototypes via exponential moving average.

        Args:
            token_embeddings: (n, D) all token embeddings (flattened).
            gating_weights: (n, E) routing probabilities.
            modality_mask: (n,) integer modality id per token (0=audio, 1=text).
            ema_decay: EMA coefficient.
        """
        E = self.config.num_experts
        if E == 0:
            return
        for m_id in range(self.config.num_modalities):
            mask = modality_mask == m_id  # (n,) bool
            if not mask.any():
                continue
            tokens_m = token_embeddings[mask]  # (n_m, D)
            weights_m = gating_weights[mask]   # (n_m, E)
            for e in range(E):
                w = weights_m[:, e]  # (n_m,)
                if w.sum() < 1e-8:
                    continue
                weighted_mean = (w.unsqueeze(-1) * tokens_m).sum(0) / (w.sum() + 1e-8)
                self.expert_prototypes[m_id, e] = (
                    ema_decay * self.expert_prototypes[m_id, e]
                    + (1 - ema_decay) * weighted_mean
                )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,
        token_ids: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | List[RoutingInfo]]:
        """
        Args:
            waveform: (B, num_samples) raw audio at ``sample_rate``.
            token_ids: (B, S_text) integer text token indices.
            text_lengths: (B,) actual text lengths (for masking padding).
                          If None, all positions are treated as valid.

        Returns:
            Dictionary with keys:
            - ``z_audio``: (B, D) projected audio representation.
            - ``z_text``:  (B, D) projected text representation.
            - ``routing_infos``: list of ``RoutingInfo`` (one per MoE block).
            - ``token_embeddings``: (B, S_total, D) after final norm.
            - ``modality_mask``: (B, S_total) int tensor, 0=audio, 1=text.
        """
        B = waveform.size(0)
        device = waveform.device
        D = self.config.hidden_dim

        # --- Embed ---
        audio_emb = self.audio_embed(waveform)  # (B, N_a, D)
        text_emb = self.text_embed(token_ids)   # (B, S_t, D)

        N_a = audio_emb.size(1)
        S_t = text_emb.size(1)

        # Add modality flags
        audio_emb = audio_emb + self.modality_embed(
            torch.zeros(B, N_a, dtype=torch.long, device=device)
        )
        text_emb = text_emb + self.modality_embed(
            torch.ones(B, S_t, dtype=torch.long, device=device)
        )

        # Concatenate: (B, S_total, D) where S_total = N_a + S_t
        x = torch.cat([text_emb, audio_emb], dim=1)  # text first, then audio
        S_total = x.size(1)

        # Modality mask: 1=text (first S_t), 0=audio (last N_a)
        modality_mask = torch.cat([
            torch.ones(B, S_t, dtype=torch.long, device=device),
            torch.zeros(B, N_a, dtype=torch.long, device=device),
        ], dim=1)  # (B, S_total)

        # --- Shared dense Transformer blocks ---
        for block in self.transformer_blocks:
            x = block(x)

        # --- MoE blocks ---
        routing_infos: List[RoutingInfo] = []
        for moe_block in self.moe_blocks:
            x, routing_info = moe_block(x)
            routing_infos.append(routing_info)

        # Final norm
        x = self.final_norm(x)

        # --- Average pool per modality ---
        # Text: positions [0, S_t)
        text_tokens = x[:, :S_t, :]  # (B, S_t, D)
        if text_lengths is not None:
            # Masked average pooling
            mask = torch.arange(S_t, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (B, S_t, 1)
            z_text = (text_tokens * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            z_text = text_tokens.mean(dim=1)

        # Audio: positions [S_t, S_total)
        audio_tokens = x[:, S_t:, :]  # (B, N_a, D)
        z_audio = audio_tokens.mean(dim=1)  # (B, D)

        # --- Projections ---
        z_audio = self.audio_proj(z_audio)  # (B, D)
        z_text = self.text_proj(z_text)     # (B, D)

        return {
            "z_audio": z_audio,
            "z_text": z_text,
            "routing_infos": routing_infos,
            "token_embeddings": x,
            "modality_mask": modality_mask,
        }
