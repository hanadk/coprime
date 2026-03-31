"""CoPRIME loss functions.

Implements all six loss components:

1. Contrastive loss  - symmetric InfoNCE
2. Importance loss   - entropy-based expert importance balance
3. Load loss         - entropy-based smooth load balance
4. Z-loss            - router logit magnitude regularizer
5. MI loss           - mutual information for expert diversity
6. ELBO loss         - expert specialization + KL to uniform

Plus the combined objective ``CoPRIMELoss``.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from coprime.model import RoutingInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _negative_entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Compute negative entropy: Σ p log(p).

    Returns a scalar that, when *minimized*, encourages a *uniform*
    distribution (high entropy).
    """
    p = probs.clamp(min=eps)
    return (p * p.log()).sum(dim)


def _entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Compute entropy: -Σ p log(p)."""
    p = probs.clamp(min=eps)
    return -(p * p.log()).sum(dim)


# ===========================================================================
# 1. Contrastive loss
# ===========================================================================

def contrastive_loss(
    z_audio: torch.Tensor,
    z_text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss.

    Args:
        z_audio: (B, D) L2-normalizable audio representations.
        z_text:  (B, D) L2-normalizable text representations.
        temperature: Scaling parameter τ.

    Returns:
        Scalar loss averaged over the batch.
    """
    z_a = F.normalize(z_audio, dim=-1)
    z_t = F.normalize(z_text, dim=-1)

    # Cosine similarity matrix scaled by temperature
    logits = z_a @ z_t.T / temperature  # (B, B)

    labels = torch.arange(logits.size(0), device=logits.device)

    # Audio-to-text + text-to-audio (symmetric)
    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_a2t + loss_t2a)


# ===========================================================================
# 2. Importance loss
# ===========================================================================

def importance_loss(routing_infos: List[RoutingInfo]) -> torch.Tensor:
    """Entropy-based importance loss averaged over MoE layers.

    imp_e(X) = Σ_x g(x)_e  ->  normalize  ->  negative entropy
    """
    losses = []
    for ri in routing_infos:
        # ri.gating_weights: (n, E)
        importance = ri.gating_weights.sum(dim=0)  # (E,)
        importance_dist = importance / (importance.sum() + 1e-8)
        losses.append(_negative_entropy(importance_dist, dim=0))
    return torch.stack(losses).mean()


# ===========================================================================
# 3. Load loss
# ===========================================================================

def load_loss(routing_infos: List[RoutingInfo]) -> torch.Tensor:
    """Entropy-based load loss averaged over MoE layers.

    load_e(X) = Σ_x p_e(x)  ->  normalize  ->  negative entropy
    """
    losses = []
    for ri in routing_infos:
        # ri.load_probs: (n, E)
        load = ri.load_probs.sum(dim=0)  # (E,)
        load_dist = load / (load.sum() + 1e-8)
        losses.append(_negative_entropy(load_dist, dim=0))
    return torch.stack(losses).mean()


# ===========================================================================
# 4. Z-loss
# ===========================================================================

def z_loss(routing_infos: List[RoutingInfo]) -> torch.Tensor:
    """Z-loss for router stability: penalises large router logits.

    L_z = (1/n) Σ_i (log Σ_e exp(a_{i,e}))²
    """
    losses = []
    for ri in routing_infos:
        # ri.raw_logits: (n, E)
        lse = torch.logsumexp(ri.raw_logits, dim=-1)  # (n,)
        losses.append((lse ** 2).mean())
    return torch.stack(losses).mean()


# ===========================================================================
# 5. Mutual information loss
# ===========================================================================

def mi_loss(
    routing_infos: List[RoutingInfo],
    modality_mask: torch.Tensor,
    num_modalities: int = 2,
) -> torch.Tensor:
    """MI loss: encourages diversity by conditioning on modality.

    L_MI = -1/M Σ_m H(p̃_m) + H(1/M Σ_m p̃_m)

    where p̃_m = mean gating weights for modality m.

    Minimizing this effectively *maximizes* mutual information between
    the modality and the expert assignment.

    Args:
        routing_infos: Per-layer routing metadata.
        modality_mask: (B, S) integer tensor, 0=audio, 1=text.
        num_modalities: Number of modalities (default 2).
    """
    mask_flat = modality_mask.reshape(-1)  # (n,)
    losses = []

    for ri in routing_infos:
        gw = ri.gating_weights  # (n, E)
        marginals = []
        for m in range(num_modalities):
            m_mask = mask_flat == m
            if not m_mask.any():
                marginals.append(torch.ones(gw.size(1), device=gw.device) / gw.size(1))
                continue
            p_m = gw[m_mask].mean(dim=0)  # (E,)
            marginals.append(p_m)

        marginals_stack = torch.stack(marginals, dim=0)  # (M, E)

        # -1/M Σ_m H(p̃_m)
        per_modality_H = torch.stack([_entropy(marginals_stack[m], dim=0)
                                      for m in range(num_modalities)])
        neg_avg_H = -per_modality_H.mean()

        # + H(1/M Σ_m p̃_m)
        avg_marginal = marginals_stack.mean(dim=0)  # (E,)
        H_avg = _entropy(avg_marginal, dim=0)

        losses.append(neg_avg_H + H_avg)

    return torch.stack(losses).mean()


# ===========================================================================
# 6. ELBO loss
# ===========================================================================

def elbo_loss(
    routing_infos: List[RoutingInfo],
    token_embeddings: torch.Tensor,
    modality_mask: torch.Tensor,
    expert_prototypes: torch.Tensor,
    sigma_sq: float = 1.0,
    num_modalities: int = 2,
) -> torch.Tensor:
    """ELBO loss for expert specialization and diversity.

    For each modality m:
        L_ELBO(X_m) = E_{p(e|x)}[log p(X_m|e)]  - KL(p_m(e) || p(e))

    where:
        log p(X_m|e) uses a Gaussian N(X_m; μ_e^(m), σ²I),
        averaged per token to be invariant to sequence length.
        p(e) is uniform

    The total loss is negated and averaged:  -1/M Σ_m L_ELBO(X_m)

    Args:
        routing_infos: Per-layer routing metadata.
        token_embeddings: (B, S, D) embeddings after the final norm.
        modality_mask: (B, S) integer tensor, 0=audio, 1=text.
        expert_prototypes: (M, E, D) EMA centroids.
        sigma_sq: Gaussian variance σ².
        num_modalities: Number of modalities.
    """
    B, S, D = token_embeddings.shape
    E = expert_prototypes.size(1)
    device = token_embeddings.device

    tok_flat = token_embeddings.reshape(-1, D)      # (n, D)
    mask_flat = modality_mask.reshape(-1)            # (n,)

    # Average routing probabilities across MoE layers for the ELBO
    avg_gw = torch.zeros(tok_flat.size(0), E, device=device)
    for ri in routing_infos:
        avg_gw = avg_gw + ri.gating_weights
    avg_gw = avg_gw / max(len(routing_infos), 1)    # (n, E)

    total_elbo = torch.tensor(0.0, device=device)

    for m in range(num_modalities):
        m_mask = mask_flat == m  # (n,)
        if not m_mask.any():
            continue

        tokens_m = tok_flat[m_mask]        # (n_m, D)
        gw_m = avg_gw[m_mask]              # (n_m, E)
        n_m = tokens_m.size(0)

        # --- Marginal expert probability for this modality ---
        p_m = gw_m.mean(dim=0)  # (E,)

        # --- Expert likelihood: log p(X_m | e) = Gaussian log-lik ---
        # Per-token average: -1/(2σ²) (1/n_m) Σ_t ||x_t - μ_e||² - (D/2)*log(2πσ²)
        prototypes_m = expert_prototypes[m]  # (E, D)

        # Pairwise squared distances: (n_m, E)
        # ||x_t - μ_e||² for all tokens and experts
        sq_dist = torch.cdist(tokens_m, prototypes_m, p=2).pow(2)  # (n_m, E)

        # Average over tokens for each expert: (E,)
        mean_sq_dist = sq_dist.mean(dim=0)

        log_lik = -mean_sq_dist / (2 * sigma_sq) - (D / 2) * math.log(2 * math.pi * sigma_sq)
        # log_lik: (E,)

        # --- E_{p(e|x)}[log p(X_m|e)] ≈ Σ_e p_m(e) * log p(X_m|e) ---
        expected_log_lik = (p_m * log_lik).sum()

        # --- KL(p_m(e) || uniform(1/E)) ---
        uniform = torch.ones(E, device=device) / E
        kl = F.kl_div(uniform.log(), p_m, reduction="sum", log_target=False)

        elbo_m = expected_log_lik - kl
        total_elbo = total_elbo + elbo_m

    # Negate and average across modalities: -1/M Σ_m L_ELBO(X_m)
    return -total_elbo / num_modalities


# ===========================================================================
# Combined objective
# ===========================================================================

class CoPRIMELoss(nn.Module):
    """Final CoPRIME objective function.

    L(X) = L_contrastive + λ_ELBO * L_ELBO
           + λ_aux * (L_imp + L_load + L_z + L_MI)
    """

    def __init__(
        self,
        lambda_elbo: float = 0.04,
        lambda_aux: float = 0.02,
        temperature: float = 0.07,
        sigma_sq: float = 1.0,
        num_modalities: int = 2,
    ):
        super().__init__()
        self.lambda_elbo = lambda_elbo
        self.lambda_aux = lambda_aux
        self.temperature = temperature
        self.sigma_sq = sigma_sq
        self.num_modalities = num_modalities

    def forward(
        self,
        z_audio: torch.Tensor,
        z_text: torch.Tensor,
        routing_infos: List[RoutingInfo],
        token_embeddings: torch.Tensor,
        modality_mask: torch.Tensor,
        expert_prototypes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss terms and the combined objective.

        Returns:
            Dictionary containing ``total``, ``contrastive``, ``elbo``,
            ``importance``, ``load``, ``z_loss``, ``mi``.
        """
        L_contrastive = contrastive_loss(z_audio, z_text, self.temperature)

        has_moe = len(routing_infos) > 0

        if has_moe:
            L_imp = importance_loss(routing_infos)
            L_load = load_loss(routing_infos)
            L_z = z_loss(routing_infos)
            L_mi = mi_loss(routing_infos, modality_mask, self.num_modalities)
            L_elbo = elbo_loss(
                routing_infos,
                token_embeddings,
                modality_mask,
                expert_prototypes,
                self.sigma_sq,
                self.num_modalities,
            )
        else:
            zero = torch.tensor(0.0, device=z_audio.device)
            L_imp = L_load = L_z = L_mi = L_elbo = zero

        total = (
            L_contrastive
            + self.lambda_elbo * L_elbo
            + self.lambda_aux * (L_imp + L_load + L_z + L_mi)
        )

        return {
            "total": total,
            "contrastive": L_contrastive.detach(),
            "elbo": L_elbo.detach() if has_moe else L_elbo,
            "importance": L_imp.detach() if has_moe else L_imp,
            "load": L_load.detach() if has_moe else L_load,
            "z_loss": L_z.detach() if has_moe else L_z,
            "mi": L_mi.detach() if has_moe else L_mi,
        }
