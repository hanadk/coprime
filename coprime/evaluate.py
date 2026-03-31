"""Zero-shot and few-shot evaluation for CoPRIME.

- ``zero_shot_evaluate``: Encode class-name text descriptions as text
  prototypes, encode test audio, predict via cosine similarity.
- ``few_shot_evaluate``: Extract embeddings for N-shot support + query,
  fit a logistic regression, report accuracy and F1.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

from coprime.model import CoPRIMEModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: CoPRIMEModel,
    loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Extract audio and text embeddings (+ optional labels) from a DataLoader.

    Returns dict with numpy arrays:
        ``z_audio``:  (N, D)
        ``z_text``:   (N, D)
        ``emotion_label`` (optional): (N,)
        ``sentiment_label`` (optional): (N,)
    """
    model.eval()
    all_audio, all_text = [], []
    all_emotion, all_sentiment = [], []

    for batch in loader:
        waveform = batch["waveform"].to(device)
        token_ids = batch["token_ids"].to(device)
        text_lengths = batch.get("text_lengths")
        if text_lengths is not None:
            text_lengths = text_lengths.to(device)

        outputs = model(waveform, token_ids, text_lengths)
        all_audio.append(outputs["z_audio"].cpu())
        all_text.append(outputs["z_text"].cpu())

        if "emotion_label" in batch:
            all_emotion.append(batch["emotion_label"])
        if "sentiment_label" in batch:
            all_sentiment.append(batch["sentiment_label"])

    result: Dict[str, np.ndarray] = {
        "z_audio": torch.cat(all_audio).numpy(),
        "z_text": torch.cat(all_text).numpy(),
    }
    if all_emotion:
        result["emotion_label"] = torch.cat(all_emotion).numpy()
    if all_sentiment:
        result["sentiment_label"] = torch.cat(all_sentiment).numpy()
    return result


# ---------------------------------------------------------------------------
# Zero-shot evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def zero_shot_evaluate(
    model: CoPRIMEModel,
    eval_loader: DataLoader,
    class_names: Sequence[str],
    tokenizer,
    device: str = "cuda",
    label_key: str = "emotion_label",
    max_text_len: int = 32,
) -> Dict[str, float]:
    """Zero-shot classification via text-prototype cosine similarity.

    For each class name, the tokenizer encodes it as a text sequence.
    The model's text encoder produces a text prototype.  Test audio
    embeddings are compared against prototypes and classified by
    highest cosine similarity.

    Args:
        model: Trained CoPRIME model.
        eval_loader: DataLoader over evaluation dataset.
        class_names: List of class-name strings (one per class).
        tokenizer: SentencePiece tokenizer with ``.encode()`` method.
        device: ``"cuda"`` or ``"cpu"``.
        label_key: Key in batch dict for ground-truth labels.
        max_text_len: Max token length for class name encoding.

    Returns:
        ``{"accuracy": float, "f1": float}``
    """
    model.eval()

    # --- Build class text prototypes ---
    class_token_ids = []
    for name in class_names:
        ids = tokenizer.encode(name.lower())[:max_text_len]
        class_token_ids.append(torch.tensor(ids, dtype=torch.long))

    # Pad to same length
    max_len = max(t.size(0) for t in class_token_ids)
    padded = torch.zeros(len(class_names), max_len, dtype=torch.long)
    text_lengths = []
    for i, t in enumerate(class_token_ids):
        padded[i, : t.size(0)] = t
        text_lengths.append(t.size(0))
    text_lengths_t = torch.tensor(text_lengths)

    # We need a dummy waveform to get text embeddings through the model.
    # Use a short silence (16000 samples = 1 sec at 16kHz).
    dummy_wav = torch.zeros(len(class_names), 16000)
    padded = padded.to(device)
    text_lengths_t = text_lengths_t.to(device)
    dummy_wav = dummy_wav.to(device)

    out = model(dummy_wav, padded, text_lengths_t)
    class_prototypes = F.normalize(out["z_text"], dim=-1)  # (C, D)

    # --- Evaluate test audio ---
    all_preds, all_labels = [], []
    for batch in eval_loader:
        waveform = batch["waveform"].to(device)
        token_ids = batch["token_ids"].to(device)
        tl = batch.get("text_lengths")
        if tl is not None:
            tl = tl.to(device)

        outputs = model(waveform, token_ids, tl)
        z_a = F.normalize(outputs["z_audio"], dim=-1)  # (B, D)

        # Cosine similarity
        sims = z_a @ class_prototypes.T  # (B, C)
        preds = sims.argmax(dim=-1).cpu()

        all_preds.append(preds)
        all_labels.append(batch[label_key])

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds, average="weighted") * 100

    logger.info("[0-shot] accuracy=%.2f%%  F1=%.2f%%", acc, f1)
    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# Few-shot evaluation (linear probe)
# ---------------------------------------------------------------------------

def few_shot_evaluate(
    model: CoPRIMEModel,
    dataset,
    n_shots: int = 10,
    device: str = "cuda",
    label_key: str = "emotion_label",
    num_trials: int = 10,
    batch_size: int = 64,
) -> Dict[str, float]:
    """N-shot evaluation via linear logistic regression on frozen embeddings.

    Samples ``n_shots`` examples per class as support, uses the rest as
    query.  Repeats ``num_trials`` times and averages metrics.

    Args:
        model: Trained CoPRIME model (will be set to eval mode).
        dataset: Dataset returning dicts with ``waveform``, ``token_ids``,
                 ``text_lengths``, and ``label_key``.
        n_shots: Number of examples per class.
        device: ``"cuda"`` or ``"cpu"``.
        label_key: Key for ground-truth labels.
        num_trials: Number of random sampling trials.
        batch_size: Batch size for embedding extraction.

    Returns:
        ``{"accuracy": float, "f1": float, "accuracy_std": float, "f1_std": float}``
    """
    from coprime.data import AudioTextCollator

    model.eval()

    # Pre-extract all embeddings
    full_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=AudioTextCollator(),
        num_workers=0,
    )
    emb_dict = extract_embeddings(model, full_loader, device)
    z_audio = emb_dict["z_audio"]  # (N, D)
    labels = emb_dict[label_key]   # (N,)

    classes = np.unique(labels)
    accs, f1s = [], []

    for trial in range(num_trials):
        rng = np.random.RandomState(trial)
        support_idx, query_idx = [], []

        for cls in classes:
            cls_idx = np.where(labels == cls)[0]
            chosen = rng.choice(cls_idx, size=min(n_shots, len(cls_idx)), replace=False)
            rest = np.setdiff1d(cls_idx, chosen)
            support_idx.extend(chosen.tolist())
            query_idx.extend(rest.tolist())

        X_train = z_audio[support_idx]
        y_train = labels[support_idx]
        X_test = z_audio[query_idx]
        y_test = labels[query_idx]

        # L2-normalize
        X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
        X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accs.append(accuracy_score(y_test, preds) * 100)
        f1s.append(f1_score(y_test, preds, average="weighted") * 100)

    acc_mean, acc_std = np.mean(accs), np.std(accs)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s)

    logger.info(
        "[%d-shot] accuracy=%.2f ± %.2f%%  F1=%.2f ± %.2f%%",
        n_shots, acc_mean, acc_std, f1_mean, f1_std,
    )
    return {
        "accuracy": float(acc_mean),
        "f1": float(f1_mean),
        "accuracy_std": float(acc_std),
        "f1_std": float(f1_std),
    }
