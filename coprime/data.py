"""Dataset classes and tokenizer utilities for CoPRIME.

Provides:
- ``get_tokenizer``: load or train a SentencePiece tokenizer.
- ``LibriSpeechDataset``: wraps torchaudio LibriSpeech for pre-training.
- ``MOSEIDataset``: loads MOSEI audio + text + labels from a processed directory.
- ``IEMOCAPDataset``: loads IEMOCAP audio + text + labels from a processed directory.
- ``AudioTextCollator``: dynamic-padding collator for DataLoader.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging

import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SentencePiece tokenizer
# ---------------------------------------------------------------------------

def get_tokenizer(
    model_path: str,
    train_text_file: Optional[str] = None,
    vocab_size: int = 32000,
):
    """Load (or train) a SentencePiece tokenizer.

    If ``model_path`` does not exist but ``train_text_file`` is given,
    a new unigram model is trained and saved.

    Args:
        model_path: Path to ``*.model`` file.
        train_text_file: Optional path to a plain-text corpus for training.
        vocab_size: Vocabulary size when training.

    Returns:
        A ``sentencepiece.SentencePieceProcessor`` instance.
    """
    import sentencepiece as spm

    if not os.path.isfile(model_path):
        if train_text_file is None:
            raise FileNotFoundError(
                f"Tokenizer model not found at {model_path} and no "
                "training corpus was supplied."
            )
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        prefix = model_path.replace(".model", "")
        logger.info("Training SentencePiece tokenizer from %s ...", train_text_file)
        spm.SentencePieceTrainer.train(
            input=train_text_file,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        logger.info("Tokenizer saved to %s", model_path)

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def extract_librispeech_transcripts(
    root: str,
    url: str = "train-clean-100",
    output_path: str = "data/librispeech_transcripts.txt",
) -> str:
    """Extract all transcripts from a LibriSpeech split to a text file.

    Each line contains one transcript.  The file is written only if it
    does not already exist.

    Args:
        root: Root directory for torchaudio LibriSpeech download.
        url: LibriSpeech subset name (e.g. ``"train-clean-100"``).
        output_path: Where to write the transcript file.

    Returns:
        ``output_path`` (absolute).
    """
    output_path = os.path.abspath(output_path)
    if os.path.isfile(output_path):
        logger.info("Transcripts already exist at %s - skipping extraction.", output_path)
        return output_path

    logger.info("Extracting LibriSpeech transcripts from %s/%s ...", root, url)
    dataset = torchaudio.datasets.LIBRISPEECH(root, url=url, download=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            # dataset[i] returns (waveform, sample_rate, transcript, ...)
            _, _, transcript, *_ = dataset[i]
            # LibriSpeech transcripts are ALL UPPERCASE - lowercase for
            # consistency with downstream mixed-case text (MOSEI, IEMOCAP).
            f.write(transcript.strip().lower() + "\n")

    logger.info("Wrote %d transcripts to %s", len(dataset), output_path)
    return output_path


def train_tokenizer_from_librispeech(
    librispeech_root: str,
    librispeech_url: str = "train-clean-100",
    tokenizer_model: str = "data/tokenizer.model",
    transcripts_path: str = "data/librispeech/librispeech_transcripts.txt",
    vocab_size: int = 21800,
):
    """End-to-end: extract LibriSpeech transcripts -> train SentencePiece.

    No-ops if the tokenizer model already exists.

    Args:
        librispeech_root: Root directory for torchaudio LibriSpeech data.
        librispeech_url: LibriSpeech subset name.
        tokenizer_model: Output path for the ``.model`` file.
        transcripts_path: Intermediate text file of transcripts.
        vocab_size: SentencePiece vocabulary size.

    Returns:
        A loaded ``sentencepiece.SentencePieceProcessor``.
    """
    if os.path.isfile(tokenizer_model):
        logger.info("Tokenizer already exists at %s", tokenizer_model)
        return get_tokenizer(tokenizer_model)

    txt_path = extract_librispeech_transcripts(
        root=librispeech_root, url=librispeech_url, output_path=transcripts_path,
    )
    return get_tokenizer(tokenizer_model, train_text_file=txt_path, vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# LibriSpeech dataset  (pre-training)
# ---------------------------------------------------------------------------

class LibriSpeechDataset(Dataset):
    """Wrapper around ``torchaudio.datasets.LIBRISPEECH`` for contrastive
    pre-training (audio-transcript pairs).

    Each sample returns:
        waveform:  (num_samples,) float tensor resampled to ``target_sr``.
        token_ids: (S,) long tensor of SentencePiece token IDs.
    """

    def __init__(
        self,
        root: str,
        url: str = "train-clean-100",
        tokenizer_model: str = "data/tokenizer.model",
        target_sr: int = 16000,
        max_audio_sec: float = 10.0,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root, url=url, download=True)

        # Auto-train tokenizer from LibriSpeech transcripts if missing
        self.tokenizer = train_tokenizer_from_librispeech(
            librispeech_root=root,
            librispeech_url=url,
            tokenizer_model=tokenizer_model,
        )
        self.target_sr = target_sr
        self.max_samples = int(max_audio_sec * target_sr)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        waveform, sr, transcript, *_ = self.dataset[idx]

        # Resample if needed
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # Mono and flatten
        waveform = waveform.mean(dim=0)  # (num_samples,)

        # Truncate to max length
        if waveform.size(0) > self.max_samples:
            waveform = waveform[: self.max_samples]

        # Tokenize transcript (lowercase for consistency with tokenizer)
        token_ids = torch.tensor(
            self.tokenizer.encode(transcript.lower()), dtype=torch.long
        )

        return {"waveform": waveform, "token_ids": token_ids}


# ---------------------------------------------------------------------------
# MOSEI metadata builder  (raw -> metadata.csv)
# ---------------------------------------------------------------------------

_MOSEI_EMOTION_KEYS = ["happiness", "sadness", "anger", "fear", "surprise", "disgust"]


def build_mosei_metadata(raw_root: str, output_csv: str) -> str:
    """Build ``metadata.csv`` from raw CMU-MOSEI download.

    Expected raw layout::

        raw_root/
            Raw/
                Audio/Full/WAV_16000/*.wav
                Transcript/Segmented/Combined/*.txt   # VIDEO___CLIP___START___END___TEXT
                Labels/*.csv                           # MTurk results with Answer.* columns

    The function aggregates per-annotator labels by majority / mean,
    then joins with transcript segments.  Each row in the output CSV
    represents one *segment* and contains:

        audio_path  - relative to ``raw_root`` (full video WAV)
        text        - segment transcript
        start       - segment start in seconds
        end         - segment end in seconds
        emotion     - int class (argmax of averaged emotion scores across annotators)
        sentiment   - int class mapped to 0-6 (raw values -3...3 -> 0...6)

    Returns:
        ``output_csv`` path.
    """
    raw_root = Path(raw_root)
    output_csv = Path(output_csv)

    if output_csv.exists():
        logger.info("MOSEI metadata already exists at %s - skipping build.", output_csv)
        return str(output_csv)

    # ------------------------------------------------------------------
    # 1. Parse transcripts  -> {(video_id, clip): {text, start, end}}
    # ------------------------------------------------------------------
    transcript_dir = raw_root / "Raw" / "Transcript" / "Segmented" / "Combined"
    segments: Dict[Tuple[str, int], Dict] = {}
    for txt_file in sorted(transcript_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("___")
                if len(parts) < 5:
                    continue
                vid = parts[0]
                clip = int(parts[1])
                start = float(parts[2])
                end = float(parts[3])
                text = parts[4]
                segments[(vid, clip)] = {
                    "text": text,
                    "start": start,
                    "end": end,
                }
    logger.info("Parsed %d transcript segments.", len(segments))

    # ------------------------------------------------------------------
    # 2. Parse labels  -> {(video_id, clip): list of annotation dicts}
    # ------------------------------------------------------------------
    label_dir = raw_root / "Raw" / "Labels"
    annotations: Dict[Tuple[str, int], List[Dict[str, float]]] = defaultdict(list)

    for csv_path in sorted(label_dir.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "Input.VIDEO_ID" not in (reader.fieldnames or []):
                continue
            for row in reader:
                vid = row.get("Input.VIDEO_ID", "").strip().strip('"')
                clip_str = row.get("Input.CLIP", "").strip().strip('"')
                if not vid or not clip_str:
                    continue
                try:
                    clip = int(clip_str)
                except ValueError:
                    continue
                ann: Dict[str, float] = {}
                for emo in _MOSEI_EMOTION_KEYS:
                    val = row.get(f"Answer.{emo}", "0").strip().strip('"')
                    try:
                        ann[emo] = float(val)
                    except ValueError:
                        ann[emo] = 0.0
                sent_val = row.get("Answer.sentiment", "0").strip().strip('"')
                try:
                    ann["sentiment"] = float(sent_val)
                except ValueError:
                    ann["sentiment"] = 0.0
                annotations[(vid, clip)].append(ann)
    logger.info("Parsed annotations for %d (video, clip) pairs.", len(annotations))

    # ------------------------------------------------------------------
    # 3. Aggregate and join -> rows
    # ------------------------------------------------------------------
    audio_dir = raw_root / "Raw" / "Audio" / "Full" / "WAV_16000"
    rows: List[Dict] = []
    for (vid, clip), seg in sorted(segments.items()):
        wav_path = audio_dir / f"{vid}.wav"
        if not wav_path.exists():
            continue

        audio_rel = str(wav_path.relative_to(raw_root))

        anns = annotations.get((vid, clip))
        if anns:
            # Average emotion scores across annotators
            avg_emo = {emo: 0.0 for emo in _MOSEI_EMOTION_KEYS}
            avg_sent = 0.0
            for a in anns:
                for emo in _MOSEI_EMOTION_KEYS:
                    avg_emo[emo] += a[emo]
                avg_sent += a["sentiment"]
            n = len(anns)
            for emo in _MOSEI_EMOTION_KEYS:
                avg_emo[emo] /= n
            avg_sent /= n

            # Emotion: argmax of averaged scores
            emotion_cls = max(range(len(_MOSEI_EMOTION_KEYS)),
                              key=lambda i: avg_emo[_MOSEI_EMOTION_KEYS[i]])
            # Sentiment: round to nearest int, shift from [-3,3] -> [0,6]
            sentiment_cls = max(0, min(6, round(avg_sent) + 3))
        else:
            emotion_cls = 0
            sentiment_cls = 3  # neutral

        # Deterministic split by hashing video ID (train 80%, val 10%, test 10%)
        h = int(hashlib.md5(vid.encode()).hexdigest(), 16) % 100
        if h < 80:
            split = "train"
        elif h < 90:
            split = "val"
        else:
            split = "test"

        rows.append({
            "audio_path": audio_rel,
            "text": seg["text"],
            "start": f"{seg['start']:.3f}",
            "end": f"{seg['end']:.3f}",
            "emotion": str(emotion_cls),
            "sentiment": str(sentiment_cls),
            "split": split,
        })

    # ------------------------------------------------------------------
    # 4. Write CSV
    # ------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "text", "start", "end", "emotion", "sentiment", "split"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote MOSEI metadata with %d segments to %s", len(rows), output_csv)
    return str(output_csv)


# ---------------------------------------------------------------------------
# MOSEI dataset  (fine-tuning / evaluation)
# ---------------------------------------------------------------------------

class MOSEIDataset(Dataset):
    """MOSEI dataset loaded from a processed directory.

    If ``metadata.csv`` does not exist under *root*, it is automatically
    built from the raw CMU-MOSEI download (expects ``Raw/`` sub-tree).

    ``metadata.csv`` columns:
        audio_path: path to the full-video WAV (relative to *root*)
        text:       segment transcript
        start:      segment start in seconds
        end:        segment end in seconds
        emotion:    integer class label (0-5)
        sentiment:  integer class label (0-6, mapped from raw -3...3)
    """

    EMOTION_CLASSES = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"]

    def __init__(
        self,
        root: str,
        tokenizer_model: str = "data/tokenizer.model",
        target_sr: int = 16000,
        max_audio_sec: float = 10.0,
        split: Optional[str] = None,
    ):
        self.root = Path(root)
        self.tokenizer = get_tokenizer(tokenizer_model)
        self.target_sr = target_sr
        self.max_samples = int(max_audio_sec * target_sr)
        self.samples: List[Dict] = []

        meta_path = self.root / "metadata.csv"

        # Auto-build metadata from raw MOSEI if needed
        if not meta_path.exists():
            logger.info("metadata.csv not found - building from raw MOSEI data ...")
            build_mosei_metadata(raw_root=str(self.root), output_csv=str(meta_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split", "") != split:
                    continue
                self.samples.append(row)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples[idx]
        audio_path = self.root / row["audio_path"]

        # Load segment slice when start/end timestamps are available
        start_sec = float(row["start"]) if "start" in row else None
        end_sec = float(row["end"]) if "end" in row else None

        if start_sec is not None and end_sec is not None:
            info = torchaudio.info(str(audio_path))
            sr = info.sample_rate
            frame_offset = max(0, int(start_sec * sr))
            num_frames = int((end_sec - start_sec) * sr)
            waveform, sr = torchaudio.load(
                str(audio_path), frame_offset=frame_offset, num_frames=num_frames,
            )
        else:
            waveform, sr = torchaudio.load(str(audio_path))

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.mean(dim=0)
        if waveform.size(0) > self.max_samples:
            waveform = waveform[: self.max_samples]

        token_ids = torch.tensor(
            self.tokenizer.encode(row["text"].lower()), dtype=torch.long
        )

        out: Dict[str, torch.Tensor] = {
            "waveform": waveform,
            "token_ids": token_ids,
        }
        if "emotion" in row:
            out["emotion_label"] = torch.tensor(int(row["emotion"]), dtype=torch.long)
        if "sentiment" in row:
            out["sentiment_label"] = torch.tensor(int(row["sentiment"]), dtype=torch.long)
        return out


# ---------------------------------------------------------------------------
# IEMOCAP metadata builder
# ---------------------------------------------------------------------------

# Standard 4-class mapping used in IEMOCAP emotion recognition literature.
_IEMOCAP_EMOTION_MAP = {
    "hap": 0,  # Happy
    "exc": 0,  # Excited -> merged with Happy
    "sad": 1,  # Sad
    "ang": 2,  # Angry
    "neu": 3,  # Neutral
}


def _parse_iemocap_transcriptions(root: Path) -> Dict[str, str]:
    """Parse all IEMOCAP transcription files into {utterance_id: text}."""
    utt2text: Dict[str, str] = {}
    for sess in range(1, 6):
        trans_dir = root / f"Session{sess}" / "dialog" / "transcriptions"
        if not trans_dir.exists():
            continue
        for txt_file in sorted(trans_dir.glob("*.txt")):
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: Ses01F_script02_1_F000 [start-end]: text
                    # Lines without brackets are stage directions, skip them
                    bracket_start = line.find("[")
                    bracket_end = line.find("]")
                    if bracket_start == -1 or bracket_end == -1:
                        continue
                    colon_pos = line.find(":", bracket_end)
                    if colon_pos == -1:
                        continue
                    utt_id = line[:bracket_start].strip()
                    text = line[colon_pos + 1:].strip()
                    if utt_id and text:
                        utt2text[utt_id] = text
    return utt2text


def build_iemocap_metadata(root: Path) -> Path:
    """Build a processsed metadata CSV from raw IEMOCAP metadata + transcripts.

    Reads the original ``metadata.csv`` (columns: session, method, gender,
    emotion, n_annotators, agreement, path), parses transcription files for
    text, maps string emotion labels to 4-class integers, and assigns
    session-based splits (Sessions 1-4 -> train, Session 5 -> test).

    Returns the path to the generated ``processed_metadata.csv``.
    """
    raw_meta = root / "metadata.csv"
    out_path = root / "processed_metadata.csv"

    logger.info("Parsing IEMOCAP transcription files ...")
    utt2text = _parse_iemocap_transcriptions(root)
    logger.info("Found transcriptions for %d utterances.", len(utt2text))

    rows_written = 0
    rows_skipped = 0
    with open(raw_meta, "r", encoding="utf-8") as fin, \
         open(out_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout, fieldnames=["audio_path", "text", "emotion", "split"]
        )
        writer.writeheader()

        for row in reader:
            emo_str = row["emotion"]
            if emo_str not in _IEMOCAP_EMOTION_MAP:
                rows_skipped += 1
                continue

            # Derive utterance ID from path filename (without .wav)
            wav_rel = row["path"]  # e.g. Session1/sentences/wav/.../Ses01F_script02_1_F000.wav
            utt_id = Path(wav_rel).stem  # Ses01F_script02_1_F000

            text = utt2text.get(utt_id, "")
            if not text:
                rows_skipped += 1
                continue

            session = int(row["session"])
            split = "test" if session == 5 else "train"

            writer.writerow({
                "audio_path": wav_rel,
                "text": text,
                "emotion": _IEMOCAP_EMOTION_MAP[emo_str],
                "split": split,
            })
            rows_written += 1

    logger.info(
        "Wrote %d rows to %s (skipped %d with unmapped emotions or missing text).",
        rows_written, out_path, rows_skipped,
    )
    return out_path


# ---------------------------------------------------------------------------
# IEMOCAP dataset  (evaluation)
# ---------------------------------------------------------------------------

class IEMOCAPDataset(Dataset):
    """IEMOCAP dataset loaded from a processed directory.

    On first use, automatically builds ``processed_metadata.csv`` from the
    raw IEMOCAP ``metadata.csv`` and transcription files.

    ``processed_metadata.csv`` columns:
        audio_path: relative path inside ``iemocap_root``
        text:       transcription string
        emotion:    integer class label (0=Happy, 1=Sad, 2=Angry, 3=Neutral)
        split:      ``train`` or ``test``
    """

    EMOTION_CLASSES = ["Happy", "Sad", "Angry", "Neutral"]

    def __init__(
        self,
        root: str,
        tokenizer_model: str = "data/tokenizer.model",
        target_sr: int = 16000,
        max_audio_sec: float = 10.0,
        split: Optional[str] = None,
    ):
        self.root = Path(root)
        self.tokenizer = get_tokenizer(tokenizer_model)
        self.target_sr = target_sr
        self.max_samples = int(max_audio_sec * target_sr)
        self.samples: List[Dict] = []

        meta_path = self.root / "processed_metadata.csv"
        if not meta_path.exists():
            logger.info("processed_metadata.csv not found - building from raw IEMOCAP data ...")
            build_iemocap_metadata(self.root)

        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("split", "") != split:
                    continue
                self.samples.append(row)

        logger.info("IEMOCAPDataset: %d samples (split=%s)", len(self.samples), split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples[idx]
        audio_path = self.root / row["audio_path"]
        waveform, sr = torchaudio.load(str(audio_path))

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.mean(dim=0)
        if waveform.size(0) > self.max_samples:
            waveform = waveform[: self.max_samples]

        token_ids = torch.tensor(
            self.tokenizer.encode(row["text"].lower()), dtype=torch.long
        )

        out: Dict[str, torch.Tensor] = {
            "waveform": waveform,
            "token_ids": token_ids,
            "emotion_label": torch.tensor(int(row["emotion"]), dtype=torch.long),
        }
        return out


# ---------------------------------------------------------------------------
# Collator (dynamic padding)
# ---------------------------------------------------------------------------

class AudioTextCollator:
    """Pads variable-length waveforms and token sequences to batch max.

    Returns a dictionary of padded tensors plus actual lengths.
    """

    def __init__(self, max_text_len: int = 128):
        self.max_text_len = max_text_len

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        waveforms = [b["waveform"] for b in batch]
        token_ids_list = [b["token_ids"] for b in batch]

        # Pad waveforms to longest in batch
        max_wav_len = max(w.size(0) for w in waveforms)
        padded_wavs = torch.zeros(len(batch), max_wav_len)
        for i, w in enumerate(waveforms):
            padded_wavs[i, : w.size(0)] = w

        # Pad token_ids to min(max_in_batch, max_text_len)
        text_lengths = torch.tensor([min(t.size(0), self.max_text_len) for t in token_ids_list])
        max_tok_len = int(text_lengths.max().item())
        padded_tokens = torch.zeros(len(batch), max_tok_len, dtype=torch.long)
        for i, t in enumerate(token_ids_list):
            length = text_lengths[i]
            padded_tokens[i, :length] = t[:length]

        result: Dict[str, torch.Tensor] = {
            "waveform": padded_wavs,
            "token_ids": padded_tokens,
            "text_lengths": text_lengths,
        }

        # Optional labels
        if "emotion_label" in batch[0]:
            result["emotion_label"] = torch.stack([b["emotion_label"] for b in batch])
        if "sentiment_label" in batch[0]:
            result["sentiment_label"] = torch.stack([b["sentiment_label"] for b in batch])

        return result
