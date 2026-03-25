from __future__ import annotations

import csv
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


FILLER_TOKENS_KEEP = {"um", "uh", "ah"}


# These are the 15 Empath/LIWC-proxy categories we want, in fixed order.
EMPATH_CATEGORIES = [
    "negative_emotion",
    "positive_emotion",
    "certainty",
    "hedging",
    "family",
    "friends",
    "money",
    "movement",
    "time",
    "body",
    "death",
    "anger",
    "sadness",
    "confusion",
    "swearing_terms",
]


OUTPUT_EMP_FLAG_NAMES = [
    "neg_emotion",
    "pos_emotion",
    "certainty",
    "hedging",
    "family",
    "friends",
    "money",
    "movement",
    "time",
    "body",
    "death",
    "anger",
    "sadness",
    "confusion",
    "swearing",
]


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def _read_and_normalize_transcripts(
    labels_csv_path: Path,
) -> Tuple[List[str], Dict[str, str], Dict[str, int], Dict[str, str]]:
    """
    Returns:
      clip_ids: clips in file order from labels.csv
      subject_id_by_clip, is_deceptive_by_clip
      text_by_clip: normalized, pre-tokenized transcript text per clip
    """
    clip_ids: List[str] = []
    subject_id_by_clip: Dict[str, str] = {}
    is_deceptive_by_clip: Dict[str, int] = {}
    text_by_clip: Dict[str, str] = {}

    with labels_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("labels.csv is empty")

        required = {"clip_id", "subject_id", "is_deceptive"}
        if not required.issubset(set(reader.fieldnames)):
            raise KeyError("labels.csv must contain `clip_id`, `subject_id`, `is_deceptive`")

        for row in reader:
            clip_id = row["clip_id"]
            clip_ids.append(clip_id)
            subject_id_by_clip[clip_id] = row["subject_id"]
            is_deceptive_by_clip[clip_id] = int(row["is_deceptive"])

    base = Path("data/Real-life_Deception_Detection_2016/Transcription")
    for clip_id in clip_ids:
        stem = Path(clip_id).stem
        if is_deceptive_by_clip[clip_id] == 1:
            txt_path = base / "Deceptive" / f"{stem}.txt"
        else:
            txt_path = base / "Truthful" / f"{stem}.txt"

        raw = txt_path.read_text(encoding="utf-8", errors="ignore").lower()

        # Strip punctuation but keep apostrophes inside words.
        # Replace other punctuation with whitespace so tokens stay separable.
        cleaned = []
        for ch in raw:
            if ch in "'":
                cleaned.append(ch)
            elif ch.isalnum() or ch.isspace():
                cleaned.append(ch)
            else:
                # punctuation -> whitespace
                cleaned.append(" ")
        cleaned_text = "".join(cleaned)

        # Ensure fillers are preserved even if tokenization later drops punctuation.
        # (They should already survive due to the cleaning logic.)
        text_by_clip[clip_id] = cleaned_text

    return clip_ids, subject_id_by_clip, is_deceptive_by_clip, text_by_clip


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _build_vocabulary(texts: Dict[str, str], min_total_freq: int = 10) -> List[str]:
    token_counts = Counter()
    for text in texts.values():
        token_counts.update(_tokenize(text))

    vocab = [tok for tok, cnt in token_counts.items() if cnt >= min_total_freq]
    vocab.sort()  # reproducible alphabetical order
    return vocab


def _compute_unigram_frequencies(text: str, vocab: List[str]) -> Tuple[List[float], int]:
    tokens = _tokenize(text)
    total_words = len(tokens)
    if total_words == 0:
        return [0.0] * len(vocab), 0

    token_counts = Counter(tokens)
    freqs = [token_counts.get(tok, 0) / total_words for tok in vocab]
    return freqs, total_words


def _compute_empath_features(text_by_clip: Dict[str, str]) -> Dict[str, List[float]]:
    from empath import Empath

    lexicon = Empath()
    features_by_clip: Dict[str, List[float]] = {}

    for clip_id, text in text_by_clip.items():
        analyzed = lexicon.analyze(text, categories=EMPATH_CATEGORIES, normalize=True)
        if analyzed is None:
            analyzed = {}

        values: List[float] = []
        for cat, out_name in zip(EMPATH_CATEGORIES, OUTPUT_EMP_FLAG_NAMES, strict=True):
            v = analyzed.get(cat)
            values.append(float(v) if v is not None else 0.0)
        features_by_clip[clip_id] = values

    return features_by_clip


def extract_linguistic_features(
    labels_csv_path: Path = Path("data/labels.csv"),
    output_csv_path: Path = Path("features/linguistic.csv"),
    min_total_freq: int = 10,
) -> None:
    clip_ids, subject_id_by_clip, is_deceptive_by_clip, text_by_clip = _read_and_normalize_transcripts(
        labels_csv_path=labels_csv_path
    )

    vocab = _build_vocabulary(texts=text_by_clip, min_total_freq=min_total_freq)

    print(f"Vocabulary size: {len(vocab)} tokens")

    # Top-10 retained tokens by corpus total frequency (ties alphabetically).
    token_counts = Counter()
    for text in text_by_clip.values():
        token_counts.update(_tokenize(text))

    retained_counts = [(tok, token_counts[tok]) for tok in vocab]
    retained_counts.sort(key=lambda x: (-x[1], x[0]))
    top10 = [tok for tok, _ in retained_counts[:10]]
    print("Top-10 tokens:", ", ".join(top10))

    empath_features_by_clip = _compute_empath_features(text_by_clip=text_by_clip)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["clip_id", "subject_id", "is_deceptive", *vocab, *OUTPUT_EMP_FLAG_NAMES]

    rows: List[List[object]] = []
    for clip_id in clip_ids:
        freqs, total_words = _compute_unigram_frequencies(text_by_clip[clip_id], vocab=vocab)
        if total_words == 0:
            print(f"WARNING: {clip_id} has zero words after preprocessing")

        empath_vals = empath_features_by_clip[clip_id]

        row: List[object] = [
            clip_id,
            subject_id_by_clip[clip_id],
            is_deceptive_by_clip[clip_id],
            *freqs,
            *empath_vals,
        ]
        rows.append(row)

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(
        f"Done — {output_csv_path.as_posix()} written ({len(rows)} rows, {len(header) - 3} features)"
    )


if __name__ == "__main__":
    extract_linguistic_features()
