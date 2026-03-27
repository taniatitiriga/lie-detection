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


# ---------------------------------------------------------------------------
# Bigram TF features
# ---------------------------------------------------------------------------

NEGATION_TOKENS: frozenset[str] = frozenset(
    {"not", "no", "never", "nobody", "nothing", "neither", "nor", "n't"}
)
SELF_TOKENS: frozenset[str] = frozenset({"i", "me", "mine", "my"})


def _build_bigram_tf(
    clip_ids: List[str],
    text_by_clip: Dict[str, str],
    min_df: int = 5,
) -> Tuple[List[str], Dict[str, List[float]]]:
    """
    Fit a bigram CountVectorizer (corpus frequency >= min_df) on all clip texts,
    then compute normalised TF (count / total_bigrams_in_clip) per clip.

    Returns:
      bigram_vocab  : list of bigram strings, alphabetically sorted (length V)
      tf_by_clip    : dict mapping clip_id -> list of V floats
    """
    from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

    # Fit on all transcripts
    corpus = [text_by_clip[cid] for cid in clip_ids]
    vec = CountVectorizer(ngram_range=(2, 2), min_df=min_df)
    X = vec.fit_transform(corpus)  # shape (n_clips, V)

    # Sorted vocab in the same order as the matrix columns
    feature_names: List[str] = vec.get_feature_names_out().tolist()

    tf_by_clip: Dict[str, List[float]] = {}
    for i, cid in enumerate(clip_ids):
        row = X[i].toarray().flatten().astype(float)
        total = row.sum()
        if total > 0:
            row = row / total  # normalise to relative frequency
        tf_by_clip[cid] = row.tolist()

    return feature_names, tf_by_clip


# ---------------------------------------------------------------------------
# Lexical count features
# ---------------------------------------------------------------------------


def _compute_lexical_features(
    clip_ids: List[str],
    text_by_clip: Dict[str, str],
) -> Dict[str, Tuple[int, int]]:
    """
    Return a dict mapping clip_id -> (negation_count, self_count).

    * negation_count: number of tokens in {not, no, never, nobody, nothing,
                      neither, nor, n't} — case-insensitive.
    * self_count    : number of tokens in {I, me, mine, my} — case-insensitive.

    Note: the cleaned text is already lower-cased by _read_and_normalize_transcripts,
    so comparisons are straightforward.
    """
    result: Dict[str, Tuple[int, int]] = {}
    for cid in clip_ids:
        tokens = _tokenize(text_by_clip[cid])
        neg = sum(1 for t in tokens if t in NEGATION_TOKENS)
        self_ = sum(1 for t in tokens if t in SELF_TOKENS)
        result[cid] = (neg, self_)
    return result


def extract_linguistic_features(
    labels_csv_path: Path = Path("data/labels.csv"),
    output_csv_path: Path = Path("features/linguistic.csv"),
    min_total_freq: int = 10,
    bigram_min_df: int = 5,
) -> None:
    clip_ids, subject_id_by_clip, is_deceptive_by_clip, text_by_clip = _read_and_normalize_transcripts(
        labels_csv_path=labels_csv_path
    )

    # --- Unigrams ---
    vocab = _build_vocabulary(texts=text_by_clip, min_total_freq=min_total_freq)

    print(f"Unigram vocabulary size: {len(vocab)} tokens")

    # Top-10 retained tokens by corpus total frequency (ties alphabetically).
    token_counts: Counter = Counter()
    for text in text_by_clip.values():
        token_counts.update(_tokenize(text))

    retained_counts = [(tok, token_counts[tok]) for tok in vocab]
    retained_counts.sort(key=lambda x: (-x[1], x[0]))
    top10 = [tok for tok, _ in retained_counts[:10]]
    print("Top-10 unigrams:", ", ".join(top10))

    # --- Bigrams ---
    bigram_vocab, bigram_tf_by_clip = _build_bigram_tf(
        clip_ids, text_by_clip, min_df=bigram_min_df
    )
    print(f"Bigram vocabulary size (min_df={bigram_min_df}): {len(bigram_vocab)} bigrams")

    # --- Empath ---
    empath_features_by_clip = _compute_empath_features(text_by_clip=text_by_clip)

    # --- Lexical counts ---
    lexical_by_clip = _compute_lexical_features(clip_ids, text_by_clip)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # bigram columns get a `bigram_` prefix so they can't collide with unigrams
    bigram_col_names = [f"bigram_{b.replace(' ', '_')}" for b in bigram_vocab]

    header = [
        "clip_id", "subject_id", "is_deceptive",
        *vocab,
        *OUTPUT_EMP_FLAG_NAMES,
        *bigram_col_names,
        "negation_count",
        "self_count",
    ]

    rows: List[List[object]] = []
    for clip_id in clip_ids:
        freqs, total_words = _compute_unigram_frequencies(text_by_clip[clip_id], vocab=vocab)
        if total_words == 0:
            print(f"WARNING: {clip_id} has zero words after preprocessing")

        empath_vals = empath_features_by_clip[clip_id]
        bigram_tf = bigram_tf_by_clip[clip_id]
        neg_count, self_count = lexical_by_clip[clip_id]

        row: List[object] = [
            clip_id,
            subject_id_by_clip[clip_id],
            is_deceptive_by_clip[clip_id],
            *freqs,
            *empath_vals,
            *bigram_tf,
            neg_count,
            self_count,
        ]
        rows.append(row)

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    n_feat = len(header) - 3  # subtract clip_id, subject_id, is_deceptive
    print(
        f"Done — {output_csv_path.as_posix()} written "
        f"({len(rows)} rows, {n_feat} features: "
        f"{len(vocab)} unigrams + {len(OUTPUT_EMP_FLAG_NAMES)} Empath + "
        f"{len(bigram_vocab)} bigrams + 2 lexical)"
    )


if __name__ == "__main__":
    extract_linguistic_features()
