"""
extract_facial.py
─────────────────────────────────────────────────────────────────────────────
Theory-grounded facial deception features from OpenFace frame-level CSVs.

Six feature groups, each with explicit deception-literature grounding:

  A. MUMIN-analog         — behavioral coding scheme proxies (Allwood 2007)
  B. Micro-expressions    — affect leakage (Ekman & Friesen 1969)
  C. Emotion prototypes   — FACS basic-emotion AU combinations
  D. AU intensity dynamics — continuous intensity signal exploitation
  E. Cognitive load        — load hypothesis cues (Vrij 2008)
  F. Temporal dynamics     — expression timing (DePaulo et al. 2003)

Output: features/facial.csv  (~75 features per clip)
"""

from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── constants ───────────────────────────────────────────────────────────────

DEFAULT_FPS: float = 25.0
DEFAULT_CONFIDENCE: float = 0.9

GAZE_CENTRE_THRESH: float = 0.15   # ~8.6°
GAZE_AVERT_THRESH: float = 0.25    # ~14°

# Micro-expression duration bounds (noise gate + Ekman upper).
MIN_MICRO_DUR: float = 1.5 / DEFAULT_FPS   # ≈0.060 s — rejects 1-frame spikes
MICRO_EXPR_MAX_SEC: float = 0.50

# Head-movement thresholds (radians).
HEAD_NOD_THRESH: float = 0.06
HEAD_SHAKE_THRESH: float = 0.06
HEAD_SIDETURN_THRESH: float = 0.20
HEAD_TILT_THRESH: float = 0.10

# Temporal dynamics thresholds.
LONG_EXPR_THRESH: float = 4.0   # seconds — posed expressions held too long
SHORT_EXPR_THRESH: float = 0.5  # seconds — suppressed leakage

# AU intensity freeze threshold.
AU_FREEZE_THRESH: float = 0.1


# ─── AU column lists ─────────────────────────────────────────────────────────

AU_C: List[str] = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]
AU_R: List[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
GAZE_COLS: List[str] = ["gaze_angle_x", "gaze_angle_y"]
POSE_COLS: List[str] = ["pose_Rx", "pose_Ry", "pose_Rz"]

ALL_SIGNAL_COLS: List[str] = AU_C + AU_R + GAZE_COLS + POSE_COLS

EXPRESSION_AU_C: List[str] = [c for c in AU_C if c != "AU45_c"]

# ─── Emotion prototype AU mappings (Ekman FACS) ─────────────────────────────
# Each entry: (label, list of required AU_c columns).
EMOTION_PROTOTYPES: List[Tuple[str, List[str]]] = [
    ("happiness",  ["AU06_c", "AU12_c"]),
    ("sadness",    ["AU01_c", "AU04_c", "AU15_c"]),
    ("surprise",   ["AU01_c", "AU02_c", "AU05_c", "AU26_c"]),
    ("fear",       ["AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU20_c"]),
    ("anger",      ["AU04_c", "AU05_c", "AU07_c", "AU23_c"]),
    ("disgust",    ["AU09_c", "AU15_c", "AU17_c"]),
    ("contempt",   ["AU12_c", "AU14_c"]),
]


# ─── helpers ─────────────────────────────────────────────────────────────────

def _safe(val: Optional[float], default: float = 0.0) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return (s[mid] + s[mid - 1]) / 2.0 if n % 2 == 0 else s[mid]


def _cv(xs: List[float]) -> float:
    """Coefficient of variation (std / mean). Returns 0 when undefined."""
    m = _mean(xs)
    if m == 0.0 or len(xs) < 2:
        return 0.0
    return _std(xs) / abs(m)


def _shannon_entropy(proportions: List[float]) -> float:
    """Shannon entropy over a distribution (need not sum to 1)."""
    total = sum(proportions)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in proportions:
        if p > 0:
            pk = p / total
            h -= pk * math.log2(pk)
    return h


def _count_events(
    binary_series: List[float],
    timestamps: List[float],
    min_dur: float = 0.0,
    max_dur: float = float("inf"),
) -> Tuple[int, List[float]]:
    """
    Count contiguous runs of 1 in *binary_series* whose inclusive duration
    falls in [min_dur, max_dur].  Returns (count, list_of_durations).
    """
    events: List[float] = []
    in_event = False
    start_idx = 0

    dt = (timestamps[-1] - timestamps[0]) / max(len(timestamps) - 1, 1) if len(timestamps) > 1 else 1.0 / DEFAULT_FPS

    for i, v in enumerate(binary_series):
        if v >= 0.5 and not in_event:
            in_event = True
            start_idx = i
        elif v < 0.5 and in_event:
            in_event = False
            dur = (timestamps[i - 1] - timestamps[start_idx]) + dt
            if min_dur <= dur <= max_dur:
                events.append(dur)
    if in_event:
        dur = (timestamps[-1] - timestamps[start_idx]) + dt
        if min_dur <= dur <= max_dur:
            events.append(dur)

    return len(events), events


def _count_events_indexed(
    binary_series: List[float],
    timestamps: List[float],
    min_dur: float = 0.0,
    max_dur: float = float("inf"),
) -> List[Tuple[int, int, float]]:
    """Like _count_events but returns (start_idx, end_idx, duration) tuples."""
    events: List[Tuple[int, int, float]] = []
    in_event = False
    start_idx = 0

    dt = (timestamps[-1] - timestamps[0]) / max(len(timestamps) - 1, 1) if len(timestamps) > 1 else 1.0 / DEFAULT_FPS

    for i, v in enumerate(binary_series):
        if v >= 0.5 and not in_event:
            in_event = True
            start_idx = i
        elif v < 0.5 and in_event:
            in_event = False
            end_idx = i - 1
            dur = (timestamps[end_idx] - timestamps[start_idx]) + dt
            if min_dur <= dur <= max_dur:
                events.append((start_idx, end_idx, dur))
    if in_event:
        end_idx = len(binary_series) - 1
        dur = (timestamps[end_idx] - timestamps[start_idx]) + dt
        if min_dur <= dur <= max_dur:
            events.append((start_idx, end_idx, dur))

    return events


def _zero_crossings(series: List[float], threshold: float = 0.0) -> int:
    """Count sign-changes across ±threshold (proxy for oscillations)."""
    if not series:
        return 0
    crossings = 0
    above = series[0] > threshold
    for v in series[1:]:
        now_above = v > threshold
        if now_above != above:
            crossings += 1
            above = now_above
    return crossings


# ─── frame reader ─────────────────────────────────────────────────────────────

class FrameData:
    """Holds all per-frame values parsed from one OpenFace CSV."""

    def __init__(self) -> None:
        self.timestamps: List[float] = []
        self.cols: Dict[str, List[float]] = {c: [] for c in ALL_SIGNAL_COLS}
        self.n_frames: int = 0

    def clip_duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def fps(self) -> float:
        d = self.clip_duration()
        return self.n_frames / d if d > 0 else DEFAULT_FPS

    def frame_dt(self) -> float:
        """Estimated inter-frame interval in seconds."""
        if len(self.timestamps) > 1:
            return (self.timestamps[-1] - self.timestamps[0]) / max(self.n_frames - 1, 1)
        return 1.0 / DEFAULT_FPS


def _load_frame_data(
    path: Path,
    confidence_threshold: float,
) -> FrameData:
    fd = FrameData()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        available = set(reader.fieldnames)
        if "confidence" not in available:
            raise KeyError(f"Missing confidence column in {path}")
        present_au_r = [c for c in AU_R if c in available]

        for row in reader:
            if float(row["confidence"]) < confidence_threshold:
                continue
            ts = float(row.get("timestamp", 0.0))
            fd.timestamps.append(ts)
            for c in AU_C + GAZE_COLS + POSE_COLS:
                fd.cols[c].append(float(row.get(c, 0.0)))
            for c in present_au_r:
                fd.cols[c].append(float(row[c]))
            for c in AU_R:
                if c not in present_au_r:
                    fd.cols[c].append(0.0)
            fd.n_frames += 1
    return fd


# ─── Group A: MUMIN-analog features ──────────────────────────────────────────
#
# AU → MUMIN mapping (Ekman FACS + deception literature):
#
#  General face:
#    Smile genuine (Duchenne) : AU06_c AND AU12_c
#    Smile social             : AU12_c AND NOT AU06_c
#    Laughter                 : AU06_c AND AU12_c AND AU25_c AND AU26_c
#    Scowl / displeasure      : AU04_c AND (AU15_c OR AU17_c)
#
#  Eyebrows:
#    Raise (both)             : AU01_c AND AU02_c
#    Inner raise only         : AU01_c AND NOT AU02_c  ← distress leakage
#    Frown                    : AU04_c
#
#  Eyes:
#    Exaggerated opening      : AU05_c
#    Blinks                   : AU45_c events
#
#  Gaze:
#    Interlocutor             : |gaze_angle_x| < GAZE_CENTRE_THRESH
#                               AND |gaze_angle_y| < GAZE_CENTRE_THRESH
#    Up                       : gaze_angle_y < −GAZE_CENTRE_THRESH
#    Down                     : gaze_angle_y >  GAZE_CENTRE_THRESH
#    Sideways                 : |gaze_angle_x| > GAZE_AVERT_THRESH
#
#  Mouth:
#    Open                     : AU25_c OR AU26_c
#    Corners up               : AU12_c
#    Corners down             : AU15_c
#    Retracted                : AU20_c
#    Protruded                : AU17_c
#
#  Head:
#    Nods   : zero-crossings in pitch (pose_Rx)
#    Shakes : zero-crossings in yaw   (pose_Ry)
#    Side turn : |pose_Ry| > HEAD_SIDETURN_THRESH
#    Tilt      : |pose_Rz| > HEAD_TILT_THRESH

def _compute_mumin_features(fd: FrameData) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    n = fd.n_frames
    if n == 0:
        return feats
    dur = fd.clip_duration()
    c = fd.cols

    # ── General face ─────────────────────────────────────────────────────────
    smile_gen = [1.0 if c["AU06_c"][i] >= 0.5 and c["AU12_c"][i] >= 0.5 else 0.0 for i in range(n)]
    smile_soc = [1.0 if c["AU12_c"][i] >= 0.5 and c["AU06_c"][i] < 0.5  else 0.0 for i in range(n)]
    laugh     = [1.0 if c["AU06_c"][i] >= 0.5 and c["AU12_c"][i] >= 0.5
                      and c["AU25_c"][i] >= 0.5 and c["AU26_c"][i] >= 0.5 else 0.0 for i in range(n)]
    scowl     = [1.0 if c["AU04_c"][i] >= 0.5 and (c["AU15_c"][i] >= 0.5 or c["AU17_c"][i] >= 0.5)
                 else 0.0 for i in range(n)]

    feats["mumin_smile_genuine_prop"] = _mean(smile_gen)
    feats["mumin_smile_social_prop"]  = _mean(smile_soc)
    feats["mumin_laughter_prop"]      = _mean(laugh)
    feats["mumin_scowl_prop"]         = _mean(scowl)

    _, sg_evts = _count_events(smile_gen, fd.timestamps)
    _, ss_evts = _count_events(smile_soc, fd.timestamps)
    _, sc_evts = _count_events(scowl,     fd.timestamps)
    feats["mumin_smile_genuine_rate"] = len(sg_evts) / dur if dur > 0 else 0.0
    feats["mumin_smile_social_rate"]  = len(ss_evts) / dur if dur > 0 else 0.0
    feats["mumin_scowl_rate"]         = len(sc_evts) / dur if dur > 0 else 0.0

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    brow_raise = [1.0 if c["AU01_c"][i] >= 0.5 and c["AU02_c"][i] >= 0.5 else 0.0 for i in range(n)]
    brow_inner = [1.0 if c["AU01_c"][i] >= 0.5 and c["AU02_c"][i] < 0.5  else 0.0 for i in range(n)]
    brow_frown = [float(c["AU04_c"][i] >= 0.5) for i in range(n)]

    feats["mumin_eyebrow_raise_prop"]       = _mean(brow_raise)
    feats["mumin_eyebrow_inner_raise_prop"] = _mean(brow_inner)
    feats["mumin_eyebrow_frown_prop"]       = _mean(brow_frown)

    _, br_evts = _count_events(brow_raise, fd.timestamps)
    _, bi_evts = _count_events(brow_inner, fd.timestamps)
    feats["mumin_eyebrow_raise_rate"]       = len(br_evts) / dur if dur > 0 else 0.0
    feats["mumin_eyebrow_inner_raise_rate"] = len(bi_evts) / dur if dur > 0 else 0.0

    # ── Eyes ──────────────────────────────────────────────────────────────────
    eyes_wide    = [float(c["AU05_c"][i] >= 0.5) for i in range(n)]
    blink_binary = [float(c["AU45_c"][i] >= 0.5) for i in range(n)]

    feats["mumin_eyes_wide_prop"] = _mean(eyes_wide)

    n_blinks, blink_durs = _count_events(
        blink_binary, fd.timestamps,
        min_dur=0.04, max_dur=0.60,
    )
    feats["mumin_blink_rate"]     = n_blinks / dur if dur > 0 else 0.0
    feats["mumin_blink_dur_mean"] = _mean(blink_durs)
    feats["mumin_blink_dur_std"]  = _std(blink_durs)

    # ── Gaze direction ────────────────────────────────────────────────────────
    gx = c["gaze_angle_x"]
    gy = c["gaze_angle_y"]

    gaze_inter = [1.0 if abs(gx[i]) < GAZE_CENTRE_THRESH and abs(gy[i]) < GAZE_CENTRE_THRESH
                  else 0.0 for i in range(n)]
    gaze_up    = [1.0 if gy[i] < -GAZE_CENTRE_THRESH else 0.0 for i in range(n)]
    gaze_down  = [1.0 if gy[i] >  GAZE_CENTRE_THRESH else 0.0 for i in range(n)]
    gaze_side  = [1.0 if abs(gx[i]) > GAZE_AVERT_THRESH else 0.0 for i in range(n)]

    feats["mumin_gaze_interlocutor_prop"] = _mean(gaze_inter)
    feats["mumin_gaze_up_prop"]           = _mean(gaze_up)
    feats["mumin_gaze_down_prop"]         = _mean(gaze_down)
    feats["mumin_gaze_side_prop"]         = _mean(gaze_side)

    # ── Mouth ─────────────────────────────────────────────────────────────────
    mouth_open    = [1.0 if c["AU25_c"][i] >= 0.5 or c["AU26_c"][i] >= 0.5 else 0.0 for i in range(n)]
    lips_up       = [float(c["AU12_c"][i] >= 0.5) for i in range(n)]
    lips_down     = [float(c["AU15_c"][i] >= 0.5) for i in range(n)]
    lips_retract  = [float(c["AU20_c"][i] >= 0.5) for i in range(n)]
    lips_protrude = [float(c["AU17_c"][i] >= 0.5) for i in range(n)]

    feats["mumin_mouth_open_prop"]    = _mean(mouth_open)
    feats["mumin_lips_up_prop"]       = _mean(lips_up)
    feats["mumin_lips_down_prop"]     = _mean(lips_down)
    feats["mumin_lips_retract_prop"]  = _mean(lips_retract)
    feats["mumin_lips_protrude_prop"] = _mean(lips_protrude)

    # ── Head movements ────────────────────────────────────────────────────────
    rx = c["pose_Rx"]
    ry = c["pose_Ry"]
    rz = c["pose_Rz"]

    rx_centred = [v - _mean(rx) for v in rx]
    ry_centred = [v - _mean(ry) for v in ry]

    nod_crossings   = _zero_crossings(rx_centred, threshold=HEAD_NOD_THRESH)
    shake_crossings = _zero_crossings(ry_centred, threshold=HEAD_SHAKE_THRESH)

    feats["mumin_head_nod_rate"]   = nod_crossings   / dur if dur > 0 else 0.0
    feats["mumin_head_shake_rate"] = shake_crossings / dur if dur > 0 else 0.0

    side_turn = [1.0 if abs(ry[i]) > HEAD_SIDETURN_THRESH else 0.0 for i in range(n)]
    head_tilt = [1.0 if abs(rz[i]) > HEAD_TILT_THRESH     else 0.0 for i in range(n)]

    feats["mumin_head_sideturn_prop"] = _mean(side_turn)
    feats["mumin_head_tilt_prop"]     = _mean(head_tilt)

    return feats


# ─── Group B: Micro-expression features ──────────────────────────────────────

def _compute_micro_expression_features(fd: FrameData) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    n = fd.n_frames
    if n == 0:
        _zero_micro_feats(feats)
        return feats

    dur = fd.clip_duration()
    c = fd.cols

    any_expr_active: List[float] = [
        1.0 if any(c[col][i] >= 0.5 for col in EXPRESSION_AU_C) else 0.0
        for i in range(n)
    ]

    micro_events: List[Tuple[int, int]] = []
    in_event = False
    start_idx = 0

    dt = fd.frame_dt()

    for i, v in enumerate(any_expr_active):
        if v >= 0.5 and not in_event:
            in_event = True
            start_idx = i
        elif v < 0.5 and in_event:
            in_event = False
            end_idx = i - 1
            dur_event = (fd.timestamps[end_idx] - fd.timestamps[start_idx]) + dt
            if MIN_MICRO_DUR <= dur_event <= MICRO_EXPR_MAX_SEC:
                micro_events.append((start_idx, end_idx))
    if in_event:
        end_idx = n - 1
        dur_event = (fd.timestamps[end_idx] - fd.timestamps[start_idx]) + dt
        if MIN_MICRO_DUR <= dur_event <= MICRO_EXPR_MAX_SEC:
            micro_events.append((start_idx, end_idx))

    count = len(micro_events)
    feats["micro_expr_count"] = float(count)
    feats["micro_expr_rate"]  = count / dur if dur > 0 else 0.0

    if count == 0:
        _zero_micro_feats(feats, skip_count_rate=True)
        return feats

    event_durations:   List[float] = []
    event_intensities: List[float] = []
    event_au_counts:   List[float] = []

    au_r_present = [col for col in AU_R if any(v != 0.0 for v in c[col])]

    for start_i, end_i in micro_events:
        event_dur = (fd.timestamps[end_i] - fd.timestamps[start_i]) + dt
        event_durations.append(event_dur)

        if au_r_present:
            peak = max(
                c[col][i]
                for i in range(start_i, end_i + 1)
                for col in au_r_present
            )
        else:
            peak = 0.0
        event_intensities.append(peak)

        active_aus = sum(
            1 for col in EXPRESSION_AU_C
            if any(c[col][i] >= 0.5 for i in range(start_i, end_i + 1))
        )
        event_au_counts.append(float(active_aus))

    feats["micro_expr_dur_mean"]        = _mean(event_durations)
    feats["micro_expr_dur_std"]         = _std(event_durations)
    feats["micro_expr_dur_median"]      = _median(event_durations)
    feats["micro_expr_intensity_mean"]  = _mean(event_intensities)
    feats["micro_expr_intensity_std"]   = _std(event_intensities)
    feats["micro_expr_intensity_max"]   = max(event_intensities)
    feats["micro_expr_au_diversity"]    = _mean(event_au_counts)

    return feats


def _zero_micro_feats(feats: Dict[str, float], skip_count_rate: bool = False) -> None:
    if not skip_count_rate:
        feats["micro_expr_count"] = 0.0
        feats["micro_expr_rate"]  = 0.0
    feats["micro_expr_dur_mean"]       = 0.0
    feats["micro_expr_dur_std"]        = 0.0
    feats["micro_expr_dur_median"]     = 0.0
    feats["micro_expr_intensity_mean"] = 0.0
    feats["micro_expr_intensity_std"]  = 0.0
    feats["micro_expr_intensity_max"]  = 0.0
    feats["micro_expr_au_diversity"]   = 0.0


# ─── Group C: Emotion prototype features ─────────────────────────────────────

def _compute_emotion_features(fd: FrameData) -> Dict[str, float]:
    """
    Map per-frame AU activations to Ekman basic-emotion prototypes.

    Per emotion: proportion of frames and event rate.
    Global: entropy over proportions, transition rate, incongruence proportion.
    """
    feats: Dict[str, float] = {}
    n = fd.n_frames
    if n == 0:
        for label, _ in EMOTION_PROTOTYPES:
            feats[f"emo_{label}_prop"] = 0.0
            feats[f"emo_{label}_rate"] = 0.0
        feats["emo_diversity"] = 0.0
        feats["emo_transition_rate"] = 0.0
        feats["emo_incongruence_prop"] = 0.0
        return feats

    dur = fd.clip_duration()
    c = fd.cols

    # Per-frame binary series for each emotion.
    emo_series: Dict[str, List[float]] = {}
    for label, au_list in EMOTION_PROTOTYPES:
        emo_series[label] = [
            1.0 if all(c[au][i] >= 0.5 for au in au_list) else 0.0
            for i in range(n)
        ]

    proportions: List[float] = []
    for label, _ in EMOTION_PROTOTYPES:
        prop = _mean(emo_series[label])
        feats[f"emo_{label}_prop"] = prop
        proportions.append(prop)

        _, evts = _count_events(emo_series[label], fd.timestamps)
        feats[f"emo_{label}_rate"] = len(evts) / dur if dur > 0 else 0.0

    feats["emo_diversity"] = _shannon_entropy(proportions)

    # Dominant emotion per frame and transition counting.
    labels = [label for label, _ in EMOTION_PROTOTYPES]
    transitions = 0
    prev_dominant: Optional[str] = None
    incongruent_frames = 0

    for i in range(n):
        active = [label for label in labels if emo_series[label][i] >= 0.5]
        if len(active) > 1:
            incongruent_frames += 1

        dominant = active[0] if len(active) == 1 else (active[0] if active else None)
        if prev_dominant is not None and dominant is not None and dominant != prev_dominant:
            transitions += 1
        if dominant is not None:
            prev_dominant = dominant

    feats["emo_transition_rate"] = transitions / dur if dur > 0 else 0.0
    feats["emo_incongruence_prop"] = incongruent_frames / n

    return feats


# ─── Group D: AU intensity dynamics ──────────────────────────────────────────

def _compute_intensity_dynamics(fd: FrameData) -> Dict[str, float]:
    """
    Exploit AU_r (continuous intensity) channels to capture expressiveness
    level, variability, suppression, and temporal smoothness.
    """
    feats: Dict[str, float] = {}
    n = fd.n_frames
    if n == 0:
        feats["au_intensity_mean"] = 0.0
        feats["au_intensity_std"] = 0.0
        feats["au_intensity_range"] = 0.0
        feats["facial_freeze_prop"] = 1.0
        feats["au_peak_to_mean"] = 0.0
        feats["au_temporal_smoothness"] = 0.0
        feats["au_activation_entropy"] = 0.0
        feats["au_coactivation_mean"] = 0.0
        return feats

    c = fd.cols

    # Per-frame mean AU_r intensity across all AU_r channels.
    au_r_with_data = [col for col in AU_R if any(v != 0.0 for v in c[col])]
    if not au_r_with_data:
        au_r_with_data = AU_R  # fall back to all (all zeros)

    per_frame_mean: List[float] = []
    per_frame_coactivation: List[float] = []

    for i in range(n):
        vals = [c[col][i] for col in au_r_with_data]
        per_frame_mean.append(_mean(vals))
        per_frame_coactivation.append(sum(1.0 for v in vals if v > 0.5))

    global_mean = _mean(per_frame_mean)
    feats["au_intensity_mean"] = global_mean
    feats["au_intensity_std"] = _std(per_frame_mean)
    feats["au_intensity_range"] = max(per_frame_mean) - min(per_frame_mean) if per_frame_mean else 0.0

    feats["facial_freeze_prop"] = sum(1.0 for v in per_frame_mean if v < AU_FREEZE_THRESH) / n

    peak = max(per_frame_mean) if per_frame_mean else 0.0
    feats["au_peak_to_mean"] = peak / global_mean if global_mean > 0 else 0.0

    # Temporal smoothness: mean abs frame-to-frame difference.
    if n > 1:
        diffs = [abs(per_frame_mean[i] - per_frame_mean[i - 1]) for i in range(1, n)]
        feats["au_temporal_smoothness"] = _mean(diffs)
    else:
        feats["au_temporal_smoothness"] = 0.0

    # Activation entropy: bin per-frame mean intensities into 10 bins [0, 0.5).
    n_bins = 10
    bin_width = 0.5 / n_bins
    bin_counts = [0.0] * n_bins
    for v in per_frame_mean:
        idx = min(int(v / bin_width), n_bins - 1) if v >= 0 else 0
        bin_counts[idx] += 1.0
    feats["au_activation_entropy"] = _shannon_entropy(bin_counts)

    feats["au_coactivation_mean"] = _mean(per_frame_coactivation)

    return feats


# ─── Group E: Cognitive load indicators ──────────────────────────────────────

def _compute_cognitive_load_features(fd: FrameData) -> Dict[str, float]:
    """
    Blink regularity, gaze stability/wandering, and head motion energy
    as proxies for cognitive load during deception.
    """
    feats: Dict[str, float] = {}
    n = fd.n_frames
    dur = fd.clip_duration()
    c = fd.cols

    if n == 0:
        feats["blink_regularity"] = 0.0
        feats["gaze_stability"] = 0.0
        feats["gaze_transition_rate"] = 0.0
        feats["head_motion_energy"] = 0.0
        feats["head_motion_range_pitch"] = 0.0
        feats["head_motion_range_yaw"] = 0.0
        feats["head_motion_range_roll"] = 0.0
        return feats

    # ── Blink regularity (CV of inter-blink intervals) ────────────────────────
    blink_binary = [float(c["AU45_c"][i] >= 0.5) for i in range(n)]
    blink_events = _count_events_indexed(
        blink_binary, fd.timestamps,
        min_dur=0.04, max_dur=0.60,
    )
    if len(blink_events) >= 2:
        # Inter-blink interval = gap between end of one blink and start of next.
        ibis: List[float] = []
        for j in range(1, len(blink_events)):
            gap = fd.timestamps[blink_events[j][0]] - fd.timestamps[blink_events[j - 1][1]]
            if gap > 0:
                ibis.append(gap)
        feats["blink_regularity"] = _cv(ibis) if ibis else 0.0
    else:
        feats["blink_regularity"] = 0.0

    # ── Gaze stability ────────────────────────────────────────────────────────
    gx = c["gaze_angle_x"]
    gy = c["gaze_angle_y"]
    gaze_mag = [math.sqrt(gx[i] ** 2 + gy[i] ** 2) for i in range(n)]
    gaze_std = _std(gaze_mag)
    feats["gaze_stability"] = max(0.0, 1.0 - gaze_std)

    # Gaze transition rate: direction changes in gaze magnitude.
    gaze_thresh = GAZE_CENTRE_THRESH
    gaze_transitions = _zero_crossings(
        [m - gaze_thresh for m in gaze_mag], threshold=0.0,
    )
    feats["gaze_transition_rate"] = gaze_transitions / dur if dur > 0 else 0.0

    # ── Head motion energy and range ──────────────────────────────────────────
    rx = c["pose_Rx"]
    ry = c["pose_Ry"]
    rz = c["pose_Rz"]

    if n > 1:
        energy_sum = 0.0
        for i in range(1, n):
            energy_sum += abs(rx[i] - rx[i - 1]) + abs(ry[i] - ry[i - 1]) + abs(rz[i] - rz[i - 1])
        feats["head_motion_energy"] = energy_sum / (n - 1)
    else:
        feats["head_motion_energy"] = 0.0

    feats["head_motion_range_pitch"] = max(rx) - min(rx) if rx else 0.0
    feats["head_motion_range_yaw"]   = max(ry) - min(ry) if ry else 0.0
    feats["head_motion_range_roll"]  = max(rz) - min(rz) if rz else 0.0

    return feats


# ─── Group F: Temporal dynamics ──────────────────────────────────────────────

def _compute_temporal_dynamics(fd: FrameData) -> Dict[str, float]:
    """
    Expression onset/offset speed, asymmetry, bout duration statistics,
    and proportions of long/short expression bouts.
    """
    feats: Dict[str, float] = {}
    n = fd.n_frames
    c = fd.cols

    if n < 2:
        feats["expr_onset_speed_mean"] = 0.0
        feats["expr_offset_speed_mean"] = 0.0
        feats["onset_offset_asymmetry"] = 0.0
        feats["expr_bout_dur_mean"] = 0.0
        feats["expr_bout_dur_std"] = 0.0
        feats["expr_bout_regularity"] = 0.0
        feats["long_expr_prop"] = 0.0
        feats["short_expr_prop"] = 0.0
        return feats

    dt = fd.frame_dt()
    au_r_with_data = [col for col in AU_R if any(v != 0.0 for v in c[col])]

    # Per-frame aggregate intensity (mean across AU_r channels).
    if au_r_with_data:
        per_frame_intensity = [_mean([c[col][i] for col in au_r_with_data]) for i in range(n)]
    else:
        per_frame_intensity = [0.0] * n

    # Detect all expression bouts (any expression AU active, no duration filter).
    any_expr = [
        1.0 if any(c[col][i] >= 0.5 for col in EXPRESSION_AU_C) else 0.0
        for i in range(n)
    ]
    bouts = _count_events_indexed(any_expr, fd.timestamps)

    onset_speeds: List[float] = []
    offset_speeds: List[float] = []
    bout_durations: List[float] = []

    for start_i, end_i, bout_dur in bouts:
        bout_durations.append(bout_dur)

        # Onset speed: intensity rise rate over first 2 frames of bout.
        if start_i < n - 1 and dt > 0:
            rise = per_frame_intensity[start_i + 1] - per_frame_intensity[max(0, start_i - 1)]
            onset_speeds.append(rise / dt)

        # Offset speed: intensity fall rate over last 2 frames of bout.
        if end_i > 0 and dt > 0:
            fall_end = min(end_i + 1, n - 1)
            fall = per_frame_intensity[end_i] - per_frame_intensity[fall_end]
            offset_speeds.append(fall / dt)

    feats["expr_onset_speed_mean"]  = _mean(onset_speeds)
    feats["expr_offset_speed_mean"] = _mean(offset_speeds)

    mean_onset = abs(_mean(onset_speeds)) if onset_speeds else 0.0
    mean_offset = abs(_mean(offset_speeds)) if offset_speeds else 0.0
    if mean_offset > 0:
        feats["onset_offset_asymmetry"] = mean_onset / mean_offset
    else:
        feats["onset_offset_asymmetry"] = 0.0

    feats["expr_bout_dur_mean"] = _mean(bout_durations)
    feats["expr_bout_dur_std"]  = _std(bout_durations)
    feats["expr_bout_regularity"] = _cv(bout_durations)

    n_bouts = len(bout_durations)
    if n_bouts > 0:
        feats["long_expr_prop"]  = sum(1.0 for d in bout_durations if d > LONG_EXPR_THRESH) / n_bouts
        feats["short_expr_prop"] = sum(1.0 for d in bout_durations if d < SHORT_EXPR_THRESH) / n_bouts
    else:
        feats["long_expr_prop"]  = 0.0
        feats["short_expr_prop"] = 0.0

    return feats


# ─── label reader ─────────────────────────────────────────────────────────────

def _read_labels(
    labels_csv_path: Path,
) -> Tuple[List[str], Dict[str, str], Dict[str, int]]:
    clip_ids: List[str] = []
    subject_id_by_clip: Dict[str, str] = {}
    is_deceptive_by_clip: Dict[str, int] = {}

    with labels_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "clip_id" not in fieldnames or "subject_id" not in fieldnames:
            raise KeyError("Expected `clip_id` and `subject_id` columns in labels.csv")

        has_is_deceptive = "is_deceptive" in fieldnames
        has_label        = "label" in fieldnames
        if not has_is_deceptive and not has_label:
            raise KeyError(
                "Expected `is_deceptive` (or fallback `label`) column in labels.csv"
            )

        for row in reader:
            clip_id = row["clip_id"]
            clip_ids.append(clip_id)
            subject_id_by_clip[clip_id] = row["subject_id"]
            if has_is_deceptive:
                is_deceptive_by_clip[clip_id] = int(row["is_deceptive"])
            else:
                label = str(row["label"]).strip().lower()
                is_deceptive_by_clip[clip_id] = 1 if label == "deceptive" else 0

    return clip_ids, subject_id_by_clip, is_deceptive_by_clip


# ─── OpenFace runner ──────────────────────────────────────────────────────────

def run_openface_feature_extraction(
    input_dir: Path = Path("data/raw_videos"),
    output_dir: Path = Path("data/extracted_AU_gaze"),
    feature_extraction_bin: Path = Path(
        "/mnt/c/users/tania/tools/openface/build/bin/FeatureExtraction"
    ),
) -> None:
    files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".mp4"}
    )
    if not files:
        print("No input files found")
        return
    cmd = [str(feature_extraction_bin)]
    for f in files:
        cmd.extend(["-f", str(f)])
    cmd.extend(["-out_dir", str(output_dir)])
    print("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ─── main builder ─────────────────────────────────────────────────────────────

# All feature-group computation functions, in order.
_FEATURE_GROUPS = [
    ("mumin",     _compute_mumin_features),
    ("micro",     _compute_micro_expression_features),
    ("emotion",   _compute_emotion_features),
    ("intensity", _compute_intensity_dynamics),
    ("cogload",   _compute_cognitive_load_features),
    ("temporal",  _compute_temporal_dynamics),
]


def build_facial_features_csv(
    labels_csv_path: Path       = Path("data/labels.csv"),
    extracted_au_dir: Path      = Path("data/extracted_AU_gaze"),
    output_csv_path: Path       = Path("features/facial.csv"),
    confidence_threshold: float = DEFAULT_CONFIDENCE,
) -> None:
    """
    Build features/facial.csv from OpenFace frame-level CSVs.

    Output schema: clip_id | subject_id | is_deceptive | <all feature groups>
    """
    clip_ids, subject_id_by_clip, is_deceptive_by_clip = _read_labels(labels_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    group_cols: Dict[str, Optional[List[str]]] = {name: None for name, _ in _FEATURE_GROUPS}

    rows: List[List[object]] = []
    total = len(clip_ids)

    for idx, clip_id in enumerate(clip_ids, start=1):
        stem = Path(clip_id).stem
        clip_csv_path = extracted_au_dir / f"{stem}.csv"

        if not clip_csv_path.exists():
            print(f"WARNING: {clip_csv_path} not found — skipping {clip_id}")
            continue

        try:
            fd = _load_frame_data(clip_csv_path, confidence_threshold)
        except Exception as exc:
            print(f"ERROR loading {clip_csv_path}: {exc} — skipping")
            continue

        if fd.n_frames == 0:
            print(f"WARNING: {clip_id} has no high-confidence frames")

        all_feats: Dict[str, float] = {}
        for group_name, compute_fn in _FEATURE_GROUPS:
            group_feats = compute_fn(fd)
            if group_cols[group_name] is None:
                group_cols[group_name] = list(group_feats.keys())
            all_feats.update(group_feats)

        row: List[object] = [
            clip_id,
            subject_id_by_clip[clip_id],
            is_deceptive_by_clip[clip_id],
        ]
        for group_name, _ in _FEATURE_GROUPS:
            cols = group_cols[group_name] or []
            for col in cols:
                row.append(_safe(all_feats.get(col), 0.0))
        rows.append(row)

        if idx % 10 == 0:
            print(f"Progress: {idx}/{total}")

    feature_cols: List[str] = []
    group_counts: List[str] = []
    for group_name, _ in _FEATURE_GROUPS:
        cols = group_cols[group_name] or []
        feature_cols.extend(cols)
        group_counts.append(f"{len(cols)} {group_name}")

    header = ["clip_id", "subject_id", "is_deceptive"] + feature_cols

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    summary = " + ".join(group_counts)
    print(
        f"\nDone — {output_csv_path.as_posix()} written "
        f"({len(rows)} clips, {len(feature_cols)} features: {summary})"
    )


if __name__ == "__main__":
    build_facial_features_csv()
