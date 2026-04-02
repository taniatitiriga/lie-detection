"""
extract_facial.py
─────────────────────────────────────────────────────────────────────────────
Extracts theory-grounded facial deception features from OpenFace frame-level
CSVs.  Produces two feature families:

  1. MUMIN-analog features
     Automatic counterparts to the MUMIN coding-scheme categories used in
     manual annotations: smile (genuine / social), scowl, eyebrow raise /
     frown, eye behaviour, gaze direction, mouth state, and head movement
     patterns.  Each category is a per-clip score (proportion of
     high-confidence frames where the pattern fires) plus a temporal density
     (number of events per second) where applicable.

  2. Micro-expression features
     Grounded in affective-leakage theory.  A micro-expression is any AU
     activation lasting between MIN_MICRO_DUR (noise floor) and
     MICRO_EXPR_MAX_SEC (Ekman upper bound, extended here to 0.5 s).

     Single-frame activations are treated as sensor noise and rejected via a
     minimum-duration noise gate (MIN_MICRO_DUR).  This gate is set to just
     above one frame at the assumed capture rate so that genuine very-short
     expressions of ~80 ms (2 frames @ 25 fps) are still captured.

     Features:
       • micro_expr_count          total count in clip
       • micro_expr_rate           count / clip_duration_sec
       • micro_expr_dur_mean       mean duration (seconds)
       • micro_expr_dur_std        std of duration
       • micro_expr_dur_median     median duration
       • micro_expr_intensity_mean mean of per-event peak AU intensity
       • micro_expr_intensity_std  std of per-event peak AU intensity
       • micro_expr_intensity_max  maximum single-event peak intensity
       • micro_expr_au_diversity   mean number of distinct AUs per event
         (breadth of each leaked expression)

Paths mirror extract_visual.py; output written to features/facial.csv.

Noise-gate rationale
────────────────────
OpenFace AU classifiers occasionally fire on a single frame due to motion
blur, slight mis-alignment, or FACS boundary ambiguity.  A 1-frame hit at
25 fps lasts 0.04 s — shorter than any physiologically plausible expression.
Ekman places the lower bound of micro-expressions at ~40 ms (≈1 frame), so
excluding exactly 1-frame events would risk losing the shortest genuine ones.
Instead we set MIN_MICRO_DUR = 1.5 / DEFAULT_FPS (≈60 ms), which requires
at least 2 consecutive frames to agree before an event is accepted.  This
eliminates isolated single-frame spikes while preserving genuine two-frame
(~80 ms) micro-expressions.
"""

from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── constants ───────────────────────────────────────────────────────────────

# Default video FPS assumed when computing rates.
DEFAULT_FPS: float = 25.0

# Confidence threshold: frames below this are discarded.
DEFAULT_CONFIDENCE: float = 0.9

# Gaze angle (radians) below which the subject is considered on-camera.
GAZE_CENTRE_THRESH: float = 0.15   # ~8.6 °

# Gaze angle beyond which we flag aversion.
GAZE_AVERT_THRESH: float = 0.25    # ~14 °

# ── Micro-expression duration bounds ─────────────────────────────────────────
# Lower bound: must span at least 1.5 frames to reject single-frame noise.
# We use 1.5 / FPS so that two consecutive frames (≈80 ms at 25 fps) pass
# while isolated single-frame spikes (~40 ms) are rejected.
MIN_MICRO_DUR: float = 1.5 / DEFAULT_FPS   # ≈ 0.060 s  (noise gate)

# Upper bound: Ekman's canonical upper limit is ~200 ms, but ambiguous
# suppressed expressions can reach ~500 ms.  We use 0.5 s to capture both.
MICRO_EXPR_MAX_SEC: float = 0.50

# Head-movement thresholds (radians).
HEAD_NOD_THRESH: float   = 0.06    # ~3.4 °
HEAD_SHAKE_THRESH: float = 0.06
HEAD_SIDETURN_THRESH: float = 0.20 # ~11.5 °
HEAD_TILT_THRESH: float  = 0.10


# ─── AU column lists ──────────────────────────────────────────────────────────

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

# AU_C channels that represent genuine facial movement (exclude blink AU45).
EXPRESSION_AU_C: List[str] = [c for c in AU_C if c != "AU45_c"]


# ─── helpers ──────────────────────────────────────────────────────────────────

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


def _count_events(
    binary_series: List[float],
    timestamps: List[float],
    min_dur: float = 0.0,
    max_dur: float = float("inf"),
) -> Tuple[int, List[float]]:
    """
    Count contiguous runs of 1 in *binary_series* whose duration falls in
    [min_dur, max_dur] seconds.

    Returns (count, list_of_durations).

    Duration is measured as  timestamps[last_frame] − timestamps[first_frame].
    For a 2-frame event at 25 fps this gives 0.04 s (1 / FPS), not 0.08 s —
    intentionally conservative so that the noise gate (min_dur ≈ 1.5/FPS)
    admits 2-frame events but not 1-frame ones:
        1-frame duration = 0.000 s  <  0.060 s  → rejected
        2-frame duration = 0.040 s  <  0.060 s  → rejected  ← wait…

    Correction: to correctly admit 2-frame events we measure duration as
    (end_timestamp − start_timestamp + 1/fps), i.e. we include the width
    of the final frame.  Call this the "inclusive" duration:
        1-frame: 1/25 = 0.040 s  <  0.060 s  → still rejected  ✓
        2-frame: 2/25 = 0.080 s  >= 0.060 s  → accepted         ✓
    """
    events: List[float] = []
    in_event = False
    start_idx = 0

    # Estimate frame interval for inclusive duration; fall back to 1/FPS.
    dt = (timestamps[-1] - timestamps[0]) / max(len(timestamps) - 1, 1) if len(timestamps) > 1 else 1.0 / DEFAULT_FPS

    for i, v in enumerate(binary_series):
        if v >= 0.5 and not in_event:
            in_event = True
            start_idx = i
        elif v < 0.5 and in_event:
            in_event = False
            # Inclusive: add one frame width so the last active frame counts.
            dur = (timestamps[i - 1] - timestamps[start_idx]) + dt
            if min_dur <= dur <= max_dur:
                events.append(dur)
    if in_event:
        dur = (timestamps[-1] - timestamps[start_idx]) + dt
        if min_dur <= dur <= max_dur:
            events.append(dur)

    return len(events), events


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


# ─── MUMIN-analog feature computation ────────────────────────────────────────
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
        min_dur=0.04, max_dur=0.60,  # realistic blink: 40–600 ms
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


# ─── micro-expression feature computation ────────────────────────────────────

def _compute_micro_expression_features(fd: FrameData) -> Dict[str, float]:
    """
    Detect and characterise micro-expressions: brief AU activations in the
    range [MIN_MICRO_DUR, MICRO_EXPR_MAX_SEC].

    Noise-gate
    ──────────
    Single-frame AU activations are common classifier artefacts (motion blur,
    landmark jitter).  We reject any event shorter than MIN_MICRO_DUR, which
    requires agreement across at least 2 consecutive frames before the event
    is accepted.  This is achieved by passing min_dur=MIN_MICRO_DUR to
    _count_events, which uses "inclusive" duration accounting (see docstring
    in _count_events).

    Intensity statistics
    ────────────────────
    For each micro-expression event we record the peak AU_r intensity across
    all active channels during that event window.  This yields per-event
    intensity values from which we derive mean, std, and max.

    AU diversity
    ────────────
    Per event we count how many distinct AU_c channels fired, giving a measure
    of expression breadth (a rich leaked expression fires many AUs; a slight
    twitch fires one or two).
    """
    feats: Dict[str, float] = {}
    n = fd.n_frames
    if n == 0:
        _zero_micro_feats(feats)
        return feats

    dur = fd.clip_duration()
    c = fd.cols

    # ── Build per-frame "any expression AU active" binary series ──────────────
    # We exclude AU45 (blink) because blinks are not facial expressions.
    any_expr_active: List[float] = [
        1.0 if any(c[col][i] >= 0.5 for col in EXPRESSION_AU_C) else 0.0
        for i in range(n)
    ]

    # ── Detect micro-expression events ────────────────────────────────────────
    # Events must be in [MIN_MICRO_DUR, MICRO_EXPR_MAX_SEC].
    # min_dur = MIN_MICRO_DUR enforces the noise gate (≥2 frames required).
    micro_events: List[Tuple[int, int]] = []  # (start_idx, end_idx) inclusive
    in_event = False
    start_idx = 0

    dt = (fd.timestamps[-1] - fd.timestamps[0]) / max(n - 1, 1) if n > 1 else 1.0 / DEFAULT_FPS

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

    # ── Per-event statistics ──────────────────────────────────────────────────
    event_durations:   List[float] = []
    event_intensities: List[float] = []  # peak AU_r intensity during event
    event_au_counts:   List[float] = []  # distinct AU_c channels that fired

    au_r_present = [col for col in AU_R if any(v != 0.0 for v in c[col])]

    for start_i, end_i in micro_events:
        # Duration (inclusive frame width already applied above at detection)
        event_dur = (fd.timestamps[end_i] - fd.timestamps[start_i]) + dt
        event_durations.append(event_dur)

        # Peak intensity: max AU_r value across frames and channels in window.
        if au_r_present:
            peak = max(
                c[col][i]
                for i in range(start_i, end_i + 1)
                for col in au_r_present
            )
        else:
            peak = 0.0
        event_intensities.append(peak)

        # AU diversity: number of distinct AU_c channels active in the window.
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
    """Fill micro-expression features with zeros (used when no events found)."""
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

def build_facial_features_csv(
    labels_csv_path: Path       = Path("data/labels.csv"),
    extracted_au_dir: Path      = Path("data/extracted_AU_gaze"),
    output_csv_path: Path       = Path("features/facial.csv"),
    confidence_threshold: float = DEFAULT_CONFIDENCE,
) -> None:
    """
    Build features/facial.csv from OpenFace frame-level CSVs.

    Output schema
    ─────────────
    clip_id | subject_id | is_deceptive
        | <mumin_*>        (MUMIN-analog proportions and rates)
        | <micro_expr_*>   (micro-expression count, rate, duration, intensity)
    """
    clip_ids, subject_id_by_clip, is_deceptive_by_clip = _read_labels(labels_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    mumin_cols: Optional[List[str]] = None
    micro_cols: Optional[List[str]] = None

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

        mumin_feats = _compute_mumin_features(fd)
        micro_feats = _compute_micro_expression_features(fd)

        if mumin_cols is None:
            mumin_cols = list(mumin_feats.keys())
        if micro_cols is None:
            micro_cols = list(micro_feats.keys())

        row: List[object] = [
            clip_id,
            subject_id_by_clip[clip_id],
            is_deceptive_by_clip[clip_id],
        ]
        for col in mumin_cols:
            row.append(_safe(mumin_feats.get(col), 0.0))
        for col in micro_cols:
            row.append(_safe(micro_feats.get(col), 0.0))
        rows.append(row)

        if idx % 10 == 0:
            print(f"Progress: {idx}/{total}")

    if mumin_cols is None:
        mumin_cols = []
    if micro_cols is None:
        micro_cols = []

    header = ["clip_id", "subject_id", "is_deceptive"] + mumin_cols + micro_cols

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(
        f"\nDone — {output_csv_path.as_posix()} written "
        f"({len(rows)} clips, {len(mumin_cols)} MUMIN + {len(micro_cols)} micro-expression features)"
    )


if __name__ == "__main__":
    build_facial_features_csv()