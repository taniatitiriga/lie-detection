from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def extract_audio_from_clips() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    clips_root = repo_root / "data" / "Real-life_Deception_Detection_2016" / "Clips"
    deceptive_dir = clips_root / "Deceptive"
    truthful_dir = clips_root / "Truthful"

    output_dir = repo_root / "data" / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = []
    video_paths.extend(sorted(deceptive_dir.glob("trial_lie_*.mp4")))
    video_paths.extend(sorted(truthful_dir.glob("trial_truth_*.mp4")))

    written = 0
    skipped = 0

    for video_path in video_paths:
        stem = video_path.stem
        out_wav = output_dir / f"{stem}.wav"

        if out_wav.exists():
            skipped += 1
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out_wav),
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as e:
            print(f"ffmpeg not found while processing {stem}: {e}")
            continue

        if proc.returncode != 0:
            # Print stderr for debugging but continue processing other clips.
            if proc.stderr:
                print(proc.stderr)
            else:
                print(f"ffmpeg failed for {stem} with return code {proc.returncode}")
            continue

        written += 1
        print(f"Extracted {stem}.wav")

    print(f"Audio extraction complete — {written} files written, {skipped} skipped.")


def _load_labels(labels_csv_path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, int]]:
    clip_ids: List[str] = []
    subject_id_by_clip: Dict[str, str] = {}
    is_deceptive_by_clip: Dict[str, int] = {}

    with labels_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"labels.csv appears empty: {labels_csv_path}")

        if "clip_id" not in reader.fieldnames or "subject_id" not in reader.fieldnames:
            raise KeyError("Expected `clip_id` and `subject_id` columns in labels.csv")

        if "is_deceptive" not in reader.fieldnames:
            raise KeyError("Expected `is_deceptive` column in labels.csv")

        for row in reader:
            clip_id = row["clip_id"]
            clip_ids.append(clip_id)
            subject_id_by_clip[clip_id] = row["subject_id"]
            is_deceptive_by_clip[clip_id] = int(row["is_deceptive"])

    return clip_ids, subject_id_by_clip, is_deceptive_by_clip


def _extract_pitch_stats(wav_path: Path) -> Tuple[float, float]:
    import numpy as np
    import parselmouth

    snd = parselmouth.Sound(str(wav_path))
    pitch_obj = snd.to_pitch()
    f0 = pitch_obj.selected_array["frequency"]
    voiced = f0[f0 > 0]  # exclude unvoiced frames

    if len(voiced) == 0:
        print(f"WARNING: {wav_path.name} has no voiced frames; pitch features set to 0.0")
        return 0.0, 0.0

    pitch_mean = float(np.mean(voiced))
    pitch_stdv = float(np.std(voiced))
    return pitch_mean, pitch_stdv


def _safe(val: object) -> float:
    """Return val as float, or 0.0 if it is NaN / None / raises."""
    import math
    try:
        v = float(val)  # type: ignore[arg-type]
        return 0.0 if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return 0.0


def _extract_intensity_stats(wav_path: Path) -> Tuple[float, float]:
    """
    Compute mean and std of the intensity (energy) contour in dB.

    Returns (intensity_mean, intensity_std), both 0.0 on failure.
    """
    import numpy as np
    import parselmouth

    try:
        snd = parselmouth.Sound(str(wav_path))
        intensity = snd.to_intensity()
        vals = intensity.values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return 0.0, 0.0
        return _safe(np.mean(vals)), _safe(np.std(vals))
    except Exception:
        return 0.0, 0.0


def _extract_jitter_local(wav_path: Path) -> float:
    """
    Compute local (cycle-to-cycle) jitter via a Praat PointProcess.

    Returns 0.0 on failure (silent clip, too-short clip, etc.).
    """
    import parselmouth
    from parselmouth.praat import call

    try:
        snd = parselmouth.Sound(str(wav_path))
        pitch = snd.to_pitch()
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        return _safe(jitter)
    except Exception:
        return 0.0


def _extract_shimmer_local(wav_path: Path) -> float:
    """
    Compute local shimmer (amplitude perturbation) via Sound + PointProcess.

    Returns 0.0 on failure.
    """
    import parselmouth
    from parselmouth.praat import call

    try:
        snd = parselmouth.Sound(str(wav_path))
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        shimmer = call(
            [snd, pp],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )
        return _safe(shimmer)
    except Exception:
        return 0.0


def _extract_hnr_mean(wav_path: Path) -> float:
    """
    Compute mean Harmonics-to-Noise Ratio over the clip.

    Returns 0.0 on failure.
    """
    import parselmouth
    from parselmouth.praat import call

    try:
        snd = parselmouth.Sound(str(wav_path))
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        return _safe(hnr)
    except Exception:
        return 0.0


def _extract_speaking_rate_and_f0_frac(wav_path: Path) -> Tuple[float, float]:
    """
    Compute two features from the pitch contour:

    * speaking_rate  — voiced frames / clip duration (Hz proxy for speaking rate)
    * f0_voiced_frac — fraction of pitch frames where f0 > 0

    Both return 0.0 on failure.
    """
    import parselmouth

    try:
        snd = parselmouth.Sound(str(wav_path))
        duration = snd.duration  # seconds
        if duration <= 0:
            return 0.0, 0.0

        pitch_obj = snd.to_pitch()
        f0 = pitch_obj.selected_array["frequency"]
        total_frames = len(f0)
        voiced_frames = int((f0 > 0).sum())

        speaking_rate = _safe(voiced_frames / duration)
        f0_voiced_frac = _safe(voiced_frames / total_frames) if total_frames > 0 else 0.0
        return speaking_rate, f0_voiced_frac
    except Exception:
        return 0.0, 0.0


def _vad_histograms_webrtcvad(wav_path: Path) -> Tuple[List[float], List[float]]:
    # `webrtcvad` imports `pkg_resources` only to read its own version.
    # Some environments (including this one) don't ship `pkg_resources`,
    # so we provide a tiny stub to keep import working.
    try:
        import webrtcvad
    except ModuleNotFoundError as e:
        if e.name != "pkg_resources":
            raise
        import sys
        import types
        from importlib.metadata import PackageNotFoundError, version

        pkg_resources_stub = types.ModuleType("pkg_resources")

        def get_distribution(dist_name: str):
            class _Dist:
                pass

            d = _Dist()
            try:
                d.version = version(dist_name)
            except PackageNotFoundError:
                d.version = "unknown"
            return d

        pkg_resources_stub.get_distribution = get_distribution  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = pkg_resources_stub

        import webrtcvad

    import wave

    vad = webrtcvad.Vad(2)  # aggressiveness/mode=2 (most aggressive)
    frame_duration_ms = 30
    sample_rate = 16000
    frame_samples = int(sample_rate * frame_duration_ms / 1000)  # 480 samples
    bytes_per_sample = 2  # pcm_s16le
    frame_byte_len = frame_samples * bytes_per_sample

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono wav for {wav_path.name}, got {wf.getnchannels()} channels")
        if wf.getframerate() != sample_rate:
            raise ValueError(
                f"Expected 16kHz wav for {wav_path.name}, got {wf.getframerate()} Hz"
            )
        if wf.getsampwidth() != 2:
            raise ValueError(
                f"Expected 16-bit wav for {wav_path.name}, got sample width {wf.getsampwidth()}"
            )

        audio_bytes = wf.readframes(wf.getnframes())

    # Pad to an integer number of 30ms frames so we always segment cleanly.
    if len(audio_bytes) % frame_byte_len != 0:
        pad_len = frame_byte_len - (len(audio_bytes) % frame_byte_len)
        audio_bytes = audio_bytes + (b"\x00" * pad_len)

    num_frames = len(audio_bytes) // frame_byte_len
    speech_flags: List[bool] = []

    for i in range(num_frames):
        start = i * frame_byte_len
        frame = audio_bytes[start : start + frame_byte_len]
        is_speech = vad.is_speech(frame, sample_rate)
        speech_flags.append(bool(is_speech))

    # Convert frame-wise flags -> contiguous speech/silence runs.
    frame_sec = frame_duration_ms / 1000.0  # 0.03
    speech_segments: List[float] = []
    silence_segments: List[float] = []

    curr_flag = speech_flags[0] if speech_flags else False
    curr_len = 0

    for flag in speech_flags:
        if flag == curr_flag:
            curr_len += 1
        else:
            dur = curr_len * frame_sec
            if curr_flag:
                speech_segments.append(dur)
            else:
                silence_segments.append(dur)
            curr_flag = flag
            curr_len = 1

    # Flush final run
    if speech_flags:
        dur = curr_len * frame_sec
        if curr_flag:
            speech_segments.append(dur)
        else:
            silence_segments.append(dur)

    # Histograms with 25 bins, range [0, 3]. Last bin absorbs everything > 3s.
    bin_width = 3.0 / 25.0

    def hist25(durations: List[float]) -> List[float]:
        counts = [0.0] * 25
        for d in durations:
            if d < 0:
                idx = 0
            else:
                idx = int(d / bin_width)
            if idx >= 25:
                idx = 24
            counts[idx] += 1.0
        return counts

    sil_hist = hist25(silence_segments)
    sp_hist = hist25(speech_segments)

    # If a clip has no speech segments or no silence segments, fill corresponding 25 bins with 0.0.
    if len(silence_segments) == 0:
        sil_hist = [0.0] * 25
    if len(speech_segments) == 0:
        sp_hist = [0.0] * 25

    return sil_hist, sp_hist


def compute_acoustic_features(
    labels_csv_path: Path = Path("data/labels.csv"),
    wav_dir: Path = Path("data/audio"),
    output_csv_path: Path = Path("features/acoustic.csv"),
) -> None:
    # Import checks are handled inside the feature extractors, since `webrtcvad`
    # may need a `pkg_resources` shim (see `_vad_histograms_webrtcvad`).
    try:
        import parselmouth  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing required dependency for acoustic features: `parselmouth` "
            "(praat-parselmouth). Install it in your Python environment (or a venv) and re-run."
        ) from e

    clip_ids, subject_id_by_clip, is_deceptive_by_clip = _load_labels(labels_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        ["clip_id", "subject_id", "is_deceptive", "pitch_mean", "pitch_stdv"]
        + [f"sil_hist_{i:02d}" for i in range(25)]
        + [f"sp_hist_{i:02d}" for i in range(25)]
        + [
            "intensity_mean",
            "intensity_std",
            "jitter_local",
            "shimmer_local",
            "hnr_mean",
            "speaking_rate",
            "f0_voiced_frac",
        ]
    )

    rows: List[List[object]] = []

    for clip_idx, clip_id in enumerate(clip_ids, start=1):
        stem = Path(clip_id).stem
        wav_path = wav_dir / f"{stem}.wav"

        pitch_mean, pitch_stdv = _extract_pitch_stats(wav_path)
        sil_hist, sp_hist = _vad_histograms_webrtcvad(wav_path)
        intensity_mean, intensity_std = _extract_intensity_stats(wav_path)
        jitter_local = _extract_jitter_local(wav_path)
        shimmer_local = _extract_shimmer_local(wav_path)
        hnr_mean = _extract_hnr_mean(wav_path)
        speaking_rate, f0_voiced_frac = _extract_speaking_rate_and_f0_frac(wav_path)

        print(
            f"[{clip_idx}/{len(clip_ids)}] {subject_id_by_clip[clip_id]} | {clip_id}: "
            f"pitch_stdv={pitch_stdv:.2f}  intensity={intensity_mean:.1f}dB  "
            f"jitter={jitter_local:.4f}  shimmer={shimmer_local:.4f}  "
            f"hnr={hnr_mean:.2f}  sr={speaking_rate:.1f}  vf={f0_voiced_frac:.3f}"
        )

        row: List[object] = [
            clip_id,
            subject_id_by_clip[clip_id],
            is_deceptive_by_clip[clip_id],
            pitch_mean,
            pitch_stdv,
            *sil_hist,
            *sp_hist,
            intensity_mean,
            intensity_std,
            jitter_local,
            shimmer_local,
            hnr_mean,
            speaking_rate,
            f0_voiced_frac,
        ]
        rows.append(row)

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    n_feat = len(header) - 3  # subtract clip_id, subject_id, is_deceptive
    print(f"Done — {output_csv_path.as_posix()} written ({len(rows)} rows, {n_feat} features)")


if __name__ == "__main__":
    # Run audio extraction first (if WAVs are already there, it will skip).
    extract_audio_from_clips()
    # Then compute acoustic features.
    compute_acoustic_features()
