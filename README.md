## Setup

### Clone this repository

```
git clone https://github.com/taniatitiriga/lie-detection
cd lie-detection
```

### Package manager

Install `uv` package manager here: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

Install dependencies:

```
uv sync
```

### OpenFace

Install OpenFace here: [https://github.com/TadasBaltrusaitis/OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)  
Required for frame-level AU extraction from video clips.

---

## Usage

### Step 1 — Feature extraction

All extractors write to `features/`. They require `data/labels.csv` (columns: `clip_id`, `subject_id`, `is_deceptive`).

#### Visual (18-dim AU means, §3.3 of Sen et al. 2020)

Averages 18 binary AU indicator columns per clip from OpenFace frame-level CSVs (confidence ≥ 0.9):

```
uv run python src/extract_visual.py
```

Output: `features/visual.csv` — 18 features per clip.

Requires: `data/extracted_AU_gaze/<clip_id>.csv` (OpenFace output). To re-run OpenFace extraction:

```
uv run python playground/face_feature_extraction.py 
```

#### Acoustic (52-dim pitch + VAD histograms)

Extracts pitch mean/stddev over voiced frames (parselmouth) and 25-bin VAD histograms for speech and silence segment lengths (webrtcvad), per clip. First extracts WAV files from videos using ffmpeg:

```
uv run python src/extract_acoustic.py
```

Output: `features/acoustic.csv` — 52 features per clip.  
Requires: ffmpeg, praat-parselmouth, webrtcvad.

#### Linguistic (unigram TF + Empath categories)

Builds corpus-wide vocabulary (tokens with frequency ≥ 10), computes normalized unigram frequencies per clip, and appends 15 Empath/LIWC-proxy category scores:

```
uv run python src/extract_linguistic.py
```

Output: `features/linguistic.csv` — variable-length vocab + 15 Empath features per clip.  
Requires: `data/Real-life_Deception_Detection_2016/Transcription/{Deceptive,Truthful}/`, empath.

---

### Step 2 — Classification & experiments

All modes use `src/main.py` via `uv run python src/main.py [flags]`.

#### Subject-aware LOSO-CV on specific feature CSVs

Run Leave-One-Subject-Out CV with RF, SVM, and NN on any set of pre-built feature CSVs
(inner-joined on `clip_id`).  Accuracy is **macro-averaged across folds** (unweighted by
clip count) to avoid bias toward subjects with many clips (e.g. S003 with 27% of all clips):

```
uv run python src/main.py \
  --feature-csvs features/visual.csv features/acoustic.csv features/linguistic.csv \
  --n-runs 3
```

Add `--subject-level` to also print subject-level majority-vote accuracy (mean posterior
per subject → binary prediction) alongside the clip-level result:

```
uv run python src/main.py \
  --feature-csvs features/visual.csv features/acoustic.csv features/linguistic.csv \
  --n-runs 3 --subject-level
```

#### Full ablation table (single-modality, early fusion, late fusion)

Runs all 20 experiments from the ablation table — single modality (RF/SVM/NN), pairwise
early/late fusion (NN), and all-three early/late fusion (RF/SVM/NN or RF/NN).
Saves `clip_level_results.csv` (column `eval_level=clip`):

```
uv run python src/main.py --ablation --n-runs 3 --out runs/my_experiment
```

Reads from `features/visual.csv`, `features/acoustic.csv`, `features/linguistic.csv` automatically.

Add `--subject-level` to append subject-level majority-vote rows (`eval_level=subject`) to
the same CSV:

```
uv run python src/main.py --ablation --subject-level --n-runs 3 --out runs/my_experiment
```

#### Sanity checks (leakage + dummy baseline)

Verifies subject-level partitioning, scaler fit only on training data, and that real
classifiers beat the majority-class dummy:

```
uv run python src/main.py --sanity
```

---

### Classifier details


| Classifier | Config                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------- |
| RF         | 100 trees, min_samples_leaf=3                                                                |
| SVM        | RBF kernel, 4-fold grid search over C∈{0.01–100}, γ∈{0.001–scale}, 3×3 mean-filter smoothing |
| NN (MLP)   | (64, 32) hidden units, ReLU, Adam, α=1e-3, early stopping (val_frac=0.15, max_iter=1000)    |

Accuracy is reported as **macro-averaged LOSO-CV** — per-subject fold accuracy averaged
across subjects — so every subject contributes equally regardless of clip count.

With `--subject-level` a second line is printed (and written to CSV) showing **subject-level
majority-vote accuracy**: each subject's clips are averaged into a single posterior and
thresholded at 0.5.

---

## Sources

Dataset: [https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf](https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf)  

Facial expression recognition: [https://www.cl.cam.ac.uk/research/rainbow/projects/openface/wacv2016.pdf](https://www.cl.cam.ac.uk/research/rainbow/projects/openface/wacv2016.pdf)

## Citations

#### Dataset

**Deception detection using real-life trial data**  
Pérez-Rosas, V., Abouelenien, M., Mihalcea, R. and Burzo, M., *Proceedings of the 2015 ACM on international conference on multimodal interaction*, 2015

#### Overall system

**OpenFace 2.0: Facial Behavior Analysis Toolkit**  
Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency,
*IEEE International Conference on Automatic Face and Gesture Recognition*, 2018

#### Facial landmark detection and tracking

**Convolutional experts constrained local model for facial landmark detection**  
A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency.
*Computer Vision and Pattern Recognition Workshops*, 2017

**Constrained Local Neural Fields for robust facial landmark detection in the wild**  
Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency.
in IEEE Int. *Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge*, 2013.

#### Eye gaze tracking

**Rendering of Eyes for Eye-Shape Registration and Gaze Estimation**  
Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling
in *IEEE International Conference on Computer Vision (ICCV)*, 2015

#### Facial Action Unit detection

**Cross-dataset learning and person-specific normalisation for automatic Action Unit detection**  
Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson
in *Facial Expression Recognition and Analysis Challenge*,
*IEEE International Conference on Automatic Face and Gesture Recognition*, 2015
