
## Setup

### Clone this repository
```
git clone https://github.com/taniatitiriga/lie-detection
cd lie-detection
```

### Package manager
Install `uv` package manager here: https://docs.astral.sh/uv/getting-started/installation/

Install dependencies:
```
uv sync
```

### OpenFace
Install OpenFace here: https://github.com/TadasBaltrusaitis/OpenFace  
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

Requires: `features/extracted_AU_gaze/<clip_id>.csv` (OpenFace output). To re-run OpenFace extraction:
```
uv run python src/extract_visual.py  # calls run_openface_feature_extraction() if needed
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

All modes use `src/classify.py` via `uv run python src/classify.py [flags]`.

#### Subject-aware LOOCV on specific feature CSVs
Run LOOCV with RF, SVM, and NN on any set of pre-built feature CSVs (inner-joined on `clip_id`):
```
uv run python src/classify.py \
  --feature-csvs features/visual.csv features/acoustic.csv features/linguistic.csv \
  --n-runs 3
```

#### Full ablation table (single-modality, early fusion, late fusion)
Runs all 20 experiments from the ablation table — single modality (RF/SVM/NN), pairwise early/late fusion (NN), and all-three early/late fusion (RF/SVM/NN or RF/NN). Saves `clip_level_results.csv`:
```
uv run python src/classify.py --ablation --n-runs 3 --out runs/my_experiment
```
Reads from `features/visual.csv`, `features/acoustic.csv`, `features/linguistic.csv` automatically.

#### Sanity checks (leakage + dummy baseline)
Verifies subject-level partitioning, scaler fit only on training data, and that real classifiers beat the majority-class dummy:
```
uv run python src/classify.py --sanity
```

#### Legacy RF experiment (per-frame aggregate features, no subject split)
```
uv run python src/classify.py --data-dir data/extracted_AU_gaze --out runs/rf_legacy
```

---

### Classifier details

| Classifier | Config |
|-----------|--------|
| RF | 100 trees, min_samples_leaf=3 |
| SVM | RBF kernel, 4-fold grid search over C∈{0.01–100}, γ∈{0.001–scale}, 3×3 mean-filter smoothing |
| NN (MLP) | (100, 500) hidden units, ReLU, Adam, α=1e-5, early stopping — matches Sen et al. 2020 |

Late fusion uses a weight sweep for the visual modality in steps of 0.1, with the remaining weight split equally between other modalities (matching Table 6 of Sen et al. 2020).

---

## Sources
Dataset: https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf  
Replication target: Sen et al., *Multimodal Deception Detection*, 2020

Facial expression recognition: https://www.cl.cam.ac.uk/research/rainbow/projects/openface/wacv2016.pdf


## Citations

#### Dataset
**Deception detection using real-life trial data**  
Pérez-Rosas, V., Abouelenien, M., Mihalcea, R. and Burzo, M., _Proceedings of the 2015 ACM on international conference on multimodal interaction_, 2015

#### Overall system

**OpenFace 2.0: Facial Behavior Analysis Toolkit**  
Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency,
_IEEE International Conference on Automatic Face and Gesture Recognition_, 2018

#### Facial landmark detection and tracking

**Convolutional experts constrained local model for facial landmark detection**  
A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency.
_Computer Vision and Pattern Recognition Workshops_, 2017

**Constrained Local Neural Fields for robust facial landmark detection in the wild**  
Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency.
in IEEE Int. _Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge_, 2013.

#### Eye gaze tracking

**Rendering of Eyes for Eye-Shape Registration and Gaze Estimation**  
Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling
in _IEEE International Conference on Computer Vision (ICCV)_, 2015

#### Facial Action Unit detection

**Cross-dataset learning and person-specific normalisation for automatic Action Unit detection**  
Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson
in _Facial Expression Recognition and Analysis Challenge_,
_IEEE International Conference on Automatic Face and Gesture Recognition_, 2015