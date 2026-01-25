
## Setup

### Package manager
Install `uv` package manager here: https://docs.astral.sh/uv/getting-started/installation/

Setup virtual env and install dependencies for experimenting:
```
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U pip setuptools wheel
uv pip install numpy pandas scikit-learn joblib
```

### Openface
Install Openface here: https://github.com/TadasBaltrusaitis/OpenFace


## Run experiments

### Facial feature extraction
Extract facial features (AUs, gaze tracking) - may need to adjust paths:
```
uv run playground/face_feature_extraction.py
```

### Random forest experiment
Run RF experiment for classification (facial data):
```
uv run python playground/experiment_RF.py
```



## Sources
Dataset: https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf

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