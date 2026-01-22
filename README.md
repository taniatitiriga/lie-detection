## Note to self

Install dependencies for experimenting:

```
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e ".[playground]"
```

Run experiments:
```
uv run playground/example.py
```
### Using OpenFace

Once compiled, OpenFace includes CLI tools that process images/videos:
```
build/bin/FeatureExtraction -f input.jpg -out_dir results/
```
You can call these from Python using subprocess:
```
import subprocess

subprocess.run([
    "./build/bin/FeatureExtraction",
    "-f", "input.jpg",
    "-out_dir", "results/"
])
```

## Sources
Dataset: https://web.eecs.umich.edu/~mihalcea/papers/perezrosas.icmi15.pdf
Facial expression recognition: https://www.cl.cam.ac.uk/research/rainbow/projects/openface/wacv2016.pdf