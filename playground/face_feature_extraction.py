import subprocess
from pathlib import Path

# Automatization for face feature extraction from dataset. Only run once.

# paths
INPUT_DIR = Path("/data/raw_videos")
OUTPUT_DIR = Path("/data/extracted_AU_gaze")

# collect media files
files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in {".mp3", ".wav", ".mp4"}
)

if not files:
    print("No input files found")

else:
    # build command
    cmd = ["/mnt/c/users/tania/tools/openface/build/bin/FeatureExtraction"]
    for f in files:
        cmd.extend(["-f", str(f)])

    cmd.extend(["-out_dir", str(OUTPUT_DIR)])

    print("Running command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)
