import subprocess
from pathlib import Path

# paths
INPUT_DIR = Path("/data/raw_videos")
OUTPUT_DIR = Path("/data/extracted_AU_gaze2")

# collect media files (adjust extensions if needed)
files = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in {".mp3", ".wav", ".mp4"}
)

if not files:
    raise RuntimeError("No input files found")

# build command
cmd = ["/mnt/c/users/tania/tools/openface/build/bin/FeatureExtraction"] # ignore this RIP
for f in files:
    cmd.extend(["-f", str(f)])

cmd.extend(["-out_dir", str(OUTPUT_DIR)])

# show command (debug-friendly)
print("Running command:")
print(" ".join(cmd))

# execute
subprocess.run(cmd, check=True)
