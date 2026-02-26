# Face Wash Pipeline

Batch filter for anime face images — keeps only **frontal faces** by measuring left/right eye-to-nose distance symmetry using the existing ONNX landmark detector (YOLOv3 + HRNetv2, 28 keypoints).

## Databricks Setup

Run this in the first notebook cell (or `%sh` block) to install dependencies on a Databricks cluster with A10G GPU:

```bash
%pip install onnxruntime-gpu opencv-python-headless numpy tqdm
```

> **`opencv-python-headless`** is preferred over `opencv-python` on Databricks — it skips GUI/Qt dependencies that don't exist on headless clusters.

If your cluster uses a custom init script instead, add to `/dbfs/init/face_wash_init.sh`:

```bash
#!/bin/bash
pip install onnxruntime-gpu opencv-python-headless numpy tqdm
```

### Verify GPU is visible

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

### Copy the repo to the driver node

```bash
# Option A: clone from git
%sh git clone <your-repo-url> /tmp/ani_face_detector

# Option B: copy from DBFS / Unity Catalog volume
%sh cp -r /dbfs/FileStore/ani_face_detector /tmp/ani_face_detector
```

Then run:

```bash
%sh cd /tmp/ani_face_detector && python -m landmark_detect.face_wash \
  --image-folder /dbfs/path/to/images \
  --output-dir /dbfs/path/to/frontal_output \
  --device cuda
```

### Dependency summary

| Package | Version | Notes |
|---|---|---|
| `onnxruntime-gpu` | >=1.14 | CUDA EP for A10G; use `onnxruntime` for CPU-only |
| `opencv-python-headless` | >=4.0 | Use `-headless` on Databricks (no GUI) |
| `numpy` | >=1.19 | Usually pre-installed on Databricks ML Runtime |
| `tqdm` | >=4.0 | Usually pre-installed on Databricks ML Runtime |

> `numpy` and `tqdm` are typically pre-installed on Databricks ML Runtime 13.x+. You may only need to install `onnxruntime-gpu` and `opencv-python-headless`.

## Quick Start

```bash
# Calibrate: visualize first 50 images to verify landmarks and threshold
python -m landmark_detect.face_wash \
  --image-folder /path/to/images \
  --output-dir /path/to/frontal_output \
  --visualize 50

# Inspect _viz/ folder, then run full pipeline
python -m landmark_detect.face_wash \
  --image-folder /path/to/images \
  --output-dir /path/to/frontal_output \
  --device cuda
```

## How It Works

For each image the detector returns 28 keypoints. The frontal classifier computes:

```
left_eye_center  = mean(keypoints[11:17])   # 6 boundary points
right_eye_center = mean(keypoints[17:23])   # 6 boundary points
nose             = keypoints[23]            # nose tip

ratio = min(d_left, d_right) / max(d_left, d_right)
is_frontal = ratio >= threshold  (default 0.70)
```

A perfectly symmetric face gives ratio=1.0. Profile views have one eye much closer to the nose, dropping the ratio well below 0.7.

## Landmark Map

| Region | Indices |
|---|---|
| Face contour | 0–4 |
| Left eyebrow | 5–7 |
| Right eyebrow | 8–10 |
| Left eye | 11–16 |
| Right eye | 17–22 |
| Nose tip | 23 |
| Mouth | 24–27 |

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--image-folder` | *(required)* | Input image directory |
| `--output-dir` | *(required)* | Output directory for frontal images |
| `--device` | `cuda` | Inference device (`cuda` or `cpu`) |
| `--frontal-threshold` | `0.70` | Min eye-nose distance ratio to be frontal |
| `--face-threshold` | `0.5` | Min face detection score |
| `--kpt-confidence` | `0.3` | Min avg keypoint confidence for eyes/nose |
| `--visualize N` | `0` | Save annotated images for first N files |
| `--viz-dir` | `output_dir/_viz` | Directory for visualization output |

## Resumability

The pipeline skips images whose filename already exists in `--output-dir`, so you can safely restart after interruption.

## Performance

- ~26 ms/image on A10G GPU (YOLOv3 + HRNetv2 + classify + copy)
- 100K images in ~50–60 min
- ~500 MB VRAM (single worker, no multiprocessing needed)

## Recommended Workflow

1. **Calibrate** — Run with `--visualize 50` and inspect `_viz/` to verify landmark indices and ratio values.
2. **Adjust** — Tweak `--frontal-threshold` if too many profiles pass or too many frontals are rejected.
3. **Run** — Full pipeline with default or tuned threshold.
4. **Check stats** — The summary printed at the end shows frontal count, profile count, and frontal rate.
