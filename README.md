# Face Wash Pipeline

Batch filter for anime face images — keeps only **frontal faces** by measuring bilateral landmark symmetry across multiple facial regions using the existing ONNX landmark detector (YOLOv3 + HRNetv2, 28 keypoints).

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
  --device cuda \
  --num-readers 8 \
  --prefetch 32
```

If your ONNX files are stored outside the package default `onnx_models/`, pass the custom folder:

```bash
%sh cd /tmp/ani_face_detector && python -m landmark_detect.face_wash \
  --image-folder /dbfs/path/to/images \
  --output-dir /dbfs/path/to/frontal_output \
  --device cuda \
  --model-path /dbfs/path/to/onnx_models
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

# Use custom ONNX model directory
python -m landmark_detect.face_wash \
  --image-folder /path/to/images \
  --output-dir /path/to/frontal_output \
  --device cuda \
  --model-path /path/to/onnx_models
```

## How It Works

For each image the detector returns 28 keypoints. The frontal classifier computes bilateral symmetry ratios from **5 landmark pairs**, each measuring the distance from a left-side and right-side landmark to the nose tip:

| Pair | Left | Right |
|---|---|---|
| Eye centers | mean(keypoints[11:17]) | mean(keypoints[17:23]) |
| Eyebrow centers | mean(keypoints[5:8]) | mean(keypoints[8:11]) |
| Mouth corners | keypoints[24] | keypoints[26] |
| Outer contour | keypoints[0] | keypoints[4] |
| Inner contour | keypoints[1] | keypoints[3] |

For each pair: `ratio_i = min(d_left, d_right) / max(d_left, d_right)`.

The final score is the **median** of all valid ratios:

```
ratio = median(ratio_1, ratio_2, ..., ratio_5)
is_frontal = ratio >= threshold  (default 0.70)
```

A perfectly symmetric face gives ratio=1.0. Profile views pull one side closer to the nose, dropping the median well below 0.7. Using the median of 5 pairs makes the classifier robust to any single noisy keypoint.

Before computing symmetry, two early-exit gates reject degenerate detections:
- **Confidence gate** — rejects faces where eye/nose keypoint confidence is too low.
- **Eye span gate** — rejects faces where one eye region collapses (likely partial/occluded).

## Landmark Map

| Region | Indices | Bilateral pairs |
|---|---|---|
| Face contour | 0–4 | (0, 4) outer jaw, (1, 3) inner jaw |
| Left eyebrow | 5–7 | center vs right eyebrow center |
| Right eyebrow | 8–10 | center vs left eyebrow center |
| Left eye | 11–16 | center vs right eye center |
| Right eye | 17–22 | center vs left eye center |
| Nose tip | 23 | reference point |
| Mouth | 24–27 | (24, 26) left vs right corner |

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--image-folder` | *(required)* | Input image directory |
| `--output-dir` | *(required)* | Output directory for frontal images |
| `--device` | `cuda` | Inference device (`cuda` or `cpu`) |
| `--model-path` | package `onnx_models/` | Custom ONNX model directory; expects `face_detector_yolov3.onnx` and `landmark_hrnetv2.onnx` |
| `--frontal-threshold` | `0.70` | Min bilateral symmetry ratio (median of 5 pairs) to be frontal |
| `--face-threshold` | `0.5` | Min face detection score |
| `--kpt-confidence` | `0.3` | Min avg keypoint confidence for eyes/nose |
| `--eye-confidence` | `0.35` | Min per-eye avg keypoint confidence |
| `--eye-span-ratio-threshold` | `0.55` | Min ratio of smaller/larger eye span to reject partial eye |
| `--visualize N` | `0` | Save annotated images for first N files |
| `--viz-dir` | `output_dir/_viz` | Directory for visualization output |
| `--num-readers` | `8` | Background threads for image prefetch (DBFS I/O parallelism) |
| `--prefetch` | `32` | Sliding-window depth; ~96 MB of decoded images in-flight |

## Resumability

The pipeline skips images whose filename already exists in `--output-dir`, so you can safely restart after interruption.

## Performance

- ~26 ms/image on A10G GPU (YOLOv3 + HRNetv2 + classify + copy)
- 100K images in ~50–60 min
- ~500 MB VRAM (single worker, no multiprocessing needed)

### I/O pipelining (prefetch thread pool)

On DBFS (cloud object storage), `cv2.imread` incurs high network latency per image. Without prefetching, the GPU sits idle between images while the main thread waits for the next read to complete.

The pipeline uses a background `ThreadPoolExecutor` to read and decode images ahead of the GPU, keeping a sliding window of `--prefetch` futures in-flight at all times:

```
Threads 1–8:  [DBFS read + PNG decode] ──► prefetch queue (32 deep)
Main thread:                                      ▼ [GPU: YOLOv3] → [GPU: HRNetv2] → repeat
```

**Tuning:**
- `--num-readers` — match to available CPU cores (A10G nodes typically have 8–16). More threads hide DBFS latency better; diminishing returns above ~16 where PNG decode becomes the CPU bottleneck.
- `--prefetch 32` — ~96 MB decoded in-flight (32 × 1024×1024×3 bytes). Increase to 64 if readers still can't keep up; decrease if memory is tight.

## Recommended Workflow

1. **Calibrate** — Run with `--visualize 50` and inspect `_viz/` to verify landmark indices and ratio values.
2. **Adjust** — Tweak `--frontal-threshold` if too many profiles pass or too many frontals are rejected.
3. **Run** — Full pipeline with default or tuned threshold.
4. **Check stats** — The summary printed at the end shows frontal count, profile count, and frontal rate.
