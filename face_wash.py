"""
Batch pipeline to classify anime face images and write a regeneration to-do list
for AnimeDiffusion.

Uses the existing ONNX landmark detector (YOLOv3 + HRNetv2, 28 keypoints)
to classify frontal vs profile by comparing left/right eye-to-nose distance
symmetry. Failed (non-frontal) images are written to a JSONL manifest so
AnimeDiffusion knows which images to regenerate as frontal faces.

Usage (from project root):
  python -m landmark_detect.face_wash --image-folder /path/to/images --output-dir /path/to/output --device cuda
  python -m landmark_detect.face_wash --image-folder /path/to/images --output-dir /path/to/output --visualize 50
  python -m landmark_detect.face_wash --image-folder /path/to/images --output-dir /path/to/output --model-path /path/to/onnx_models
"""
import argparse
import collections
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from anime_face_detector import create_detector, create_detector_onnx

# ── Landmark indices (hysts/anime-face-detector, 28 keypoints) ──────────────
# Face contour: 0-4, Left eyebrow: 5-7, Right eyebrow: 8-10
# Left eye: 11-16, Right eye: 17-22, Nose tip: 23, Mouth: 24-27
LEFT_EYE_INDICES = range(11, 17)
RIGHT_EYE_INDICES = range(17, 23)
NOSE_TIP_INDEX = 23
LEFT_EYEBROW_INDICES = range(5, 8)
RIGHT_EYEBROW_INDICES = range(8, 11)
MOUTH_LEFT_INDEX = 24
MOUTH_RIGHT_INDEX = 26
CONTOUR_PAIRS = [(0, 4), (1, 3)]  # bilateral jaw pairs

# ── Default thresholds ──────────────────────────────────────────────────────
DEFAULT_FRONTAL_THRESHOLD = 0.70
DEFAULT_FACE_SCORE_THRESHOLD = 0.5
DEFAULT_KPT_CONFIDENCE = 0.3
DEFAULT_EYE_CONFIDENCE = 0.35
DEFAULT_EYE_SPAN_RATIO_THRESHOLD = 0.55

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def _prefetch_images(image_paths, n_workers: int = 8, prefetch: int = 32):
    """Yield (path, img) pairs, reading images in background threads.

    Uses a sliding window of futures so at most `prefetch` images are
    in-flight at once, bounding peak memory use.
    """
    def _load(path):
        return path, cv2.imread(str(path))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        it = iter(image_paths)
        window = collections.deque()
        for path in itertools.islice(it, prefetch):
            window.append(pool.submit(_load, path))
        while window:
            path, img = window.popleft().result()
            yield path, img
            try:
                window.append(pool.submit(_load, next(it)))
            except StopIteration:
                pass


def classify_frontal(
    keypoints: np.ndarray,
    threshold: float = DEFAULT_FRONTAL_THRESHOLD,
    kpt_confidence: float = DEFAULT_KPT_CONFIDENCE,
    eye_confidence: float = DEFAULT_EYE_CONFIDENCE,
    eye_span_ratio_threshold: float = DEFAULT_EYE_SPAN_RATIO_THRESHOLD,
) -> tuple[bool, float]:
    """
    Classify whether a face is frontal based on eye-nose distance symmetry.

    Args:
        keypoints: (N, 3) array of [x, y, score] per keypoint.
        threshold: Minimum ratio of min(d_left, d_right)/max(...) to be frontal.
        kpt_confidence: Minimum average confidence for eye/nose keypoints.
        eye_confidence: Minimum confidence for each eye (left/right) independently.
        eye_span_ratio_threshold: Minimum ratio between smaller/larger eye span.

    Returns:
        (is_frontal, ratio)
          - ratio == -1.0: keypoints confidence too low
          - ratio == -2.0: eye geometry indicates partial/occluded eye
    """
    left_eye_kpts = keypoints[list(LEFT_EYE_INDICES)]
    right_eye_kpts = keypoints[list(RIGHT_EYE_INDICES)]
    nose_kpt = keypoints[NOSE_TIP_INDEX]

    # Check confidence of relevant keypoints
    relevant_scores = np.concatenate([
        left_eye_kpts[:, 2], right_eye_kpts[:, 2], [nose_kpt[2]]
    ])
    if relevant_scores.mean() < kpt_confidence:
        return False, -1.0
    left_eye_conf = float(left_eye_kpts[:, 2].mean())
    right_eye_conf = float(right_eye_kpts[:, 2].mean())
    if left_eye_conf < eye_confidence or right_eye_conf < eye_confidence:
        return False, -1.0

    # Reject likely partial-eye/occluded-eye predictions: one eye region collapses.
    left_eye_span = left_eye_kpts[:, :2].max(axis=0) - left_eye_kpts[:, :2].min(axis=0)
    right_eye_span = right_eye_kpts[:, :2].max(axis=0) - right_eye_kpts[:, :2].min(axis=0)
    left_eye_size = float(np.linalg.norm(left_eye_span))
    right_eye_size = float(np.linalg.norm(right_eye_span))
    if max(left_eye_size, right_eye_size) < 1e-6:
        return False, -2.0
    eye_size_ratio = min(left_eye_size, right_eye_size) / max(left_eye_size, right_eye_size)
    if eye_size_ratio < eye_span_ratio_threshold:
        return False, -2.0

    # Compute bilateral symmetry ratios across all available pairs.
    nose = nose_kpt[:2]
    pairs = []

    # Eye centers to nose (existing logic, now one of many)
    left_eye_center = left_eye_kpts[:, :2].mean(axis=0)
    right_eye_center = right_eye_kpts[:, :2].mean(axis=0)
    pairs.append((np.linalg.norm(left_eye_center - nose),
                  np.linalg.norm(right_eye_center - nose)))

    # Eyebrow centers to nose
    left_brow = keypoints[list(LEFT_EYEBROW_INDICES), :2].mean(axis=0)
    right_brow = keypoints[list(RIGHT_EYEBROW_INDICES), :2].mean(axis=0)
    pairs.append((np.linalg.norm(left_brow - nose),
                  np.linalg.norm(right_brow - nose)))

    # Mouth corners to nose
    mouth_left = keypoints[MOUTH_LEFT_INDEX, :2]
    mouth_right = keypoints[MOUTH_RIGHT_INDEX, :2]
    pairs.append((np.linalg.norm(mouth_left - nose),
                  np.linalg.norm(mouth_right - nose)))

    # Contour pairs to nose
    for li, ri in CONTOUR_PAIRS:
        dl = np.linalg.norm(keypoints[li, :2] - nose)
        dr = np.linalg.norm(keypoints[ri, :2] - nose)
        pairs.append((dl, dr))

    # Compute per-pair ratios, skip degenerate pairs
    ratios = []
    for dl, dr in pairs:
        dmax = max(dl, dr)
        if dmax < 1e-6:
            continue
        ratios.append(min(dl, dr) / dmax)

    if not ratios:
        return False, 0.0

    ratio = float(np.median(ratios))
    is_frontal = ratio >= threshold
    return is_frontal, ratio


# ── Landmark region colors (BGR) ────────────────────────────────────────────
_REGION_COLORS = {
    'contour':       (180, 180, 180),  # gray
    'left_eyebrow':  (0, 200, 200),    # yellow
    'right_eyebrow': (200, 200, 0),    # cyan
    'left_eye':      (0, 255, 0),      # green
    'right_eye':     (255, 0, 0),      # blue
    'nose':          (0, 0, 255),      # red
    'mouth':         (255, 0, 255),    # magenta
}


def _kpt_region(idx: int) -> str:
    if idx <= 4:
        return 'contour'
    if idx <= 7:
        return 'left_eyebrow'
    if idx <= 10:
        return 'right_eyebrow'
    if idx <= 16:
        return 'left_eye'
    if idx <= 22:
        return 'right_eye'
    if idx == 23:
        return 'nose'
    return 'mouth'


def visualize_landmarks(
    image: np.ndarray,
    keypoints: np.ndarray,
    is_frontal: bool,
    ratio: float,
    name: str,
    output_dir: Path,
) -> None:
    """Draw color-coded landmarks, eye centers, nose, and frontal label onto image and save."""
    vis = image.copy()
    h, w = vis.shape[:2]
    scale = max(1, min(h, w) // 256)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35 * scale
    thickness = max(1, scale)

    # Draw each keypoint with index number
    for i, (x, y, score) in enumerate(keypoints):
        color = _REGION_COLORS[_kpt_region(i)]
        cx, cy = int(x), int(y)
        cv2.circle(vis, (cx, cy), 2 * scale, color, -1)
        cv2.putText(vis, str(i), (cx + 3 * scale, cy - 3 * scale),
                    font, font_scale, color, thickness)

    # Draw eye centers and nose tip
    left_eye_center = keypoints[list(LEFT_EYE_INDICES), :2].mean(axis=0).astype(int)
    right_eye_center = keypoints[list(RIGHT_EYE_INDICES), :2].mean(axis=0).astype(int)
    nose = keypoints[NOSE_TIP_INDEX, :2].astype(int)

    marker_size = 4 * scale
    cv2.drawMarker(vis, tuple(left_eye_center), (0, 255, 0), cv2.MARKER_CROSS, marker_size, thickness)
    cv2.drawMarker(vis, tuple(right_eye_center), (255, 0, 0), cv2.MARKER_CROSS, marker_size, thickness)
    cv2.drawMarker(vis, tuple(nose), (0, 0, 255), cv2.MARKER_CROSS, marker_size, thickness)

    # Draw lines from eye centers to nose
    cv2.line(vis, tuple(left_eye_center), tuple(nose), (0, 255, 0), thickness)
    cv2.line(vis, tuple(right_eye_center), tuple(nose), (255, 0, 0), thickness)

    # Annotate ratio and label
    label = "FRONTAL" if is_frontal else "PROFILE"
    color = (0, 200, 0) if is_frontal else (0, 0, 200)
    text = f"{label}  ratio={ratio:.3f}"
    cv2.putText(vis, text, (5, 20 * scale), font, font_scale * 1.2, color, thickness + 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"viz_{name}.png"), vis)


def run(
    image_folder: str | Path,
    output_dir: str | Path,
    device: str = 'cuda',
    require_cuda: bool = False,
    model_path: str | Path | None = None,
    frontal_threshold: float = DEFAULT_FRONTAL_THRESHOLD,
    face_score_threshold: float = DEFAULT_FACE_SCORE_THRESHOLD,
    kpt_confidence: float = DEFAULT_KPT_CONFIDENCE,
    eye_confidence: float = DEFAULT_EYE_CONFIDENCE,
    eye_span_ratio_threshold: float = DEFAULT_EYE_SPAN_RATIO_THRESHOLD,
    visualize: int = 0,
    viz_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    num_readers: int = 8,
    prefetch: int = 32,
) -> dict:
    """
    Classify anime face images and write a JSONL regeneration to-do list for AnimeDiffusion.

    Only failed (non-frontal) images are written to the manifest. Passed images are not
    recorded. AnimeDiffusion uses sample_id to locate each file in its known source folder.

    Returns dict with stats: total, no_face, low_score, low_confidence, partial_eye,
    profile, frontal.
    """
    image_folder = Path(image_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if viz_dir is None:
        viz_dir = output_dir / '_viz'
    else:
        viz_dir = Path(viz_dir)
    if manifest_path is None:
        manifest_path = output_dir / 'clean_manifest.jsonl'
    else:
        manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover images
    image_paths = sorted(
        [p for p in image_folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if not image_paths:
        print("No images found.", file=sys.stderr)
        return {'total': 0}

    n_total = len(image_paths)

    # Init detector
    if model_path is None:
        detector = create_detector('yolov3', device=device, require_cuda=require_cuda)
    else:
        onnx_dir = Path(model_path)
        detector = create_detector_onnx(
            face_name='yolov3',
            face_onnx_path=onnx_dir / 'face_detector_yolov3.onnx',
            landmark_onnx_path=onnx_dir / 'landmark_hrnetv2.onnx',
            device=device,
            require_cuda=require_cuda,
        )
    if hasattr(detector, 'get_runtime_info'):
        info = detector.get_runtime_info()
        print(
            f"Detector runtime providers: face={info.get('face_providers')}, "
            f"landmark={info.get('landmark_providers')}",
            file=sys.stderr,
        )

    stats = {
        'total': n_total,
        'no_face': 0,
        'low_score': 0,
        'low_confidence': 0,
        'partial_eye': 0,
        'profile': 0,
        'frontal': 0,
    }
    viz_count = 0

    # Suppress stdout (detector prints), keep stderr for tqdm
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = open(os.devnull, 'w')
    manifest_file = manifest_path.open('w', encoding='utf-8')
    try:
        pbar = tqdm(total=n_total, desc='face_wash', unit='img', file=_stderr)
        for path, img in _prefetch_images(image_paths, n_workers=num_readers, prefetch=prefetch):
            sample_id = path.stem
            pbar.update(1)
            if img is None:
                stats['no_face'] += 1
                manifest_file.write(json.dumps({'sample_id': sample_id, 'ratio': None}) + '\n')
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue

            preds = detector(img)
            if not preds:
                stats['no_face'] += 1
                manifest_file.write(json.dumps({'sample_id': sample_id, 'ratio': None}) + '\n')
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue

            best = preds[0]
            if best['bbox'].size >= 5 and best['bbox'][4] < face_score_threshold:
                stats['low_score'] += 1
                manifest_file.write(json.dumps({'sample_id': sample_id, 'ratio': None}) + '\n')
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue

            kpts = best['keypoints'].astype(np.float32)
            is_frontal, ratio = classify_frontal(
                kpts,
                frontal_threshold,
                kpt_confidence,
                eye_confidence,
                eye_span_ratio_threshold,
            )

            if ratio == -1.0:
                stats['low_confidence'] += 1
                manifest_file.write(json.dumps({'sample_id': sample_id, 'ratio': ratio}) + '\n')
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue
            if ratio == -2.0:
                stats['partial_eye'] += 1
                manifest_file.write(json.dumps({'sample_id': sample_id, 'ratio': ratio}) + '\n')
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue

            # Visualize first N images for calibration
            if visualize > 0 and viz_count < visualize:
                visualize_landmarks(img, kpts, is_frontal, ratio, path.stem, viz_dir)
                viz_count += 1

            if is_frontal:
                stats['frontal'] += 1
            else:
                stats['profile'] += 1
                manifest_file.write(json.dumps({'sample_id': sample_id, 'ratio': ratio}) + '\n')

            pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
    except KeyboardInterrupt:
        sys.stdout = _stdout
        print('\nInterrupted by user (Ctrl+C)', file=sys.stderr)
        _print_stats(stats)
        sys.exit(130)
    finally:
        manifest_file.close()
        sys.stdout = _stdout

    stats['manifest_path'] = str(manifest_path)
    return stats


def _print_stats(stats: dict) -> None:
    print("\n=== Face Wash Summary ===", file=sys.stderr)
    for key, val in stats.items():
        print(f"  {key:20s}: {val}", file=sys.stderr)
    total_processed = stats.get('frontal', 0) + stats.get('profile', 0)
    if total_processed > 0:
        pct = stats['frontal'] / total_processed * 100
        print(f"  {'frontal_rate':20s}: {pct:.1f}%", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Filter anime face images, keeping only frontal faces.'
    )
    parser.add_argument('--image-folder', type=str, required=True,
                        help='Input image directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for frontal face images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference (default: cuda)')
    parser.add_argument('--require-cuda', action='store_true',
                        help='Fail fast if CUDAExecutionProvider is not active for ONNX sessions')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Custom ONNX model directory (expects face_detector_yolov3.onnx and landmark_hrnetv2.onnx)')
    parser.add_argument('--frontal-threshold', type=float, default=DEFAULT_FRONTAL_THRESHOLD,
                        help=f'Min eye-nose distance ratio for frontal (default: {DEFAULT_FRONTAL_THRESHOLD})')
    parser.add_argument('--face-threshold', type=float, default=DEFAULT_FACE_SCORE_THRESHOLD,
                        help=f'Min face detection score (default: {DEFAULT_FACE_SCORE_THRESHOLD})')
    parser.add_argument('--kpt-confidence', type=float, default=DEFAULT_KPT_CONFIDENCE,
                        help=f'Min avg keypoint confidence for eye/nose (default: {DEFAULT_KPT_CONFIDENCE})')
    parser.add_argument('--eye-confidence', type=float, default=DEFAULT_EYE_CONFIDENCE,
                        help=f'Min per-eye avg keypoint confidence (default: {DEFAULT_EYE_CONFIDENCE})')
    parser.add_argument('--eye-span-ratio-threshold', type=float, default=DEFAULT_EYE_SPAN_RATIO_THRESHOLD,
                        help=f'Min ratio of smaller/larger eye size to reject partial eye (default: {DEFAULT_EYE_SPAN_RATIO_THRESHOLD})')
    parser.add_argument('--visualize', type=int, default=0, metavar='N',
                        help='Save annotated landmark images for first N files (for calibration)')
    parser.add_argument('--viz-dir', type=str, default=None,
                        help='Directory for visualization output (default: output_dir/_viz)')
    parser.add_argument('--manifest-path', type=str, default=None,
                        help='Path to clean_manifest.jsonl (default: output_dir/clean_manifest.jsonl)')
    parser.add_argument('--num-readers', type=int, default=8,
                        help='Number of background threads for image prefetch (default: 8)')
    parser.add_argument('--prefetch', type=int, default=32,
                        help='Sliding window size for prefetch (default: 32)')
    args = parser.parse_args()

    stats = run(
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        device=args.device,
        require_cuda=args.require_cuda,
        model_path=args.model_path,
        frontal_threshold=args.frontal_threshold,
        face_score_threshold=args.face_threshold,
        kpt_confidence=args.kpt_confidence,
        eye_confidence=args.eye_confidence,
        eye_span_ratio_threshold=args.eye_span_ratio_threshold,
        visualize=args.visualize,
        viz_dir=args.viz_dir,
        manifest_path=args.manifest_path,
        num_readers=args.num_readers,
        prefetch=args.prefetch,
    )
    _print_stats(stats)


if __name__ == '__main__':
    main()
