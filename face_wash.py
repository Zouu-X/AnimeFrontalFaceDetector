"""
Batch pipeline to filter anime face images, keeping only frontal faces.

Uses the existing ONNX landmark detector (YOLOv3 + HRNetv2, 28 keypoints)
to classify frontal vs profile by comparing left/right eye-to-nose distance
symmetry. Frontal faces are copied to the output directory.

Usage (from project root):
  python -m landmark_detect.face_wash --image-folder /path/to/images --output-dir /path/to/output --device cuda
  python -m landmark_detect.face_wash --image-folder /path/to/images --output-dir /path/to/output --visualize 50
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from anime_face_detector import create_detector

# ── Landmark indices (hysts/anime-face-detector, 28 keypoints) ──────────────
# Face contour: 0-4, Left eyebrow: 5-7, Right eyebrow: 8-10
# Left eye: 11-16, Right eye: 17-22, Nose tip: 23, Mouth: 24-27
LEFT_EYE_INDICES = range(11, 17)
RIGHT_EYE_INDICES = range(17, 23)
NOSE_TIP_INDEX = 23

# ── Default thresholds ──────────────────────────────────────────────────────
DEFAULT_FRONTAL_THRESHOLD = 0.70
DEFAULT_FACE_SCORE_THRESHOLD = 0.5
DEFAULT_KPT_CONFIDENCE = 0.3
DEFAULT_EYE_CONFIDENCE = 0.35
DEFAULT_EYE_SPAN_RATIO_THRESHOLD = 0.55

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


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

    left_eye_center = left_eye_kpts[:, :2].mean(axis=0)
    right_eye_center = right_eye_kpts[:, :2].mean(axis=0)
    nose = nose_kpt[:2]

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

    d_left = np.linalg.norm(left_eye_center - nose)
    d_right = np.linalg.norm(right_eye_center - nose)

    if max(d_left, d_right) < 1e-6:
        return False, 0.0

    ratio = min(d_left, d_right) / max(d_left, d_right)
    is_frontal = ratio >= threshold
    return is_frontal, float(ratio)


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
    frontal_threshold: float = DEFAULT_FRONTAL_THRESHOLD,
    face_score_threshold: float = DEFAULT_FACE_SCORE_THRESHOLD,
    kpt_confidence: float = DEFAULT_KPT_CONFIDENCE,
    eye_confidence: float = DEFAULT_EYE_CONFIDENCE,
    eye_span_ratio_threshold: float = DEFAULT_EYE_SPAN_RATIO_THRESHOLD,
    visualize: int = 0,
    viz_dir: str | Path | None = None,
) -> dict:
    """
    Filter anime face images, keeping only frontal faces.

    Returns dict with stats: total, skipped_existing, no_face, low_score,
    low_confidence, partial_eye, profile, frontal.
    """
    image_folder = Path(image_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if viz_dir is None:
        viz_dir = output_dir / '_viz'
    else:
        viz_dir = Path(viz_dir)

    # Discover images
    image_paths = sorted(
        [p for p in image_folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if not image_paths:
        print("No images found.", file=sys.stderr)
        return {'total': 0}

    # Skip already-copied for resumability
    n_total = len(image_paths)
    image_paths = [p for p in image_paths if not (output_dir / p.name).exists()]
    n_already = n_total - len(image_paths)
    if n_already > 0:
        print(f"Already copied: {n_already}, remaining: {len(image_paths)}", file=sys.stderr)
    if not image_paths:
        print("All images already processed, nothing to do.", file=sys.stderr)
        return {'total': n_total, 'skipped_existing': n_already}

    # Init detector
    detector = create_detector('yolov3', device=device)

    stats = {
        'total': n_total,
        'skipped_existing': n_already,
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
    try:
        pbar = tqdm(image_paths, desc='face_wash', unit='img', file=_stderr)
        for path in pbar:
            img = cv2.imread(str(path))
            if img is None:
                stats['no_face'] += 1
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue

            preds = detector(img)
            if not preds:
                stats['no_face'] += 1
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue

            best = preds[0]
            if best['bbox'].size >= 5 and best['bbox'][4] < face_score_threshold:
                stats['low_score'] += 1
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
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue
            if ratio == -2.0:
                stats['partial_eye'] += 1
                pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
                continue

            # Visualize first N images for calibration
            if visualize > 0 and viz_count < visualize:
                visualize_landmarks(img, kpts, is_frontal, ratio, path.stem, viz_dir)
                viz_count += 1

            if is_frontal:
                stats['frontal'] += 1
                shutil.copy2(str(path), str(output_dir / path.name))
            else:
                stats['profile'] += 1

            pbar.set_postfix(f=stats['frontal'], p=stats['profile'])
    except KeyboardInterrupt:
        sys.stdout = _stdout
        print('\nInterrupted by user (Ctrl+C)', file=sys.stderr)
        _print_stats(stats)
        sys.exit(130)
    finally:
        sys.stdout = _stdout

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
    args = parser.parse_args()

    stats = run(
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        device=args.device,
        frontal_threshold=args.frontal_threshold,
        face_score_threshold=args.face_threshold,
        kpt_confidence=args.kpt_confidence,
        eye_confidence=args.eye_confidence,
        eye_span_ratio_threshold=args.eye_span_ratio_threshold,
        visualize=args.visualize,
        viz_dir=args.viz_dir,
    )
    _print_stats(stats)


if __name__ == '__main__':
    main()
