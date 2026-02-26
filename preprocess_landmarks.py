"""
从 image_folder 读取图像，用 anime_face_detector 检测人脸并提取 landmark，
归一化后保存到 output_dir，供后续 MLP 使用。
每张图对应一个 .npy 文件，文件名与图像 stem 一致（便于与 label 按 id 对齐）。

三种 embedding 模式:
  1) landmark_to_embedding（默认）：keypoint 相对 bbox 归一化 + bbox 4 维，共 88 维。
  2) landmark_to_embedding_nose_relative（--use-nose-relative）：各点相对鼻尖( index 23 )的偏移，用 bbox 宽高归一化，共 56 维；不含 score（默认 landmark 完全置信），不使用与 bbox 的相对关系。
  3) --use-model-layer：pose 模型中间层（如 neck）输出，约 270 维。

从项目根目录运行（需在 witches-online-face 下）:
  python -m service.preprocess_landmarks --image-folder 图片目录 --output-dir landmark_embeddings_88
  python -m service.preprocess_landmarks --image-folder 图片目录 --output-dir landmark_embeddings_nose --use-nose-relative
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .anime_face_detector import create_detector

# 多进程 worker 内使用的全局 detector（每进程一个）
_detector = None


# 28 个 landmark，鼻尖为 0-based index 23
NOSE_TIP_INDEX = 23


def landmark_to_embedding(pred: dict, image_width: int, image_height: int) -> np.ndarray:
    """
    将单张图的一个检测结果转为固定长度向量。
    使用 bbox 归一化 keypoints：先减 bbox 左上角，再除以 bbox 宽高，得到 [0,1] 相对坐标；
    再拼上 bbox 中心与宽高归一化到 [0,1] 的 4 维，共 28*3 + 4 = 88 维。
    """
    bbox = pred['bbox']
    x0, y0, x1, y1 = bbox[:4]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)

    kpts = pred['keypoints'].astype(np.float32)
    rel_x = (kpts[:, 0] - x0) / bw
    rel_y = (kpts[:, 1] - y0) / bh
    scores = kpts[:, 2]

    flat = np.concatenate([rel_x, rel_y, scores], axis=0)
    box_norm = np.array([
        cx / image_width, cy / image_height,
        bw / image_width, bh / image_height
    ], dtype=np.float32)
    return np.concatenate([flat, box_norm], axis=0)


def landmark_to_embedding_nose_relative(pred: dict, image_width: int, image_height: int) -> np.ndarray:
    """
    将单张图的一个检测结果转为固定长度向量：各点相对鼻尖（index 23）的偏移，不用与 bbox 的相对关系。
    28 个 landmark，鼻尖为 index 23。对每个点：dx = (x_i - nose_x) / bw, dy = (y_i - nose_y) / bh，
    用 bbox 宽高做尺度归一化。不含 score（默认 landmark 完全置信）。共 28*2 = 56 维。
    """
    bbox = pred['bbox']
    x0, y0, x1, y1 = bbox[:4]
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)

    kpts = pred['keypoints'].astype(np.float32)
    nose_x = float(kpts[NOSE_TIP_INDEX, 0])
    nose_y = float(kpts[NOSE_TIP_INDEX, 1])

    rel_dx = (kpts[:, 0] - nose_x) / bw
    rel_dy = (kpts[:, 1] - nose_y) / bh

    return np.concatenate([rel_dx, rel_dy], axis=0).astype(np.float32)


def extract_embedding_from_model_layer(
    detector,
    image: np.ndarray | str | Path,
    face_score_threshold: float = 0.5,
    layer: str = 'neck',
) -> np.ndarray | None:
    """
    从 pose 模型预测时取指定中间层（如 neck / backbone）输出作为 embedding。
    使用 detector.predict_with_embedding(image, layer=layer)，取第一张脸的
    该层输出（经 GAP 后的向量）；无人脸或分数过低时返回 None。
    """
    preds = detector.predict_with_embedding(image, layer=layer)
    if not preds:
        return None
    best = preds[0]
    if best['bbox'].size >= 5 and best['bbox'][4] < face_score_threshold:
        return None
    emb = best.get('embedding')
    if emb is None:
        return None
    if hasattr(emb, 'cpu'):
        emb = emb.cpu().numpy().astype(np.float32).flatten()
    return np.asarray(emb, dtype=np.float32)


def _init_worker(device: str) -> None:
    """多进程 worker 初始化：每进程创建一次 detector；屏蔽 worker 内 log 避免冲刷主进程 tqdm。"""
    global _detector
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    _detector = create_detector('yolov3', device=device)


def _process_one(
    args: tuple,
) -> int:
    """
    处理单张图：读图、检测、提 embedding、保存 .npy。
    args: (path_str, output_dir_str, face_score_threshold, use_model_layer, embedding_layer, use_nose_relative)
    返回 1 表示保存成功，0 表示跳过。
    """
    global _detector
    path_str, output_dir_str, face_score_threshold, use_model_layer, embedding_layer, use_nose_relative = args
    path = Path(path_str)
    output_dir = Path(output_dir_str)
    img = cv2.imread(path_str)
    if img is None:
        return 0
    h, w = img.shape[:2]
    if use_model_layer:
        emb = extract_embedding_from_model_layer(
            _detector, img,
            face_score_threshold=face_score_threshold,
            layer=embedding_layer,
        )
    else:
        preds = _detector(img)
        if not preds:
            return 0
        best = preds[0]
        if best['bbox'].size >= 5 and best['bbox'][4] < face_score_threshold:
            return 0
        if use_nose_relative:
            emb = landmark_to_embedding_nose_relative(best, w, h)
        else:
            emb = landmark_to_embedding(best, w, h)
    if emb is None:
        return 0
    out_path = output_dir / f"{path.stem}.npy"
    np.save(out_path, emb)
    return 1


def run(
    image_folder: str | Path,
    output_dir: str | Path,
    device: str = 'cpu',
    face_score_threshold: float = 0.5,
    use_model_layer: bool = False,
    embedding_layer: str = 'neck',
    use_nose_relative: bool = False,
    workers: int = 1,
):
    """
    use_model_layer=True 时，embedding 取自 pose 模型的指定中间层（默认 neck）；
    use_nose_relative=True 时，使用各点相对鼻尖的 embedding（56 维）；
    否则沿用 landmark_to_embedding（88 维，相对 bbox）。
    workers>1 时使用多进程并行；device=cuda 时建议 workers=1 以免显存不足。
    """
    image_folder = Path(image_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in image_folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
    )
    if not image_paths:
        return 0

    # 跳过已存在 .npy 的图片，不再重复提取
    n_total = len(image_paths)
    image_paths = [p for p in image_paths if not (output_dir / f"{p.stem}.npy").exists()]
    n_already = n_total - len(image_paths)
    if n_already > 0:
        print(f"Already done: {n_already}, remaining: {len(image_paths)}", file=sys.stderr)
    if not image_paths:
        print("All images already have embeddings, nothing to do.", file=sys.stderr)
        return 0

    if workers <= 1:
        detector = create_detector('yolov3', device=device)
        saved = 0
        _stdout, _stderr = sys.stdout, sys.stderr
        sys.stdout = open(os.devnull, 'w')  # 屏蔽其他 print，避免冲刷 tqdm
        try:
            pbar = tqdm(image_paths, desc='preprocess', unit='img', file=_stderr)
            for path in pbar:
                img = cv2.imread(str(path))
                if img is None:
                    pbar.set_postfix(saved=saved)
                    continue
                h, w = img.shape[:2]
                if use_model_layer:
                    emb = extract_embedding_from_model_layer(
                        detector, img,
                        face_score_threshold=face_score_threshold,
                        layer=embedding_layer,
                    )
                else:
                    preds = detector(img)
                    if not preds:
                        pbar.set_postfix(saved=saved)
                        continue
                    best = preds[0]
                    if best['bbox'].size >= 5 and best['bbox'][4] < face_score_threshold:
                        pbar.set_postfix(saved=saved)
                        continue
                    emb = landmark_to_embedding_nose_relative(best, w, h) if use_nose_relative else landmark_to_embedding(best, w, h)
                if emb is None:
                    pbar.set_postfix(saved=saved)
                    continue
                out_path = output_dir / f"{path.stem}.npy"
                np.save(out_path, emb)
                saved += 1
                pbar.set_postfix(saved=saved)
        finally:
            sys.stdout = _stdout
        return saved

    tasks = [
        (str(p), str(output_dir), face_score_threshold, use_model_layer, embedding_layer, use_nose_relative)
        for p in image_paths
    ]
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = open(os.devnull, 'w')  # 屏蔽主进程内其他 print，避免冲刷 tqdm
    pool = Pool(
        workers,
        initializer=_init_worker,
        initargs=(device,),
    )
    try:
        results = list(tqdm(
            pool.imap_unordered(_process_one, tasks, chunksize=min(32, max(1, len(tasks) // (workers * 4)))),
            total=len(tasks),
            desc='preprocess',
            unit='img',
            file=_stderr,
        ))
        n_saved = sum(results)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.stdout = _stdout
        print('Interrupted by user (Ctrl+C)', file=sys.stderr)
        sys.exit(130)
    finally:
        sys.stdout = _stdout
        pool.close()
        pool.join()
    return n_saved


def main():
    parser = argparse.ArgumentParser(description='Extract landmark embeddings from image_folder')
    parser.add_argument('--image-folder', type=str, default='D:/witches-img-new',
                        help='Input image directory')
    parser.add_argument('--output-dir', type=str, default='landmark_embeddings_new_nose',
                        help='Output directory for .npy embeddings')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--face-threshold', type=float, default=0.5)
    parser.add_argument('--use-model-layer', action='store_true',
                        help='Use pose model intermediate layer (e.g. neck) as embedding instead of landmark vector')
    parser.add_argument('--use-nose-relative', action='store_true',
                        help='Use landmark relative to nose tip (index 23): 56-dim, no score, no bbox-relative coords')
    parser.add_argument('--embedding-layer', type=str, default='neck',
                        help='Layer name to extract (neck or backbone). Used when --use-model-layer')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of processes for parallel extraction (default: 4). Use 1 for single-thread. GPU 时建议 1 以免显存不足')
    parser.add_argument('--delay', type=float, default=0,
                        help='Delay start by N seconds (e.g. 3600 = 1 hour). Default: 0')
    args = parser.parse_args()

    if args.delay > 0:
        print(f'Waiting {args.delay}s ({args.delay / 3600:.1f}h) before starting...', file=sys.stderr)
        time.sleep(args.delay)
        img_dir = Path(args.image_folder)
        if img_dir.exists():
            n_img = len([p for p in img_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')])
            print(f'Delay over. Reading folder (now): {img_dir} ({n_img} images)', file=sys.stderr)
        else:
            print(f'Delay over. Folder: {img_dir}', file=sys.stderr)
    n = run(
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        device=args.device,
        face_score_threshold=args.face_threshold,
        use_model_layer=args.use_model_layer,
        embedding_layer=args.embedding_layer,
        use_nose_relative=args.use_nose_relative,
        workers=args.workers,
    )
    print(f"Saved {n} embeddings to {args.output_dir}")


if __name__ == '__main__':
    main()
