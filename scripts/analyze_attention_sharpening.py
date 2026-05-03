#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


def parse_layer_list(value: str) -> List[int]:
    value = value.strip()
    if "-" in value:
        start_s, end_s = value.split("-", 1)
        return list(range(int(start_s), int(end_s) + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def parse_start_end(path: str) -> Tuple[Optional[int], Optional[int]]:
    base = os.path.basename(path)
    parts = base.replace(".npy", "").split("_")
    start = None
    end = None
    for part in parts:
        if part.startswith("start"):
            try:
                start = int(part.replace("start", ""))
            except ValueError:
                start = None
        if part.startswith("end"):
            try:
                end = int(part.replace("end", ""))
            except ValueError:
                end = None
    return start, end


def choose_attn_file(sample_dir: str, mode: str, layer: int) -> Optional[str]:
    pattern = os.path.join(sample_dir, f"{mode}_{layer}_start*_end*.npy")
    matches = sorted(glob(pattern))
    if not matches:
        return None
    best_path = None
    best_span = -1
    for path in matches:
        start, end = parse_start_end(path)
        if start is None or end is None:
            continue
        if start < 0 or end < 0 or end < start:
            continue
        span = end - start
        if span > best_span:
            best_span = span
            best_path = path
    if best_path is not None:
        return best_path
    return matches[0]


def load_controlled_images_json(dataset: str, root_dir: str) -> str:
    if dataset == "Controlled_Images_A":
        return os.path.join(root_dir, "controlled_images_dataset.json")
    if dataset == "Controlled_Images_B":
        return os.path.join(root_dir, "controlled_clevr_dataset.json")
    raise ValueError(f"Unsupported dataset {dataset} for YOLO analysis.")


def load_image_path(dataset: str, root_dir: str, index: int) -> Optional[str]:
    annotation_path = load_controlled_images_json(dataset, root_dir)
    if not os.path.exists(annotation_path):
        return None
    with open(annotation_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    if index < 0 or index >= len(data):
        return None
    image_path = data[index].get("image_path")
    if not image_path:
        return None
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    if os.path.exists(image_path):
        return image_path
    candidate = os.path.join(root_dir, image_path)
    if os.path.exists(candidate):
        return candidate
    return image_path


def load_correct_map(res_path: str, dataset: str, method: str) -> Dict[int, bool]:
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Missing results file: {res_path}")
    last = None
    with open(res_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("dataset") == dataset and entry.get("method") == method:
                last = entry
    if not last:
        raise ValueError(f"No entries for dataset={dataset}, method={method} in {res_path}")
    correct_ids = set(last.get("correct_id", []))
    return {int(idx): (int(idx) in correct_ids) for idx in correct_ids}


def load_attention_vector(attn_path: str, head: str) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    attn = np.load(attn_path)
    if attn.ndim == 3:
        if attn.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for attention array, got shape {attn.shape}")
        attn = attn[0]
    if attn.ndim != 2:
        raise ValueError(f"Expected 2D attention array, got shape {attn.shape}")
    if head == "mean":
        vec = attn.mean(axis=0)
    elif head == "max":
        vec = attn.max(axis=0)
    else:
        head_idx = int(head)
        if head_idx < 0 or head_idx >= attn.shape[0]:
            raise ValueError(f"Head index {head_idx} out of range for shape {attn.shape}")
        vec = attn[head_idx]
    start, end = parse_start_end(attn_path)
    if start is not None and end is not None and start >= 0 and end >= start:
        end_inclusive = min(end + 1, vec.shape[-1])
        start_clamped = min(max(start, 0), vec.shape[-1])
        vec = vec[start_clamped:end_inclusive]
    return vec, start, end


def normalize_vector(vec: np.ndarray, use_minmax: bool) -> np.ndarray:
    if use_minmax:
        vmin = float(vec.min())
        vmax = float(vec.max())
        if vmax - vmin > 1e-8:
            vec = (vec - vmin) / (vmax - vmin)
        else:
            vec = np.zeros_like(vec)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def mask_from_boxes(boxes: List[List[float]], size: Tuple[int, int], grid: Tuple[int, int]) -> np.ndarray:
    width, height = size
    grid_h, grid_w = grid
    mask = np.zeros((grid_h, grid_w), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        start_col = int((x1 / width) * grid_w)
        end_col = int(np.ceil((x2 / width) * grid_w)) - 1
        start_row = int((y1 / height) * grid_h)
        end_row = int(np.ceil((y2 / height) * grid_h)) - 1
        start_col = max(0, min(grid_w - 1, start_col))
        end_col = max(0, min(grid_w - 1, end_col))
        start_row = max(0, min(grid_h - 1, start_row))
        end_row = max(0, min(grid_h - 1, end_row))
        mask[start_row : end_row + 1, start_col : end_col + 1] = 1.0
    return mask


def load_or_run_yolo(
    image_path: str,
    cache: Dict[str, List[List[float]]],
    model,
    conf: float,
    iou: float,
    class_ids: Optional[List[int]],
) -> List[List[float]]:
    if image_path in cache:
        return cache[image_path]
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)
    if not results:
        cache[image_path] = []
        return []
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        cache[image_path] = []
        return []
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    if class_ids is not None and boxes.cls is not None:
        classes = boxes.cls.cpu().numpy().tolist()
        filtered = []
        for box, cls_id in zip(xyxy, classes):
            if int(cls_id) in class_ids:
                filtered.append(box)
        xyxy = filtered
    cache[image_path] = xyxy
    return xyxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--attn-dir", required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--mode", choices=["pre", "post", "diff"], default="post")
    parser.add_argument("--head", default="mean", help="Head index, 'mean', or 'max'")
    parser.add_argument("--layers", default="12-20")
    parser.add_argument("--grid", nargs=2, type=int, default=[24, 24])
    parser.add_argument("--res", default="./outputs/res.json")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-classes", default=None, help="Comma-separated class ids to keep")
    parser.add_argument("--yolo-cache", default=None)
    parser.add_argument("--out-prefix", default="./output/attn_overlap")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Missing ultralytics. Install with: pip install ultralytics") from exc

    if not os.path.isdir(args.attn_dir):
        raise FileNotFoundError(f"Attention directory not found: {args.attn_dir}")

    layer_list = parse_layer_list(args.layers)
    grid = (int(args.grid[0]), int(args.grid[1]))
    class_ids = None
    if args.yolo_classes:
        class_ids = [int(x) for x in args.yolo_classes.split(",") if x.strip()]

    if args.yolo_cache:
        cache_path = args.yolo_cache
    else:
        cache_path = f"./output/yolo_boxes_{args.dataset}.json"

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fin:
            yolo_cache = json.load(fin)
    else:
        yolo_cache = {}

    yolo_model = YOLO(args.yolo_model)
    correct_map = load_correct_map(args.res, args.dataset, args.method)

    sample_ids = [
        int(name)
        for name in os.listdir(args.attn_dir)
        if os.path.isdir(os.path.join(args.attn_dir, name)) and name.isdigit()
    ]
    sample_ids.sort()

    per_layer_scores = {layer: [] for layer in layer_list}
    per_layer_labels = {layer: [] for layer in layer_list}
    per_layer_samples = {layer: [] for layer in layer_list}

    for sample_id in sample_ids:
        if sample_id not in correct_map:
            continue
        image_path = load_image_path(args.dataset, args.image_root, sample_id)
        if not image_path or not os.path.exists(image_path):
            continue
        image = Image.open(image_path)
        width, height = image.size
        boxes = load_or_run_yolo(
            image_path, yolo_cache, yolo_model, args.yolo_conf, args.yolo_iou, class_ids
        )
        mask = mask_from_boxes(boxes, (width, height), grid)
        mask_vec = mask.reshape(-1)
        mask_vec = normalize_vector(mask_vec, use_minmax=False)

        for layer in layer_list:
            sample_dir = os.path.join(args.attn_dir, str(sample_id))
            attn_path = choose_attn_file(sample_dir, args.mode, layer)
            if not attn_path:
                continue
            attn_vec, start, end = load_attention_vector(attn_path, args.head)
            if attn_vec.shape[0] != grid[0] * grid[1]:
                continue
            use_minmax = args.mode == "diff"
            attn_vec = normalize_vector(attn_vec, use_minmax=use_minmax)
            score = float(np.dot(attn_vec, mask_vec))
            per_layer_scores[layer].append(score)
            per_layer_labels[layer].append(1 if correct_map[sample_id] else 0)
            per_layer_samples[layer].append(sample_id)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    layer_rows = []
    for layer in layer_list:
        scores = per_layer_scores[layer]
        labels = per_layer_labels[layer]
        if len(set(labels)) < 2:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(labels, scores))
        layer_rows.append({
            "layer": layer,
            "num_samples": len(scores),
            "auroc": auroc,
            "avg_score": float(np.mean(scores)) if scores else float("nan"),
        })

    json_path = f"{args.out_prefix}_layers.json"
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump({
            "dataset": args.dataset,
            "method": args.method,
            "mode": args.mode,
            "head": args.head,
            "grid": list(grid),
            "layers": layer_rows,
        }, fout, indent=2)

    csv_path = f"{args.out_prefix}_layers.csv"
    with open(csv_path, "w", encoding="utf-8") as fout:
        fout.write("layer,num_samples,auroc,avg_score\n")
        for row in layer_rows:
            fout.write(f"{row['layer']},{row['num_samples']},{row['auroc']},{row['avg_score']}\n")

    plot_path = f"{args.out_prefix}_auroc.png"
    try:
        import matplotlib.pyplot as plt

        layers = [row["layer"] for row in layer_rows]
        aurocs = [row["auroc"] for row in layer_rows]
        plt.figure(figsize=(7, 4))
        plt.plot(layers, aurocs, marker="o")
        plt.title("Attention overlap AUROC by layer")
        plt.xlabel("Layer")
        plt.ylabel("AUROC")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    except Exception as exc:
        print(f"Plot skipped: {exc}")

    with open(cache_path, "w", encoding="utf-8") as fout:
        json.dump(yolo_cache, fout, indent=2)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    if os.path.exists(plot_path):
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
