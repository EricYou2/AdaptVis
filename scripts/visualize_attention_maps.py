#!/usr/bin/env python3
import argparse
import json
import math
import os
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def parse_sample_ids(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            ids.extend(list(range(start, end + 1)))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def resolve_controlled_images_json(dataset: str, root_dir: str) -> str:
    if dataset == "Controlled_Images_A":
        return os.path.join(root_dir, "controlled_images_dataset.json")
    if dataset == "Controlled_Images_B":
        return os.path.join(root_dir, "controlled_clevr_dataset.json")
    raise ValueError(f"Unsupported dataset for overlays: {dataset}")


def load_controlled_image_path(dataset: str, root_dir: str, index: int) -> Optional[str]:
    annotation_path = resolve_controlled_images_json(dataset, root_dir)
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
    # Try as-is relative to repo root
    if os.path.exists(image_path):
        return image_path
    # Try relative to root_dir
    candidate = os.path.join(root_dir, image_path)
    if os.path.exists(candidate):
        return candidate
    return image_path


def parse_start_end(path: str) -> Tuple[Optional[int], Optional[int]]:
    base = os.path.basename(path)
    # Expected format: pre_12_start1_end576.npy
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
    # Prefer a valid image token span over sentinel -1 spans.
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


def reshape_attention(vec: np.ndarray, grid: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    if grid is not None:
        h, w = grid
        if h * w != vec.shape[-1]:
            return None
        return vec.reshape(h, w)
    n = vec.shape[-1]
    # Find a factor pair close to sqrt
    root = int(math.sqrt(n))
    for h in range(root, 0, -1):
        if n % h == 0:
            w = n // h
            return vec.reshape(h, w)
    return None


def normalize_map(attn_map: np.ndarray) -> np.ndarray:
    min_v = float(attn_map.min())
    max_v = float(attn_map.max())
    if max_v - min_v < 1e-8:
        return np.zeros_like(attn_map)
    return (attn_map - min_v) / (max_v - min_v)


def apply_colormap(value_map: np.ndarray) -> np.ndarray:
    # Simple blue->yellow->red colormap
    v = np.clip(value_map, 0.0, 1.0)
    r = np.clip(2.0 * v, 0.0, 1.0)
    b = np.clip(2.0 * (1.0 - v), 0.0, 1.0)
    g = np.clip(2.0 * (1.0 - np.abs(v - 0.5)), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def load_attention_map(attn_path: str, head: str) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    attn = np.load(attn_path)
    # Saved attention can be (heads, seq) or (batch, heads, seq)
    if attn.ndim == 3:
        if attn.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 3D attention array, got shape {attn.shape}")
        attn = attn[0]
    if attn.ndim != 2:
        raise ValueError(f"Expected 2D attention array, got shape {attn.shape}")
    if head == "mean":
        vec = attn.mean(axis=0)
    else:
        head_idx = int(head)
        if head_idx < 0 or head_idx >= attn.shape[0]:
            raise ValueError(f"Head index {head_idx} out of range for shape {attn.shape}")
        vec = attn[head_idx]
    start, end = parse_start_end(attn_path)
    if start is not None and end is not None and start >= 0 and end >= start:
        # The indices are inclusive in filenames
        end_inclusive = min(end + 1, vec.shape[-1])
        start_clamped = min(max(start, 0), vec.shape[-1])
        vec = vec[start_clamped:end_inclusive]
    return vec, start, end


def save_visuals(
    image_path: Optional[str],
    attn_map: np.ndarray,
    out_dir: str,
    basename: str,
    overlay_alpha: float,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    value_map = normalize_map(attn_map)
    heatmap_rgb = apply_colormap(value_map)
    heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB")

    heatmap_path = os.path.join(out_dir, f"{basename}_heatmap.png")
    heatmap_img.save(heatmap_path)

    if not image_path:
        return
    if not os.path.exists(image_path):
        return

    base_img = Image.open(image_path).convert("RGB")
    heatmap_resized = heatmap_img.resize(base_img.size, resample=Image.BILINEAR)
    overlay = Image.blend(base_img, heatmap_resized, alpha=overlay_alpha)
    overlay_path = os.path.join(out_dir, f"{basename}_overlay.png")
    overlay.save(overlay_path)


def build_basename(sample_id: int, mode: str, layer: int, head: str, tag: str) -> str:
    return f"{tag}_sample{sample_id}_layer{layer}_head{head}_{mode}"


def list_sample_dirs(attn_dir: str) -> List[int]:
    ids = []
    for name in os.listdir(attn_dir):
        full = os.path.join(attn_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            ids.append(int(name))
        except ValueError:
            continue
    return sorted(ids)


def render_from_dir(
    attn_dir: str,
    dataset: str,
    image_root: str,
    sample_ids: List[int],
    mode: str,
    layer: int,
    head: str,
    out_dir: str,
    overlay_alpha: float,
    tag: str,
    grid: Optional[Tuple[int, int]],
    flip_vertical: bool,
    flip_horizontal: bool,
    transpose: bool,
) -> None:
    for sample_id in sample_ids:
        sample_dir = os.path.join(attn_dir, str(sample_id))
        attn_path = choose_attn_file(sample_dir, mode, layer)
        if not attn_path:
            print(f"Skip sample {sample_id}: no {mode} files found for layer {layer} in {sample_dir}")
            continue

        vec, start, end = load_attention_map(attn_path, head)
        attn_map = reshape_attention(vec, grid=grid)
        if attn_map is None:
            print(f"Skip sample {sample_id}: cannot reshape attention vector of length {vec.shape[-1]}")
            continue
        if transpose:
            attn_map = attn_map.T
        if flip_vertical:
            attn_map = np.flipud(attn_map)
        if flip_horizontal:
            attn_map = np.fliplr(attn_map)

        image_path = load_controlled_image_path(dataset, image_root, sample_id)
        basename = build_basename(sample_id, mode, layer, head, tag)
        save_visuals(image_path, attn_map, out_dir, basename, overlay_alpha)


def make_side_by_side(left_path: str, right_path: str, out_path: str) -> None:
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        return
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    height = max(left.size[1], right.size[1])
    left = left.resize((left.size[0], height), resample=Image.BILINEAR)
    right = right.resize((right.size[0], height), resample=Image.BILINEAR)
    combined = Image.new("RGB", (left.size[0] + right.size[0], height))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.size[0], 0))
    combined.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Controlled_Images_A")
    parser.add_argument("--attn-dir", required=True)
    parser.add_argument("--compare-dir", default=None)
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--sample-ids", default=None, help="Comma list or ranges like 0,2,5-8")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--mode", choices=["pre", "post", "diff"], default="post")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", default="mean", help="Head index or 'mean'")
    parser.add_argument("--grid", nargs=2, type=int, default=None, help="Force grid H W, e.g. --grid 24 24")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--out-dir", default="./output/attn_vis")
    args = parser.parse_args()

    if not os.path.isdir(args.attn_dir):
        raise FileNotFoundError(f"Attention directory not found: {args.attn_dir}")

    sample_ids = parse_sample_ids(args.sample_ids)
    if sample_ids is None:
        all_ids = list_sample_dirs(args.attn_dir)
        sample_ids = all_ids[: args.max_samples]

    grid = tuple(args.grid) if args.grid else None
    render_from_dir(
        attn_dir=args.attn_dir,
        dataset=args.dataset,
        image_root=args.image_root,
        sample_ids=sample_ids,
        mode=args.mode,
        layer=args.layer,
        head=args.head,
        out_dir=args.out_dir,
        overlay_alpha=args.overlay_alpha,
        tag="base",
        grid=grid,
        flip_vertical=args.flip_vertical,
        flip_horizontal=args.flip_horizontal,
        transpose=args.transpose,
    )

    if args.compare_dir:
        render_from_dir(
            attn_dir=args.compare_dir,
            dataset=args.dataset,
            image_root=args.image_root,
            sample_ids=sample_ids,
            mode=args.mode,
            layer=args.layer,
            head=args.head,
            out_dir=args.out_dir,
            overlay_alpha=args.overlay_alpha,
            tag="compare",
            grid=grid,
            flip_vertical=args.flip_vertical,
            flip_horizontal=args.flip_horizontal,
            transpose=args.transpose,
        )
        # Create simple side-by-side overlays when possible
        for sample_id in sample_ids:
            left = os.path.join(
                args.out_dir,
                build_basename(sample_id, args.mode, args.layer, args.head, "base") + "_overlay.png",
            )
            right = os.path.join(
                args.out_dir,
                build_basename(sample_id, args.mode, args.layer, args.head, "compare") + "_overlay.png",
            )
            combined = os.path.join(
                args.out_dir,
                build_basename(sample_id, args.mode, args.layer, args.head, "side_by_side") + "_overlay.png",
            )
            make_side_by_side(left, right, combined)

    print(f"Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
