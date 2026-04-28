#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from glob import glob

import numpy as np


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def load_correct_map(selected_path):
    if not os.path.exists(selected_path):
        return {}
    with open(selected_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    return {int(d["selected_index"]): bool(d["correct"]) for d in data}


def summarize_sample(pre_files, post_files):
    pre_stats = []
    post_stats = []

    for pf in pre_files:
        base = os.path.basename(pf).replace("pre_", "post_", 1)
        post_path = os.path.join(os.path.dirname(pf), base)
        if not os.path.exists(post_path):
            continue

        pre = np.load(pf)
        post = np.load(post_path)

        # pre/post shape: (heads, seq_len)
        pre_ent = entropy(pre)
        post_ent = entropy(post)

        pre_max = np.max(pre, axis=-1)
        post_max = np.max(post, axis=-1)

        pre_stats.append((np.mean(pre_ent), np.mean(pre_max)))
        post_stats.append((np.mean(post_ent), np.mean(post_max)))

    if not pre_stats or not post_stats:
        return None

    pre_ent = float(np.mean([s[0] for s in pre_stats]))
    post_ent = float(np.mean([s[0] for s in post_stats]))
    pre_max = float(np.mean([s[1] for s in pre_stats]))
    post_max = float(np.mean([s[1] for s in post_stats]))

    return {
        "pre_entropy": pre_ent,
        "post_entropy": post_ent,
        "entropy_delta": pre_ent - post_ent,
        "pre_peak": pre_max,
        "post_peak": post_max,
        "peak_delta": post_max - pre_max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="adapt_vis")
    parser.add_argument("--option", default="four")
    parser.add_argument("--attn-root", default="./output/attn")
    parser.add_argument("--selected", default=None)
    parser.add_argument("--out", default="./output/attn_analysis.json")
    args = parser.parse_args()

    selected_path = args.selected
    if selected_path is None:
        selected_path = os.path.join("./output", f"selected_indices_{args.dataset}.json")

    correct_map = load_correct_map(selected_path)

    # Find attention directories for the dataset and method
    method_prefix = f"{args.method}_"
    base_dir = os.path.join(args.attn_root, args.dataset)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Attention root not found: {base_dir}")

    candidates = [
        d for d in os.listdir(base_dir)
        if d.startswith(method_prefix) and f"_{args.option}opt_" in d
    ]
    if not candidates:
        raise ValueError(f"No attention directories for method={args.method} under {base_dir}")

    # Use most recent (lexicographic) tag if multiple
    candidates.sort()
    tag = candidates[-1]
    run_dir = os.path.join(base_dir, tag)

    sample_dirs = sorted([d for d in glob(os.path.join(run_dir, "*")) if os.path.isdir(d)])

    results = []
    for sample_dir in sample_dirs:
        sample_id = int(os.path.basename(sample_dir))
        pre_files = sorted(glob(os.path.join(sample_dir, "pre_*.npy")))
        post_files = sorted(glob(os.path.join(sample_dir, "post_*.npy")))
        if not pre_files or not post_files:
            continue

        stats = summarize_sample(pre_files, post_files)
        if stats is None:
            continue

        stats["sample_id"] = sample_id
        if sample_id in correct_map:
            stats["correct"] = correct_map[sample_id]
        results.append(stats)

    if not results:
        raise ValueError("No attention stats computed. Check saved attention files.")

    # Compute simple correlation between entropy delta and correctness (if labels exist)
    labeled = [r for r in results if "correct" in r]
    summary = {}
    if labeled:
        ent_delta = np.array([r["entropy_delta"] for r in labeled], dtype=np.float32)
        correct = np.array([1.0 if r["correct"] else 0.0 for r in labeled], dtype=np.float32)
        corr = float(np.corrcoef(ent_delta, correct)[0, 1]) if len(labeled) > 1 else float("nan")
        summary["entropy_delta_vs_correct_corr"] = corr
        summary["avg_entropy_delta_correct"] = float(ent_delta[correct == 1].mean())
        summary["avg_entropy_delta_incorrect"] = float(ent_delta[correct == 0].mean())

        # Examples for "sharpened but wrong" and "unchanged but correct"
        sharpened_wrong = [r for r in labeled if (not r["correct"]) and r["entropy_delta"] > 0]
        sharpened_wrong.sort(key=lambda r: r["entropy_delta"], reverse=True)
        unchanged_correct = [r for r in labeled if r["correct"]]
        unchanged_correct.sort(key=lambda r: abs(r["entropy_delta"]))

        summary["sharpened_but_wrong"] = sharpened_wrong[:3]
        summary["correct_but_unchanged"] = unchanged_correct[:3]

    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "option": args.option,
        "run_dir": run_dir,
        "num_samples": len(results),
        "summary": summary,
        "samples": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)

    print(f"Wrote analysis to {args.out}")


if __name__ == "__main__":
    main()
