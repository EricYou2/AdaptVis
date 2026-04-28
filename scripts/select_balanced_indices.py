#!/usr/bin/env python3
import argparse
import json
import os
import random
import numpy as np

from dataset_zoo import get_dataset


def load_records(path):
    with open(path, "r", encoding="utf-8") as fin:
        text = fin.read().strip()
        if not text:
            raise ValueError(f"Empty results file: {path}")
        if text.startswith("["):
            return json.loads(text)
        if text.startswith("{") and text.endswith("}") and "\n" not in text:
            return [json.loads(text)]
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--results", default="./outputs/filtered_results.json")
    parser.add_argument("--method", default="base")
    parser.add_argument("--model", default="llava1.5")
    parser.add_argument("--option", default="four")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="./output")
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results file not found: {args.results}")

    records = load_records(args.results)
    matched = [
        r for r in records
        if r.get("dataset") == args.dataset
        and str(r.get("method")) == args.method
        and str(r.get("model")) == args.model
        and str(r.get("option")) == args.option
    ]
    if not matched:
        raise ValueError(
            "No matching results entry for dataset/method/model/option. "
            "Check filtered_results.json or adjust filters."
        )

    correct_id = matched[-1].get("correct_id")
    if correct_id is None:
        raise ValueError("Matched results entry does not include correct_id")

    dataset = get_dataset(args.dataset, image_preprocess=None, download=False)
    total = len(dataset)

    all_indices = list(range(total))
    correct_set = set(correct_id)
    correct_indices = sorted([i for i in all_indices if i in correct_set])
    incorrect_indices = sorted([i for i in all_indices if i not in correct_set])

    n = args.n
    n_per_side = n // 2
    pick_correct = min(n_per_side, len(correct_indices))
    pick_incorrect = min(n - pick_correct, len(incorrect_indices))

    sel_correct = random.sample(correct_indices, pick_correct) if pick_correct else []
    sel_incorrect = random.sample(incorrect_indices, pick_incorrect) if pick_incorrect else []

    selected = sorted(sel_correct + sel_incorrect)
    if len(selected) < n:
        remaining = n - len(selected)
        pool = [i for i in all_indices if i not in selected]
        extra = random.sample(pool, min(len(pool), remaining)) if pool else []
        selected = sorted(selected + extra)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"sampled_idx_{args.dataset}.npy")
    np.save(out_path, np.array(selected))

    details = [{"selected_index": int(i), "correct": (i in correct_set)} for i in selected]
    details_path = os.path.join(args.out_dir, f"selected_indices_{args.dataset}.json")
    with open(details_path, "w", encoding="utf-8") as fout:
        json.dump(details, fout, indent=2)

    print(f"Wrote {len(selected)} indices to {out_path}")


if __name__ == "__main__":
    main()
