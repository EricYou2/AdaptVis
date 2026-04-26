import argparse
import csv
import json
from pathlib import Path

DATASETS = [
    "Controlled_Images_A",
    "Controlled_Images_B",
    "COCO_QA_one_obj",
    "COCO_QA_two_obj",
    "VG_QA_one_obj",
    "VG_QA_two_obj",
]
WEIGHTS = [0.5, 0.8, 1.2, 1.5, 2.0]


def read_acc(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # score files created by this repo include "acc"
    if isinstance(data, dict) and "acc" in data:
        try:
            return float(data["acc"])
        except Exception:
            return None
    return None


def maybe_plot(rows, out_png: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib is not available; skipping PNG plot generation")
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        drows = [r for r in rows if r["dataset"] == dataset]
        drows.sort(key=lambda x: x["weight"])

        xs = [r["weight"] for r in drows]
        ys = [r["acc"] if r["acc"] is not None else float("nan") for r in drows]

        ax.plot(xs, ys, marker="o")
        ax.set_title(dataset)
        ax.set_xlabel("alpha")
        ax.set_ylabel("validation acc")
        ax.set_xticks(WEIGHTS)
        ax.grid(True, alpha=0.3)

    for j in range(len(DATASETS), len(axes)):
        axes[j].axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[ok] wrote plot to {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate ScalingVis validation sweep and choose best alpha.")
    parser.add_argument("--root", default="sweeps/scalingvis", help="Root directory for sweep artifacts")
    args = parser.parse_args()

    root = Path(args.root)
    val_root = root / "raw" / "val"
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best = {}

    for dataset in DATASETS:
        dataset_rows = []
        for weight in WEIGHTS:
            score_path = val_root / dataset / f"weight_{weight}.json"
            acc = read_acc(score_path)
            row = {
                "dataset": dataset,
                "weight": weight,
                "acc": acc,
                "score_file": str(score_path),
            }
            rows.append(row)
            dataset_rows.append(row)

        # pick best by max acc, then smaller alpha on ties
        valid = [r for r in dataset_rows if r["acc"] is not None]
        if valid:
            chosen = sorted(valid, key=lambda r: (-r["acc"], r["weight"]))[0]
            best[dataset] = {
                "weight": chosen["weight"],
                "val_acc": chosen["acc"],
                "score_file": chosen["score_file"],
            }
        else:
            best[dataset] = {
                "weight": None,
                "val_acc": None,
                "score_file": None,
            }

    csv_path = report_dir / "val_sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "weight", "acc", "score_file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote table to {csv_path}")

    best_path = report_dir / "best_weights.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"[ok] wrote best weights to {best_path}")

    maybe_plot(rows, report_dir / "val_sweep_plot.png")


if __name__ == "__main__":
    main()
