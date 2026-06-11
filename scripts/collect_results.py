"""Collect all metrics_*.json into one comparison table (CSV + console).

Reads every ``metrics_<backbone>_<pe>_seed<seed>.json`` written by train_hrd.py
and prints a single table sorted by participant-level AUC, so all positional-
encoding variants are compared consistently. Optionally writes a CSV.

Run:  python collect_results.py --results-dir results_hrd [--csv summary.csv]
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Summarise CoST PE-variant results")
    p.add_argument("--results-dir", default="results_hrd")
    p.add_argument("--csv", default=None, help="Optional path to write the table as CSV")
    args = p.parse_args()

    rows = []
    # new layout: <results-dir>/<run_id>/<variant>/metrics.json
    for fp in sorted(Path(args.results_dir).rglob("metrics*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        win, pid = d.get("window_level", {}), d.get("participant_level", {})
        rows.append({
            "run_id": fp.parent.parent.name,
            "backbone": d.get("backbone", "?"),
            "pe": d.get("pe", "?"),
            "seed": d.get("config", {}).get("seed", "?"),
            "win_auc": win.get("auc_roc", float("nan")),
            "win_f1": win.get("f1", float("nan")),
            "win_acc": win.get("accuracy", float("nan")),
            "pid_auc": pid.get("auc_roc", float("nan")),
            "pid_f1": pid.get("f1", float("nan")),
            "pid_acc": pid.get("accuracy", float("nan")),
        })

    if not rows:
        raise SystemExit(f"No metrics_*.json found in {args.results_dir}")

    rows.sort(key=lambda r: r["pid_auc"], reverse=True)

    header = f"{'backbone':<12} {'pe':<11} {'pid_AUC':>8} {'pid_F1':>7} {'pid_Acc':>8} {'win_AUC':>8} {'win_F1':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['backbone']:<12} {r['pe']:<11} {r['pid_auc']:>8.3f} "
              f"{r['pid_f1']:>7.3f} {r['pid_acc']:>8.3f} {r['win_auc']:>8.3f} {r['win_f1']:>7.3f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
