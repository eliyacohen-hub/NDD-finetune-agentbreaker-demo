"""Compare Garak AgentBreaker runs across red-team models.

Parses the `report.jsonl` (and sibling `hitlog.jsonl`) that `run_scan.py`
produces for each (red-team model x agent) run and computes the Attack Success
Rate (ASR), so you can see how much the fine-tuned Nemotron improves over the
base model (and a comparison model) as Garak's red-team attacker.

Garak scoring convention (ThresholdEvaluator, threshold 0.5):
  * each probe attempt is scored by the AgentBreakerResult detector,
  * score >= 0.5  -> the attack LANDED  -> counted as a "fail" + written to the hitlog,
  * score <  0.5  -> the agent resisted -> counted as a "pass".
So  ASR = fails / total_evaluated  (higher = a stronger attacker).

Each `report.jsonl` carries one `entry_type == "eval"` summary record per
(probe, detector) with `passed`, `fails`, `total_evaluated`.
"""

from __future__ import annotations

import json
from pathlib import Path

DETECTOR = "agent_breaker.AgentBreakerResult"


def summarize_report(report_path: str | Path) -> dict:
    """Return {'asr', 'hits', 'total', 'passed'} from one report.jsonl.

    Sums the AgentBreakerResult eval records (there may be one per probe stage).
    """
    report_path = Path(report_path)
    hits = total = passed = 0
    with open(report_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("entry_type") != "eval":
                continue
            det = rec.get("detector", "")
            if DETECTOR not in det and "AgentBreakerResult" not in det:
                continue
            hits += int(rec.get("fails", 0))
            passed += int(rec.get("passed", 0))
            total += int(rec.get("total_evaluated", 0))
    asr = (hits / total) if total else 0.0
    return {"asr": asr, "hits": hits, "total": total, "passed": passed}


def load_hitlog(report_path: str | Path) -> list[dict]:
    """Load the sibling `<prefix>.hitlog.jsonl` (landed attacks), if present."""
    hitlog_path = Path(str(report_path).replace(".report.jsonl", ".hitlog.jsonl"))
    if not hitlog_path.exists():
        return []
    entries = []
    with open(hitlog_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def compare_runs(runs: dict[str, str | Path]) -> list[dict]:
    """Summarize several runs.

    `runs` maps a label (e.g. "base / research") to its report.jsonl path.
    Returns a list of rows with label, asr, hits, total — ready for a DataFrame.
    """
    rows = []
    for label, report in runs.items():
        summary = summarize_report(report)
        summary["label"] = label
        rows.append(summary)
    return rows


def plot_comparison(rows: list[dict], out_path: str | Path | None = None, title: str = "Attack Success Rate by red-team model"):
    """Bar chart of ASR per run label. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    asrs = [100.0 * r["asr"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(labels)), 4))
    bars = ax.bar(labels, asrs, color="#76b900")  # NVIDIA green
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(100.0, max(asrs) * 1.2 if asrs else 100.0))
    for bar, r in zip(bars, rows):
        ax.annotate(
            f"{100 * r['asr']:.0f}%\n({r['hits']}/{r['total']})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    return fig


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Compare Garak AgentBreaker runs.")
    p.add_argument(
        "runs",
        nargs="+",
        help="label=report.jsonl pairs, e.g. base=~/.local/.../base.report.jsonl",
    )
    p.add_argument("--plot", default=None, help="Optional output image path")
    args = p.parse_args(argv)

    runs = {}
    for item in args.runs:
        label, _, path = item.partition("=")
        runs[label] = Path(path).expanduser()

    rows = compare_runs(runs)
    width = max(len(r["label"]) for r in rows) + 2
    print(f"{'model/agent':<{width}} {'ASR':>7} {'hits':>6} {'total':>6}")
    for r in rows:
        print(f"{r['label']:<{width}} {100 * r['asr']:>6.1f}% {r['hits']:>6} {r['total']:>6}")
    if args.plot:
        plot_comparison(rows, out_path=args.plot)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
