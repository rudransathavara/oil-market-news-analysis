"""EVALUATE — probability quality, not just accuracy.

The original program only reported directional accuracy (~50% = a coin
flip). An honest probability model must also be asked: are the numbers
themselves trustworthy? This module walks the model forward over recent
history, then computes:

  - accuracy          share of days the >50% rule called the direction right
  - precision         of days we called UP, how many actually went up
  - recall            of days that actually went up, how many we called
  - Brier score       mean (probability - outcome)^2; 0 = perfect, 0.25 = always guessing 50%
  - reliability bins  grouped by predicted probability: does a "60%" really rise 60% of the time?

and saves two charts into results/: a reliability curve with the forecasts
over time, and the model's learned feature weights.

Run via the CLI:  python3 one_file.py --evaluate 30
"""

import os

import matplotlib
matplotlib.use("Agg")            # no display server needed — we only save files
import matplotlib.pyplot as plt

from .config import RESULTS_DIR
from .model import forecast_on_date


def collect_forecasts(prices, n_days=60):
    """Walk the model forward over the last n_days trading days and collect
    (date, predicted_probability, actually_rose, next_close). Training for
    each day uses ONLY data from before that day (no look-ahead)."""
    n_days = max(1, min(int(n_days), len(prices) - 46))
    rows = []
    for loc in range(len(prices) - n_days, len(prices)):
        day = prices.index[loc].date()
        try:
            info = forecast_on_date(prices, str(day))
        except ValueError:
            continue
        if info["rose"] is None:      # most recent day may have no outcome yet
            continue
        rows.append((day, float(info["prob"]), int(bool(info["rose"])),
                     float(info["next_close"])))
    return rows


def metrics(rows):
    """Accuracy / precision / recall / Brier score for collect_forecasts rows."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": None, "precision": None,
                "recall": None, "brier": None, "mean_prob": None}
    probs = [r[1] for r in rows]
    truth = [r[2] for r in rows]
    preds = [int(p > 0.5) for p in probs]   # int on purpose: False == 0 is True in Python
    tp = sum(1 for p, t in zip(preds, truth) if p and t)
    called_up = sum(1 for p in preds if p)
    actual_up = sum(1 for t in truth if t)
    return {
        "n": n,
        "accuracy": sum(1 for p, t in zip(preds, truth) if p == t) / n,
        "precision": tp / called_up if called_up else None,
        "recall": tp / actual_up if actual_up else None,
        "brier": sum((p - t) ** 2 for p, t in zip(probs, truth)) / n,
        "mean_prob": sum(probs) / n,
    }


def reliability_bins(rows, n_bins=5):
    """Group forecasts by predicted probability and compare each group's
    mean prediction with its observed up-rate. Trustworthy = the two match."""
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        pts = [r for r in rows if lo <= r[1] < hi]
        if i == n_bins - 1:   # include an exact-1.0 prediction in the last bin
            pts += [r for r in rows if r[1] == hi]
        if pts:
            bins.append({"lo": lo, "hi": hi, "n": len(pts),
                         "mean_pred": sum(r[1] for r in pts) / len(pts),
                         "mean_actual": sum(r[2] for r in pts) / len(pts)})
        else:
            bins.append({"lo": lo, "hi": hi, "n": 0,
                         "mean_pred": (lo + hi) / 2, "mean_actual": None})
    return bins


def print_report(rows, m, log=print):
    """Human-readable summary of the evaluation."""
    print(f"\n{'=' * 62}")
    print("  PROBABILITY EVALUATION — quality of the price model's own numbers")
    print(f"{'=' * 62}")
    if m["n"] == 0:
        log("  Not enough graded days to evaluate. (Needs a few days of newer prices.)")
        return
    log(f"  Graded days:            {m['n']}")
    log(f"  Directional accuracy:   {m['accuracy'] * 100:5.1f}%   (50% = coin flip)")
    if m["precision"] is not None:
        log(f"  Precision (called UP):  {m['precision'] * 100:5.1f}%   of the days we said UP, how many rose")
    if m["recall"] is not None:
        log(f"  Recall (real UP days):  {m['recall'] * 100:5.1f}%   of all up-days, how many we caught")
    if m["brier"] is not None:
        quality = ("excellent" if m["brier"] < 0.2 else
                   "decent" if m["brier"] < 0.24 else
                   "no better than always guessing 50%" if abs(m["brier"] - 0.25) < 0.01 else
                   "poor")
        log(f"  Brier score:            {m['brier']:.4f}   (0 = perfect, 0.25 = meaningless) -> {quality}")
    if m["mean_prob"] is not None:
        drift = (m["mean_prob"] - .5) * 100
        log(f"  Mean predicted prob:    {m['mean_prob'] * 100:5.1f}%   "
            f"({'biased ' + f'{drift:+.0f}pp toward UP' if abs(drift) > 2 else 'roughly unbiased'})")
    print("  RELIABILITY (does a stated 60% actually rise ~60% of the time?):")
    for b in reliability_bins(rows):
        if b["n"] == 0:
            continue
        print(f"   bins {b['lo']:.0%}-{b['hi']:.0%}: {b['n']:>3} forecasts   "
              f"predicted {b['mean_pred']:5.1%}   actually rose {b['mean_actual']:5.1%}")


def plot_evaluation(rows, math_report=None, out_dir=RESULTS_DIR):
    """Save two PNG charts into results/: (1) reliability + hit-rate over
    time, (2) the currently learned feature weights."""
    os.makedirs(out_dir, exist_ok=True)

    # Chart 1 — reliability curve (left) and running hit-rate (right)
    fig, (ax_rel, ax_run) = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = [b for b in reliability_bins(rows) if b["n"] > 0]
    if bins:
        ax_rel.plot([b["mean_pred"] for b in bins], [b["mean_actual"] for b in bins],
                    "o-", label="observed up-rate")
        ax_rel.plot([0, 1], [0, 1], "k:", label="perfect calibration")
        ax_rel.set_xlim(0, 1); ax_rel.set_ylim(0, 1)
        ax_rel.set_xlabel("predicted probability of rise")
        ax_rel.set_ylabel("observed share that rose")
        ax_rel.set_title("Reliability curve (calibration)")
        ax_rel.legend(); ax_rel.grid(alpha=.3)
    if rows:
        days = [r[0] for r in rows]
        runs, hits = [], 0
        for r in rows:
            hits += (r[1] > 0.5) == bool(r[2])
            runs.append(hits)
        ax_run.plot(days, runs, "o-")
        ax_run.set_title("Correct calls over time (cumulative)")
        ax_run.set_ylabel("days graded correct"); ax_run.grid(alpha=.3)
        fig.autofmt_xdate()
    fig.suptitle("Oil Market AI — probability evaluation (walk-forward, no look-ahead)")
    fig.tight_layout()
    path1 = os.path.join(out_dir, "evaluation.png")
    fig.savefig(path1, dpi=150); plt.close(fig)

    # Chart 2 —feature weights of today's model
    path2 = None
    if math_report:
        w = math_report["weights"]
        order = sorted(w, key=lambda k: w[k])
        fig2, ax = plt.subplots(figsize=(7, max(3, .35 * len(order) + 1.5)))
        ax.barh(order, [w[k] for k in order])
        ax.axvline(0, color="black", lw=.8)
        ax.set_title("Learned feature weights (rich feature set)")
        ax.set_xlabel("weight on the up-probability")
        fig2.tight_layout()
        path2 = os.path.join(out_dir, "feature_weights.png")
        fig2.savefig(path2, dpi=150); plt.close(fig2)

    print(f"\n  Charts saved: {path1}" + (f" and {path2}" if path2 else ""))
    return path1, path2


def evaluate_run(prices, math_report=None, n_days=60, log=print):
    """Full evaluation pass: collect walk-forward forecasts, print the report,
    save the charts, and return a machine-readable summary (for --json)."""
    rows = collect_forecasts(prices, n_days)
    m = metrics(rows)
    print_report(rows, m, log=log)
    paths = plot_evaluation(rows, math_report)
    m["results_png"] = paths[0]
    if paths[1]:
        m["weights_png"] = paths[1]
    if rows:
        m["reliability_bins"] = reliability_bins(rows)
    return m
