"""CLI — the command-line entry point. `one_file.py` is now a thin shim that
calls main() here; the original program's behavior and flags are unchanged,
plus the new ones: --json, --evaluate, --archive-stats."""

import argparse
import json
import os

from dotenv import load_dotenv

from . import archive, evaluate, report
from .config import ARCHIVE_DB, RESULTS_DIR, llm_config
from .model import show_forecast_math, train_auto_model
from .news import analyse_news, fetch_headlines
from .prices import load_prices


def build_parser():
    parser = argparse.ArgumentParser(
        description="Oil Market AI — auto-tuned price model, per-driver news, "
                    "AI analysis, forecast, backtest, evaluation, news archive")
    parser.add_argument("--headline", nargs="?", const=True, default=False,
                        help="analyse this one headline instead of fetched news")
    parser.add_argument("--skip-news", action="store_true", help="forecast only, no news")
    parser.add_argument("--compare", nargs="?", const=True, default=False,
                        help="run BOTH local models (gemma3:4b and qwen2.5:14b) on the news")
    parser.add_argument("--headlines", type=int, default=12,
                        help="how many recent headlines to analyse (default 12)")
    parser.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                        help="time travel: the forecast the model would have made on that past date")
    parser.add_argument("--backtest", nargs="?", const=10, default=None, type=int, metavar="N",
                        help="grade the model on the last N trading days (default 10)")
    parser.add_argument("--evaluate", nargs="?", const=60, default=None, type=int, metavar="N",
                        help="probability quality over the last N days (Brier, calibration, precision/recall) + charts")
    parser.add_argument("--archive-stats", action="store_true",
                        help="summarize what the SQLite news archive has collected")
    parser.add_argument("--json", action="store_true",
                        help="also write a machine-readable summary to results/run.json")
    return parser


def summarize(report_data, analyses, eval_summary=None):
    """Since the program stores state, build a plain dict for --json output."""
    from .news import aggregate, drivers_table
    m = report_data["math"]
    return {
        "as_of": str(report_data["date"]),
        "brent_close": report_data["last"],
        "model": m["config"],
        "prob_up": round(report_data["prob"], 4),
        "direction": "up" if report_data["prob"] > 0.5 else "down",
        "accuracy_unseen": round(report_data["accuracy"], 4),
        "naive_baseline": round(report_data["baseline"], 4),
        "score_logit": round(m["logit"], 4),
        "news": (lambda ag: {"n_analysed": len(analyses), "bullish": ag[0], "bearish": ag[1],
                             "avg_score": round(ag[2], 1), "spread": round(ag[3], 1),
                             "tone": ag[4],
                             "drivers": [{"driver": d, "n": n, "net": round(a, 1)}
                                         for d, n, _b, _be, a, _t in drivers_table(analyses)]}
                 )(aggregate(analyses)),
        "evaluation": eval_summary,
    }


def main(argv=None):
    load_dotenv()
    opts = vars(build_parser().parse_args(argv))

    prices = load_prices()

    if opts["archive_stats"]:
        archive.print_stats()
        return

    if opts["date"] or opts["backtest"]:
        if opts["date"]:
            report.time_travel_report(prices, str(opts["date"]), llm_config()[1])
        if opts["backtest"]:
            report.backtest(prices, int(opts["backtest"]))
        return

    print("Training the auto-tuned price model (this runs the experiments inside)...")
    report_data = train_auto_model(prices)
    prob = report_data["prob"]
    show_forecast_math(report_data)

    eval_summary = None
    if opts["evaluate"] is not None:
        eval_summary = evaluate.evaluate_run(prices, report_data["math"], int(opts["evaluate"]))

    if opts["skip_news"]:
        report.outlook(prob, report_data["last"], report_data["date"], report_data["accuracy"],
                       [], llm_config()[1], report_data["baseline"])
        report.plain_english(prob, [])
        report.backtest(prices, 10)
        report.final_paragraph(prob, [])
        if opts["json"]:
            write_json(summarize(report_data, [], eval_summary))
        return

    if opts["headline"]:
        articles = [{"title": opts["headline"], "published": "pasted by you"}]
    else:
        print(f"Fetching up to {opts['headlines']} recent headlines across ALL drivers "
              f"(past few days + today)...")
        articles = fetch_headlines(opts["headlines"])

    models = ["gemma3:4b", "qwen2.5:14b"] if opts["compare"] else \
             [llm_config()[1] or "gemma3:4b"]
    analyses = []
    for model in models:
        print(f"\n#### AI model: {model} ####")
        analyses = analyse_news(articles, model=model, save_to_archive=ARCHIVE_DB)
        report.outlook(prob, report_data["last"], report_data["date"], report_data["accuracy"],
                       analyses, model, report_data["baseline"])
        report.plain_english(prob, analyses)

    report.backtest(prices, 10)
    report.final_paragraph(prob, analyses)
    if opts["json"]:
        write_json(summarize(report_data, analyses, eval_summary))


def write_json(summary):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "run.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"  Machine-readable summary written to: {path}")
