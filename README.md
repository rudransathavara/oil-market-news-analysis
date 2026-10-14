# OIL MARKET AI — Beginner Prototype (one-file edition)

A project folder kept as small as possible for learning + daily use:

| File | Purpose | Run it? |
|---|---|---|
| **`one_file.py`** | The main program: downloads Brent prices, trains the logistic-regression model, fetches news **per market driver** (geopolitics, OPEC+, US inventories, dollar/macro, market expectations), analyses each headline with a local LLM (Ollama), and prints a plain-English outlook | **YES — this is the one** |
| `experiments.py` | The accuracy lab: compares feature sets + regularization honestly (train/validation/test discipline) | optional |
| `simple_model.py` | The ~30-line learning model for interview practice | optional study |
| `requirements.txt` | Python packages needed | no |
| `.env` / `.env.example` | AI provider/model config (default: your local Ollama) | no |
| `data/brent_prices.csv` | Cached price history (auto-refreshes every 6 h) | no |
| `README.md` | This file | no |

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # default works with local Ollama, no API key needed
```

Install Ollama (https://ollama.com), pull a model once (`ollama pull gemma3:4b`),
and keep the Ollama app (or `ollama serve`) running while you use the project.

## Daily use

```bash
source .venv/bin/activate
python3 one_file.py                       # news per driver + forecast + plain-English verdict
python3 one_file.py --headlines 8         # fewer headlines, faster run
python3 one_file.py --headline "OPEC+ announces production cuts."
python3 one_file.py --compare             # run gemma3:4b AND qwen2.5:14b back-to-back
python3 one_file.py --skip-news           # price-model forecast only
python3 one_file.py --date 2026-09-15     # what the model would have said on a past date
python3 one_file.py --backtest 20         # grade the model over the last N trading days
python3 experiments.py                    # accuracy-improvement experiments
python3 simple_model.py                   # the minimal learn-everything model
```

## Honest performance summary (re-measured out-of-sample)

- Price-only features: ~47.6% → ~52% directional accuracy after the experiments
  (validation-selected recipe), vs. a ~51.8% naive baseline. Near coin-flip; the
  program says so itself in every run.
- The biggest remaining lever is the news-fused model (Model C), which needs a
  labelled historical news dataset (Stage 4) — news scores are shown, not yet fused.
- Everything here is a research toy, NOT financial advice.
