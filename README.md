# Oil Market News Analysis

A research project that explores the relationship between financial news and Brent crude oil price movements. It combines:

- an **auto-tuned logistic-regression model** trained on Brent futures prices,
- **per-driver news retrieval** (OPEC+, geopolitics, US inventories, macro, price action, market expectations),
- **LLM-based headline analysis** that scores each headline's likely impact on oil prices,
- **evaluation tools** (backtesting, probability calibration, accuracy experiments) and a **local SQLite archive** of analysed headlines.

> **Disclaimer:** This is a research and learning project. It is not financial advice. The model's directional accuracy is close to a coin flip, and the program reports this in every run.

## Features

- Downloads five years of Brent (`BZ=F`) daily prices from Yahoo Finance and caches them locally. The cache is reused on later runs.
- Fetches recent headlines from Google News RSS, one feed per market driver.
- Analyses each headline with an LLM (OpenAI, Anthropic, Gemini, or any OpenAI-compatible server such as [Ollama](https://ollama.com)).
- Produces a forecast, a plain-English outlook, and a backtest summary.
- Supports time-travel forecasts for past dates, to check what the model would have said.
- Writes evaluation charts and an optional JSON summary to `results/`.

## Project Structure

```
.
├── one_file.py          # Main entry point (thin wrapper around oil_ai.cli)
├── oil_ai/              # Application package
│   ├── cli.py           # Command-line interface and orchestration
│   ├── config.py        # Settings, market drivers, LLM prompt, provider config
│   ├── prices.py        # Price download and cache
│   ├── model.py         # Auto-tuned logistic-regression model
│   ├── news.py          # Headline fetching and LLM analysis
│   ├── archive.py       # SQLite archive of analysed headlines
│   ├── evaluate.py      # Accuracy, Brier score, calibration, charts
│   └── report.py        # Console output: outlook, backtest, summaries
├── experiments.py       # Feature-set and regularization experiments
├── simple_model.py      # Minimal, annotated model for learning
├── tests/               # Unit tests (pytest)
├── requirements.txt     # Python dependencies
└── .env.example         # Configuration template
```

Generated at runtime (not tracked in version control):

| Path | Contents |
|---|---|
| `data/brent_prices.csv` | Cached price history (downloaded once; delete it to re-download) |
| `data/news_archive.sqlite` | Archive of analysed headlines |
| `results/` | Charts (`evaluation.png`, `feature_weights.png`) and `run.json` |

## Prerequisites

- **Python 3.12** (the version the project was developed and tested with)
- **Internet access** for price and news downloads
- **An LLM**, only for news analysis. Choose one:
  - A cloud API key (OpenAI, Anthropic, or Gemini), or
  - A local model via Ollama (free, runs on your machine)

The price model and `--skip-news` mode need no API key.

## Installation

```bash
# 1. Clone the repository and enter the project folder
cd oil-market-news-analysis

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your configuration file
cp .env.example .env
```

## Configuration

Edit `.env` to choose your AI provider. The file is read at startup.

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openai`, `anthropic`, `gemini`, or `custom` (any OpenAI-compatible API) |
| `LLM_API_KEY` | API key for the provider. Required for news analysis. |
| `LLM_MODEL` | Model name. Optional; a sensible default is used if blank. |
| `LLM_BASE_URL` | Base URL for `custom` providers only. |

### Option A: Cloud provider

```ini
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL=                # optional, e.g. gpt-4o-mini
```

### Option B: Local model with Ollama (no API key)

1. Install Ollama from https://ollama.com.
2. Start the server and pull a model:

   ```bash
   ollama serve               # keep running, or open the Ollama app
   ollama pull gemma3:4b
   ```

3. Set the following in `.env`:

   ```ini
   LLM_PROVIDER=custom
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_API_KEY=ollama
   LLM_MODEL=gemma3:4b
   ```

> **Note:** `LLM_API_KEY` must be set to any non-empty value, even for Ollama, or news analysis is skipped.

## Usage

Activate the virtual environment first (`source .venv/bin/activate`).

### Main program

```bash
python3 one_file.py                        # price forecast + news analysis + verdict
python3 one_file.py --headlines 8          # analyse fewer headlines (faster)
python3 one_file.py --headline "OPEC+ announces production cuts."   # analyse one headline
python3 one_file.py --compare              # run gemma3:4b and qwen2.5:14b side by side (both must be pulled in Ollama)
python3 one_file.py --skip-news            # price-model forecast only, no LLM needed
```

### Historical analysis

```bash
python3 one_file.py --date 2026-09-15      # what the model would have forecast on a past date
python3 one_file.py --backtest 20          # grade the model over the last 20 trading days
python3 one_file.py --evaluate 60          # probability quality (Brier, calibration) + charts
```

### Reporting and archive

```bash
python3 one_file.py --json                 # also write results/run.json
python3 one_file.py --archive-stats        # summarise the headline archive
```

### Learning and experiments

```bash
python3 experiments.py                     # compare feature sets and regularization
python3 simple_model.py                    # minimal, annotated model
```

`experiments.py` and `simple_model.py` read the cached price file. If `data/brent_prices.csv` does not exist yet, run `python3 one_file.py --skip-news` once first.

Run `python3 one_file.py --help` for the full list of options.

## Testing

```bash
python3 -m pytest tests/ -v
```

The tests cover the pure helper functions: news aggregation, date parsing, feature construction, chronological data splitting, look-ahead-bias checks, the SQLite archive, and evaluation metrics. They need no network access or API key.

## Methodology

The price model uses strict chronological discipline:

- **Train** on the oldest data.
- **Validate** to compare configurations and select the best.
- **Test** once, on the most recent unseen data, for the final result.

Tuning on the test set would produce overly optimistic results ("data snooping"), so the project avoids it.

## Results

Measured out-of-sample:

- Price-only features reach roughly **52%** directional accuracy after experiments, against a **~51.8%** naive baseline. This is near random.
- The most promising next step is a news-fused model, which requires a labelled historical news dataset. News scores are currently displayed but not yet used in the forecast.

## Troubleshooting

| Problem | Solution |
|---|---|
| `AI news analysis not configured` | Create `.env` from `.env.example` and set `LLM_API_KEY`. |
| `The AI server is not reachable` | Start Ollama with `ollama serve`, and confirm the model is pulled. |
| `No price cache found` | Run `python3 one_file.py --skip-news` to download prices. |
| Prices look outdated | The cache is not refreshed automatically. Delete `data/brent_prices.csv` and run again to re-download. |
| `Driver feed ... failed` | Check your internet connection. The run continues with the other feeds. |
| `AI analysis failed for this headline` | Check the model name and provider settings in `.env`. |
