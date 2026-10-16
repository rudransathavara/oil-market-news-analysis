# How This Project Works (in Plain English)

This guide explains what the project does, how the pieces fit together, and what each part of the code is for. It is written for someone who is new to Python, machine learning, or finance.

Read it top to bottom once, then use the section headings to jump to the part you need.

---

## 1. The big picture

The project asks one question every day:

> **"Will the price of Brent crude oil go up or down tomorrow?"**

It tries to answer using two kinds of information:

1. **Price history.** The past prices of Brent oil. For example, "the price went up 2% yesterday and has been jumpy this week." A small machine-learning model looks at these patterns and gives a probability that tomorrow's price will be higher.
2. **News.** Recent news headlines about oil, grouped into six "market drivers" (see section 3). An AI model reads each headline and gives it a score from **-100** (pushes oil prices down) to **+100** (pushes them up).

The program then prints:

- the current price,
- the price model's probability of a rise,
- the average news score, broken down by driver,
- a plain-English summary, and
- an honest check of how well the model has done in the past.

**Important:** the price model is only slightly better than a coin flip. The program says this every time it runs. This is a learning and research project, not financial advice.

---

## 2. The big-picture flow

When you run `python3 one_file.py`, this is what happens, in order:

```
1. Load settings from the .env file          (cli.py)
2. Load Brent prices from a CSV cache,       (prices.py)
   or download them if the cache is missing
3. Train the price model, testing a few      (model.py)
   versions and keeping the best one
4. Show the math behind today's forecast     (model.py, report.py)
5. Fetch recent news headlines               (news.py)
6. Ask the AI to score each headline         (news.py)
7. Save the scored headlines to a database   (archive.py)
8. Print the outlook, the verdict, and a     (report.py)
   one-paragraph summary
```

Each step lives in its own file. That is deliberate: each file does one job, so it is easier to understand and to test.

---

## 3. Key words you will see

| Word | Meaning |
|---|---|
| **Brent crude** | A global benchmark price for oil. The project uses its futures ticker, `BZ=F`. |
| **Close** | The price at the end of a trading day. This is the number the model uses. |
| **Feature** | A number we compute from the past prices to describe the market, such as "yesterday's return." |
| **Return** | How much the price changed, as a fraction. `0.02` means +2%. |
| **Target** | What we want to predict. Here: "did the price rise the next day? (1 = yes, 0 = no)". |
| **Logistic regression** | A simple model that turns a weighted sum of features into a probability between 0 and 1. |
| **Weight (coefficient)** | How strongly one feature pushes the prediction up or down. |
| **Standardize** | Re-express a value as "how many typical steps away from average it is." |
| **Sigmoid** | The formula `1 / (1 + e^-score)` that turns any score into a probability. |
| **Train / validation / test** | Three chronological slices of history. Train = learn, validation = pick the best setting, test = grade once at the end. |
| **Look-ahead bias** | Accidentally letting the model see the future. This makes results look better than they really are. The project works hard to prevent it. |
| **Accuracy** | The share of days where the model's "up" or "down" call was right. 50% = a coin flip. |
| **Brier score** | How good the probabilities are, not just the up/down calls. 0 is perfect; 0.25 is what you get by always guessing 50%. |
| **Calibration** | Whether a "60%" forecast really goes up about 60% of the time. |
| **Driver** | One of six themes that move oil: OPEC+ policy, geopolitics, US inventories, dollar and macro, price action, market expectations. |
| **LLM** | Large Language Model: the AI that reads a headline and scores it. |
| **Ollama** | A free program that runs an LLM on your own computer. |
| **RSS** | A simple feed format that news sites publish. The project reads Google News RSS feeds. |
| **Backtest** | Replaying the model on past days to see how it would have done. |
| **Time-travel forecast** | Asking "what would the model have said on date X?" using only data available on that date. |
| **Archive** | The SQLite database (`data/news_archive.sqlite`) that stores every scored headline. |

---

## 4. The project's files

```
one_file.py          The program you run. Just calls the main code.
experiments.py       A standalone lab that compares model settings honestly.
simple_model.py      A short, heavily commented model for learning.
oil_ai/              The main code, split into modules:
  __init__.py          Describes the package. No logic.
  config.py            All settings and constants in one place.
  cli.py               Reads your command, runs the steps in order.
  prices.py            Downloads and caches Brent prices.
  model.py             The auto-tuned price model and its math.
  news.py              Fetches headlines and asks the AI to score them.
  archive.py           Saves scored headlines to a SQLite database.
  evaluate.py          Measures how trustworthy the probabilities are; draws charts.
  report.py            All the printed output.
tests/test_pipeline.py Automated checks of the core math and logic.
requirements.txt     The Python packages the project needs.
.env / .env.example  Your private settings (API keys, which AI to use).
```

---

## 5. File by file

### 5.1 `one_file.py`: the front door

This file is deliberately tiny. Its job is to start the real program.

- **The docstring (the text in triple quotes at the top):** a comment block describing the program and its newer options. Python ignores it when running.
- **`from oil_ai.cli import main`:** brings in the `main` function from the `oil_ai` package.
- **`if __name__ == "__main__":`:** means "only run this when the file is started directly, not when another file imports it."
- **`main()`:** starts the whole program.

Why keep it so small? It means the main logic can be imported and tested in other places without running the whole program.

---

### 5.2 `oil_ai/config.py`: all the settings

This file holds every setting in one place, so you change a value once and it applies everywhere.

- **`TICKER = "BZ=F"`:** the Yahoo Finance code for Brent crude futures.
- **`PRICE_CSV`, `RESULTS_DIR`, `ARCHIVE_DB`:** where the price cache, charts, and the news database are stored.
- **`DRIVERS`:** a dictionary (a list of name-to-value pairs). Each name is a market driver, and each value is the search words used to find news about that driver. For example, `"OPEC+ policy"` searches for OPEC meetings and production cuts.
- **`DRIVER_ORDER`:** the same driver names as a plain list, so they always print in the same order.
- **`RSS_BASE`:** the start of the Google News search link.
- **`rss_url(query)`:** builds a full search link from a query. `quote_plus` makes the words safe for a web link (spaces become `+`). `when:3d` limits results to the last 3 days.
- **`SYSTEM_PROMPT`:** the instructions given to the AI. It tells the AI to act as an oil analyst, to sort each headline into exactly one driver, to score its impact from -100 to +100, and to reply only in JSON (a structured text format). The prompt also says: a headline about a price *increase* in supply is bearish (bad for prices), and speculation is not fact.
- **`DEFAULT_MODELS`:** if you don't name a model, these are used for each provider (OpenAI, Anthropic, Gemini).
- **`OPENAI_BASE`:** the web addresses for OpenAI and Gemini.
- **`llm_config()`:** reads `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_BASE_URL` from your `.env` file and works out which AI to use. If a value is missing, it falls back to a default. The `os.getenv` calls read environment variables, which `.env` fills in.

---

### 5.3 `oil_ai/cli.py`: the conductor

This file reads your command-line options and runs the steps in the right order. It does not do the heavy work itself; it calls the other modules.

**`build_parser()`** defines the options you can type after `python3 one_file.py`. Each `add_argument` line creates one option:

| Option | What it does |
|---|---|
| `--headline` | Analyse one headline you type, instead of fetching news |
| `--skip-news` | Only run the price model, no news or AI |
| `--compare` | Run two AI models (`gemma3:4b` and `qwen2.5:14b`) and compare them |
| `--headlines N` | How many headlines to fetch (default 12) |
| `--date YYYY-MM-DD` | Show what the model would have forecast on that past day |
| `--backtest N` | Grade the model on the last N trading days |
| `--evaluate N` | Measure probability quality over the last N days and draw charts |
| `--archive-stats` | Summarise the news database |
| `--json` | Also save a machine-readable summary to `results/run.json` |

**`summarize(report_data, analyses, eval_summary)`** packs the main results into a plain Python dictionary, which `--json` then writes to a file. The `(lambda ag: {...})(aggregate(analyses))` trick computes the news summary once and then builds the dictionary from it.

**`main(argv=None)`** is the heart of the program. In order, it:

1. Calls `load_dotenv()`, which reads your `.env` file into the environment.
2. Parses your options into a dictionary called `opts`.
3. Loads the prices (cached or downloaded) with `load_prices()`.
4. If `--archive-stats` was given, prints the database summary and stops.
5. If `--date` or `--backtest` was given, runs that report and stops. This skips news entirely.
6. Otherwise, trains the price model and prints the forecast math.
7. If `--evaluate` was given, runs the probability evaluation.
8. If `--skip-news` was given, prints the outlook and stops after the price part.
9. Otherwise, gets headlines: either your `--headline` text or `fetch_headlines()`.
10. Picks the AI model(s). With `--compare` it uses both; otherwise the one from `.env`.
11. For each model, scores the headlines with `analyse_news()` and prints the outlook and verdict.
12. Prints a 10-day backtest, the final paragraph, and, if asked, writes the JSON file.

**`write_json(summary)`** creates the `results` folder if needed, then writes the dictionary as formatted JSON.

---

### 5.4 `oil_ai/prices.py`: getting the price data

This file gets the historical Brent prices and saves them to a CSV file, so later runs are fast and work offline.

**`load_prices()`** does this:

- **If the CSV exists and has rows:** it reads the file and returns it. Dates are read as real date values (`parse_dates`), and the date column becomes the row label (`index_col`).
- **Otherwise:** it creates the `data` folder if needed, then downloads five years of daily data with `yfinance.Ticker(TICKER).history(period="5y", interval="1d")`.
- It keeps only the date and closing price, converts dates to plain dates with no time zone, removes missing rows and duplicate dates, sorts by date, and saves the result to the CSV.

**Important:** the cache is **never refreshed automatically**. To get newer prices, delete `data/brent_prices.csv` and run the program again.

---

### 5.5 `oil_ai/model.py`: the price model

This is the most mathematical file. It answers: "Given the recent price pattern, what is the chance of a rise tomorrow?"

**`CANDIDATES = (0.001, 0.01, 0.1)`** lists three strengths of a setting called *regularization*. A small number forces the model to stay simple, which helps avoid memorising noise. The program tries all three and keeps the best.

**`split_ranges(n, test_len=250, valid_len=250)`** splits `n` days into three blocks, in date order:

- **train:** the oldest days, used to learn the weights.
- **valid:** the next 250 days, used to choose between candidates.
- **test:** the final 250 days, used once to grade the chosen model.

Because the blocks come in time order, the model never trains on the future. The tests in `tests/` check that the blocks never overlap.

**`_features(kind, prices)`** builds the features, one row per day:

- **basic** (3 features): yesterday's return, the 5-day return, and the 10-day volatility (how jumpy recent prices are).
- **rich** (12 features): adds more time windows (2, 10, 20 days), a 5- and 20-day volatility, how far the price sits from its 20-day and 50-day averages (`vs_ma20`, `vs_ma50`), where the price sits inside its 20-day high-low range (`pos_in_range20`), and a volume measure (`volume_z`).

Every feature is computed using only past data. For example, `pct_change(5)` compares today with five days ago.

**`_pipeline(C)`** builds a two-step recipe:

1. `StandardScaler()`: rescales every feature so its average is 0 and its spread is 1. This lets features in different units be compared fairly.
2. `LogisticRegression(C=C, max_iter=5000, class_weight="balanced")`: the actual model. `C` is the regularization strength. `max_iter` is how many attempts the solver makes. `class_weight="balanced"` stops the model from ignoring the less common outcome.

**`train_auto_model(prices, log=print)`** is the main training function. Step by step:

1. `y_all` is the target: `1` if tomorrow's close is higher than today's, otherwise `0`.
2. For each feature set (basic, rich):
   - it builds the features and joins them to the target,
   - drops rows with missing values,
   - skips the set if there are fewer than 700 rows,
   - splits into train, valid and test,
   - trains one model for each candidate `C`,
   - records how accurate each one is on the **validation** block.
3. Sorts all the results and keeps the best on validation.
4. Grades that winner **once** on the **test** block. This is the honest score you see printed.
5. Prints the chosen setup and the scores.
6. Refits the winner on **all** history, so today's forecast uses the most data possible.
7. Computes today's probability with `predict_proba`.
8. Builds the **math report**: today's feature values, their standardized values, each feature's weight, each feature's "push" (standardized value × weight), and the sum of pushes plus the intercept (the score). It then checks that the sigmoid of that score matches the probability, using an `assert` (a self-check that stops the program if the numbers disagree).

> **Note on `baseline`:** the code sets the "naive baseline" to `max(valid_acc, 1 - valid_acc)`. That is the validation accuracy itself or its opposite, not a true baseline such as "always guess up." Treat the number with caution.

**`show_forecast_math(report)`** prints the calculation as a table: feature, value, standardized value, weight, and push. It then shows the score and the final probability, so nothing is hidden.

**Time travel: `forecast_on_date(prices, date_str)`** answers "what would the model have said on this past date?" Using a simpler model with only two features (yesterday's return and the 5-day return):

- `searchsorted` finds the last trading day on or before the date.
- It refuses to forecast if there are fewer than 40 days of history.
- It trains only on days whose outcome was already known by that date. The final row is dropped on purpose, because its target would peek at the next day.
- It predicts the chosen day's probability.
- Only after the forecast is made does it look at the next day's price, to grade the forecast. This is clearly marked in the code with a warning symbol.

---

### 5.6 `oil_ai/news.py`: headlines and the AI

This file has three jobs: fetch headlines, ask the AI to score them, and combine the scores.

**`llm_analysis(headline, model=None)`** scores one headline:

1. If no `LLM_API_KEY` is set, it prints a warning and returns nothing. (Ollama still needs a key value, even a fake one like `ollama`.)
2. It reads the provider and model name from `llm_config()`.
3. **Anthropic** uses its own API format. Everything else uses the OpenAI-style format, which also covers Ollama and Gemini.
4. It sends the system prompt and the headline to the AI, with a low `temperature` (0.2) so answers are more consistent.
5. It takes the text of the reply and pulls out the JSON part, even if the AI wrapped it in code fences or added extra words. It does this by finding the first `{` and the last `}`.
6. It clamps `impact_score` to the range -100 to +100, so a silly value cannot break the maths.
7. If the server is not running, it prints a helpful message. For any other error, it skips that headline and continues.

**`fetch_headlines(total=10)`** gets headlines:

- It divides the requested total by the number of drivers. With the default of 12 and six drivers, that is 2 headlines per driver. Note that with `--headlines 8`, the integer division gives 1 per driver, so you get 6, not 8.
- For each driver, it downloads that driver's Google News RSS feed, parses the XML, and reads each `<item>`.
- It removes the trailing ` - Publisher name` from each title with `rsplit(" - ", 1)[0]`.
- It tags every article with its driver. This is more reliable than trusting the AI to say which driver a headline belongs to.
- If one feed fails, it prints a message and continues with the others.

**`pub_day(pub)`** turns the feed's date (for example `Tue, 13 Oct 2026 19:04:00 GMT`) into a short day label (`Tue 13 Oct`). The `GMT` and `UTC` text is swapped for `+0000` first, because Python's date parser does not understand those words. If the date cannot be read, it returns the original text.

**`aggregate(analyses)`** combines a list of scores into one summary:

- `n_bull` and `n_bear`: how many headlines were bullish and how many bearish.
- `avg`: the average score.
- `spread`: the sample variance, a measure of how much the scores disagree.
- `tone`: "bullish" if the average is above +5, "bearish" if below -5, otherwise "neutral". The gap around zero stops tiny moves from counting as a trend.

**`drivers_table(analyses)`** runs `aggregate` once per driver, skipping drivers with no headlines. Anything without a known driver goes into "other."

**`day_trends(analyses)`** does the same, grouped by day, in date order. It is used to see how the tone changed over time.

**`analyse_news(articles, model=None, save_to_archive=None)`** loops over every headline:

1. Prints the headline and its date.
2. Asks the AI to score it.
3. If the AI returned a result, adds the day, driver and headline to it, prints a short summary, and keeps it.
4. At the end, saves all the results to the archive database, if a path was given.

---

### 5.7 `oil_ai/archive.py`: the news database

This file stores every scored headline in a small SQLite database. SQLite is a database that lives in a single file, so no server is needed. The stored data will later be used to train a model that includes news.

**`_SCHEMA`** is the table's blueprint. It creates a table called `headlines` with one column per field: when it was saved, the day, the raw date, driver, title, direction, score, time horizon, uncertainty, and model name. `IF NOT EXISTS` means it is safe to run again.

**`connect(db_path)`** creates the folder if needed, opens the database file, and makes sure the table exists.

**`save(analyses, model, db_path)`** writes one row per headline. It uses `executemany` to insert many rows at once, and `with conn:` makes the insert all-or-nothing. It returns how many rows were saved.

**`stats(db_path, limit_days=30)`** runs two queries:

- average score and count per day (the 30 busiest days),
- average score and count per driver, sorted by the size of the average.

If the file does not exist yet, it returns empty lists.

**`print_stats(db_path)`** prints those summaries in a readable form.

---

### 5.8 `oil_ai/evaluate.py`: is the probability trustworthy?

Accuracy alone is not enough. This file asks whether the probabilities themselves can be trusted. It is run with `--evaluate`.

**`collect_forecasts(prices, n_days=60)`** walks the model forward one day at a time over the last N trading days. For each day it makes a forecast using only earlier data (through `forecast_on_date`), and records the forecast, whether the price actually rose, and the next close. Days without a known outcome are skipped.

**`metrics(rows)`** calculates the scores:

- **Accuracy:** share of days where "above 50%" matched the outcome.
- **Precision:** of the days the model said "up", how many really rose.
- **Recall:** of the days the price really rose, how many the model caught.
- **Brier score:** the average of `(probability − outcome)²`. Lower is better.
- **Mean predicted probability:** the average forecast. Far from 50% suggests the model leans one way.

The line `preds = [int(p > 0.5) ...]` uses `int` on purpose. In Python, `False == 0` is true, and the conversion makes the comparison clearer.

**`reliability_bins(rows, n_bins=5)`** groups forecasts into buckets (0–20%, 20–40%, and so on). For each bucket it compares the average predicted probability with how often the price really rose. If a bucket predicting "60%" rose about 60% of the time, the model is well calibrated.

**`print_report(rows, m, log)`** prints these numbers in plain language, including a verdict on the Brier score and whether the model leans up or down on average.

**`plot_evaluation(rows, math_report, out_dir)`** draws two charts and saves them as PNG files in `results/`:

1. `evaluation.png`: a calibration curve (left) and a running count of correct calls (right).
2. `feature_weights.png`: a horizontal bar chart of the current model's weights.

`matplotlib.use("Agg")` lets charts be saved without a screen, so the program works on servers too.

**`evaluate_run(...)`** ties it together: collect, measure, print, draw, and return a summary for `--json`.

---

### 5.9 `oil_ai/report.py`: everything you read on screen

This file only prints. It takes results from the other modules and turns them into text for you.

**`outlook(...)`** prints the main summary box: the closing price, the accuracy on unseen days (with the baseline), the probability and whether it points up or down, the number of headlines, how many lean up or down, the average news score, the spread, the overall tone, and a breakdown by driver. Each line is followed by a short "->" explanation.

**`plain_english(prob_rise, analyses)`** gives the verdict in five numbered parts:

1. How the news tone changed day by day, comparing the first and last day's tone category (not raw averages, which can mislead when headline counts differ).
2. The newest day's net tone and the strongest driver. If some drivers push up and others push down, it says they disagree.
3. The price model's probability and which way it leans.
4. Whether the news and the price model **agree** or **disagree**, and what that means for confidence.
5. A reminder that this is a probability, not a promise.

It also explains in words how each number was built.

**`final_paragraph(prob_rise, analyses)`** writes one short paragraph at the end: the likely direction, the price model's reason, and today's news. It also says whether the news supports the call, is lukewarm, or disagrees, and adjusts the confidence wording to match.

**`time_travel_report(prices, date_str, model_name)`** prints the `--date` output: the forecast, what actually happened next, and whether it was right. It also explains that old news cannot be re-fetched, because the free feeds only cover about three days.

**`backtest(prices, n_days=10)`** replays the price model over the last N days, prints each forecast beside the real outcome, marks each one with `+` (right) or `x` (wrong), and reports the hit rate.

> **Note:** the backtest prints the line "An honest target to compare against: ~57%" as fixed text. That number is not calculated, so it should not be read as a measured result.

---

### 5.10 `experiments.py`: the accuracy lab

This is a separate script, not called by the main program. It answers: "Can we honestly beat 50% accuracy?"

- **`load_brent_prices()`** reads the cache. If it is missing, it tells you how to create it.
- **`features_basic` / `features_rich`** build the same kinds of features described in section 5.5.
- **`FEATURE_SETS`** and **`REGULARIZATIONS`** list the options to try.
- **`run()`** does the comparison:
  1. Builds each feature set and the target.
  2. Splits the data into train (all but the last 500), valid (250 days), and test (last 250 days).
  3. Trains a model for each combination of feature set and regularization.
  4. Ranks the models by **validation** accuracy.
  5. Shows the top four, with their **test** accuracy. Only the top four are shown, and the test scores are reported as the honest final result.

Its message is the lesson: if test scores are close to validation scores, the finding is trustworthy. If they swing widely, any "improvement" was probably noise.

The split here is 500/250/250, while `model.py` uses 250/250 for validation and test. Both approaches follow the same rule of never letting the model see the future.

---

### 5.11 `simple_model.py`: the learning version

This script is short and heavily commented, and it is meant to be read line by line. It does the same job as the main model with fewer features:

1. Load the cached prices.
2. Build three features: yesterday's return, the 5-day return, and volatility.
3. Build the target: did tomorrow close higher?
4. Split the data in date order: first 80% for learning, last 20% for testing.
5. Train a logistic regression with balanced class weights.
6. Print the accuracy on the unseen 20%.
7. Print each feature's weight.
8. Show today's forecast step by step: feature value, times weight, sum, then the sigmoid formula.

Read it if you want the core idea without the extra machinery.

---

### 5.12 `oil_ai/__init__.py`

This file only has a docstring. It describes the package layout. Python uses it to recognise `oil_ai` as a package, so the other files can import from it.

---

### 5.13 `requirements.txt`

A list of exact package versions the project needs. The most important ones:

- `pandas`, `numpy`: data handling and maths.
- `scikit-learn`: the logistic regression and the scaler.
- `yfinance`: downloads prices from Yahoo Finance.
- `requests`: talks to the AI server and news feeds.
- `python-dotenv`: reads the `.env` file.
- `matplotlib`: draws the charts.
- `beautifulsoup4`, `lxml`, `curl_cffi`, and others: supporting packages. The project does not import them directly.
- `pytest`: runs the tests.

Install them all with `pip install -r requirements.txt`.

---

### 5.14 `tests/test_pipeline.py`

Automated checks that confirm the core logic works. They run without internet access or an AI. The file has 27 tests in these groups:

- **Date parsing:** `pub_day` reads GMT/UTC timestamps correctly and handles bad input.
- **News maths:** `aggregate` counts directions, averages scores, and applies the tone thresholds.
- **Grouping:** `drivers_table` skips empty drivers, and `day_trends` sorts by date.
- **Features:** the feature builders produce the right columns and no missing values at the end.
- **Splitting:** train, valid and test do not overlap and keep time order.
- **No look-ahead:** a key test replaces all prices after a chosen day with absurd values. The forecast for that day must not change. If it does, the model has cheated.
- **Archive:** saving and summarising work, and an empty archive is handled safely.
- **Evaluation maths:** perfect forecasts score a Brier of 0; always guessing 50% scores 0.25; precision, recall and bins behave as defined.

Run them with `python3 -m pytest tests/ -v`.

---

## 6. Glossary of the maths, in one place

- **Return:** `today's close ÷ yesterday's close − 1`. Example: 80 → 82 is a return of `0.025`, or +2.5%.
- **Volatility:** the standard deviation of recent returns. Higher means jumpier prices.
- **Standardize:** `(value − average) ÷ spread`. Shows how unusual a value is.
- **Score (logit):** the sum of each feature's standardized value times its weight, plus the intercept.
- **Probability:** `1 ÷ (1 + e^(−score))`. A score of 0 gives 50%; a large positive score gives a value near 100%.
- **Brier score:** the average of `(probability − outcome)²`, where the outcome is 1 for a rise and 0 for a fall.

---

## 7. The statistical models, explained in depth

This section explains every statistical model and calculation in the project, from raw prices to the printed probability. It is the "how does the maths actually work" part of the guide.

The project uses **no deep learning**. Everything is classical statistics: simple summary numbers, one type of classifier (logistic regression), and standard measures of how good the predictions are. The news analysis uses a language model, which is covered separately in section 7.12.

### 7.1 What kind of problem this is

The question "up or down tomorrow?" is a **binary classification** problem. There are two outcomes, 1 (rose) or 0 (did not rise). The model does not just pick a side; it outputs a **probability** between 0 and 1. A probability is more useful than a yes/no answer because it shows how confident the model is.

This is also **supervised learning**: the model learns from past examples where the outcome is already known.

### 7.2 Step 1: turning prices into features

Raw prices are not useful to a model on their own. A price of $80 means little without context. So the program converts prices into **features**, numbers that describe the recent market.

**Returns.** The basic building block is the percentage change:

```
return_k(t) = P(t) / P(t - k) - 1
```

For example, if the price was 80 five days ago and is 82 today, `return_5d = 82 / 80 - 1 = 0.025`, or +2.5%. In code this is `close.pct_change(k)`.

**Volatility.** Volatility measures how jumpy recent prices are. It is the **sample standard deviation** of the last 10 one-day returns:

```
vol_10d(t) = sqrt( sum over the last 10 days of (r_i - mean_r)^2 / 9 )
```

The divisor is 9 (one less than 10) because this is a sample, not the whole population. Pandas does this by default. In code: `ret.rolling(10).std()`.

**Moving-average distance** (rich set). `close / close.rolling(20).mean() - 1` measures how far today's price sits above (positive) or below (negative) its 20-day average. A common trading idea is that prices tend to drift back toward their average. This is a hypothesis the model can test, not a fact.

**Position in range** (rich set). `(close - min20) / (max20 - min20)` gives a number between 0 and 1. It is 0 when today's price is the lowest of the last 20 days and 1 when it is the highest.

**Volume z-score** (rich set). `volume / average_volume_20 - 1`. Positive values mean unusually heavy trading.

Each feature is computed **using only past data**. None of them peek at tomorrow.

The **basic** set has 3 features. The **rich** set has 12 features. More features give the model more to work with, but also more chances to fit noise.

### 7.3 Step 2: the target (what we predict)

```
y(t) = 1   if  P(t+1) > P(t)
y(t) = 0   otherwise (fell or stayed flat)
```

The target uses tomorrow's price, which is the thing we want to forecast. That is why the last row of data has no target and is dropped before training.

About half the days are up and half are down, so the two outcomes are close to balanced. This matters for the baseline (section 7.10).

### 7.4 Step 3: standardization (the StandardScaler)

Features live on different scales. A 10-day volatility might be `0.01`, while a volume z-score might be `2.3`. Regularization (section 7.5) penalises large weights, and a feature on a bigger scale would get a smaller weight for the same effect. To make the comparison fair, each feature is standardized:

```
z = (x - mu) / sigma
```

where `mu` is the average and `sigma` is the standard deviation of that feature.

The key rule: **`mu` and `sigma` are learned from the training data only.** The `StandardScaler` sits inside the pipeline, so when the model fits on the training block, it learns the training averages. Validation and test data are then transformed using those same numbers. If the scaler saw the test data, the test would leak information.

The one exception is the final forecast. After the winner is chosen, the whole pipeline is refitted on all history (section 7.7), so the scaler's averages and spreads are then recalculated from all the data. This is safe because the forecast is for a day that comes after all of that data.

After standardizing, a value of `z = 0` means "average" and `z = 2` means "two typical steps above average."

### 7.5 Step 4: logistic regression (the model itself)

**The model.** Logistic regression computes a weighted sum of the standardized features, then squeezes it into a probability:

```
score = b + w1*z1 + w2*z2 + ... + wk*zk
probability_up = 1 / (1 + e^(-score))
```

- `w1 ... wk` are the **weights** (coefficients). The model learns them.
- `b` is the **intercept**. It is the starting score before any feature is considered.
- The function `1 / (1 + e^(-score))` is the **sigmoid** (or logistic) function. It turns any number, from minus infinity to plus infinity, into a value between 0 and 1. A score of 0 gives 50%. A large positive score approaches 100%. A large negative score approaches 0%.

**How to read the weights.** Because the features are standardized, the weights can be compared directly. A weight of `+0.20` on `ret_5d` means that when the 5-day return is one standard deviation above average, the score rises by 0.20. Equivalently, the odds of an up day are multiplied by `e^0.20`, about 1.22.

**How the weights are learned.** The model is fitted by **maximum likelihood**: it adjusts the weights to make the observed outcomes as probable as possible. For each training day, the **log-loss** measures how wrong the prediction was:

```
loss = -[ y * log(p) + (1 - y) * log(1 - p) ]
```

If the outcome was 1 and `p` was 0.9, the loss is small. If the outcome was 1 and `p` was 0.1, the loss is large. The training process minimises the total loss across all training days.

**Regularization (the C parameter).** Scikit-learn's version minimises:

```
0.5 * (sum of squared weights)  +  C * (total log-loss)
```

The first term is a **penalty** that pulls weights toward zero. The parameter `C` controls the trade-off:

- **Small C** (for example 0.001): strong penalty, weights stay small, the model is simpler and less likely to memorise noise.
- **Large C** (for example 0.1): weak penalty, weights can grow, the model fits the training data more closely.

The project tries `C` values of 0.001, 0.01 and 0.1 and lets the validation set choose. This is the standard way to avoid **overfitting**, where a model learns the quirks of the training period that will not repeat.

**Balanced class weights.** `class_weight="balanced"` gives each outcome a weight inversely proportional to how common it is:

```
weight_for_class_c = n_total / (2 * n_class_c)
```

If up days were 60% of the data, up-day mistakes would count less per day than down-day mistakes, so the model cannot score well just by always guessing "up."

**Solver and iterations.** The default solver (`lbfgs`) is an iterative method that adjusts the weights step by step. `max_iter=5000` gives it up to 5,000 steps to settle. If it does not converge, scikit-learn prints a warning.

**Decision rule.** The model predicts "up" when `probability > 0.5`, otherwise "down."

### 7.6 Step 5: choosing the best setup (validation)

The program builds six candidates: two feature sets (basic, rich) times three regularization strengths. Each candidate is trained on the **training block** (the oldest data, all but the last 500 days in the auto-tuned model).

Each candidate is then scored on the **validation block** (the next 250 days). The highest validation accuracy wins.

Then the winner is graded **once** on the **test block** (the final 250 days). This test score is the honest result that gets printed.

Two things to keep in mind:

- **Selection bias.** Picking the best of six candidates on validation makes the validation score look a little better than it really is. That is why the test score is reported separately.
- **Sampling noise.** The accuracy on 250 days is itself uncertain. The **standard error** of an accuracy of about 50% on 250 days is:

```
sqrt(0.5 * 0.5 / 250) = 0.032   (about 3.2 percentage points)
```

A 95% range is roughly plus or minus 6 points. So a result of 52% is well inside the range you would see from pure chance. The project's conclusion, that the price model is near a coin flip, follows directly from this.

### 7.7 Step 6: refitting and forecasting today

After the winner is chosen, it is **refitted on all the history** (every row that has a known target). This uses the most data for today's forecast. The model then takes today's feature values and returns `predict_proba`, the probability of "up."

### 7.8 The arithmetic, shown step by step

Because logistic regression is a weighted sum, the final probability can be broken into parts that add up exactly. For each feature, the **push** is:

```
push_j = z_j * w_j      (today's standardized value times its weight)
```

The score is the sum of all pushes plus the intercept, and the probability is the sigmoid of that score. The program prints this table (the function `show_forecast_math`) and includes an `assert` (a self-check) that confirms the hand-calculated probability matches the library's answer. If they differed, the program would stop.

### 7.9 Evaluating the model over time (walk-forward)

A single test block is one snapshot. A **walk-forward** evaluation gives a more realistic picture. For each of the last N days:

1. Train a model using only data from before that day.
2. Predict that day's probability.
3. Record whether the price actually rose the next day.

This is what `--backtest` and `--evaluate` do, and it is run through `forecast_on_date`.

> **Important difference:** `forecast_on_date` uses a **simpler, 2-feature model** (yesterday's return and the 5-day return). It does **not** use the auto-tuned model that produces the headline probability. So the backtest and the evaluation grade the simpler model, not the one in the main forecast. The only exception is the feature-weight chart from `--evaluate`, which shows the auto-tuned model's weights.

The 2-feature model in `forecast_on_date` is trained as follows:

- It uses every trading day up to the chosen date.
- It drops the final row, because that row's target is whether the price rose on the day *after* the chosen date. That is the outcome the forecast is trying to predict, so it must not be used for training.
- It requires at least 40 days of history.

### 7.10 The metrics, explained

**Accuracy** is the share of forecasts on the right side of 50%:

```
accuracy = (number of correct calls) / (number of days)
```

**Precision** is: of the days the model said "up," how many really rose:

```
precision = true positives / (days predicted up)
```

**Recall** is: of the days that really rose, how many the model caught:

```
recall = true positives / (days that actually rose)
```

**Brier score** measures the quality of the probabilities, not just the up/down call:

```
Brier = average of (probability - outcome)^2
```

It is 0 for perfect forecasts. Always guessing 50% gives `(0.5 - y)^2 = 0.25` every day, so the Brier score is 0.25. Anything above 0.25 is worse than guessing. The verdict words in the program ("excellent", "decent", "poor") use cut-off points of 0.20 and 0.24 chosen in the code. They are rules of thumb, not formal statistical tests.

**Naive baseline.** This is the accuracy you would get by always predicting whichever outcome is most common. Because the outcomes are close to balanced, this is close to 50%. A model must beat this to add any value.

**Calibration bins.** Forecasts are grouped into buckets by their probability (for example 40–60%). In each bucket the program compares the **average predicted probability** with the **observed share that rose.** A well-calibrated model that says "60%" should see about 60% of those days rise. This is a different question from accuracy: a model can be right 52% of the time and still have badly calibrated probabilities.

### 7.11 The other statistics in the project

Some calculations in the project are **descriptive statistics**. They summarise, but do not learn anything:

- **Average score (mean):** the sum of the scores divided by the count.
- **Spread (sample variance):** the sum of squared distances from the mean, divided by `n - 1`. High spread means the headlines disagree.
- **Tone:** "bullish" if the average is above +5, "bearish" if below -5, otherwise "neutral." The dead zone around zero stops tiny averages from being treated as a real trend.
- **Counts:** how many headlines were bullish or bearish, per day and per driver.

### 7.12 The news AI is not a statistical model

The headline scoring is done by a **large language model** (LLM), not by a formula the project trained. The project does not train or fine-tune it. It only:

1. sends the system prompt and one headline,
2. asks for a JSON reply with an impact score,
3. parses the JSON and clamps the score to -100..+100.

This is called **zero-shot classification**: the model classifies text it has never been specifically trained on, guided only by instructions. The `temperature` setting of 0.2 makes its answers more consistent, but the scores are still the model's judgment. They have not been checked for calibration, so they should not be treated as probabilities.

### 7.13 The other scripts' models

- **`simple_model.py`:** the same logistic regression with 3 features (yesterday's return, 5-day return, volatility). It splits the data 80% for training and 20% for testing, with no validation step, because it has only one model to compare. It prints every weight and every step of today's calculation.
- **`experiments.py`:** a fair comparison of 2 feature sets times 3 regularization strengths. It uses 500 days for training, then 250 for validation and 250 for testing. It ranks by validation accuracy and shows the top four with their test scores.
- **`forecast_on_date`:** a 2-feature logistic regression refitted for each chosen date, using only data up to that date (see section 7.9).

### 7.14 How the pieces fit together

```
Prices (CSV)
   |
   v
Returns and features  --(basic or rich)-->  feature table  (one row per day)
   |
   v
Standardize (fit on training data only)
   |
   v
Logistic regression, C chosen on validation
   |
   v
Probability of rise tomorrow  --> printed forecast, math table, and charts

Headlines (RSS)
   |
   v
LLM gives each headline a score (-100..+100)
   |
   v
Descriptive stats: counts, average, spread, tone  --> printed verdict
   |
   v
SQLite archive  --> future news-aware model (not built yet)
```

### 7.15 Why the model is so weak (the statistical reasons)

Several reasons explain why the accuracy stays near 50%:

- **Weak signal.** Past price patterns explain very little of tomorrow's move. Many economists believe that prices already reflect the information available, which leaves little for a simple model to find. This is a widely discussed hypothesis, not a settled result.
- **Small sample.** Five years of trading days is roughly 1,250 rows. After dropping rows where the rolling windows are incomplete, the rich set has around 1,200. With a standard error of about 3 points at 250 days, small improvements cannot be told apart from luck.
- **Changing conditions.** Oil markets shift between regimes (calm periods, supply crises, price wars). A pattern that worked in one regime may not work in the next.
- **Few and simple features.** The models only see past prices and volume. They do not see the news, which is probably where much of the real information lies.
- **Honest testing.** The project refuses to tune on the test set, so its results are modest. A tuned model would look better on paper, but worse in use.

---

## 8. Why the results are honest (and modest)

Three habits keep the project honest:

1. **Chronological splits.** The model never learns from data that comes after the day it is predicting.
2. **Choose on validation, grade on test once.** Tuning on the test set would make the results look better than they are.
3. **Report the boring truth.** Accuracy is close to 50%, and the program says so every time.

The main lever for real improvement is the news. The archive is collecting labelled headlines so that a future model can learn from news as well as prices. At the moment, news scores are displayed but not used in the forecast.

---

## 9. Known limitations and quirks

- **The price cache never refreshes automatically.** Delete `data/brent_prices.csv` to get newer data.
- **Free news feeds cover only about the last 3 days.** Old headlines cannot be re-fetched for backtests.
- **The "baseline" number is not a true baseline.** See the note in section 5.5.
- **The backtest's "~57%" line is fixed text,** not a measured value. See the note in section 5.9.
- **Headline counts are rounded down.** With `--headlines 8`, the program fetches 6 headlines (one per driver), not 8.
- **The news is not used in the forecast yet.** The verdict reports news separately from the price probability.
- **The `--evaluate` charts reuse the same walk-forward model as `--backtest`.** They are a diagnostic, not a separate model.

---

## 10. Where to go next

- To **run it**: see `README.md`.
- To **change what news is searched**: edit `DRIVERS` in `oil_ai/config.py`.
- To **change the AI's instructions**: edit `SYSTEM_PROMPT` in `oil_ai/config.py`.
- To **try different model settings**: edit `CANDIDATES` in `oil_ai/model.py`, or run `experiments.py`.
- To **learn the core idea first**: read `simple_model.py` from top to bottom.
