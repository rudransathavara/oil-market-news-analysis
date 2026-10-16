# How This Project Works (Simple Guide)

This guide explains the project in plain English. You do not need to know Python, machine learning, or oil trading to read it.

Read it from top to bottom once. After that, use the section titles to find what you need.

> **This project is for learning and research. It is not financial advice.**

---

## 1. What does this project do?

Every day it tries to answer one question:

> **"Will the price of Brent oil go UP or DOWN tomorrow?"**

Brent is a famous type of crude oil. Its price changes every day, and people who buy and sell oil watch it closely.

The project uses two kinds of information to answer:

1. **Old prices.** It looks at how the price moved in the past. For example: "It went up yesterday, and it has been jumping around a lot this week."
2. **News.** It reads recent news headlines about oil. An AI reads each headline and gives it a score from **-100** (bad for prices) to **+100** (good for prices).

At the end, it prints:

- today's price,
- its guess for a rise tomorrow (as a percentage),
- the average news score,
- a short plain-English summary,
- an honest check of how well its guesses did in the past.

**The big secret:** the price guess is only a little better than flipping a coin. The project says this every time it runs. Please keep that in mind.

---

## 2. Words you need to know

| Word | Simple meaning |
|---|---|
| **Price** | What one barrel of oil costs. |
| **Close** | The price at the end of a trading day. The project only uses this number. |
| **Trading day** | A day when the market is open. Weekends and holidays are skipped. |
| **Return** | How much the price changed, as a fraction. `0.02` means up 2%. |
| **Probability** | How likely something is, from 0 to 1. `0.60` means 60% likely. |
| **Feature** | A number we work out from old prices to describe the market. This is the model's input. |
| **Target** | The answer the model is trying to guess. Here: "did the price go up tomorrow?" |
| **Model** | A formula that turns features into a guess. |
| **Weight** | A number that says how much one feature matters in the formula. |
| **Training** | Letting the model find good weights by studying old data. |
| **Test** | Checking the model on days it has never seen. |
| **Accuracy** | The share of guesses that were right. `0.50` is the same as flipping a coin. |
| **Overfitting** | When a model memorises the random noise in old data. It looks great on old data but fails on new data. |
| **Look-ahead bias** | Cheating by letting the model see the future. This makes results look better than they really are. |
| **LLM** | A large language model. It is the AI that reads and scores each headline. |
| **Brier score** | A score for how good the probabilities are. Lower is better. `0` is perfect. `0.25` is what you get by always saying 50%. |
| **Calibration** | Whether the guesses are honest. If the model says "60%" many times, about 60% of those should really go up. |
| **Driver** | One of six topics that move oil prices. (See section 6.) |
| **RSS** | A simple feed format that news sites use to publish headlines. |
| **SQLite** | A small database that lives in one file. No server needed. |
| **Backtest** | Replaying the model on past days to see how it would have done. |

---

## 3. How the program works, step by step

When you run `python3 one_file.py`, this happens in order:

```
1. Read your settings (the .env file)
2. Load the oil prices
      - from the saved CSV file, if it exists
      - otherwise, download them
3. Train the price model
      - try 6 versions
      - keep the best one
4. Show the math behind today's guess
5. Get recent news headlines
6. Ask the AI to score each headline
7. Save the scored headlines to a database
8. Print the outlook, the verdict, and a short summary
```

Each step is in its own file. This is on purpose. Each file does one job, so each part is easier to understand and test.

---

## 4. Part A: The price model (how the guess is made)

This part uses only old prices. It answers: *"Given how the price has been moving, what is the chance it goes up tomorrow?"*

### Step A1: Turn prices into features

Raw prices alone are hard to use. The program turns them into numbers that describe the market.

Here is a simple example. Suppose the price was 80 five days ago, and it is 82 today.

- **Return over 5 days** = 82 ÷ 80 − 1 = 0.025, which is **+2.5%**.

**Volatility** means how jumpy the prices are. It is the typical size of the daily ups and downs over the last 10 days. A high number means a jumpy market. A low number means a calm market.

The program has two feature sets:

- **Basic (3 features):**
  1. yesterday's return,
  2. the return over the last 5 days,
  3. volatility over the last 10 days.
- **Rich (12 features):** the 3 basic ones, plus more time windows (2, 10, and 20 days), more volatility windows, how far the price is from its average (20-day and 50-day), where the price sits in its recent high-low range, and a volume number.

**Important:** every feature uses only past days. None of them can see tomorrow.

### Step A2: Decide what to predict (the target)

For each day, the target is:

- **1** if tomorrow's close is higher than today's close,
- **0** if not (down, or exactly the same).

This is the answer the model learns from.

### Step A3: Make the numbers comparable (scaling)

Features use different units. Volume might be in thousands, while a return might be 0.02. The program adjusts each feature so that its average is 0 and its typical spread is 1. This is called **standardizing**.

After this step:
- `0` means "normal, average for this market,"
- `+2` means "about two typical steps above normal."

The averages used for this adjustment come **only from the training data**. This matters, because the test data must stay unseen.

### Step A4: The formula (logistic regression)

The model is a simple formula. Here is how it works:

1. Multiply each standardized feature by its **weight**.
2. Add all of those results together.
3. Add a starting number called the **intercept**.
4. This final number is called the **score**.
5. Turn the score into a probability with the **sigmoid** formula:

```
probability = 1 / (1 + e^(-score))
```

The sigmoid always gives a number between 0 and 1:

- score = 0 → probability = 50%
- score = large and positive → probability close to 100%
- score = large and negative → probability close to 0%

**Example:** if the score is 0.4, the probability is about 0.60, or **60%**. If the score is −0.4, the probability is about 40%.

**What the weights mean:** a positive weight means that feature pushes the guess toward "up." A negative weight pushes it toward "down." A big weight means the feature matters more.

**Important:** a weight does not prove that a feature *causes* price moves. It only shows that the feature helped the model's guesses on the old data.

### Step A5: Choose the best version (the 6 candidates)

The program tries **6 versions**:

- 2 feature sets (basic and rich),
- 3 strengths of a setting called **regularization** (0.001, 0.01, and 0.1).

Regularization is a rule that keeps weights from getting too big. Small values keep the model simple. This helps stop overfitting.

The data is split into three blocks, in date order:

| Block | Which days | What it is used for |
|---|---|---|
| **Train** | The oldest days (everything except the last 500) | Learn the weights |
| **Validation** | The next 250 days | Pick the best of the 6 versions |
| **Test** | The newest 250 days | Grade the chosen version **one time** |

The rule is: **always train on the past and test on the future.** Random shuffling would let the model see the future by accident, so the program never does that.

### Step A6: Make today's guess

1. The winning version is trained again, this time on **all** the known history. This gives it the most data.
2. It is given today's features.
3. It prints today's probability of a rise tomorrow.

### Step A7: Show the math

The program prints a table so you can check the calculation yourself. For each feature, it shows:

- today's value,
- the standardized value,
- the weight,
- the **push** (standardized value × weight).

Then it adds all the pushes and the intercept to get the score, and turns the score into the probability. A built-in check makes sure the table's answer matches the model's answer. If they did not match, the program would stop.

---

## 5. Part B: The news part (how headlines are scored)

This part answers: *"What is the news saying about oil right now?"*

### Step B1: Get headlines

The program downloads headlines from **Google News RSS feeds**. It searches once for each of the six drivers (see section 6).

- It asks for 12 headlines by default, which is 2 per driver.
- If one feed fails, it prints a message and keeps going with the others.
- It removes the publisher name at the end of each title. For example, "Oil rises - Reuters" becomes "Oil rises."
- It tags each headline with its driver. The program does this itself. It does not ask the AI, because the AI can get that wrong.

### Step B2: Ask the AI to score each headline

The AI gets:

- **instructions** (called a system prompt), and
- **one headline**.

The instructions tell the AI to:

- act like an oil market analyst,
- pick one driver for the headline,
- give an impact score from **-100** to **+100**,
- reply only in a structured format called **JSON** (a neat text format that a program can read).

**Simple meaning of the score:**

- **Negative** (for example −35): the headline is bearish, which means it pushes prices **down**.
- **Positive** (for example +40): the headline is bullish, which means it pushes prices **up**.
- **Near zero:** little effect, or mixed news.

**Important:** a score of +80 does **not** mean an 80% chance of a rise. It is a rating, not a probability.

The program also:

- sets the AI's "creativity" low (0.2), so answers are more consistent,
- pulls the JSON out of the reply, even if the AI added extra words,
- limits every score to the range −100 to +100,
- skips a headline if something goes wrong, and says so.

If `LLM_API_KEY` is missing, the AI step is skipped. For Ollama (a free AI that runs on your computer), you still need to set the key to any value, such as `ollama`.

### Step B3: Summarise the news

The program combines all the scores:

- **Average score:** the middle point of all scores.
- **Bullish and bearish counts:** how many headlines lean up and how many lean down.
- **Spread:** how much the scores disagree with each other. A big spread means mixed news.
- **Tone:**
  - above +5 on average → **bullish**,
  - below −5 on average → **bearish**,
  - in between → **neutral**.
  The small "neutral" zone stops tiny averages from looking like a trend.
- **By driver:** the same summary for each of the six drivers.
- **By day:** how the tone changed from day to day.

### Step B4: Save the headlines

Every scored headline is saved in the SQLite database `data/news_archive.sqlite`. Each row stores the day, driver, title, direction, score, and which AI model was used. Over time this builds a history that could be used later.

---

## 6. The six drivers

A "driver" is a topic that can push oil prices. The program searches news for each one:

| Driver | The question it asks |
|---|---|
| **OPEC+ policy** | Will the oil-producing group change how much oil it pumps? |
| **Geopolitics** | Could a war, sanctions, or trouble at sea disrupt oil supply? |
| **US inventories** | Are US oil stocks growing or shrinking? |
| **Dollar and macro** | Do currency, interest rates, or growth worries change oil demand? |
| **Price action** | Does the news describe price moves or chart levels? |
| **Market expectations** | Is the news a surprise, or was it already expected? |

**Bullish** means "good for higher prices." **Bearish** means "good for lower prices." These words describe the likely price effect. They do not mean good or bad for people.

Example: more oil supply is usually bearish for price. A supply disruption is usually bullish. But prices react to *surprises*, so the market may already expect the news.

---

## 7. The files in this project

```
one_file.py            The program you run. It only starts the main code.
experiments.py         A separate lab. It compares model versions fairly.
simple_model.py       A short, easy version of the price model. Best place to start learning.
oil_ai/                The main code, split into smaller files:
  __init__.py            Just a description of the folder. No logic.
  config.py              All settings in one place (ticker, news searches, AI settings).
  cli.py                 Reads your options and runs the steps in order.
  prices.py              Loads and saves the oil prices.
  model.py               The price model: features, training, guesses, and the math.
  news.py                Gets headlines, asks the AI to score them, and summarises.
  archive.py             Saves scored headlines to the database.
  evaluate.py            Checks how honest the probabilities are, and draws charts.
  report.py              Prints all the results on screen.
tests/test_pipeline.py Automatic checks that the core logic works.
requirements.txt       The Python packages the project needs.
.env                   Your private settings, such as API keys. Never share this file.
```

### What the most important files do

- **`one_file.py`**: the front door. It just calls `main()`.
- **`oil_ai/cli.py`**: the boss. It decides which steps to run and in what order.
- **`oil_ai/prices.py`**: loads the prices. If the CSV file exists, it reads that. If not, it downloads five years of prices and saves them.
- **`oil_ai/model.py`**: the price model. This is where the features, the 6 versions, and the probability are made.
- **`oil_ai/news.py`**: gets headlines and asks the AI to score them.
- **`oil_ai/config.py`**: the place to change settings, such as the driver searches and the AI's instructions.
- **`oil_ai/evaluate.py`**: checks whether the probabilities can be trusted.
- **`oil_ai/archive.py`**: the database for scored headlines.
- **`oil_ai/report.py`**: prints the results. It does not make any calculations.

---

## 8. How to run it

### Set up (one time)

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Add your settings

Create a `.env` file with your AI settings. For a free local AI with Ollama:

```
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:4b
LLM_API_KEY=ollama
```

Never put a real secret key into a shared file or a public place.

### Run it

| Command | What happens |
|---|---|
| `python3 one_file.py` | Full run: price model and news |
| `python3 one_file.py --skip-news` | Price model only. No news and no AI |
| `python3 one_file.py --headline "OPEC+ announces a production cut"` | Score one headline you type |
| `python3 one_file.py --headlines 12` | Fetch 12 headlines |
| `python3 one_file.py --date 2026-10-01` | What the model would have guessed on that past day |
| `python3 one_file.py --backtest 10` | Check the model on the last 10 trading days |
| `python3 one_file.py --evaluate 60` | Check how honest the probabilities are, over 60 days, and draw charts |
| `python3 one_file.py --archive-stats` | Summary of the news database |
| `python3 one_file.py --json` | Also save a summary to `results/run.json` |
| `python3 one_file.py --compare` | Run two local AI models on the news and compare them |

To see every option, run `python3 one_file.py --help`.

### Files it creates

- `data/brent_prices.csv`: the saved prices.
- `data/news_archive.sqlite`: the saved headline scores.
- `results/evaluation.png`: the calibration chart.
- `results/feature_weights.png`: a chart of the model's weights.
- `results/run.json`: a summary, if you used `--json`.

### Run the tests

```bash
python3 -m pytest tests/ -v
```

The tests check the core logic. They do not need internet or an AI. A passing test means the checked rules work. It does not mean the model makes money.

---

## 9. Why the results are honest (and modest)

The project uses three habits to avoid fooling itself:

1. **Train on the past, test on the future.** The model never learns from days after the day it is guessing.
2. **Pick on validation, grade on test once.** If you keep checking the test days and changing the model, the test days stop being "unseen."
3. **Report the plain truth.** The accuracy is close to 50%, and the program says so.

**Why is the accuracy so close to 50%?**

- **The signal is weak.** Old prices explain very little about tomorrow's move. Many experts think prices already include most public information.
- **The sample is small.** About 250 test days is not many. A result of 52% could easily be luck.
- **Markets change.** A pattern that worked last year may stop working.
- **The features are simple.** The model sees only prices and volume. It does not see news.

---

## 10. Known problems (be careful)

These are real limits you should know about. Each one is a good place to improve the project.

1. **The "baseline" number is not a real baseline.** The code shows `max(validation accuracy, 1 − validation accuracy)`. A real baseline would be a simple rule, such as "always guess up," tested on the same days.
2. **The "~57%" line in the backtest is fixed text.** It is not measured. Do not treat it as a real target.
3. **The price cache never updates by itself.** The saved CSV can get old, so the "current" price may be out of date. To refresh it, delete `data/brent_prices.csv` and run again.
4. **Free news feeds only cover about 3 days.** Old news cannot be downloaded again for testing.
5. **Headline counts are rounded down.** `--headlines 8` gives 6 headlines, because 8 ÷ 6 rounds down to 1 per driver.
6. **The news does not change the forecast.** The news score and the price probability are shown side by side. They are not combined into one number.
7. **News scores are not probabilities.** A score of +60 does not mean a 60% chance of a rise.
8. **Duplicates can distort news averages.** The same story repeated many times counts many times.
9. **Repeated testing makes results look better.** Trying many settings and picking the best one makes the result look better than it really is.
10. **Direction is not profit.** Being right 55% of the time can still lose money after costs. This project does not simulate trades.

---

## 11. Plain-English math summary

- **Return:** today's close ÷ yesterday's close − 1. Example: 80 → 82 is +2.5%.
- **Volatility:** how much the daily returns jump around. Higher means jumpier.
- **Standardize:** (value − average) ÷ typical spread. Shows how unusual a value is.
- **Score:** the sum of (standardized value × weight) for each feature, plus the intercept.
- **Probability:** 1 ÷ (1 + e^(−score)). Score 0 gives 50%.
- **Accuracy:** correct guesses ÷ total guesses.
- **Precision:** of the days the model said "up," how many really went up.
- **Recall:** of the days that really went up, how many the model caught.
- **Brier score:** the average of (probability − actual outcome)². Lower is better.

---

## 12. Glossary of the project's files and terms

- **Ticker `BZ=F`:** the Yahoo Finance code for Brent oil futures. A futures contract is an agreement to buy or sell oil at a set price on a later date.
- **CSV:** a plain text table file. Easy to open in Excel.
- **JSON:** a structured text format with named fields, such as `{"direction": "bearish"}`.
- **`.env`:** a private settings file for things like API keys.
- **`requirements.txt`:** the list of Python packages to install.
- **Virtual environment (`.venv`):** a private box for this project's packages, so they do not clash with other projects.

---

## 13. What to learn next, and how to improve the project

### Start learning here

1. Read `simple_model.py` from top to bottom. It is the smallest version of the price model.
2. Read `oil_ai/model.py`, section by section.
3. Read `oil_ai/news.py`, to see how the AI is asked for scores.
4. Run the tests, and read a few of them to see what they check.

### Ideas to make it better (in order)

1. **Make the data trustworthy.** Record the date of the last price. Add a refresh option, and warn when the data is old.
2. **Compare with a real baseline.** For example, "always guess up." Test both on the same days.
3. **Report sample sizes and dates** next to every result.
4. **Measure Brier score too,** not just accuracy.
5. **Save each headline's link, publisher, and time,** and remove duplicates.
6. **Check the AI's answers.** Make sure every field exists and has the right type.
7. **Test whether news helps.** Compare three versions on the same days: price only, news only, and price + news. Only trust news if it clearly beats price only.

---

## 14. Quick check: can you explain these?

- [ ] What question the project asks, and why the answer is a probability.
- [ ] The difference between a feature and a target.
- [ ] How to calculate a return, and what volatility means.
- [ ] What the sigmoid does with a score.
- [ ] Why the data is split into train, validation, and test in date order.
- [ ] Why a news score of +60 is **not** a 60% chance of a rise.
- [ ] What accuracy and Brier score each tell you.
- [ ] Why a 52% accuracy on 250 days is weak evidence.

If you can explain all of these, you understand the project.

---

## 15. Final word

The project is best seen as two simple tools working side by side:

- a **price model** that estimates the chance of a rise tomorrow, and
- a **news reader** that summarises recent headlines.

Its main value is that it shows each step of its math and tests itself on days it has not seen. Treat every result as a hypothesis to test, not a fact. No model can promise tomorrow's price.
