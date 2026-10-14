"""
THE SIMPLEST POSSIBLE OIL MODEL  (~30 essential lines)
======================================================

The whole idea in one sentence:
    "We give the model a list of examples: (what the oil market looked like
    each day) and (whether oil went up the next day). It learns how the
    two are connected."

The three ideas behind it, in simple words:

1. FEATURES (X)  -> simple numbers that describe the past,
                    like "how much did oil move yesterday?"
2. TARGET (y)    -> the answer we want to predict:
                    1 = oil went up the next day, 0 = it did not.
3. LOGISTIC REGRESSION -> gives each feature a WEIGHT (how important it is),
   adds the weighted numbers together, and turns that sum into a
   probability between 0 and 1 that oil goes up tomorrow.

We keep the data in date order: the model learns from the older days and is
tested on the newer days it has never seen.
(The common beginner mistake is to mix all days together. Then the model
accidentally sees future prices while learning, and its test score looks
much better than it really is. That mistake is called look-ahead bias.)

Run:  source .venv/bin/activate && python3 simple_model.py
"""

import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---- 1. Load the historical prices (already downloaded and saved) --------
prices = pd.read_csv("data/brent_prices.csv", index_col="date", parse_dates=["date"])
close = prices["close"]

# ---- 2. FEATURES: what the market looked like ON each day ---------------
X = pd.DataFrame({
    "yesterday_return": close.pct_change(),    # yesterday's move
    "week_return": close.pct_change(5),        # the last 5 trading days' move
    "volatility": close.pct_change().rolling(10).std(),  # how jumpy recent days are
})

# ---- 3. TARGET: is tomorrow's close above today's? -----------------------
y = (close.shift(-1) > close).astype(int)      # 1 = went up, 0 = went down or flat

data = pd.concat([X, y.rename("rose_next_day")], axis=1).dropna()
print(f"Learning examples (trading days): {len(data)}\n")

X, y = data[X.columns], data["rose_next_day"]

# ---- 4. Split in date order: learn from the past, test on the future ----
split = int(len(data) * 0.8)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]
print(f"First {split} days -> for LEARNING. Last {len(data) - split} days -> "
      f"for TESTING (the model never saw them while learning).\n")

# ---- 5. TRAIN: the model picks a weight for each feature ----------------
model = LogisticRegression(class_weight="balanced")
model.fit(X_tr, y_tr)

# ---- 6. TEST: accuracy on days the model has never seen -----------------
accuracy = accuracy_score(y_te, model.predict(X_te))
print(f"How often it gets the direction right on unseen future days: "
      f"{accuracy * 100:.1f}%  (50% would be a coin flip)\n")

# ---- 7. WHAT DID IT LEARN? ----------------------------------------------
print("The weight it picked for each feature "
      "(a bigger positive number = pushes 'goes up' harder):")
for name, weight in zip(X.columns, model.coef_[0]):
    print(f"  weight for {name:<18}: {weight:+.3f}")

# ---- 8. TODAY'S FORECAST, WITH THE FULL CALCULATION SHOWN ---------------
today_values = X_te.iloc[-1]                    # yesterday's market picture
weights = model.coef_[0]
intercept = model.intercept_[0]                 # the starting score before features

print("\nTHE FULL CALCULATION for tomorrow (no hidden steps):")
print(f"  step 1: today's feature values (what the market looked like):")
for name, value in today_values.items():
    print(f"          {name:<18} = {value:+.4f}")

print(f"  step 2: multiply each feature value by its weight:")
pushes = {}
for name, value, weight in zip(X.columns, today_values, weights):
    pushes[name] = value * weight
    print(f"          {name:<18} ({value:+.4f}) x ({weight:+.3f}) = {pushes[name]:+.4f}")

score = sum(pushes.values()) + intercept
print(f"  step 3: add all the pushes together, plus the starting score "
      f"({intercept:+.3f}):")
print(f"          total score = {score:+.4f}")

prob_up = model.predict_proba(X_te.iloc[[-1]])[0, 1]
print(f"  step 4: turn the score into a probability "
      f"(the formula is 1 / (1 + e^-score)):")
print(f"          probability = 1 / (1 + e^({score:+.4f})) = {prob_up * 100:.1f}%")

print(f"\nANSWER: Brent is predicted to be HIGHER on the next trading day "
      f"with {prob_up * 100:.0f}% probability "
      f"-> it leans {'UP' if prob_up > 0.5 else 'DOWN' if prob_up < 0.5 else 'NEITHER'}.")
print(f"Remember: 49% vs 51% is a very weak signal — near a coin flip. "
      f"Not financial advice.")
