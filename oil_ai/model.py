"""MODEL — auto-tuned logistic regression on "does Brent rise tomorrow?".

Instead of one fixed model, every run compares several honest candidates on
a validation window (never on the test window) and keeps the best.
The recipe, the features and the choice all live here. This is the port of
the embedded experiments logic from the original one_file.py.
"""

import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATES = (0.001, 0.01, 0.1)   # regularization strengths to compare


def split_ranges(n, test_len=250, valid_len=250):
    """Chronological index slices: train -> validation -> test.
    Kept as its own (testable) function: the split must never overlap and
    must never let training data come after validation/test data."""
    return {
        "train": slice(0, n - test_len - valid_len),
        "valid": slice(n - test_len - valid_len, n - test_len),
        "test":  slice(n - test_len, n),
    }


def _features(kind, prices):
    """Basic = the original 3; Rich = the 12-feature momentum/volume set."""
    close = prices["close"]
    volume = prices["volume"] if "volume" in prices else pd.Series(0.0, index=prices.index)
    ret = close.pct_change()
    shared = {
        "ret_1d": ret, "ret_5d": close.pct_change(5),
        "vol_10d": ret.rolling(10).std(),
    }
    if kind == "rich":
        shared.update({
            "ret_2d": close.pct_change(2), "ret_10d": close.pct_change(10),
            "ret_20d": close.pct_change(20),
            "vol_5d": ret.rolling(5).std(), "vol_20d": ret.rolling(20).std(),
            "vs_ma20": close / close.rolling(20).mean() - 1,
            "vs_ma50": close / close.rolling(50).mean() - 1,
            "pos_in_range20": (close - close.rolling(20).min()) /
                              (close.rolling(20).max() - close.rolling(20).min()),
            "volume_z": volume / volume.rolling(20).mean() - 1,
        })
    return pd.DataFrame(shared, index=prices.index)


def _pipeline(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def train_auto_model(prices, log=print):
    """Compare candidates on VALIDATION days, pick a winner, grade ONCE on the
    untouched TEST block, then refit on all history for today's forecast."""
    close = prices["close"]
    y_all = (close.shift(-1) > close).astype(int).rename("rose")

    results, test_sets = [], {}
    for kind in ("basic", "rich"):
        feats = _features(kind, prices)
        data = pd.concat([feats, y_all], axis=1).dropna()
        if len(data) < 700:
            continue
        X, t = data.drop(columns="rose").reset_index(drop=True), data["rose"].reset_index(drop=True)
        n = len(data)
        sl = split_ranges(n)
        X_tr, X_va, X_te = X.iloc[sl["train"]], X.iloc[sl["valid"]], X.iloc[sl["test"]]
        y_tr, y_va, y_te = t.iloc[sl["train"]], t.iloc[sl["valid"]], t.iloc[sl["test"]]
        for C in CANDIDATES:
            pipe = _pipeline(C)
            pipe.fit(X_tr, y_tr)
            results.append((accuracy_score(y_va, pipe.predict(X_va)), kind, C, pipe))
            test_sets[(kind, C)] = (X_te, y_te)

    if not results:
        raise RuntimeError("Not enough price history to train (need ~700 rows in the CSV).")
    results.sort(key=lambda r: -r[0])
    valid_acc, kind, C, winner = results[0]
    X_te, y_te = test_sets[(kind, C)]
    test_acc = accuracy_score(y_te, winner.predict(X_te))  # graded ONCE, like experiments.py
    baseline = max(valid_acc, 1 - valid_acc)
    log(f"Auto-tuned model: features={kind} C={C}   "
        f"validation {valid_acc*100:.1f}%   test {test_acc*100:.1f}%   "
        f"naive baseline ~{baseline*100:.1f}%")

    # Refit on ALL known history, then forecast the latest day (features are
    # all available today; the target (tomorrow's close) is what we predict).
    feats = _features(kind, prices)
    data = pd.concat([feats, y_all], axis=1).dropna()
    winner.fit(data[list(feats.columns)], data["rose"])
    prob = winner.predict_proba(feats.iloc[[-1]])[0, 1]

    # Keep every step of the arithmetic so the output can SHOW the calculation
    # (probability = sigmoid of the weighted feature sum for TODAY's market):
    clf = winner.named_steps["logisticregression"]
    scaler = winner.named_steps["standardscaler"]
    today = feats.iloc[-1]
    z = (today - scaler.mean_) / scaler.scale_     # standardized feature values
    pushes = z * clf.coef_[0]                      # each feature's push on the score
    logit = float(pushes.sum() + clf.intercept_[0])
    math_report = {"config": f"{kind} C={C}",
                   "features": dict(zip(feats.columns, [float(v) for v in today])),
                   "z": dict(zip(feats.columns, [float(v) for v in z])),
                   "weights": dict(zip(feats.columns, [float(v) for v in clf.coef_[0]])),
                   "pushes": dict(zip(feats.columns, [float(v) for v in pushes])),
                   "intercept": float(clf.intercept_[0]), "logit": logit}
    assert abs(float(1 / (1 + math.exp(-logit))) - prob) < 1e-6   # the math must match

    return {"prob": float(prob), "last": float(close.iloc[-1]),
            "date": prices.index[-1].date(), "accuracy": test_acc,
            "baseline": baseline, "config": f"{kind} C={C}", "math": math_report}


def show_forecast_math(report):
    """Print the forecast's full arithmetic, so no step stays hidden."""
    m = report["math"]
    print(f"{'-' * 62}\n")
    print(f"  HOW THE PRICE MODEL GOT ITS NUMBER — the full calculation ({m['config']})")
    print(f"{'-' * 62}")
    print("  The model gives each feature a weight. For every feature:")
    print("  today's value -> standardize it (how far from normal) -> x weight = push")
    print(f"\n  {'feature':<16} {'value':>9} {'standardized':>13} {'weight':>8} {'push':>9}")
    for col in m["features"]:
        print(f"  {col:<16} {m['features'][col]:>9.4f} {m['z'][col]:>13.2f} "
              f"{m['weights'][col]:>8.3f} {m['pushes'][col]:>9.4f}")
    print(f"\n  step 1: add all the pushes together, plus the starting score "
          f"({m['intercept']:+.4f}):")
    print(f"          score = {m['logit']:+.4f}")
    print(f"  step 2: turn the score into a probability: 1 / (1 + e^(-score))")
    print(f"          probability = 1 / (1 + e^({m['logit']:+.4f})) "
          f"= {report['prob'] * 100:.1f}%  "
          f"-> leans {'UP' if report['prob'] > 0.5 else 'DOWN' if report['prob'] < 0.5 else 'NEITHER'}")
    print(f"{'-' * 62}\n")


# =========================================================================
# TIME TRAVEL: what WOULD the model have said on a past date?
# =========================================================================
# The golden rule for grading a dated prediction: on the forecast evening the
# model may only train on data whose OUTCOME was already known by then, and
# it may only see features of the chosen day. Never the day after. This is the
# same look-ahead prevention as the chronological split, applied to one date.


def forecast_on_date(prices, date_str):
    """Forecast made AS OF the chosen date (trains strictly on older data)."""
    ts = pd.Timestamp(date_str)
    loc = prices.index.searchsorted(ts, side="right") - 1   # last trading day on/before date
    if loc < 40:
        raise ValueError(f"Need ~40 trading days of history before {date_str} to train the model.")
    close = prices["close"].iloc[: loc + 1]
    features = pd.DataFrame({"yesterday_return": close.pct_change(),
                             "week_return": close.pct_change(5)})
    target = (close.shift(-1) > close).astype(int)
    data = pd.concat([features, target.rename("rose")], axis=1).dropna()
    usable = data.iloc[:-1]   # the final row's target would peek past the forecast date
    model = LogisticRegression(class_weight="balanced")
    model.fit(usable[["yesterday_return", "week_return"]], usable["rose"])
    chosen_day = prices.index[loc]
    prob = model.predict_proba(features.loc[[chosen_day]])[0, 1]
    actual = {"next_close": None, "next_date": None, "rose": None}
    if loc + 1 < len(prices):   # ⚠ only the OUTCOME fields read future data
        actual["next_close"] = prices["close"].iloc[loc + 1]
        actual["next_date"] = prices.index[loc + 1].date()
        actual["rose"] = actual["next_close"] > prices["close"].iloc[loc]
    return {"day": chosen_day.date(), "prob": prob, "close": float(close.iloc[-1]), **actual}
