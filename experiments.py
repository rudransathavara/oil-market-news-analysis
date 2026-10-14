"""
EXPERIMENTS — can we honestly beat ~50% directional accuracy?
=============================================================

The golden rule this script enforces (this IS the interview lesson):
  TRAIN  -> the model learns here
  VALID  -> we COMPARE configurations here and pick the best
  TEST   -> used ONCE, at the end, only for the final pick.
If we tuned on the test set, we would fool ourselves into thinking we
improved — that's called data snooping, and it's why backtests lie.

Run:  python3 experiments.py        (uses the cached Brent CSV)
"""

import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_brent_prices():
    """Read the price CSV cache (kept fresh by one_file.py). If it is missing,
    tell the user how to create it instead of failing with a cryptic error."""
    path = "data/brent_prices.csv"
    if not os.path.exists(path):
        raise RuntimeError("No price cache found. Run `python3 one_file.py --skip-news` "
                           "once to download Brent prices, then re-run experiments.")
    return pd.read_csv(path).dropna(subset=["close"])


# ---- feature builders: each returns X = "what the market looked like" -----
def features_basic(df, volume):
    """The 3 features the simple model used."""
    close = df["close"]
    return pd.DataFrame({
        "ret_1d": close.pct_change(),
        "ret_5d": close.pct_change(5),
        "volatility": close.pct_change().rolling(10).std(),
    }, index=df.index)


def features_rich(df, volume):
    """A wider view: several momentum horizons, ranges and trend distances."""
    close = df["close"]
    ret = close.pct_change()
    return pd.DataFrame({
        "ret_1d": ret, "ret_2d": close.pct_change(2), "ret_5d": close.pct_change(5),
        "ret_10d": close.pct_change(10), "ret_20d": close.pct_change(20),
        "vol_5d": ret.rolling(5).std(), "vol_10d": ret.rolling(10).std(),
        "vol_20d": ret.rolling(20).std(),
        "vs_ma20": close / close.rolling(20).mean() - 1,
        "vs_ma50": close / close.rolling(50).mean() - 1,
        "pos_in_range20": (close - close.rolling(20).min()) /
                          (close.rolling(20).max() - close.rolling(20).min()),
        "volume_z": volume / volume.rolling(20).mean() - 1,
    }, index=df.index)


FEATURE_SETS = {"basic": features_basic, "rich": features_rich}
REGULARIZATIONS = [0.001, 0.01, 0.1]  # smaller = stronger penalty = simpler model


def run():
    prices = load_brent_prices().dropna(subset=["close"])
    close = prices["close"]
    volume = prices["volume"] if "volume" in prices else pd.Series(0.0, index=prices.index)
    y = (close.shift(-1) > close).astype(int).rename("rose")

    results = []
    for set_name, builder in FEATURE_SETS.items():
        data = pd.concat([builder(prices, volume), y], axis=1).dropna()
        X, target = data.drop(columns="rose"), data["rose"]
        n = len(data)
        slices = {"train": (slice(0, n - 500)),
                  "valid": (slice(n - 500, n - 250)),
                  "test": (slice(n - 250, n))}
        X_tr, y_tr = X.iloc[slices["train"]], target.iloc[slices["train"]]
        X_va, y_va = X.iloc[slices["valid"]], target.iloc[slices["valid"]]
        X_te, y_te = X.iloc[slices["test"]], target.iloc[slices["test"]]

        for C in REGULARIZATIONS:
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            pipe.fit(X_tr, y_tr)
            valid_acc = accuracy_score(y_va, pipe.predict(X_va))
            # keep this config's OWN test slice so we never mix feature sets
            results.append((set_name, C, valid_acc, pipe, X_te, y_te))

    # ---- pick the top 3 on VALIDATION, score them ONCE on the untouched test set
    results.sort(key=lambda r: -r[2])
    baseline = max(y.iloc[: n - 250].mean(), 1 - y.iloc[: n - 250].mean())
    print(f"{'=' * 66}")
    print(f"  Leaderboard (validated on 250 unseen VALIDATION days, then tested ONCE\n"
          f"  on 250 untouched TEST days). Baseline (always most frequent): {baseline*100:.1f}%")
    print(f"{'=' * 66}")
    for rank, (set_name, C, valid_acc, pipe, X_te_cfg, y_te_cfg) in enumerate(results[:4], 1):
        test_acc = accuracy_score(y_te_cfg, pipe.predict(X_te_cfg))
        print(f"  {rank}. features={set_name:<6} C={C:<6} "
              f"validation {valid_acc*100:5.1f}%   test {test_acc*100:5.1f}%")
    print("\n  IMPORTANT: only the top-3's TEST scores are shown ONCE.")
    print("  If they cluster near validation, the finding is trustworthy;")
    print("  if they jump around, the 'improvement' was noise. No cherry-picking.")


if __name__ == "__main__":
    run()
