"""Test suite for the oil_ai package.

Focus: the small pure helpers (no network, no LLM, no downloads) — the
pieces where silent math bugs would poison every printed number:
pub_day, aggregate / drivers_table / day_trends, feature builders, the
chronological split, no-look-ahead forecasting, the SQLite archive, and
the evaluation math.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oil_ai import archive, evaluate
from oil_ai.model import _features, forecast_on_date, split_ranges
from oil_ai.news import aggregate, day_trends, drivers_table, pub_day


# ---------- synthetic price data used by several tests -----------------------
def synthetic_prices(n=300, seed=7):
    """Deterministic random-walk closes, indexed by business days."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(steps))
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"close": close, "volume": 1000.0}, index=idx)


# ---------- pub_day ----------------------------------------------------------
def test_pub_day_parses_gmt_rss_timestamp():
    assert pub_day("Tue, 13 Oct 2026 19:04:00 GMT") == "Tue 13 Oct"

def test_pub_day_parses_utc_timestamp():
    assert pub_day("Tue, 13 Oct 2026 19:04:00 UTC") == "Tue 13 Oct"

def test_pub_day_invalid_string_returns_it_unchanged():
    assert pub_day("pasted by you") == "pasted by you"

def test_pub_day_empty_string_gets_fallback_label():
    assert pub_day("") == "pasted/unknown"


# ---------- aggregate: the news-tone math ------------------------------------
def _mk(score, direction):
    return {"impact_score": score, "impact_direction": direction}

def test_aggregate_empty_news_is_neutral_zero():
    assert aggregate([]) == (0, 0, 0.0, 0.0, "neutral")

def test_aggregate_counts_directions_and_averages():
    n_bull, n_bear, avg, _spread, _tone = aggregate([_mk(80, "bullish"), _mk(-30, "bearish")])
    assert (n_bull, n_bear) == (1, 1)
    assert abs(avg - 25.0) < 1e-9

def test_aggregate_tone_thresholds():
    assert aggregate([_mk(10, "bullish")])[4] == "bullish"      # > +5
    assert aggregate([_mk(-10, "bearish")])[4] == "bearish"     # < -5
    assert aggregate([_mk(3, "neutral")])[4] == "neutral"       # inside the dead zone

def test_aggregate_spread_zero_for_one_headline_positive_for_many():
    assert aggregate([_mk(50, "bullish")])[3] == 0.0
    assert aggregate([_mk(50, "bullish"), _mk(-50, "bearish")])[3] > 0


# ---------- drivers_table / day_trends ---------------------------------------
def test_drivers_table_skips_empty_groups():
    rows = drivers_table([_mk(20, "bullish") | {"driver": "Geopolitics"}])
    assert [r[0] for r in rows] == ["Geopolitics"]
    assert rows[0][1] == 1 and rows[0][4] == 20

def test_drivers_table_defaults_unknown_driver_to_other():
    rows = drivers_table([{**_mk(-20, "bearish"), "driver": "mystery"}])
    assert rows[0][0] == "other"

def test_day_trends_groups_by_day_in_date_order():
    a = {**_mk(+40, "bullish"), "day": "Mon 12 Oct"}
    b = {**_mk(-10, "bearish"), "day": "Tue 13 Oct"}
    rows = day_trends([b, a])          # deliberately out of order
    assert [r[0] for r in rows] == ["Mon 12 Oct", "Tue 13 Oct"]
    assert rows[0][2] == 1 and rows[1][3] == 1


# ---------- feature builders --------------------------------------------------
def test_features_basic_has_three_expected_columns():
    feats = _features("basic", synthetic_prices())
    assert list(feats.columns) == ["ret_1d", "ret_5d", "vol_10d"]

def test_features_rich_has_twelve_columns_and_finite_tail():
    feats = _features("rich", synthetic_prices())
    assert len(feats.columns) == 12
    tail = feats.tail(50)
    assert np.isfinite(tail.to_numpy()).all()   # no NaN/inf where it matters

def test_features_unknown_kind_falls_back_to_basic():
    feats = _features("nonsense-kind", synthetic_prices())
    assert len(feats.columns) == 3


# ---------- chronological split (the discipline guarantee) --------------------
def test_split_ranges_are_ordered_and_disjoint():
    sl = split_ranges(1000, test_len=250, valid_len=250)
    tr, va, te = sl["train"], sl["valid"], sl["test"]
    assert tr.stop <= va.start and va.stop <= te.start and te.stop == 1000
    assert tr.start == 0          # training on the OLDEST data
    assert (tr.stop - tr.start) + (va.stop - va.start) + (te.stop - te.start) == 1000


# ---------- no look-ahead in dated forecasting --------------------------------
def test_forecast_on_date_needs_history():
    prices = synthetic_prices(60)
    with pytest.raises(ValueError):
        forecast_on_date(prices, "2025-01-05")     # too early: only ~3 days behind

def test_forecast_on_date_ignores_future_prices():
    """The gold standard test: a forecast for day X must not change even if
    every price AFTER day X is replaced — if it did, the model peeked."""
    prices = synthetic_prices(260)
    mid = prices.index[150]
    before = forecast_on_date(prices, str(mid.date()))
    tampered = prices.copy()
    tampered.iloc[151:] = 500.0                    # absurd future prices
    after = forecast_on_date(tampered, str(mid.date()))
    assert before["prob"] == pytest.approx(after["prob"], abs=1e-9)
    assert before["close"] == after["close"]

def test_forecast_on_date_reports_true_outcome():
    prices = synthetic_prices(260)
    mid = prices.index[150]
    info = forecast_on_date(prices, str(mid.date()))
    rose = prices["close"].iloc[151] > prices["close"].iloc[150]
    assert info["rose"] == rose
    assert info["next_close"] == prices["close"].iloc[151]


# ---------- SQLite news archive -----------------------------------------------
def test_archive_save_and_stats_roundtrip(tmp_path):
    db = tmp_path / "archive.sqlite"
    saved = archive.save([
        {"day": "Tue 13 Oct", "published": "x", "driver": "OPEC+ policy",
         "headline": "OPEC+ cuts", "impact_direction": "bearish",
         "impact_score": -40, "expected_horizon": "24 hours", "uncertainty": "low"},
        {"day": "Tue 13 Oct", "published": "y", "driver": "Geopolitics",
         "headline": "Tanker attacked", "impact_direction": "bullish",
         "impact_score": 60, "expected_horizon": "intraday", "uncertainty": "high"},
    ], model="gemma3:4b", db_path=str(db))
    assert saved == 2
    by_day, by_driver = archive.stats(str(db))
    assert sum(n for _d, n, _s in by_day) == 2
    drivers = {d: (n, round(s, 1)) for d, n, s in by_driver}
    assert drivers["OPEC+ policy"] == (1, -40.0)
    assert drivers["Geopolitics"] == (1, 60.0)

def test_archive_save_of_empty_list_writes_nothing(tmp_path):
    db = tmp_path / "archive.sqlite"
    assert archive.save([], db_path=str(db)) == 0
    assert not db.exists()

def test_archive_print_stats_empty_is_safe(tmp_path, capsys):
    archive.print_stats(str(tmp_path / "missing.sqlite"))
    assert "empty" in capsys.readouterr().out.lower()


# ---------- evaluation math ----------------------------------------------------
def test_metrics_perfect_probabilities_score_zero_brier():
    rows = [("2026-10-01", 1.0, 1, 101.0), ("2026-10-02", 0.0, 0, 99.0)]
    m = evaluate.metrics(rows)
    assert m["brier"] == pytest.approx(0.0)
    assert m["accuracy"] == 1.0

def test_metrics_always_fifty_percent_is_a_coin_flip():
    rows = [("2026-10-01", 0.5, 1, 101.0), ("2026-10-02", 0.5, 0, 99.0)]
    m = evaluate.metrics(rows)
    assert m["accuracy"] == 0.5
    assert m["brier"] == pytest.approx(0.25)
    assert m["precision"] is None  # a 0.5 prediction is NOT an UP call — undefined
    assert m["recall"] == 0.0      # no real up-day was caught
    assert m["n"] == 2

def test_metrics_precision_and_recall_semantics():
    rows = [("d1", 0.9, 1, 0), ("d2", 0.9, 0, 0), ("d3", 0.2, 1, 0)]
    m = evaluate.metrics(rows)
    assert m["precision"] == 0.5   # 1 of the 2 UP calls was right
    assert m["recall"] == 0.5      # caught 1 of the 2 real up-days

def test_metrics_empty_rows_returns_nones():
    m = evaluate.metrics([])
    assert m["n"] == 0 and m["brier"] is None

def test_reliability_bins_group_by_predicted_probability():
    rows = [("d1", 0.45, 1, 0), ("d2", 0.55, 1, 0), ("d3", 0.55, 0, 0)]
    b = evaluate.reliability_bins(rows, n_bins=10)
    b_45, b_55 = b[4], b[5]          # 0.4-0.5 and 0.5-0.6 bins
    assert b_45["n"] == 1 and b_45["mean_actual"] == 1
    assert b_55["n"] == 2 and b_55["mean_actual"] == 0.5
    assert b_55["hi"] == pytest.approx(0.6)

def test_reliability_bins_skip_empty_bins_in_output():
    b = evaluate.reliability_bins([("d1", 0.95, 1, 0)], n_bins=10)
    assert [x["n"] for x in b if x["n"] > 0] == [1]
