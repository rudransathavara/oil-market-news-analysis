"""PRICES — download historical Brent crude prices and cache them in a CSV."""
import os

import pandas as pd
import yfinance

from .config import TICKER, PRICE_CSV


def load_prices():
    """Return daily Brent closes as a date-indexed DataFrame. Reads the CSV
    cache if it exists; otherwise downloads 5 years from Yahoo Finance and
    saves it for the next runs."""
    if os.path.exists(PRICE_CSV):
        prices = pd.read_csv(PRICE_CSV, index_col="date", parse_dates=["date"])
        if len(prices) > 0:
            return prices
    os.makedirs(os.path.dirname(PRICE_CSV), exist_ok=True)
    raw = yfinance.Ticker(TICKER).history(period="5y", interval="1d", auto_adjust=False).reset_index()
    prices = (pd.DataFrame({"date": pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize(),
                            "close": raw["Close"].astype(float)})
              .dropna().drop_duplicates("date").sort_values("date").set_index("date"))
    prices.to_csv(PRICE_CSV)
    return prices
