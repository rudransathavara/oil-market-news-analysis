"""SHARED CONFIG — every constant the pipeline needs lives here, so there is
exactly one place to change a setting (the original one_file.py put these at
the top of the file)."""

import os
import urllib.parse

TICKER = "BZ=F"          # Brent crude front-month futures (Yahoo Finance)
PRICE_CSV = "data/brent_prices.csv"
RESULTS_DIR = "results"
ARCHIVE_DB = "data/news_archive.sqlite"   # analysed-headline archive (Stage 2)

# NOTE: PDF/PNG outputs from the evaluation module land in RESULTS_DIR.

DRIVERS = {   # the six market drivers the analyst focuses on (matches SYSTEM_PROMPT)
    "OPEC+ policy":        'OPEC+ OR "OPEC meeting" OR "production cuts" OR "production increase"',
    "Geopolitics":         'crude oil sanctions OR tanker OR "strait of hormuz" OR "oil supply disruption" OR war oil',
    "US inventories":     '"crude oil inventories" OR "EIA weekly petroleum" OR "API crude stocks"',
    "Dollar & macro":      '"US dollar" oil OR "Federal Reserve" oil OR "interest rates" oil OR "economic data" oil',
    "Price action":        'oil price rally OR crude oil selloff OR "oil prices surge" OR "oil prices slide"',
    "Market expectations": 'Brent outlook OR "oil price forecast" OR "oil demand outlook" OR traders expect',
}
DRIVER_ORDER = list(DRIVERS.keys())
RSS_BASE = "https://news.google.com/rss/search?q="


def rss_url(query):
    """Build a Google News RSS feed URL, filtered to the last 3 days so
    'past days + today' is covered."""
    return RSS_BASE + urllib.parse.quote_plus(query + " when:3d") + "&hl=en-US&gl=US&ceid=US:en"


# =========================================================================
# CONFIG: which AI should analyse the news? (read from .env at run time)
# =========================================================================
SYSTEM_PROMPT = """You are an analyst specialising in crude oil markets.
You analyse news strictly through SIX market drivers:
  1. geopolitics      - wars, sanctions, attacks, shipping chokepoints, tanker risks
  2. opec_plus        - OPEC+ production policy, quota decisions, meetings
  3. inventories      - US crude oil inventory reports (EIA/API): builds are bearish, draws bullish
  4. macro            - US dollar strength (a stronger dollar pressures oil prices), interest rates,
                        Fed decisions, growth/recession data (incl. China demand)
  5. price_action     - oil's own moves: rallies, selloffs, momentum, breakouts
  6. expectations     - market-expectation changes: demand forecasts, analyst consensus, forward curves

Rules:
- Classify the headline into EXACTLY ONE driver: geopolitics, opec_plus, inventories, macro,
  price_action, or expectations (use 'other' only if it fits none).
- Do not assume a negative-sounding event is bearish for oil: judge its actual effect on
  supply, demand and market expectations (e.g. an output INCREASE is bearish; report of
  shipping-cost spikes from disruption is often bullish as a risk premium). - Distinguish confirmed facts from speculation; a headline may itself REPORT a market expectation,
- Do not invent numbers or probabilities.

Return ONLY a JSON object with fields:
event_summary, category (use a driver slug from the list above),
impact_direction (bullish/bearish/neutral), impact_score (-100..+100),
expected_horizon (intraday/24 hours/5 trading days), supply_impact, demand_impact,
uncertainty (low/medium/high), reasoning, key_assumption."""

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-4-5", "gemini": "gemini-2.0-flash"}
OPENAI_BASE = {"openai": "https://api.openai.com/v1", "gemini": "https://generativelanguage.googleapis.com/v1beta/openai"}


def llm_config():
    """Resolve (provider, model, base_url) from the environment."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    base = os.getenv("LLM_BASE_URL") or OPENAI_BASE.get(provider, "http://localhost:11434/v1")
    return provider, model, base
