"""NEWS — fetch live headlines (one feed per driver) + analyse each with an LLM."""

import json
import os
from datetime import datetime
from statistics import variance

import requests
import xml.etree.ElementTree as ET

from .config import DRIVERS, rss_url, SYSTEM_PROMPT, llm_config


def llm_analysis(headline, model=None):
    """Send one headline to the configured LLM and get structured JSON back."""
    if not os.getenv("LLM_API_KEY"):
        print("! AI news analysis not configured — copy .env.example to .env (Ollama needs no real key).")
        return None
    provider, default_model, base = llm_config()
    model = model or default_model

    try:
        if provider == "anthropic":  # Anthropic has its own API shape.
            resp = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"},
                json={"model": model, "max_tokens": 1024, "temperature": 0.2, "system": SYSTEM_PROMPT,
                      "messages": [{"role": "user", "content": headline}]}, timeout=120)
        else:                        # OpenAI-compatible API (includes your local Ollama).
            resp = requests.post(f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                json={"model": model, "temperature": 0.2,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": headline}]}, timeout=120)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"] if provider != "anthropic" \
                 else "".join(b["text"] for b in resp.json()["content"])
        text = answer.strip().strip("`")
        text = text[4:] if text.lower().startswith("json") else text
        result = json.loads(text[text.find("{"): text.rfind("}") + 1])
        result["impact_score"] = max(-100, min(100, int(result.get("impact_score", 0))))
        return result
    except requests.exceptions.ConnectionError:
        print("! The AI server is not reachable — start Ollama (or run: ollama serve)")
        return None
    except Exception as err:
        print(f"! AI analysis failed for this headline ({type(err).__name__}: {err}) — it was skipped.")
        return None


def fetch_headlines(total=10):
    """One dedicated feed PER DRIVER, so the mix covers all six focus areas.
    Every article is tagged with its driver at fetch time — more reliable
    than trusting the LLM's own category label."""
    per_driver = max(1, total // len(DRIVERS))
    articles = []
    for driver, query in DRIVERS.items():
        try:
            resp = requests.get(rss_url(query),
                                headers={"User-Agent": "Mozilla/5.0 (oil ai prototype)"}, timeout=20)
            resp.raise_for_status()
            items = list(ET.fromstring(resp.content).iter("item"))
        except Exception as err:
            print(f"! Driver feed '{driver}' failed: {err}")
            continue
        for item in items[:per_driver]:
            articles.append({"title": item.findtext("title", "").rsplit(" - ", 1)[0],
                             "published": item.findtext("pubDate", ""),
                             "driver": driver})
    return articles


def pub_day(pub):
    """'Tue, 13 Oct 2026 19:04:00 GMT' -> 'Tue 13 Oct' — which calendar day the
    news broke, so headlines from the past few days group into daily rows."""
    try:
        # RSS timestamps look like "Tue, 13 Oct 2026 19:04:00 GMT" — Python's
        # %z does not understand the letters "GMT", so swap them for +0000 first.
        zulu = pub.replace("GMT", "+0000").replace("UTC", "+0000").strip()
        # group by DAY only — headlines from the same calendar day share a row
        return datetime.strptime(zulu, "%a, %d %b %Y %H:%M:%S %z").strftime("%a %d %b")
    except ValueError:
        return pub or "pasted/unknown"


def aggregate(analyses):
    """The math behind the news tone: count directions, average the scores,
    and measure how far apart they are (sample variance)."""
    scores = [a["impact_score"] for a in analyses]
    n_bull = sum(a["impact_direction"] == "bullish" for a in analyses)
    n_bear = sum(a["impact_direction"] == "bearish" for a in analyses)
    avg = sum(scores) / len(scores) if scores else 0.0
    spread = variance(scores) if len(scores) > 1 else 0.0
    tone = "bullish" if avg > 5 else "bearish" if avg < -5 else "neutral"
    return n_bull, n_bear, avg, spread, tone


def drivers_table(analyses):
    """Per-driver scorecard: how hard is each of the six drivers pushing price?
    Headlines whose driver tag is unknown bucket into 'other'."""
    rows = []
    for driver in list(DRIVERS.keys()) + ["other"]:
        subset = [a for a in analyses
                  if (a.get("driver") if a.get("driver") in DRIVERS else "other") == driver]
        if not subset:
            continue
        n_bull, n_bear, avg, _spread, tone = aggregate(subset)
        rows.append((driver, len(subset), n_bull, n_bear, avg, tone))
    return rows


def day_trends(analyses):
    """Group the analysed news PER DAY, so we can see how the tone evolved."""
    by_day = {}
    for a in analyses:
        by_day.setdefault(a["day"], []).append(a)
    rows = []
    for day in sorted(by_day):
        n_bull, n_bear, avg, spread, tone = aggregate(by_day[day])
        rows.append((day, len(by_day[day]), n_bull, n_bear, avg, tone))
    return rows


def analyse_news(articles, model=None, save_to_archive=None):
    """Analyse every headline with the LLM. When `save_to_archive` is passed
    (the application database path from archive.py), each result is stored so
    a future news-fused model can use labelled history (Stage 2 groundwork)."""
    analyses = []
    for art in articles:
        print(f"\nHeadline: {art['title']}")
        print(f"Published: {art['published']}   (now's forecast may only use facts available already)")
        result = llm_analysis(art["title"], model=model)
        if result:
            result["day"] = pub_day(art["published"])     # which day the news belongs to
            result["driver"] = art.get("driver", "other")  # which market driver it hits
            result["headline"] = art["title"]
            analyses.append(result)
            print(f"  -> {result['impact_direction']:<7} score {result['impact_score']:+4.0f}"
                  f"  horizon: {result['expected_horizon']}")
            print(f"     {result['reasoning'][:180]}")
    if save_to_archive is not None and analyses:
        from . import archive
        archive.save(analyses, model, db_path=save_to_archive)
    return analyses
