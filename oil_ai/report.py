"""REPORT — everything the program prints: the outlook table, the
plain-English verdict, the backtest, the time-travel report and the
one-paragraph takeaway at the very end."""

from .model import forecast_on_date
from .news import aggregate, drivers_table, day_trends


def outlook(prob, last, date, accuracy, analyses, model_name, baseline=None):
    ag = aggregate(analyses)
    n_bull, n_bear, avg, spread, tone = ag
    down_or_up = "more likely DOWN" if prob < 0.5 else "more likely UP"
    base_line = f"   (a naive baseline to beat: {baseline * 100:.1f}%)" if baseline else ""
    print(f"""
{'=' * 62}
  OIL MARKET AI  (AI model: {model_name})
{'=' * 62}
  Brent closing price:             ${last:,.2f} ({date})
     -> the oil price this whole program works from.
  Accuracy on days it never saw:   {accuracy * 100:.1f}%{base_line}
     -> how often the model's UP/DOWN call was right on unseen days; 50% = random.
  Chance of going up (price only): {prob * 100:.0f}%  -> {down_or_up}
     -> this comes straight from the weighted-feature math shown above.

  Headlines analysed:              {len(analyses)}
     -> each headline was given a score from -100 (pushes oil down) to
        +100 (pushes oil up) by the AI.
  Pushing up / pushing down:       {n_bull} / {n_bear} headlines
     -> headlines whose score leaned positive / negative.
  Average impact score:            {avg:+.1f} (-100 = very bearish, +100 = very bullish)
     -> the mean of every headline score: the overall news push.
  How much headlines disagree:     {spread:.0f}  (higher = a noisier news day)
     -> how far apart the individual scores are.
  Overall news tone:               {tone}
     -> bullish if the average is above +5, bearish below -5, else neutral.

  BREAKDOWN BY MARKET DRIVER (one row per theme: OPEC+, geopolitics, etc.):
""")
    for driver, n, bull, bear, davg, dtone in drivers_table(analyses):
        print(f"   {driver:<20} {n:>2} headlines   {bull} bull / {bear} bear   "
              f"net {davg:+6.1f}   ({dtone})")
    print("""
     bull/bear = headlines pushing oil UP / DOWN; net = their average score.
  NOTE: the probability is trained on price features only (Model A); news is
  reported separately until the news archive has enough labelled history
  (Stage 4) to retrain with news included (Model C). Research tool, NOT
  financial advice.
""")


def plain_english(prob_rise, analyses):
    """The final verdict, said in words anyone can follow."""
    rows = day_trends(analyses)
    print(f"{'-' * 62}")
    print("  WHAT THIS MEANS, IN PLAIN ENGLISH")
    print(f"{'-' * 62}")
    print("  1. How the news tone moved, day by day:")
    if rows:
        for day, n, bull, bear, avg, tone in rows:
            print(f"     {day:<12} {n:>3} headlines   {bull} push up / {bear} push down   "
                  f"net score {avg:+7.1f}   ({tone})")
        first, last = rows[0], rows[-1]
        # compare TONE categories (bullish/bearish/neutral), not raw averages:
        # averaging different headline counts across days creates fake trends.
        direction = (f"stayed {last[5]}" if first[5] == last[5]
                     else f"changed: it was {first[5]} early on, and is {last[5]} now")
        print(f"     -> So the news tone has {direction}.")
    else:
        print("     (no news analysed)")

    today_day = rows[-1][0] if rows else ""
    today_news = [a for a in analyses if a["day"] == today_day]
    n_bull, n_bear, avg, _spread, tone = aggregate(today_news)
    print(f"  2. The newest news ({today_day or 'none available'}) is net "
          f"{tone.upper()} ({n_bull} bullish vs {n_bear} bearish, score {avg:+.1f}).")
    driver_rows = drivers_table(today_news)   # today's drivers only, not all days
    if driver_rows:
        strongest = max(driver_rows, key=lambda r: abs(r[4]))
        print(f"     The biggest push comes from '{strongest[0]}' (net {strongest[4]:+.1f}).")
        ups = [r for r in driver_rows if r[4] >= 5]
        downs = [r for r in driver_rows if r[4] <= -5]
        if ups and downs:
            print("     The drivers DISAGREE: "
                  + ", ".join(f"{r[0]} ({r[4]:+.0f})" for r in ups)
                  + " push the price up, while "
                  + ", ".join(f"{r[0]} ({r[4]:+.0f})" for r in downs)
                  + " push it down — so one headline alone would mislead you.")
    # safety net: a probability must always sit between 0 and 1
    prob_rise = prob_rise if prob_rise > 0 else 0.49
    print(f"  3. Tomorrow's chance of oil being HIGHER (from price data only): "
          f"{prob_rise * 100:.0f}% -> this leans "
          f"{'UP' if prob_rise > 0.5 else 'DOWN' if prob_rise < 0.5 else 'NEITHER'}.")
    if tone == "neutral":
        verdict = "the news is lukewarm, so the call follows the price model"
    elif (tone == "bullish") == (prob_rise > 0.5):
        verdict = "the news and the price model AGREE — slightly more confidence"
    else:
        verdict = ("the news and the price model DISAGREE — weak signal: follow the "
                   "price model, but trust it less")
    print(f"  4. PREDICTION: oil most likely {'rises' if prob_rise > 0.5 else 'falls'} "
          f"a little on the next trading day ({verdict}).")
    print(f"  5. REMEMBER: this is a probability, not a promise — the opposite can happen,")
    print(f"     and past accuracy is around 50% (a coin flip). Not financial advice.")
    print("  HOW EVERY NUMBER ABOVE WAS BUILT (the recipe):")
    print("     Price side: yesterday's market numbers (features) x learned weights, all")
    print("                 summed -> score -> probability (full steps in the math table).")
    print("     News side:  every headline gets a -100..+100 score from the AI; scores are")
    print("                 then averaged per day (section 1) and per driver (section 2).")
    print("     Verdict:    sections 3-4 just compare the news push with the price")
    print("                 probability and say whether they agree.")
    print(f"{'-' * 62}\n")


def final_paragraph(prob_rise, analyses):
    """The very last thing the program prints: one short paragraph — likely
    direction plus the reasons (from the price model and today's news)."""
    today_day = day_trends(analyses)[-1][0] if analyses else None
    today_news = [a for a in analyses if a["day"] == today_day] if today_day else []
    _b, _be, _avg, _s, today_tone = aggregate(today_news)
    driver_rows = sorted(drivers_table(today_news), key=lambda r: -abs(r[4]))
    prob_rise = prob_rise if prob_rise > 0 else 0.49   # safety net
    lean = "UP" if prob_rise > 0.5 else "DOWN"
    price_reason = (f"the price model gives a {prob_rise * 100:.0f}% chance of rising, "
                    f"just {'above' if prob_rise > 0.5 else 'below'} the 50% coin-flip line")
    if not analyses:
        body = (f"Oil is likely to go {lean.lower()} a little tomorrow. "
                f"Why: {price_reason}, and there was no news analysed "
                f"(nothing connected the news to the call).")
    else:
        top = ", ".join(f"{r[0]} (net {r[4]:+.0f})" for r in driver_rows[:2])
        if today_tone == ("bullish" if prob_rise > 0.5 else "bearish"):
            news_bit = f"Today's news agrees: it is net {today_tone} ({today_day}, "
            news_bit += f"average score {aggregate(today_news)[2]:+.0f}), mainly because of {top}."
            conf = "So the call has a bit more confidence behind it."
        elif today_tone == "neutral":
            news_bit = (f"Today's news was lukewarm ({today_day}: {aggregate(today_news)[2]:+.0f} on "
                        f"average{(', mostly ' + top) if top else ''}), so the call comes from the price side.")
            conf = ""
        else:
            news_bit = (f"Today's news pushes the {today_tone} way instead "
                        f"({today_day}, average {aggregate(today_news)[2]:+.0f}, mainly because of {top}) — "
                        f"when the two disagree we follow the price model.")
            conf = "So treat this call as low-confidence."
        body = (f"Oil is likely to go {lean.lower()} a little tomorrow. "
                f"Why: {price_reason}. {news_bit} {conf}")
    print(f"{'-' * 62}")
    print("  IN ONE SHORT PARAGRAPH")
    print(f"{'-' * 62}")
    print(f"  {body}")
    print("  As always: past accuracy is ~50% — this is not financial advice.")
    print(f"{'-' * 62}\n")


def time_travel_report(prices, date_str, model_name):
    info = forecast_on_date(prices, date_str)
    print(f"\n{'=' * 62}")
    print(f"  TIME-TRAVEL FORECAST — what the model would have said on {info['day']}")
    print(f"{'=' * 62}")
    print(f"\n  It only used closes from BEFORE that date to train (no peeking).\n")
    print(f"  Brent close on {info['day']}:      ${info['close']:,.2f}")
    print(f"  Probability of rising (price-only): {info['prob'] * 100:.0f}%")
    print(f"  Prediction: {'RISE' if info['prob'] > 0.5 else 'FALL'} over the next trading day\n")
    if info["rose"] is None:
        print("  Outcome: no next trading day in the data yet — cannot grade.")
        return
    print(f"  WHAT ACTUALLY HAPPENED: Brent {'ROSE' if info['rose'] else 'FELL'}"
          f" to ${info['next_close']:,.2f} on {info['next_date']}")
    correct = (info['prob'] > 0.5) == info['rose']
    print(f"  VERDICT: prediction was {'CORRECT' if correct else 'WRONG'}")
    print(f"\n  (news was not re-analysed for this date: free RSS feeds only cover")
    print(f"  roughly the last 3 days — a real system must ARCHIVE news (Stage 2/4),")
    print(f"  which data/news_archive.sqlite is now collecting.)")
    print(f"  AI model; forecast uses the price-only model: {model_name}.")


def backtest(prices, n_days=10):
    """Grades the price-only model on the last n_days trading days, day by day."""
    asked = n_days
    skipped = 0
    n_days = min(n_days, len(prices) - 45)
    results = []
    for loc in range(len(prices) - n_days, len(prices)):
        day = prices.index[loc].date()
        try:
            info = forecast_on_date(prices, str(day))
        except ValueError:
            continue
        if info["rose"] is None:          # no next-day price yet = nothing to grade
            skipped += 1
            continue
        results.append((day, info["prob"], info["rose"], info["next_close"]))
    hits = sum(1 for _d, p, r, _c in results if (p > 0.5) == r)

    print(f"\n{'=' * 62}")
    print(f"  BACKTEST — the price-only model's last {len(results)} forecasts,")
    print(f"  made exactly as they would have been, vs what actually happened")
    print(f"{'=' * 62}")
    print("  (+ = the model got the direction right, x = wrong)\n")
    for day, prob, rose, close_next in results:
        predicted = "RISE" if prob > 0.5 else "FALL"
        happened = "ROSE" if rose else "FELL"
        mark = "+" if (prob > 0.5) == rose else "x"
        print(f"  {day}  forecast: {predicted:<4} ({prob*100:2.0f}%)   "
              f"actual: {happened:<4} (next close ${close_next:,.2f})   {mark}")
    rate = hits / len(results) * 100 if results else 0.0
    print(f" \n  HIT-RATE MATH: {hits} correct out of {len(results)} graded days = {rate:.0f}%")
    if skipped:
        print(f"  ({skipped} recent day(s) skipped: their next-day price is not in the data yet)")
    if asked != n_days:
        print(f"  (you asked for {asked} days; only {n_days} fit with this much history)")
    print(f"  An honest target to compare against: ~57% (always guessing the recent majority).")
    print(f"  A ~50% rate on a small sample is a coin flip — that IS the finding,")
    print(f"  and it is exactly what Stage 4 (news data) and more features are meant to fix.")
