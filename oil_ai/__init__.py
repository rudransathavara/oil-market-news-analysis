"""oIL MARKET AI — package modules (ported from the original one_file.py).

Layout:
    config.py    shared settings: ticker, cache paths, drivers, AI prompt
    prices.py    download + cache Brent prices
    model.py     the auto-tuned logistic-regression model and its math
    news.py      fetch headlines + analyse each with an LLM
    archive.py   SQLite archive of analysed headlines (Stage 2 groundwork)
    evaluate.py  probability quality: accuracy, precision/recall, Brier,
                 reliability curve + charts
    report.py    all printed output: outlook, plain English, backtest
    cli.py       the command-line entry point (`one_file.py` calls this)
"""
