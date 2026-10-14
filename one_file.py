"""ONE-FILE OIL MARKET AI — thin compatibility shim.

All logic moved into the `oil_ai` package (config / prices / model / news /
archive / evaluate / report / cli). Running this file still launches the
exact same program as before:

    source .venv/bin/activate && python3 one_file.py

New flags since the modularization:
    --evaluate [N]     probability quality (Brier, calibration, precision/recall) + charts
    --archive-stats    summarize the SQLite news archive
    --json             write a machine-readable summary to results/run.json
"""

from oil_ai.cli import main

if __name__ == "__main__":
    main()
