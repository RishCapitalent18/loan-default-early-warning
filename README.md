# Consumer Loan Delinquency Early-Warning System

Rebuilds a messy consumer-loan origination export into a clean, model-ready
book, then trains a gradient-boosted classifier to flag loans likely to reach
60-day delinquency, and analyses where the risk concentrates. The point is the
whole loop, not just the model: unusable raw data in, ranked risk and a review
queue out.

Built with Python (pandas, scikit-learn) and pytest. Data is synthetic,
generated locally with a fixed seed, so every number below reproduces.

## The problem

Origination systems rarely hand you clean data. In `data/raw_loans.csv` the
income column is stored as strings like `"$31,917.63"`, employment length is
free text (`"< 1 year"`, `"10+ years"`, `"n/a"`), rate and utilisation carry
percent signs, missing FICO pulls are coded as `9999`, home ownership casing is
inconsistent, and several fields are simply blank. A strict numeric load would
reject a large share of the file. The job is to rebuild it, keep the records,
and get a usable risk signal out.

## What it does

| Stage | Module | What happens |
| --- | --- | --- |
| Generate | `src/ews/generate_data.py` | Synthesizes 40,000 loans with a latent delinquency signal, then damages the export the way real systems do. |
| Repair | `src/ews/pipeline.py` (repair layer) | Parses money, percent, and free-text fields, drops FICO sentinels, imputes, and derives `loan_to_income`. |
| Model | `src/ews/pipeline.py` | HistGradientBoosting on a time-ordered split (early vintages train, latest vintages test). |
| Analyse | `reports/` | Decile capture, a top-20% review queue, a rules-based baseline, and a per-purpose risk table. |

## Headline results (committed run, seed 42)

- Rebuilt and retained **16.88%** of records (6,753 of 40,000) that a strict validation load would have dropped as unparseable or missing.
- **ROC-AUC 0.79**, PR-AUC 0.44 on a held-out later vintage (test default rate 16.1%).
- The top two risk deciles capture **51.58%** of all eventual delinquencies, against 20% expected by chance.
- Flagging the riskiest 20% for review gives **precision 0.414, recall 0.516**, versus a rules screen (FICO < 640 or DTI > 28 or rate > 19%) at precision 0.327 for the same recall band, roughly a **27% cut in false positives** for the same catch rate.

Highest-risk segments by actual delinquency: small business, major purchase, and medical loans.

## Run it

```
pip install -r requirements.txt
PYTHONPATH=. python3 run_pipeline.py     # writes reports/metrics.json and segment_lift.csv
PYTHONPATH=. python3 -m pytest -q        # repair-layer unit tests
```

## Layout

```
loan-default-early-warning/
|-- run_pipeline.py
|-- requirements.txt
|-- src/ews/
|   |-- generate_data.py     # synthetic messy loan book
|   `-- pipeline.py          # repair + features + model + analysis
|-- tests/test_repair.py     # unit tests for the rebuild
|-- data/raw_loans.csv       # committed messy input
`-- reports/                 # metrics.json, segment_lift.csv
```

Data is fully synthetic and generated locally. No proprietary or vendor data is used.
