# Phase 5: Statute Identification

**Task:** for each case, list which of 100 statutes apply.
**Metric:** macro-F1 over statutes.
**Best:** average of 5 runs with one threshold per statute, **0.3841** on test.

## Method

- The model reads up to four 512-token chunks per case, combines them with attention, and has 100 yes/no outputs.
- **Thresholds:** with a 0.5 cut the model rarely says "yes" for rare statutes. One threshold per statute, chosen
  on dev, raised macro-F1 from about 0.24 to 0.38.
- The thresholds are chosen on one half of dev and checked on the other half, so the dev score stays honest.
- **Ensemble members:** `B07_full`, `D06_partial` and `B03_qlora` (notebook 2), and `D07_full` and `D03_qlora`
  (notebook 3).

## Files

| File | Purpose |
|---|---|
| `lsi_data.py` | Tokenizes the cases into chunks |
| `lsi_model.py` | Chunk encoder, attention pooling and 100 outputs |
| `lsi_train.py` | Trains one run and writes its probabilities and submission |
| `lsi_eval.py` | Macro-F1 and per-statute thresholds |
| `experiments_p5.json` | All runs and their settings |
| `tests_lsi.py` | CPU checks |

## Notebooks

| Notebook | Purpose |
|---|---|
| `cs781-p5-{1..4}` | Training runs |
| `cs781-p5-final` | Ensembles, thresholds and submission files (CPU) |
