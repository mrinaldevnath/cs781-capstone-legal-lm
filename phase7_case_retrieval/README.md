# Phase 7: Prior Case Retrieval

**Task:** for each query judgment, find the earlier cases it cites from a pool of candidates, and return the
top 7.
**Metric:** micro-F1@7.
**Best:** BM25 plus the average of 14 trained retrievers (`ENS_all_trained`), **0.4755** on test.

## Method

- **Word matching (BM25):**
  - BM25 over word 1–6-grams;
  - BM25 of each citation passage on its own;
  - a match on reporter citations (e.g. "2 SCR 148");
  - a rule that a cited case cannot be newer than the query.
- **Trained retriever:** a Phase-1 model turns the query's citation passages and the candidate's text chunks into
  vectors. It is trained with a contrastive loss (InfoNCE).
- **Fusion:** both scores are standardized per query and added, with a weight chosen on dev. The final system
  averages 14 trained runs.

## Files

| File | Purpose |
|---|---|
| `pcr_lex.py` | BM25 and the other word-matching scores |
| `pcr_data.py` | Data reading, top-7 selection, micro-F1@7, writing `prediction.json` |
| `pcr_model.py` | The retriever (Phase-1 model as an encoder) |
| `pcr_train.py` | Trains one retriever and fuses it with BM25 |
| `pcr_ensemble.py` | Averages all trained runs and fuses them with BM25 |
| `experiments_p7.json` | All runs and their settings |
| `tests_pcr.py` | CPU checks |

## Notebooks

| Notebook | Purpose |
|---|---|
| `cs781-p7-{1..6}` | Build the BM25 scores and train the retrievers |
| `cs781-p7-ensemble` | Builds the final ensemble (CPU) |
| `cs781-p7-submissions` | Checks and ranks all submission files (CPU) |
