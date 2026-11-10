# Phase 6: Summarization

**Task:** write a summary of a long judgment.
**Metric:** BERTScore F1 (ROUGE as a secondary score).
**Best:** Model B with LoRMA, chunk-and-join input, summary cut at 425 words (run `B15_lorma_chunk`), **0.8488** on
test.

## Method

- A judgment is much longer than the model's 1,024-token window. It is split into chunks, each chunk is summarized,
  and the chunk summaries are joined.
- The joined summary is cut to the length that scored best on a held-out part of train.
- Other input strategies were also tested: cutting the input short, picking key sentences first, long-input
  encoding for Model A, and a two-stage rewrite.
- **Distillation:** small Model E learns from Model D and keeps 99.9% of its test score.

## Files

| File | Purpose |
|---|---|
| `summ_data.py` | Sentences, the held-out split, the input strategies and the metrics |
| `summ_model.py` | Wraps a Phase-1 model for summary generation |
| `summ_train.py` | Trains one run, generates the summaries and writes the submissions |
| `summ_coarse.py` | First stage of the two-stage rewrite |
| `experiments_p6.json` | All runs and their settings |
| `tests_summ.py` | CPU checks |

## Notebooks

| Notebook | Purpose |
|---|---|
| `cs781-p6-{1..10}` | Training runs (`cs781-p6-7` contains `B15_lorma_chunk`) |
| `cs781-p6-length-rule` | Finds the best summary length (CPU) |
| `cs781-p6-submission-list` | Collects and ranks the submission files (CPU) |
