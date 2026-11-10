# Phase 2: Named Entity Recognition

**Task:** find 12 entity types (petitioner, respondent, judge, court, statute, precedent, date, …) as exact
character spans.
**Metric:** strict macro-F1.
**Best:** average of the 10 best runs, **0.6189** on test.

## Method

- Each word is split at letter/digit/punctuation edges before BPE, so entity boundaries fall on token edges.
- A Phase-1 model reads 512-token windows; a BiLSTM layer and a linear layer predict a BIO tag for each token.
- **Decoding:** Viterbi decoding allows only valid BIO sequences and added about +0.10 macro-F1 over picking the
  best tag per token.
- **Ensemble:** the token probabilities of several runs are averaged, then decoded once.

## Files

| File | Purpose |
|---|---|
| `ner_train.py` | Trains one run and writes its submission |
| `ner_post.py` | Viterbi decoding and token-to-character offsets |
| `ner_ensemble.py` | Averages several runs and decodes |
| `ner_eval.py` | Strict span-level macro-F1 |
| `tests_ner.py` | CPU checks |
| `../common/ner_data.py`, `../common/ner_model.py` | Data reading and the tagging model |

## Notebooks

| Notebook | Purpose |
|---|---|
| `cs781-p2-tech-{a..e}` | One per model: trains each adaptation method |
| `cs781-p2-ensemble` | Ranks the runs on dev and builds the ensembles (CPU) |
