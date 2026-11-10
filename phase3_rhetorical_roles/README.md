# Phase 3: Rhetorical Role Labelling

**Task:** label each sentence of a judgment with one of 13 roles (Fact, Issue, Argument, Ratio, Dissent, …).
**Metric:** macro-F1.
**Best:** average of 50 runs plus a vote rule for "Dissent", **0.6498** on test.

## Method

- Sentences are packed into 512-token windows. Each sentence becomes one vector (the mean of its tokens), plus
  position features.
- A BiLSTM runs over the sentence vectors, so each label sees the sentences around it.
- **Dissent vote:** Dissent is rare, and averaging hides it. A sentence is labelled Dissent when at least one run
  predicts it.
- **Extra runs:** 40 more runs come from 5-fold cross-validation on train, to give the vote more members.

## Files

| File | Purpose |
|---|---|
| `rr_data.py` | Splits documents into sentences and windows |
| `rr_model.py` | The sentence model (Phase-1 model + BiLSTM) |
| `rr_train.py` | Trains one run and saves sentence probabilities |
| `rr_ensemble.py` | Averages runs, applies the Dissent vote, writes the submission |
| `rr_eval.py`, `evaluate_phase3.py` | Macro-F1 (ours and the official script) |
| `tests_rr.py` | CPU checks |

## Notebooks

| Notebook | Purpose |
|---|---|
| `cs781-p3-{A..E}`, `cs781-p3-extra` | Main runs, one notebook per model |
| `cs781-p3-ensemble`, `cs781-p3-submissions` | Ensembles of the main runs (CPU) |
| `cross_validation/cs781-p3-cv-*` | Cross-validation runs for models D and E |
| `cross_validation/cs781-p3-cv-analysis` | Results of the cross-validation runs |
| `cross_validation/cs781-p3-cv-submissions` | Builds the best submission (CPU) |
