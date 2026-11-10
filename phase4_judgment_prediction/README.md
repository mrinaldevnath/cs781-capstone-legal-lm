# Phase 4: Judgment Prediction

**Task:** read a Supreme Court case and predict whether the appeal was accepted (1) or rejected (0).
**Metric:** macro-F1.
**Best:** Model D, full fine-tuning (run `D07_full`), **0.7660** on test.

## Method

- The decision is written near the end of a judgment, so the model reads the **last** 1,024 tokens as two
  512-token chunks. This beat reading the first 1,024 tokens by 0.08–0.09 F1.
- Each chunk is mean-pooled, a learned attention combines the chunks, and a linear layer gives the label.
- Training cases that also appear in dev or test are removed.

## Files

| File | Purpose |
|---|---|
| `cjpe_data.py` | Tokenizes the cases and picks the chunks |
| `cjpe_model.py` | Chunk encoder, attention pooling and classifier |
| `cjpe_train.py` | Trains one run and writes its submission |
| `cjpe_eval.py` | Macro-F1 and the dev threshold search |
| `experiments_p4.json` | All runs and their settings |
| `tests_cjpe.py` | CPU checks |

## Notebooks

| Notebook | Purpose |
|---|---|
| `cs781-p4-{1..4}` | Training runs (`cs781-p4-3` contains `D07_full`) |
| `cs781-p4-final` | Result tables, ensembles and submission files (CPU) |

## Run

```bash
python cjpe_train.py --ckpt <model D final.pt> --tok_root <data folder> --data_dir <phase-4 data> --cache p4.npz \
    --select last --n_chunks 2 --pool attn --method full --out runs/D07_full --model_dir models
```
