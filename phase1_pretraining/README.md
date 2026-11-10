# Phase 1: Language Model Pretraining

**Task:** given the first 512 words of a legal document, write the next 128 words.
**Metric:** BERTScore F1.
**Best:** Model B with greedy decoding and repetition penalty 1.2, **0.8389** on test.

## Method

- **Data:** the four official sources (Acts, Supreme Court, High Courts, Tribunals). Text is cleaned and
  de-duplicated, and documents that are mostly non-English are dropped. The sources are mixed
  40% Supreme Court / 35% High Courts / 15% Tribunals / 10% Acts.
- **Tokenizer:** byte-level BPE with 32,768 tokens, shared by every model and phase.
- **Training:** next-token prediction on 1,024-token windows, 131,072 tokens per step, AdamW, learning rate 6e-4,
  300 warm-up steps, then constant, then a linear decay over the last 20%. Each model trains for 11 hours on
  2 × T4 GPUs (DDP).
- **Decoding:** greedy decoding with repetition penalty 1.2 raised dev BERTScore from 0.815 to 0.842.

## Files

| File | Purpose |
|---|---|
| `prep_p1.py` | Cleans the data and writes the token files |
| `train.py` | Trains one model on 2 GPUs |
| `sampler.py` | Mixes the data sources by weight |
| `generate.py` | Writes the 128-word continuations and the submission |
| `tests.py` | CPU checks: model shapes, masking, cache, no built-in attention |
| `inspect_p1.py`, `data_analysis/` | Data statistics that guided the cleaning and the source mix |
| `bench_t4.py` | Speed test on a T4 (results in `data_analysis/bench_results.txt`) |

## Notebooks

| Notebook | Purpose |
|---|---|
| `cs781-train-ddp-model-{a..e}` | Train each model for 11 hours |
| `cs781-train-pair` | Train two models at once, one per GPU |
| `cs781-continual-pretrain` | Continue training a model to a common step count |
| `cs781-generate-model-{a..e}` | Generate the test submission for each model |

## Run

```bash
TOK_PATH=<tokenizer json> python prep_p1.py      # data from Hugging Face (HF_TOKEN) or LOCAL_DATA_DIR
torchrun --standalone --nproc_per_node=2 train.py --config ../common/configs/b.json --data_root <data folder> --micro_batch 16 --compile
python generate.py --help
```
