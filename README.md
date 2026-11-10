# Legal Language Models From Scratch (CS781 Capstone)

We pretrain five small language models (58M–124M parameters) on Indian legal text, starting from random weights.
We then adapt them to six legal NLP tasks. Training used only the course data and free Kaggle GPUs (2 × T4); no outside
pretrained model, tokenizer or library for attention or fine-tuning was used.

## Results (Codabench test set)

| Phase | Task | Metric | Score | Best system |
|---|---|---|---|---|
| 1 | [Language model pretraining](phase1_pretraining) | BERTScore F1 | 0.8389 | Model B, greedy decoding + repetition penalty 1.2 |
| 2 | [Named entity recognition](phase2_ner) | strict macro-F1 | 0.6189 | Ensemble of 10 runs, BIO-Viterbi decoding |
| 3 | [Rhetorical role labelling](phase3_rhetorical_roles) | macro-F1 | 0.6498 | Ensemble of 50 runs + vote rule for "Dissent" |
| 4 | [Judgment prediction](phase4_judgment_prediction) | macro-F1 | 0.7660 | Model D, full fine-tuning, last 1,024 tokens |
| 5 | [Statute identification](phase5_statute_identification) | macro-F1 | 0.3841 | Ensemble of 5 runs, one threshold per statute |
| 6 | [Summarization](phase6_summarization) | BERTScore F1 | 0.8488 | Model B + LoRMA, summarize chunks and join |
| 7 | [Prior case retrieval](phase7_case_retrieval) | micro-F1@7 | 0.4755 | BM25 + average of 14 trained retrievers |

## Models

All five models use 12 layers, width 768 (512 for E), SwiGLU feed-forward, RMSNorm and rotary position embeddings.
They differ only in the attention layer.

| Model | Design | Parameters |
|---|---|---|
| A | Encoder–decoder (BART-style) | 124.3M |
| B | GPT with multi-head attention | 110.1M |
| C | GPT with local grouped-query attention (window 256) + global attention | 103.0M |
| D | Gated DeltaNet (linear attention) + multi-head latent attention | 114.8M |
| E | Small GPT | 57.7M |

All models share one byte-level BPE tokenizer with 32,768 tokens, trained on the Phase-1 data.

**Adaptation methods** (all written from scratch): full fine-tuning, partial fine-tuning, gradual unfreezing, LoRA,
LoRMA, AdaLoRA, QLoRA, IA³, adapters, norm-only tuning, prefix tuning, prompt tuning and knowledge distillation.

## Repository layout

```
common/            shared code: models, attention layers, adaptation methods, tokenizer, model configs
phase1_pretraining/ … phase7_case_retrieval/
                   one folder per phase: scripts, tests, and the Kaggle notebooks that were run
tools/             make_kaggle_zip.py: packs common/ + one phase into a single folder for Kaggle
```

Each phase folder has its own README with the method, the files and how to run it.

## How to run

**Requirements:** Python 3.10+, `pip install -r requirements.txt`, and a GPU for training (we used 2 × T4).

**Locally:** run the scripts from inside a phase folder; they find `common/` on their own.

```bash
cd phase4_judgment_prediction
python tests_cjpe.py          # quick checks on CPU
python cjpe_train.py --help   # all options
```

**On Kaggle** (how all results were produced):

1. Build the code folder for a phase: `python tools/make_kaggle_zip.py phase4_judgment_prediction`
   (or `all`). Upload the zip from `dist/` as a Kaggle dataset.
2. Open the phase notebook, attach the code dataset, the data and the Phase-1 checkpoints, and run all cells.
3. Each run writes a `submission.zip` ready for Codabench.

## Not included

Model weights and the course data are too large for GitHub. The Phase-1 notebooks produce the weights; the data comes
from the course's Hugging Face dataset `Exploration-Lab/CS-781-Capstone`.
