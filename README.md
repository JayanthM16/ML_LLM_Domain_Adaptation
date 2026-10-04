
# ML-LLM: Proving Domain Adaptation in GPT-2 Through Structured Q&A Fine-Tuning

Fine-tunes GPT-2 Small on a machine learning question-and-answer dataset, then tests whether the model really adapted to the domain using five independent metrics, each with a pass threshold fixed before evaluation.

Master's project (DSC 550), University of Massachusetts Dartmouth, Spring 2026. Built by a team of two: Jayanth Mekala and Rugwesh Reddy Gankidi. Advisor: Dr. Amir Akhavan Masoumi.

## Results

All five metrics passed their thresholds.

| # | Metric | GPT-2 baseline | Fine-tuned |
|---|---|---|---|
| 1 | Perplexity on held-out ML text (lower is better) | 93.4 | 55.2 (41.1% lower) |
| 1 | Perplexity change on off-topic control sentences | n/a | 1.8% |
| 2 | Multiple-choice answer ranking | 3/15 | 15/15 |
| 3 | Log-probability win rate on domain questions | n/a | 18/20 (90%) |
| 4 | Embedding domain separation gap | threshold 0.08 | 0.543 |
| 5 | Cosine similarity of generated answers to references | 0.391 | 0.772 |

The off-topic control matters: perplexity on unrelated sentences barely moved, which shows the model specialised in the ML domain rather than changing across the board.

**Limits.** The evaluation sets are small (20 held-out pairs, 15 multiple-choice questions), so these results show the effect clearly but are not a large-scale benchmark.

## Approach

1. **Dataset.** 1,247 ML Q&A pairs covering core concepts, evaluation metrics, distribution shift and advanced topics, formatted as `Question: {question}\nAnswer: {answer}<|endoftext|>`.
2. **Split.** Pairs 1 to 700 for training; pairs 701 to 720 held out and never seen in training.
3. **Fine-tuning.** GPT-2 Small (124M parameters) trained for 5 epochs on a Google Colab T4 GPU.
4. **Evaluation.** Five metrics run against both the base and fine-tuned models, compared with pre-defined thresholds.

## Training configuration

| Setting | Value |
|---|---|
| Base model | GPT-2 Small (124M parameters) |
| Epochs | 5 |
| Learning rate | 3e-5, cosine decay, 10% warmup |
| Batch size | 8, with gradient accumulation over 4 steps (effective 32) |
| Block size / stride | 512 tokens / 64 |
| Optimizer | AdamW, weight decay 0.01 |
| Precision | float32 weights with fp16 autocast and GradScaler |
| Stability | Gradient clipping at 1.0, gradient checkpointing |

## Repository structure

```
.
├── notebooks/
│   └── ml_driftnet.ipynb          # Training and all five evaluation metrics
├── data/
│   └── ml_qa_pairs.txt            # The Q&A dataset
├── app/                           # Flask application
├── docs/
│   └── final_report.pdf           # Full project report
├── requirements.txt
└── README.md
```

## How to run

1. Open `notebooks/ml_driftnet.ipynb` in Google Colab and select a T4 GPU runtime.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the cells in order: data loading, fine-tuning, then the five metric sections.

## Tech stack

Python, PyTorch, Hugging Face Transformers, Sentence Transformers (all-MiniLM-L6-v2), Flask, Google Colab.

## Future work

- Evaluate on a larger held-out set.
- Compare against parameter-efficient methods such as LoRA.
- Repeat with a larger base model.
