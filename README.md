# Sentiment: Baseline vs Fine-Tuned Transformer

Comparing a classic TF-IDF + Logistic Regression baseline against a fine-tuned
DistilBERT model on IMDB movie reviews.

## What's here

`sentiment_classification.ipynb` has the whole thing, built incrementally:

- load a 2000/500 train/test subset of IMDB
- TF-IDF + Logistic Regression baseline, evaluated with precision/recall/F1
  and a confusion matrix (not just accuracy)
- DistilBERT fine-tuned with Hugging Face `Trainer`
- side-by-side comparison + manual error analysis on where the two models
  actually disagree

## Results

| model | accuracy | precision | recall | f1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.830 | 0.816 | 0.846 | 0.830 |
| Fine-tuned DistilBERT | 0.818 | 0.797 | 0.846 | 0.821 |

DistilBERT doesn't actually beat the baseline here. Validation F1 peaked
after epoch 1 and dropped after epoch 2 while training loss kept falling,
classic overfitting on a small (2k example) training set. Reviews were also
truncated to 128 tokens to keep CPU training time reasonable, which loses
information on longer reviews. Left it as-is because the point was to see
and understand the tradeoff, not chase a leaderboard number.

## Running it

```
pip install -r requirements.txt
jupyter notebook sentiment_classification.ipynb
```

No GPU used, trained on CPU, ~20 min for 2 epochs on the subset above.
