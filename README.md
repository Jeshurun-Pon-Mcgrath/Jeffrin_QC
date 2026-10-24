# TitleGuard — SIH1782

A third-internal Data Science project for checking new publication titles against an existing registry. Built for a CPU Google Colab session with an interactive form, explainable similarity ranking, and honest held-out evaluation.

## Run in Google Colab

1. Download [notebooks/TitleGuard_SIH1782.ipynb](notebooks/TitleGuard_SIH1782.ipynb).
2. Open Google Colab, select **File → Upload notebook**, and upload it.
3. Select a Python CPU runtime and **Runtime → Run all**.
4. Scroll to **Interactive online submission demo**, enter a title, and press **Verify title**.

The notebook is self-contained: its data and model code are embedded, so no repository clone, Drive mount, credentials, GPU, or model download is necessary. A connection may be needed to install missing Python packages. Actual Google-hosted Colab was not accessed during development; every cell was executed locally in a Jupyter kernel. Re-running the notebook rebuilds the model deterministically. Saved outputs are illustrative execution evidence; use your own run for submission.

## What it does

Unicode normalization → word/character TF-IDF retrieval → five ranked candidates → logistic regression pair scoring → exact-match / review / low-similarity indication. Review is a screening recommendation, not legal acceptance or rejection. New submissions are processed in memory and not added to the registry automatically.

## Measured results

72 invented registry titles; 24 subject families; 432 labeled example pairs. Families are split 14/5/5 into training/validation/test. Negative candidate families also remain within the selected partition. Threshold is selected only on validation F1. Test set: 90 pairs.

| Metric | Measured value |
|---|---:|
| Test accuracy | 93.33% |
| Precision | 100% |
| Recall | 90% |
| F1 | 94.74% |
| Recall@5 / MRR@5 | 100% / 1.00 |
| Token Jaccard baseline F1 | 66.67% |

The classifier missed 6 positive examples and produced no false positives on this small test set. **These synthetic results do not establish real-world accuracy.** See [data card](data/README.md), [metrics](reports/metrics.json) and [individual predictions](reports/test_predictions.csv). Perfect accuracy is not a defensible promise. Lexical methods can miss semantic, phonetic and multilingual similarities.

## Course coverage

| Module | Project evidence |
|---|---|
| Basics of Data Science | Problem formulation, data provenance, reproducible workflow |
| Understanding Data | CSV import, missing values, normalization, deduplication, filtering, sorting, grouping, plots |
| Statistical Inference | Mean, median, variance, standard deviation, correlations, explicit hypotheses, exploratory Welch procedure with zero-variance guard, family bootstrap |
| Machine Learning | TF-IDF, cosine/Jaccard distances, logistic regression, K-means, leakage-aware evaluation, confusion matrix |
| Recommender System | Content-based top-five candidate recommendations, Recall@5 and MRR@5 |
| Ethics | Privacy, synthetic-data limitations, false decisions, language bias, explainability, human review |

The notebook explains why constant word-count correlations are undefined and why zero-variance synthetic family averages cannot support a valid Welch test. These limitations are reported rather than hidden.

## Local development

From this repository root, using Python 3.10+:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python evaluate.py
```

The `src/title_verifier` package mirrors the self-contained notebook implementation. After changing the package, update the matching notebook cells too. Dependencies use compatible version ranges, not a frozen cross-platform lock. No server is required; Colab provides the browser interface. This prototype has no production authentication, persistent database or government registry integration.

## Exam presentation

Read [the project report](reports/PROJECT_REPORT.md). Add your name, register number and department to your submission. Explain the pipeline, the split, the errors and limitations in your own words. Follow your institution's requirements for disclosing assistance and tools; do not claim data provenance or authorship that you cannot support.
