# 🧠 BERT + Stacking Ensemble for Multi‑Class Mental‑Health Sentiment Analysis

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EB5B00)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-3B7DDD)](https://lightgbm.readthedocs.io/)
[![Transformers/BERT](https://img.shields.io/badge/Transformers-BERT-ffcc00)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A hybrid BERT + TF‑IDF representation with a stacking ensemble (LightGBM + Random Forest → XGBoost meta‑learner) for seven‑class mental‑health sentiment classification.

</div>

---

## Overview

This repository implements a novel stacking framework that:
- Combines deep contextual BERT embeddings with TF‑IDF lexical features
- Addresses class imbalance using Random Over‑Sampling (ROS)
- Trains a stacking ensemble with LightGBM and Random Forest as base learners and XGBoost as the meta‑classifier
- Targets seven mental‑health sentiment states (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder)

Use cases include research on patient feedback, mental‑health monitoring, and decision support. Not a clinical device.

---

## Key Features

- Hybrid features: BERT [CLS]/pooled embeddings + TF‑IDF vectors
- Stacking ensemble: LightGBM + Random Forest → XGBoost meta‑learner
- Class balancing: ROS applied only on the training split
- Robust evaluation: 5‑fold CV, ablation on feature sets, comparisons with strong baselines
- Reproducible pipeline: fixed seeds, leakage‑safe preprocessing, code/data links

---

## Results (from the paper)

- Accuracy: 93.21%
- Macro F1: 93.24%
- 5‑fold CV accuracy (mean): 91.24% (91.43, 91.15, 91.36, 91.07, 91.21)

Ablations (macro‑F1):
- TF‑IDF + ROS: 0.5523
- BERT + ROS: 0.5667
- Hybrid (BERT + TF‑IDF) + ROS: 0.6100

Comparative baselines (accuracy):
- Logistic Regression: 87.61%
- Random Forest: 88.61%
- LightGBM: 91.50%
- Naive Bayes: 52.41%
- AdaBoost: 65.55%
- XGBoost: 90.87%
- Stacking (proposed): 93.21%

Note: ROS can inflate minority‑class metrics if not carefully controlled; in this work, ROS is applied only to the training set to avoid leakage.

---

## Project Structure

```plaintext
bert-ensemble-mental-health/
├── notebooks/                     # End-to-end experiments
│   └── bert_stacking_pipeline.ipynb
├── src/
│   ├── data.py                    # Load/split, ROS, leakage-safe transforms
│   ├── features.py                # TF-IDF + BERT embedding builders
│   ├── models.py                  # Base learners and stacking meta-learner
│   ├── train.py                   # Training/CV/ablation
│   └── evaluate.py                # Reports & metrics
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset

- Source: Sentiment Analysis for Mental Health (53,043 entries)
  - https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- Working subset: 20,000 samples, ROS‑balanced across 7 classes
- Public artifacts and instructions:
  - Harvard Dataverse (BertEnsemble): https://doi.org/10.7910/DVN/6TZJJP

Labels: Anxiety, Normal, Depression, Suicidal, Stress, Bipolar, Personality Disorder

---

## Method

- Preprocessing: lowercasing, URL/mention/hashtag cleanup, punctuation/number removal, stopwords, optional stemming/lemmatization
- Features:
  - BERT embeddings (768‑D pooled/CLS) from a pretrained model
  - TF‑IDF (fit on train only), then horizontally stacked with BERT
- Imbalance handling: Random Over‑Sampling (train split only)
- Stacking:
  - Base: LightGBM, Random Forest (predict_proba outputs concatenated)
  - Meta: XGBoost on base‑probability features
- Metrics: accuracy, precision, recall, F1 (macro/weighted), confusion matrix; 5‑fold CV

---

## Getting Started

Requirements
- Python 3.10
- scikit‑learn, xgboost, lightgbm, transformers or TF‑Hub (for BERT), pandas, numpy

Install
```bash
pip install -r requirements.txt
# or
pip install scikit-learn xgboost lightgbm transformers pandas numpy nltk
```

Run (typical flow)
- Load dataset, split (stratified), fit TF‑IDF on train, extract BERT embeddings
- Apply ROS on train only
- Train base learners → collect out‑of‑fold probabilities → train XGBoost meta‑learner
- Evaluate on test

Example (pseudo‑code)
```python
X_train, X_test, y_train, y_test = split(df, labels)
tfidf = build_tfidf().fit(X_train_text)
X_train_tfidf = tfidf.transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)

X_train_bert = bert_embed(X_train_text)   # (N, 768)
X_test_bert  = bert_embed(X_test_text)

X_train_hybrid = hstack([X_train_tfidf, X_train_bert])
X_test_hybrid  = hstack([X_test_tfidf,  X_test_bert])

X_res, y_res = ROS().fit_resample(X_train_hybrid, y_train)

stack = Stacking(base=[lgbm(), rf()], meta=xgb())
stack.fit(X_res, y_res)
report = evaluate(stack, X_test_hybrid, y_test)
print(report)
```

---

## Reproducibility

- Use fixed random_state=42 for splits and learners
- Fit TF‑IDF and any scalers on train only
- Apply ROS/augmentation strictly on train
- Save artifacts (vectorizer, class mapping, model)

---

## Ethics, Safety, and Scope

- Research/education only; not a medical device or diagnostic tool
- Evaluate with clinicians before any deployment
- Guard against biases, privacy risks, and false negatives in sensitive classes
- Prefer human‑in‑the‑loop review for any real‑world use

---

## License

MIT License. See LICENSE.

---

## Authors

- Gunda Sai Shivananda
- G. Muneeswari
