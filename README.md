# GridSentinel AI

A governed agentic AI platform for extreme-weather grid reliability and major-outage risk assessment.

GridSentinel combines machine learning, retrieval-augmented generation, structured knowledge, LLM reasoning, human-in-the-loop governance, observability, and automated cloud deployment in a single end-to-end system.

## Live Demo

Cloud Run application:

https://gridsentinel-ai-1013046614784.us-central1.run.app

API documentation:

https://gridsentinel-ai-1013046614784.us-central1.run.app/docs

---

## Problem

Extreme-weather events can create rapid and safety-critical changes in electric-grid reliability.

The goal of GridSentinel is not to autonomously operate grid infrastructure. Instead, it acts as a decision-support system that:

- estimates major-outage risk,
- retrieves relevant historical evidence,
- connects structured domain knowledge,
- generates evidence-grounded explanations,
- identifies high-risk cases that require human engineering review,
- records model and LLM runtime behavior for auditability.

The current prototype focuses on Harris County, Texas.

---

## Data

The project integrates public historical data from:

- ORNL EAGLE-I power outage data
- NOAA Storm Events data
- curated regional extreme-weather event flags

The modeling dataset includes:

- current severe-weather indicators,
- recent 7-day weather indicators,
- calendar features,
- previous-day outage levels,
- rolling outage statistics.

Data-quality checks reject invalid outage-rate calculations when customer totals are missing, non-positive, or inconsistent.

---

## Machine Learning

Three candidate models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The final promoted model was Logistic Regression.

### Final model performance

| Metric | Value |
|---|---:|
| ROC-AUC | 0.994 |
| PR-AUC | 0.949 |
| Precision | 0.846 |
| Recall | 0.917 |
| F1 | 0.880 |
| Brier Score | 0.0179 |

A time-based split was used:

- Train: January-June 2024
- Test: July-December 2024

The major-outage threshold was learned only from the training period to reduce leakage risk.

---

## Exact Model Reasoning

Because the production model is Logistic Regression, GridSentinel does not need an approximate surrogate for local explanations.

The prediction is decomposed directly into additive log-odds contributions:

```text
log_odds =
intercept
+ feature_1 * coefficient_1
+ ...
+ feature_n * coefficient_n