# Phase 1: Data Preparation & Preprocessing — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2025-05-26
**Phase:** 1-Data Preparation & Preprocessing
**Areas discussed:** Lemmatization vs Stemming, Notebook Architecture, Stopword Handling, Dataset Acquisition

---

## Lemmatization vs Stemming

| Option | Description | Selected |
|--------|-------------|----------|
| Lemmatization | NLTK WordNet lemmatizer with POS tagging | |
| Stemming | NLTK Porter stemmer | |
| Both — compare | Run both pipelines, feed both into downstream model training | ✓ |

**User's choice:** Both — compare
**Notes:** NLTK WordNet lemmatizer (not spaCy) with POS tagging enabled. Both preprocessing pipelines flow into downstream model training in Phases 2-3.

---

## Notebook Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Notebook-only | All code in Jupyter Notebook with markdown section headers | ✓ |
| Hybrid: .py utils + notebook | Core functions in Python module, notebook orchestrates | |

**User's choice:** Notebook-only
**Notes:** Sections: 1. Setup/Imports, 2. Load & Inspect, 3. Text Cleaning Pipeline, 4. Stemming Output, 5. Lemmatization Output.

---

## Stopword Handling

| Option | Description | Selected |
|--------|-------------|----------|
| NLTK defaults (~179 words) | Standard set | ✓ |
| NLTK + domain additions | Also remove 'movie', 'film', 'br', etc. | |

**User's choice:** NLTK defaults (~179 words)
**Notes:** Default set only — no domain-specific additions.

---

## Dataset Acquisition

| Option | Description | Selected |
|--------|-------------|----------|
| Kaggle API (programmatic) | kagglehub Python library with token auth | |
| Manual download + local path | Download CSV from Kaggle web UI, read from local | ✓ |
| Include both options | Auto-download with API fallback to local path | |

**User's choice:** Manual download + local path
**Notes:** CSV saved to `data/` directory. No Kaggle API setup needed.

---

## Agent's Discretion

- Preprocessing order within the pipeline (standard NLP order: lowercasing → HTML removal → punctuation removal → tokenization → stopword removal → stemming/lemmatization)
- NLTK resource download management (WordNet, stopwords, averaged_perceptron_tagger)

## Deferred Ideas

None.
