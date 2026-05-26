# AIT303 Assignment 1 — Aspect-Based Sentiment Analysis

## What This Is

An academic NLP assignment that implements a complete aspect-based sentiment analysis pipeline. It trains four sentiment classification models (SVM with CountVectorizer/TfidfVectorizer, BiGRU with CBOW/Skip-Gram) on the IMDB 50K movie reviews dataset, then performs aspect extraction on scraped Best Buy speaker reviews using LDA, BERTopic, and CorEx to produce ranked product recommendations.

## Core Value

Deliver a working aspect-based sentiment analysis pipeline with trained models, product rankings, and a well-documented Jupyter Notebook that demonstrates the full NLP cycle from preprocessing through evaluation and interpretation.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] **SENT-01**: Implement text preprocessing pipeline (lowercasing, HTML tag removal, punctuation removal, stopword removal, tokenization, lemmatization/stemming)
- [ ] **SENT-02**: Train SVM model with CountVectorizer features
- [ ] **SENT-03**: Train SVM model with TfidfVectorizer features
- [ ] **SENT-04**: Train BiGRU model with CBOW word embeddings
- [ ] **SENT-05**: Train BiGRU model with Skip-Gram word embeddings
- [ ] **SENT-06**: Apply cross-validation (K-Fold / Stratified K-Fold) to all models
- [ ] **SENT-07**: Evaluate all models (accuracy, precision, recall, F1, confusion matrix)
- [ ] **SENT-08**: Save all trained models to disk (.pkl/.joblib/.h5)
- [ ] **ABSA-01**: Scrape 30+ speaker products (100+ reviews each) from Best Buy using polite Python scraper
- [ ] **ABSA-02**: Run unsupervised aspect extraction using LDA
- [ ] **ABSA-03**: Run unsupervised aspect extraction using BERTopic
- [ ] **ABSA-04**: Analyze topics, restructure into 6+ meaningful aspects
- [ ] **ABSA-05**: Train semi-supervised CorEx model with anchored aspects
- [ ] **ABSA-06**: Label reviews with aspect and sentiment labels
- [ ] **ABSA-07**: Rank top 5 products per aspect by positive sentiment score
- [ ] **ABSA-08**: Select and justify one best product
- [ ] **ABSA-09**: Save all aspect models to disk
- [ ] **RPT-01**: Write academic report with proper sections (Introduction, Methodology, Results, Conclusion)
- [ ] **RPT-02**: Include model comparison table, confusion matrices, evaluation metrics
- [ ] **RPT-03**: Create 6 bar charts showing top 5 products per aspect
- [ ] **RPT-04**: Justify best sentiment analysis model and best product selection
- [ ] **DLVR-01**: Deliver Jupyter Notebook (.ipynb) with clear comments, section headers, readable structure
- [ ] **DLVR-02**: Upload report (PDF), notebooks, models, Excel files to cloud storage

### Out of Scope

- Real-time/streaming sentiment analysis — static dataset analysis only
- Web-based UI or dashboard — Jupyter Notebook + PDF report only
- Languages other than English — IMDB and Best Buy reviews are English

## Context

- **Course:** AIT303 Advanced Issues of AI (Natural Language Processing)
- **Academic Session:** 2026/04
- **Deadline:** 5 June 2025, 5:00 PM
- **Product category:** Speaker (assigned to student)
- **Approach:** Python scraper with polite delays (no MCP/skills-based scraping) — rate limiting with randomized intervals to avoid blocking
- **Deliverables:** PDF report, Jupyter Notebooks, model files, Excel files
- **Marking:** Sentiment Analysis (6 marks), ABSA (9 marks), Code & Report Quality (5 marks) = 20 total

## Constraints

- **Tech Stack**: Python, Jupyter Notebook, scikit-learn, TensorFlow/Keras (or PyTorch), gensim, transformers, BERTopic
- **Timeline**: Submission by 5 June 2025
- **Scraping**: Must use Python scripts (not MCP skills); implement polite scraping with delays
- **Report**: PDF format, submitted to Moodle; Notebooks/models uploaded to cloud storage with link in report

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Product category: Speaker | Assigned by lecturer | — Pending |
| Python scraper (not MCP) | Scraping code must be student's own work | — Pending |
| Polite scraping with delays | Avoid rate limiting and IP blocking | — Pending |
| Jupyter Notebook | Required by assignment spec | — Pending |
| BiGRU over BiLSTM | GRU simpler/faster than LSTM for this scale | — Pending |

---

*Last updated: 2025-05-26 after initialization*
