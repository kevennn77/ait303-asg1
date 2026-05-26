# Requirements: AIT303 Assignment 1 — Aspect-Based Sentiment Analysis

**Defined:** 2025-05-26
**Core Value:** Deliver a working aspect-based sentiment analysis pipeline with trained models, product rankings, and a well-documented Jupyter Notebook.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Sentiment Analysis (6 marks)

- [ ] **SENT-01**: Load and inspect IMDB dataset, check missing values, balance dataset
- [ ] **SENT-02**: Implement text preprocessing (lowercasing, HTML tags removal, punctuation removal, stopword removal, tokenization, lemmatization/stemming)
- [ ] **SENT-03**: Extract features with CountVectorizer for SVM
- [ ] **SENT-04**: Extract features with TfidfVectorizer for SVM
- [ ] **SENT-05**: Train CBOW word embeddings for BiGRU
- [ ] **SENT-06**: Train Skip-Gram word embeddings for BiGRU
- [ ] **SENT-07**: Train SVM + CountVectorizer model
- [ ] **SENT-08**: Train SVM + TfidfVectorizer model
- [ ] **SENT-09**: Train BiGRU + CBOW model
- [ ] **SENT-10**: Train BiGRU + Skip-Gram model
- [ ] **SENT-11**: Apply cross-validation (K-Fold / Stratified K-Fold) to all models
- [ ] **SENT-12**: Evaluate all models (accuracy, precision, recall, F1, confusion matrix)
- [ ] **SENT-13**: Save all trained models to disk (.pkl/.joblib/.h5)

### Aspect-Based Sentiment Analysis (9 marks)

- [ ] **ABSA-01**: Scrape 30+ speaker products with 100+ reviews each from Best Buy using Python scraper with polite delays
- [ ] **ABSA-02**: Train LDA model for unsupervised aspect extraction
- [ ] **ABSA-03**: Train BERTopic model for unsupervised aspect extraction
- [ ] **ABSA-04**: Analyze LDA and BERTopic topics, identify meaningful keywords
- [ ] **ABSA-05**: Restructure keywords into 6+ meaningful aspects (e.g., Design, Quality, Comfort, Performance, Price, Battery Life)
- [ ] **ABSA-06**: Train semi-supervised CorEx model with anchored aspects
- [ ] **ABSA-07**: Save all aspect extraction models to disk
- [ ] **ABSA-08**: Label all reviews with aspect labels (CorEx) and sentiment labels (best Task 1 model)
- [ ] **ABSA-09**: Calculate positive sentiment scores per aspect per product
- [ ] **ABSA-10**: Generate top 5 product rankings for each of the 6 aspects

### Product Selection & Visualization (part of ABSA)

- [ ] **VIZ-01**: Create 6 bar charts showing top 5 products per aspect with sentiment scores
- [ ] **VIZ-02**: Select one best product with detailed justification

### Report (part of Code & Report Quality, 5 marks)

- [ ] **RPT-01**: Write workflow discussion (data prep, preprocessing, feature extraction, training, CV, evaluation)
- [ ] **RPT-02**: Compare all 4 models with accuracy table, evaluation metrics, confusion matrices
- [ ] **RPT-03**: Justify the best performing model
- [ ] **RPT-04**: Discuss LDA and BERTopic outputs (topic coherence, interpretability)
- [ ] **RPT-05**: Explain how keywords were restructured into meaningful aspects
- [ ] **RPT-06**: Discuss CorEx output and compare unsupervised vs semi-supervised results
- [ ] **RPT-07**: Discuss top 5 products for each aspect and final best product selection
- [ ] **RPT-08**: Proper academic structure (Introduction, Methodology, Results, Conclusion)

### Code Quality (part of Code & Report Quality, 5 marks)

- [ ] **CODE-01**: Clear comments throughout all code
- [ ] **CODE-02**: Section headers in Jupyter Notebook
- [ ] **CODE-03**: Proper variable names and readable structure

### Deliverables

- [ ] **DLVR-01**: Jupyter Notebook (.ipynb) with all code
- [ ] **DLVR-02**: PDF report submitted to Moodle
- [ ] **DLVR-03**: Notebooks, models, Excel files uploaded to cloud storage (OneDrive/Google Drive/Box)
- [ ] **DLVR-04**: Shared folder link highlighted in report

## v2 Requirements

(None — this is a single-deliverable assignment.)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Model deployment / API | Not required by assignment |
| Web dashboard or UI | Assignment requires Jupyter Notebook and PDF report only |
| Real-time streaming analysis | Static dataset analysis only |
| Non-English reviews | IMDB and Best Buy reviews are English |
| Additional product categories | Speaker is the assigned category |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SENT-01 | Phase 1 | Pending |
| SENT-02 | Phase 1 | Pending |
| SENT-03 | Phase 2 | Pending |
| SENT-04 | Phase 2 | Pending |
| SENT-05 | Phase 3 | Pending |
| SENT-06 | Phase 3 | Pending |
| SENT-07 | Phase 2 | Pending |
| SENT-08 | Phase 2 | Pending |
| SENT-09 | Phase 3 | Pending |
| SENT-10 | Phase 3 | Pending |
| SENT-11 | Phase 2, 3 | Pending |
| SENT-12 | Phase 2, 3 | Pending |
| SENT-13 | Phase 3 | Pending |
| ABSA-01 | Phase 4 | Pending |
| ABSA-02 | Phase 4 | Pending |
| ABSA-03 | Phase 4 | Pending |
| ABSA-04 | Phase 4 | Pending |
| ABSA-05 | Phase 4 | Pending |
| ABSA-06 | Phase 4 | Pending |
| ABSA-07 | Phase 4 | Pending |
| ABSA-08 | Phase 5 | Pending |
| ABSA-09 | Phase 5 | Pending |
| ABSA-10 | Phase 5 | Pending |
| VIZ-01 | Phase 5 | Pending |
| VIZ-02 | Phase 5 | Pending |
| RPT-01 | Phase 6 | Pending |
| RPT-02 | Phase 6 | Pending |
| RPT-03 | Phase 6 | Pending |
| RPT-04 | Phase 6 | Pending |
| RPT-05 | Phase 6 | Pending |
| RPT-06 | Phase 6 | Pending |
| RPT-07 | Phase 6 | Pending |
| RPT-08 | Phase 6 | Pending |
| CODE-01 to CODE-03 | Phase 1–6 | Pending |
| DLVR-01 | Phase 6 | Pending |
| DLVR-02 | Phase 6 | Pending |
| DLVR-03 | Phase 6 | Pending |
| DLVR-04 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 35
- Unmapped: 0 ✓

---

*Requirements defined: 2025-05-26*
*Last updated: 2025-05-26 after initial definition*
