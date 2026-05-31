# Roadmap: AIT303 Assignment 1

**6 phases** | **35 requirements mapped** | All v1 requirements covered ✓

## Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Data Preparation & Preprocessing | Load IMDB dataset and build text cleaning pipeline | SENT-01, SENT-02, CODE-01, CODE-02, CODE-03 | 4 |
| 2 | SVM Sentiment Models | Train and evaluate both SVM variants | SENT-03, SENT-04, SENT-07, SENT-08, SENT-11, SENT-12 | 5 |
| 3 | BiGRU Sentiment Models | Train and evaluate both BiGRU variants | SENT-05, SENT-06, SENT-09, SENT-10, SENT-11, SENT-12, SENT-13 | 5 |
| 4 | Web Scraping & Aspect Extraction | Scrape reviews and extract aspects | ABSA-01, ABSA-02, ABSA-03, ABSA-04, ABSA-05, ABSA-06, ABSA-07 | 5 |
| 5 | Labeling, Ranking & Visualization | Label reviews, rank products, create charts | ABSA-08, ABSA-09, ABSA-10, VIZ-01, VIZ-02 | 5 |
| 6 | Report & Deliverables | Compile report, finalize notebooks, submit | RPT-01 to RPT-08, DLVR-01 to DLVR-04 | 5 |

---

## Phase Details

### Phase 1: Data Preparation & Preprocessing
**Goal:** Load and inspect the IMDB 50K dataset, then build a complete text preprocessing pipeline.

**Mode:** mvp

**Requirements:** SENT-01, SENT-02, CODE-01, CODE-02, CODE-03

**Plans:** 2 plans in 2 waves

**Success Criteria:**
1. IMDB dataset loaded and inspected — shape, missing values, class balance confirmed
2. Dataset is balanced (or balancing strategy documented if already balanced)
3. Text preprocessing pipeline applied: lowercasing, HTML tag removal, punctuation removal, stopword removal, tokenization, lemmatization
4. Cleaned text samples verified and ready for feature extraction

**Risks:**
- Kaggle dataset download may require authentication — cache local copy

**Plans:**
- [ ] 01-01-PLAN.md — Dataset loading, inspection, and notebook scaffold
- [ ] 01-02-PLAN.md — Text cleaning, stemming, and lemmatization pipelines

---

### Phase 2: SVM Sentiment Models
**Goal:** Extract features using CountVectorizer and TfidfVectorizer, train both SVM models, and evaluate with cross-validation.

**Mode:** mvp

**Requirements:** SENT-03, SENT-04, SENT-07, SENT-08, SENT-11, SENT-12

**Success Criteria:**
1. CountVectorizer and TfidfVectorizer features extracted with appropriate parameters
2. SVM + CountVectorizer model trained
3. SVM + TfidfVectorizer model trained
4. Cross-validation (K-Fold or Stratified K-Fold) applied to both models
5. Both models evaluated with accuracy, precision, recall, F1-score, and confusion matrices

**Dependencies:** Phase 1

**Plans:**
- [ ] 02-01-PLAN.md — SVM + CountVectorizer: training, 5-fold CV, evaluation, and results
- [ ] 02-02-PLAN.md — SVM + TfidfVectorizer: training, 5-fold CV, and consolidated 4-model comparison

---

### Phase 3: BiGRU Sentiment Models
**Goal:** Train CBOW and Skip-Gram word embeddings, build and train both BiGRU models, save all models.

**Mode:** mvp

**Requirements:** SENT-05, SENT-06, SENT-09, SENT-10, SENT-11, SENT-12, SENT-13

**Plans:** 2 plans in 2 waves

**Success Criteria:**
1. CBOW word embeddings trained on preprocessed reviews
2. Skip-Gram word embeddings trained on preprocessed reviews
3. BiGRU + CBOW model trained with proper architecture
4. BiGRU + Skip-Gram model trained with proper architecture
5. Both BiGRU models evaluated (accuracy, precision, recall, F1, confusion matrices) with cross-validation
6. All 4 models saved to disk (.pkl/.joblib for SVM, .h5 for BiGRU)

**Dependencies:** Phase 1

**Plans:**
- [ ] 03-01-PLAN.md — Word2Vec embedding training (CBOW + Skip-Gram), vocabulary building, BiGRU model builder
- [ ] 03-02-PLAN.md — 5-fold CV training for both BiGRU variants, evaluation, model saving, comparison with Phase 2

---

### Phase 4: Web Scraping & Aspect Extraction
**Goal:** Scrape 30+ speaker products from Best Buy, then perform unsupervised and semi-supervised aspect extraction.

**Mode:** mvp

**Requirements:** ABSA-01, ABSA-02, ABSA-03, ABSA-04, ABSA-05, ABSA-06, ABSA-07

**Plans:** 4 plans in 3 waves

**Success Criteria:**
1. Python scraper collects 30+ speaker products with 100+ reviews each, with polite delays
2. LDA model trained and produces interpretable topic clusters
3. BERTopic model trained and produces semantically coherent topic clusters
4. Topics analyzed and keywords identified from both unsupervised outputs
5. Keywords restructured into 6+ meaningful aspects
6. CorEx model trained with anchored aspects
7. All aspect models saved

**Risks:**
- Best Buy may block aggressive scraping — use randomized delays (1-3 second intervals)
- BERTopic requires sentence-transformers download — plan for first-run latency

**Plans:**
- [ ] 04-01-PLAN.md — Scraper script + Wave 0 test infrastructure
- [ ] 04-02-PLAN.md — Notebook scaffold + preprocessing + SpaCy keyphrase extraction
- [ ] 04-03-PLAN.md — Unsupervised topic models (LDA + BERTopic)
- [ ] 04-04-PLAN.md — CorEx + model persistence + CSV export for Phase 5

---
### Phase 5: Labeling, Ranking & Visualization
**Goal:** Label all reviews with aspect and sentiment, rank products per aspect, create bar charts.

**Mode:** mvp

**Requirements:** ABSA-08, ABSA-09, ABSA-10, VIZ-01, VIZ-02

**Success Criteria:**
1. All scraped reviews labeled with aspect (CorEx) and sentiment (best Task 1 model)
2. Positive sentiment scores calculated per aspect per product
3. Top 5 products ranked for each of the 6 aspects
4. 6 bar charts created showing top 5 products per aspect
5. One best product selected with clear justification

**Dependencies:** Phase 3 (best sentiment model), Phase 4 (aspect models)

---

### Phase 6: Report & Deliverables
**Goal:** Write the academic report, finalize notebook, submit everything.

**Mode:** mvp

**Requirements:** RPT-01, RPT-02, RPT-03, RPT-04, RPT-05, RPT-06, RPT-07, RPT-08, DLVR-01, DLVR-02, DLVR-03, DLVR-04

**Success Criteria:**
1. Report organized into proper sections (Introduction, Methodology, Sentiment Analysis, ABSA, Results, Conclusion)
2. Workflow discussion covers all steps from data prep to evaluation
3. All 4 models compared with accuracy table, metrics, confusion matrices
4. Best model justified with reasoning
5. LDA/BERTopic outputs discussed with topic coherence analysis
6. Keyword-to-aspect restructuring explained
7. CorEx output discussed; unsupervised vs semi-supervised compared
8. Top 5 product rankings per aspect explained
9. Best product selection with detailed reasoning
10. Jupyter Notebook has clear comments, section headers, proper variable names
11. PDF report and all deliverables uploaded

**Dependencies:** Phase 2, Phase 3, Phase 4, Phase 5

---

## Requirement-Phase Mapping

| Requirement | Phase | Status |
|-------------|-------|--------|
| SENT-01 | Phase 1 | Pending |
| SENT-02 | Phase 1 | Pending |
| SENT-03 | Phase 2 | Pending |
| SENT-04 | Phase 2 | Pending |
| SENT-07 | Phase 2 | Pending |
| SENT-08 | Phase 2 | Pending |
| SENT-11 | Phase 2, 3 | Pending |
| SENT-12 | Phase 2, 3 | Pending |
| SENT-05 | Phase 3 | Pending |
| SENT-06 | Phase 3 | Pending |
| SENT-09 | Phase 3 | Pending |
| SENT-10 | Phase 3 | Pending |
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
| RPT-01 to RPT-08 | Phase 6 | Pending |
| CODE-01 to CODE-03 | Phase 1–6 | Pending |
| DLVR-01 to DLVR-04 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 35
- Unmapped: 0 ✓

---

*Created: 2025-05-26*
