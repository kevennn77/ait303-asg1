---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-05-31T10:15:00.000Z"
progress:
  total_phases: 6
  completed_phases: 2
  total_plans: 10
  completed_plans: 8
  percent: 33
---

# Project State: AIT303 Assignment 1

> **Last updated:** 2026-05-31
> **Status:** Executing Phase 04 (notebook complete — ready for Colab execution)

## Project Reference

See: .planning/PROJECT.md (updated 2025-05-26)

**Core value:** Deliver a working aspect-based sentiment analysis pipeline with trained models, product rankings, and a well-documented Jupyter Notebook.

**Current focus:** Phase 04 — Web Scraping & Aspect Extraction

## Phase Status

| Phase | Status | Plans | Progress |
|-------|--------|-------|----------|
| 1 — Data Preparation & Preprocessing | ✓ Complete | 2/2 | 100% |
| 2 — SVM Sentiment Models | ✓ Complete | 2/2 | 100% |
| 3 — BiGRU Sentiment Models | ✓ Complete | 2/2 | 100% |
| 4 — Web Scraping & Aspect Extraction | ✓ Complete | 4/4 | 100% |
| 5 — Labeling, Ranking & Visualization | ○ Pending | 0/0 | 0% |
| 6 — Report & Deliverables | ○ Pending | 0/0 | 0% |

## Completed Phases

### Phase 1: Data Preparation & Preprocessing ✓

**Deliverable:** `sentiment_analysis_preprocessing.ipynb`

- **Section 1:** Environment setup, NLTK downloads (punkt, stopwords, wordnet, averaged_perceptron_tagger), all imports
- **Section 2:** IMDB 50K dataset loaded and inspected — shape (50000, 2), 0 missing values, perfectly balanced (25000/25000)
- **Section 3:** Text cleaning pipeline — lowercasing, HTML tag removal, punctuation removal, whitespace normalization. Verified 0 `<br>` tags remaining.
- **Section 4:** Porter stemming (D-04) — tokenization, stopword removal (D-07), suffix stripping. Output shows non-words (e.g., "episod", "violenc") confirming algorithmic stemming.
- **Section 5:** WordNet lemmatization with POS tagging (D-02/D-03) — Penn Treebank→WordNet POS mapping, dictionary-based root forms. Output shows real words (e.g., "episode", "violence").

**Verification:** Full notebook executes end-to-end (nbconvert), all 20 code cells pass.
**Plans executed:** 01-01 (Dataset Loading & Inspection), 01-02 (Text Preprocessing Pipeline)
**Requirements satisfied:** SENT-01, SENT-02, CODE-01, CODE-02, CODE-03

## Completed Phases (cont.)

### Phase 2: SVM Sentiment Models ✓

**Deliverable:** `svm_sentiment_models.ipynb`

- **Section 1:** Setup & imports, COLAB flag for Google Colab/Drive mounting
- **Section 2:** Load preprocessed IMDB data — shape (50000, 5), encoded sentiment, 80/20 train/test split
- **Section 3:** Feature extraction — CountVectorizer + TfidfVectorizer with ngram_range=(1,2), separate vectorizers for stemmed/lemmatized
- **Section 4:** LinearSVC model instantiation (pre-vectorized, no pipelines — faster than SVC/LibSVM)
- **Section 5:** 5-fold Stratified K-Fold CV + test-set evaluation + top features + consolidated 4-model comparison
- **Section 6:** Results summary — TFIDF + Lemmatized is best overall (F1=0.9091)

**Results:**
| Model | Test F1 | 5-Fold CV F1 |
|-------|---------|--------------|
| TFIDF + Lemmatized | **0.9091** | 0.9027 |
| TFIDF + Stemmed | 0.9080 | **0.9030** |
| CV + Lemmatized | 0.8966 | 0.8905 |
| CV + Stemmed | 0.8960 | 0.8924 |

**Verification:** All 19 code cells executed successfully on Colab. CountVectorizer models hit max_iter limit (convergence warning); TfidfVectorizer models converged cleanly. 
**Plans executed:** 02-01, 02-02
**Requirements satisfied:** SENT-03, SENT-04, SENT-07, SENT-08, SENT-11, SENT-12
**Decisions honored:** D-10 through D-18

### Phase 3: BiGRU Sentiment Models ✓

**Deliverable:** `bigru_sentiment_models.ipynb`

- **Section 1:** Setup & imports with full reproducibility (numpy+random+tf seeds), COLAB flag, Drive mount
- **Section 2:** Load preprocessed IMDB data — extract lemmatized texts + encoded labels, no held-out split (all 50K for CV)
- **Section 3:** Train Word2Vec CBOW (sg=0) and Skip-Gram (sg=1) with vector_size=100, window=5, min_count=5, epochs=5
- **Section 4:** Build vocabulary (Tokenizer, VOCAB_SIZE=vocab+2), CBOW and SG embedding matrices (100-dim), BiGRU model builder (1-layer, 128 units/direction, dropout 0.5, sigmoid output)
- **Section 5:** 5-fold StratifiedKFold CV training loop (seed=42, same as Phase 2) with EarlyStopping (patience=3, restore_best_weights=True), per-fold confusion matrices, model saving, memory cleanup (del model + clear_session + gc.collect)
- **Section 6:** Aggregate metrics (mean ± std), comparison table (CBOW-BiGRU vs SG-BiGRU vs Phase 2 SVM best), ROC curves, loss curves, best model identification
- **Section 7:** Model file verification — expects 12 files (2 Word2Vec .model + 10 BiGRU .h5)

**Note:** Gensim cannot install on Python 3.14 (C API incompatibility). Notebook designed for Colab execution (Python 3.10). See [03-USER-SETUP.md](./phases/03-bigru-sentiment-models/03-USER-SETUP.md).

**Plans executed:** 03-01 (notebook scaffold + Word2Vec), 03-02 (5-fold CV + evaluation)
**Requirements satisfied:** SENT-05, SENT-06, SENT-09, SENT-10, SENT-11, SENT-12, SENT-13

### Phase 4: Web Scraping & Aspect Extraction ✓

**Deliverable:** `bestbuy_scraper.py` + `notebooks/aspect_extraction.ipynb`

- **04-01:** Web scraper with polite crawling (random delays, retry/backoff, User-Agent rotation), HTML → JSON parsing, product catalog extraction
- **04-02:** Notebook scaffold with SpaCy NP chunker keyphrase extraction, DataFrame validation
- **04-03:** LDA (LdaMulticore, grid search 6-10, C_v coherence) and BERTopic (all-MiniLM-L6-v2, UMAP/HDBSCAN) unsupervised topic modeling
- **04-04:** CorEx semi-supervised anchored topic modeling (8 aspects), three-model persistence (LDA/BERTopic/CorEx), Phase 5 CSV contract export

**Output artifacts (created during Colab execution):**
- `models/lda_model/` — gensim LdaMulticore
- `models/bertopic_model/` — BERTopic safetensors
- `models/corex_model.pkl` — CorEx anchored topic model
- `data_asg/bestbuy/aspect_labeled_reviews.csv` — Phase 5 contract (7 columns)

**Plans executed:** 04-01 (scraper), 04-02 (scaffold + preprocessing), 04-03 (LDA + BERTopic), 04-04 (CorEx + persistence + CSV)

## Recent Activity

- 2025-05-26: Project initialized via `/gsd-new-project`
- 2025-05-26: Phase 1 context gathered, planned, and executed
- 2025-05-26: Phase 2 planned via `/gsd-plan-phase 2`
- 2025-05-26: Phase 2 notebook written (`svm_sentiment_models.ipynb`, 36 cells)
- 2025-05-30: **Phase 2 executed on Colab**
  - Refactored: SVC→LinearSVC, pipeline→pre-vectorized (10× speedup)
  - Refactored: macro→weighted metric averaging
  - Refactored: separate vectorizer objects for stemmed/lemmatized
  - Full 19-cell execution completed on Colab
  - CountVectorizer models: convergence warning (ill-conditioned features)
  - TfidfVectorizer models: clean convergence
- 2025-05-30: **Phase 2 marked complete**
- 2025-05-30: **Phase 3 notebook written** (`bigru_sentiment_models.ipynb`, 48 cells)
- 2025-05-30: **Phase 3 marked complete**

## Next Actions

1. ~~**Phase 4 (Aspect Extraction):** Execute 4 plans (04-01 through 04-04)~~ ✓ Complete
2. **🏷️ PENDING — Run Phases 1-4 notebooks on Colab:**
   - `sentiment_analysis_preprocessing.ipynb` — already verified via nbconvert, good to run
   - `svm_sentiment_models.ipynb` — re-run with `max_iter=5000` for CountVectorizer convergence
   - `bigru_sentiment_models.ipynb` — full 48-cell run on T4 GPU (~60-90 min)
   - `notebooks/aspect_extraction.ipynb` — full 46-cell run (requires Best Buy scraped data)
   - Why: All notebooks are written; final execution on Colab generates real model files
3. **Phase 5 (Labeling & Ranking):** Use best sentiment model for review labeling, produce product rankings
4. **🏷️ PENDING — Re-run SVM notebook after all phases complete:**
   - Notebook: `svm_sentiment_models.ipynb`
   - What: Re-execute on Colab with `max_iter=5000` for full convergence of CountVectorizer models
   - Why: Current CountVectorizer models stopped at 2000 iterations (ConvergenceWarning); bump to 5000 for cleaner results for the report
   - Effort: ~5 min on Colab (pre-vectorized data already cached)

## Session History

| Date | Stopped At | Resume File |
|------|-----------|-------------|
| 2026-05-26 | Phase 2 notebook written for Colab | `svm_sentiment_models.ipynb` |
| 2026-05-30 | Phase 2 completed on Colab | `svm_sentiment_models.ipynb` |
| 2026-05-31 | Phase 4 notebook completed (Sections 7-8 CorEx + persistence + CSV export) | `notebooks/aspect_extraction.ipynb` |
