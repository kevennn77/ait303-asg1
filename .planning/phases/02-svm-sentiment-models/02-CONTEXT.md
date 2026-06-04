# Phase 2: SVM Sentiment Models — Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Train and evaluate two SVM sentiment classification models using features extracted from preprocessed IMDB reviews. CountVectorizer and TfidfVectorizer each produce feature vectors from both stemmed and lemmatized text (4 model variants total). All models evaluated with cross-validation and standard classification metrics.

**Requirements:** SENT-03, SENT-04, SENT-07, SENT-08, SENT-11, SENT-12

**Dependencies:** Phase 1 — consumes `cleaned`, `stemmed`, `lemmatized` columns from `sentiment_analysis_preprocessing.ipynb`. The Phase 1 notebook must be executed first to produce the preprocessed DataFrame.
</domain>

<decisions>
## Implementation Decisions

### Preprocessing Input
- **D-10:** Feature extraction uses **both stemmed and lemmatized** text. This produces 4 SVM model variants: CountVectorizer + stemmed, CountVectorizer + lemmatized, TfidfVectorizer + stemmed, TfidfVectorizer + lemmatized. Enables comparison of how preprocessing strategy affects SVM performance.

### Feature Vectorization
- **D-11:** Both CountVectorizer and TfidfVectorizer use **n-gram range (1,2)** — unigrams + bigrams — to capture common sentiment phrases (e.g., "not good").

### Model Configuration
- **D-12:** SVM kernel: **Linear** — standard for high-dimensional sparse text features. Fast training, interpretable coefficients, proven performance on BoW representations.
- **D-13:** Hyperparameters: sklearn defaults initially (C=1.0). GridSearchCV may be used for tuning if baseline performance is poor.

### Cross-Validation
- **D-14:** **5-fold Stratified K-Fold** cross-validation — preserves class distribution per fold, standard for balanced classification tasks.

### Evaluation
- **D-15:** Metrics: accuracy, precision, recall, F1-score, confusion matrix — per SENT-12.
- **D-16:** Evaluation done on held-out test set (80/20 split) in addition to cross-validation scores for final model assessment.

### Notebook Architecture
- **D-17:** **New notebook** — `svm_sentiment_models.ipynb` — separate from Phase 1 preprocessing notebook. Loads preprocessed data from Phase 1 output and builds SVM models.
- **D-18:** Notebook sections: 1. Setup & Imports, 2. Load Preprocessed Data, 3. Feature Extraction (CountVectorizer, TfidfVectorizer), 4. Model Training, 5. Cross-Validation & Evaluation, 6. Results Summary

### the Agent's Discretion
- Specific sklearn imports and pipeline construction (use of `Pipeline` class, `make_pipeline`)
- Train/test split ratio beyond the 80/20 baseline
- GridSearchCV parameter grid (if used)
- Whether to save intermediate feature matrices or compute on-the-fly
- Visualization style for confusion matrices and metrics tables
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Assignment Spec
- `assignment.md` — Full assignment brief defining all requirements for all tasks

### Project Planning
- `.planning/PROJECT.md` — Project definition, key decisions, constraints
- `.planning/REQUIREMENTS.md` — Full requirements traceability matrix
- `.planning/ROADMAP.md` — Phase breakdown, dependencies, success criteria

### Phase 1 Artifacts
- `.planning/phases/01-data-preparation-preprocessing/01-CONTEXT.md` — Phase 1 locked decisions (D-01 through D-09)
- `.planning/phases/01-data-preparation-preprocessing/SKELETON.md` — Walking Skeleton with architectural decisions
- `sentiment_analysis_preprocessing.ipynb` — Phase 1 notebook output (preprocessed DataFrame with cleaned, stemmed, lemmatized columns)

### Documentation
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
</canonical_refs>

<specifics>
## Specific Ideas

- Use `sklearn.pipeline.Pipeline` to chain vectorizer + SVM into a single estimator for clean cross-validation
- Compare performance across all 4 variants systematically with a results table
- Include confusion matrix visualizations for the best-performing variant per vectorizer
- Show most informative features (top positive/negative coefficients) from the linear SVM for interpretability
</specifics>

<deferred>
## Deferred Ideas

None — phase scope covers all SVM requirements.
</deferred>

---

*Phase: 2-SVM Sentiment Models*
*Context gathered: 2026-05-26*
