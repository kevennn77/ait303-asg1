# Plan 02-01 Summary: SVM + CountVectorizer Notebook

## Status
✅ **Written — Requires Colab execution for heavy cells**

## Deliverables
- `svm_sentiment_models.ipynb` — Complete notebook (36 cells: 19 code + 17 markdown)

## Notebook Structure
| Section | Content | Cells |
|---------|---------|-------|
| 1. Setup & Imports | Environment, imports, config (COLAB flag) | 2-3 |
| 2. Load Preprocessed Data | Load `preprocessed_imdb.csv`, encode labels, train/test split (80/20) | 4-5 |
| 3. Feature Extraction | CountVectorizer + TfidfVectorizer (ngram_range=(1,2)) on stemmed + lemmatized | 6-8 |
| 4. Model Training | 4 Pipeline objects (CV+stemmed, CV+lemmatized, TFIDF+stemmed, TFIDF+lemmatized) | 9-11 |
| 5. Cross-Validation & Evaluation | 5-fold StratifiedKFold CV, test metrics, confusion matrices, top features, 4-model comparison | 12-24 |
| 6. Results Summary | Rankings, takeaways for report | 25-26 |

## Decisions Implemented
- D-10: Both stemmed AND lemmatized text used (4 model variants)
- D-11: `ngram_range=(1,2)` on both vectorizers
- D-12: `SVC(kernel='linear')`
- D-13: `C=1.0`
- D-14: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- D-15: Accuracy, precision, recall, F1, confusion matrix + heatmaps
- D-16: 80/20 train/test split with stratification
- D-17: New notebook (`svm_sentiment_models.ipynb`)
- D-18: 6-section notebook structure

## Heavy Computation Cells
| Cell | What | Est. Time |
|------|------|-----------|
| Cell 20 (5.1 CV) | 5-fold CV — CountVectorizer SVMs | 15-30 min |
| Cell 25 (5.3 CV) | 5-fold CV — TfidfVectorizer SVMs | 15-30 min |

## To Complete
Run in Colab (see Colab instructions below).
