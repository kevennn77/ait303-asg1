# Phase 03: BiGRU Sentiment Models — Design Context

## Phase Goal

Train two BiGRU classifiers (CBOW-initialized + Skip-Gram-initialized) on the IMDB 50K dataset, evaluate via 5-fold CV, and save trained models for downstream aspect-based sentiment analysis on Best Buy speaker reviews.

## Requirements Traceability

| Req ID | Description | Status |
|--------|-------------|--------|
| SENT-05 | Train Word2Vec CBOW on IMDB lemmatized text | Awaiting |
| SENT-06 | Train Word2Vec Skip-Gram on IMDB lemmatized text | Awaiting |
| SENT-09 | Build + train BiGRU classifier (CBOW-init embeddings) | Awaiting |
| SENT-10 | Build + train BiGRU classifier (Skip-Gram-init embeddings) | Awaiting |
| SENT-11 | Cross-validate on IMDB (StratifiedKFold, n=5) | Awaiting |
| SENT-12 | Evaluate (accuracy, precision, recall, F1, ROC-AUC) | Awaiting |
| SENT-13 | Save trained models for inference | Awaiting |

## Design Decisions (All Captured)

### Word Embedding Config
- **Vector size**: 100 (CBOW + Skip-Gram, same dim for fair comparison)
- **Window**: 5 (standard sentiment context window)
- **min_count**: 5 (filters rare tokens, reduces vocab noise)
- **Training text**: lemmatized only (single preprocessing pipeline)

### BiGRU Architecture
- **Layers**: 1 (sufficient for sentiment — deeper = diminishing returns + overfit risk)
- **Hidden units per direction**: 128 (256 total for bidirectional)
- **Dropout**: 0.5 (applied between BiGRU and classifier)
- **Classifier**: `Dense(1, sigmoid)` — no extra hidden layer; standard for binary sentiment
- **Embedding training**: fine-tune (initialize from word2vec, update during BiGRU training)

### Training Strategy
- **Epochs**: 20 (safety margin; early stopping will cut early if converged)
- **Batch size**: 64 (good throughput/update balance)
- **Optimizer**: Adam with LR 1e-3 (standard, well-tuned for RNNs)
- **Early stopping**: patience 3 on val_loss; restore best weights
- **Loss**: binary crossentropy

### CV Strategy
- **Method**: 5-fold StratifiedKFold (identical to Phase 2 SVM for fair comparison)
- **Consequence**: 10 total training runs (2 embeddings × 5 folds)
- **Validation**: held-out fold per run

### GPU Execution
- **Platform**: Google Colab (T4 GPU, `COLAB=True` flag)
- **Storage**: Google Drive mount, `DATA_DIR` pointing to Drive path
- **Rationale**: matches Phase 2 infrastructure; free GPU for 10 training runs

## Key Constraints

1. **Fair comparison with SVM (Phase 2)**: Use same CV splits (same seed), same evaluation metrics, same preprocessing output.
2. **10 training runs**: 2 embeddings × 5 folds — must fit within Colab T4 memory and time limits.
3. **Reproducibility**: Seeds for numpy, random, tensorflow must be set at notebook start.
4. **Model persistence**: Save `.h5` (or SavedModel) per fold per embedding — 10 models total.

## Reference Points

- **Phase 2 notebook**: `notebooks/02-svm-sentiment-models/svm_sentiment.ipynb` — patterns for Colab setup, Drive mount, CV loop, evaluation
- **ROADMAP.md**: Phase 3 requirements and successor phases
- **REQUIREMENTS.md**: Full SENT requirement specification
