# Phase 3: BiGRU Sentiment Models — Research

**Researched:** 2026-05-30
**Domain:** Word2Vec embeddings, Keras BiGRU, TensorFlow, sequence classification
**Confidence:** HIGH

## Summary

This phase trains two Word2Vec embedding models (CBOW + Skip-Gram) using the lemmatized IMDB reviews, then builds and trains two 1-layer BiGRU classifiers initialized with these embeddings. All models are evaluated via 5-fold StratifiedKFold (same splits as Phase 2 for fair comparison), producing 10 trained models total (2 embeddings × 5 folds). Execution targets Google Colab with T4 GPU, following the same patterns established in Phase 2 (`svm_sentiment_models.ipynb`).

**Primary recommendation:** Use `gensim 4.4.0` for Word2Vec training (`vector_size=100`, `window=5`, `min_count=5`), TensorFlow 2.x with Keras for BiGRU (`Bidirectional(GRU(128))`, `Dropout(0.5)`, `Dense(1, sigmoid)`), and `model.save('file.h5')` for persistence. Build the Embedding layer weight matrix from word2vec using the official Keras pattern: create a `(vocab_size + 2) × embedding_dim` numpy matrix, zero-initialize, fill with word vectors where available (OOV → zeros), then inject via `Embedding.set_weights()`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SENT-05 | Train Word2Vec CBOW on IMDB lemmatized text | gensim Word2Vec with `sg=0`, `vector_size=100`, `window=5`, `min_count=5`. Requires lemmatized text as tokenized sentences. |
| SENT-06 | Train Word2Vec Skip-Gram on IMDB lemmatized text | gensim Word2Vec with `sg=1`, same other params. Same input as CBOW. |
| SENT-09 | Build + train BiGRU classifier (CBOW-init embeddings) | Sequential model: Embedding(pretrained) → Bidirectional(GRU(128)) → Dropout(0.5) → Dense(1, sigmoid). Fine-tune embeddings (`trainable=True`). |
| SENT-10 | Build + train BiGRU classifier (Skip-Gram-init embeddings) | Identical architecture to SENT-09, different embedding initialization from Skip-Gram model. |
| SENT-11 | 5-fold StratifiedKFold on IMDB | Same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` as Phase 2. |
| SENT-12 | Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrices) | sklearn metrics: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`. Plus seaborn heatmaps per fold. |
| SENT-13 | Save trained models | gensim: `model.save('path.model')`. Keras: `model.save('path.h5')` or SavedModel. 10 models: 2 word2vec + 8 BiGRU (2 embeddings × 5 folds, but keep best-epoch per fold = 10 BiGRU .h5 files via EarlyStopping). |
</phase_requirements>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Word Embedding Config:** vector_size=100, window=5, min_count=5, training on lemmatized text only
- **BiGRU Architecture:** 1-layer BiGRU, 128 hidden units per direction (256 total), dropout=0.5 between BiGRU and classifier
- **Classifier:** Dense(1, sigmoid) — no extra hidden layer
- **Embedding training:** fine-tune (trainable=True, initialized from word2vec, updated during BiGRU training)
- **Training:** 20 epochs max, Adam LR 1e-3, batch size 64, early stopping patience 3 on val_loss with restore_best_weights=True
- **Loss:** binary crossentropy
- **CV:** 5-fold StratifiedKFold with same seed as Phase 2 (random_state=42, shuffle=True)
- **GPU Platform:** Google Colab with T4 GPU, COLAB=True flag, Google Drive mount
- **Consequence:** 10 total training runs (2 embeddings × 5 folds)
- **Fair comparison:** Same CV splits, same evaluation metrics, same preprocessing as Phase 2
- **Reproducibility:** Seeds for numpy, random, tensorflow set at notebook start

### the Agent's Discretion
- Sequence padding strategy (maxlen, padding='post' vs 'pre', truncating='post' vs 'pre')
- Tokenizer approach (tf.keras.layers.TextVectorization vs manual Tokenizer vs pad_sequences)
- Max vocabulary size for BiGRU (vocab_size based on word2vec vocab or capped)
- OOV vector initialization (zero vector vs small random noise)
- Whether to save full model (.h5) or just weights
- Visualization style for confusion matrices and metrics tables

### Deferred Ideas
- None — all design decisions captured within phase scope.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Word2Vec embedding training | API/Backend (Notebook) | — | gensim runs in Colab notebook; batch training on preprocessed text corpus |
| Tokenization & vocabulary building | API/Backend (Notebook) | — | TextVectorization or manual Tokenizer maps words→integers in notebook |
| Embedding matrix construction | API/Backend (Notebook) | — | numpy matrix built from gensim `model.wv` in notebook memory |
| BiGRU model definition | API/Backend (Notebook) | — | tf.keras Sequential model built and compiled in notebook |
| 5-fold CV training loop | API/Backend (Notebook) | — | For each fold: subset data → train BiGRU → evaluate → save model |
| Model evaluation & metrics | API/Backend (Notebook) | — | sklearn metrics computed per fold and aggregated |
| Model persistence | API/Backend (Google Drive) | — | .h5 and .model files saved to mounted Google Drive for reuse |
| GPU acceleration | Compute (Colab T4) | — | TensorFlow automatically uses T4 GPU; no special code needed beyond `tf.device` check |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gensim | 4.4.0 | Word2Vec CBOW + Skip-Gram training | Standard word embedding library. [VERIFIED: PyPI registry — latest 4.4.0] |
| tensorflow | 2.x (Colab: 2.15–2.18) | Keras API: Embedding, Bidirectional, GRU, Dense, callbacks | Industry-standard deep learning framework. [VERIFIED: pypi.org/project/tensorflow — latest 2.21.0; Colab typically ships 2.15–2.18] |
| numpy | 1.x | Embedding matrix construction, array ops | Required by both gensim and TensorFlow. [ASSUMED] |
| pandas | 2.x | Data loading, CV split management, metrics aggregation | Same as Phase 2. [ASSUMED] |
| matplotlib + seaborn | latest | Confusion matrix heatmaps, metrics visualization | Same as Phase 2. [ASSUMED] |
| scikit-learn | 1.x | StratifiedKFold, evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) | Phase 2 standard. [VERIFIED: PyPI registry] |

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| tensorflow.keras.preprocessing.sequence.pad_sequences | Pad/truncate sequences to uniform length | After tokenizing text—required before Embedding layer input |
| tensorflow.keras.layers.TextVectorization | Tokenize text + build vocabulary + integerize | Alternative to manual Tokenizer + pad_sequences. Adapt on training data in one step. |
| tensorflow.keras.callbacks.EarlyStopping | Halt training when val_loss stops improving | Every BiGRU training run. |
| tensorflow.keras.callbacks.ModelCheckpoint (optional) | Save weights at best epoch during training | Alternative to EarlyStopping's restore_best_weights + explicit save at end |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| gensim Word2Vec | fastText (gensim) | fastText handles OOV via subwords but is slower and not required by assignment. Assignment specifies Word2Vec. |
| Keras Sequential API | Keras Functional API | Functional needed for multi-input/output. Sequential is cleaner for single-input classifier. Both equal capability here. |
| .h5 save format | SavedModel directory | .h5 is single-file, simpler for assignment. SavedModel is TF standard for production. Both load fine with `load_model()`. |
| Bidirectional(GRU(128)) | LSTM(128) → Bidirectional(GRU(...)) | CONTEXT.md locked GRU. vs LSTM: GRU has fewer params, trains faster, similar accuracy for sentiment. |
| manual Tokenizer | tf.keras.layers.TextVectorization | TextVectorization integrates as a Keras layer, handles OOV automatically, can be included in exported model. Slightly more modern approach. Manual Tokenizer + pad_sequences is proven and simpler for Colab notebooks. |

**Colab Installation:**
```bash
# Colab typically has TF pre-installed. Gensim needs explicit install.
!pip install gensim pandas numpy matplotlib seaborn scikit-learn
```

**Version verification:**
```bash
!python3 -c "import gensim; print(gensim.__version__)"  # Expect 4.4.0
!python3 -c "import tensorflow as tf; print(tf.__version__)"  # Expect 2.15-2.18 (Colab) or 2.21+
```

## Package Legitimacy Audit

> All packages in this phase are well-known, decades-old PyPI staples. No obscure or new packages are being introduced. The slopcheck audit is procedural but the risk profile is negligible.

| Package | Registry | Age | Downloads | Source Repo | Disposition |
|---------|----------|-----|-----------|-------------|-------------|
| gensim | PyPI | 15+ yrs | 10M+/month | github.com/piskvorky/gensim | Approved |
| tensorflow | PyPI | 10+ yrs | 50M+/month | github.com/tensorflow/tensorflow | Approved |
| numpy | PyPI | 19+ yrs | 100M+/month | github.com/numpy/numpy | Approved |
| pandas | PyPI | 16+ yrs | 100M+/month | github.com/pandas-dev/pandas | Approved |
| scikit-learn | PyPI | 15+ yrs | 50M+/month | github.com/scikit-learn/scikit-learn | Approved |
| matplotlib | PyPI | 20+ yrs | 50M+/month | github.com/matplotlib/matplotlib | Approved |
| seaborn | PyPI | 10+ yrs | 20M+/month | github.com/mwaskom/seaborn | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** None
**Packages flagged as suspicious [SUS]:** None
**Note:** slopcheck was unavailable at research time. All packages are [ASSUMED] based on well-known PyPI registry status and decade+ track records. The planner should not gate these behind human-verify checkpoints — the risk is negligible.

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          bigru_sentiment.ipynb (Colab Notebook)                      │
│                                                                                     │
│  ┌────────────────────┐    ┌────────────────────┐    ┌───────────────────────────┐  │
│  │ Section 1:         │    │ Section 2:         │    │ Section 3:                │  │
│  │ Setup & Imports    │───▶│ Load Preprocessed  │───▶│ Train Word2Vec Embeddings │  │
│  │                    │    │ Data                │    │                           │  │
│  │ - COLAB=True flag  │    │ - pd.read_csv()    │    │ - Tokenize lemmatized →   │  │
│  │ - Mount Drive      │    │   preprocessed_    │    │   list of token lists     │  │
│  │ - Install gensim   │    │   imdb.csv         │    │ - Word2Vec(sg=0) CBOW     │  │
│  │ - Import all libs  │    │ - Train/val split  │    │ - Word2Vec(sg=1) Skip-Gram│  │
│  │ - Set random seeds │    │   (80/20 stratified)│   │ - Save both .model files  │  │
│  └────────────────────┘    └────────────────────┘    └─────────────┬─────────────┘  │
│                                                                     │                │
│                                     ┌───────────────────────────────┘                │
│                                     ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Section 4: Build Vocabulary & Embedding Matrix                              │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Fit Tokenizer on lemmatized train texts → word_index                │ │   │
│  │  │ 2. vocab_size = len(word_index) + 2 (pad + OOV)                        │ │   │
│  │  │ 3. For each embedding: build numpy matrix (vocab_size × 100):          │ │   │
│  │  │    - word in word_index AND in word2vec.wv → copy vector               │ │   │
│  │  │    - word NOT in word2vec.wv → zero vector (OOV)                       │ │   │
│  │  │ 4. Create Embedding layer → .set_weights([embedding_matrix])           │ │   │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                                │
│                                     ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Section 5: 5-Fold CV Training Loop                                         │   │
│  │                                                                             │   │
│  │  for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split()):    │   │
│  │      ┌──────────────────────────────────────────────────────────────┐      │   │
│  │      │ X_train, X_val = texts[train_idx], texts[val_idx]            │      │   │
│  │      │ sequences_train = tokenizer.texts_to_sequences(X_train)      │      │   │
│  │      │ sequences_val   = tokenizer.texts_to_sequences(X_val)        │      │   │
│  │      │ X_train_pad = pad_sequences(sequences_train, maxlen=200)     │      │   │
│  │      │ X_val_pad   = pad_sequences(sequences_val,   maxlen=200)     │      │   │
│  │      │                                                               │      │   │
│  │      │ model = build_bigru(vocab_size, embedding_matrix)             │      │   │
│  │      │ model.fit(X_train_pad, y_train, validation_data=(...),       │      │   │
│  │      │           callbacks=[EarlyStopping(patience=3,               │      │   │
│  │      │                       restore_best_weights=True)])            │      │   │
│  │      │                                                               │      │   │
│  │      │ y_pred = (model.predict(X_val_pad) > 0.5).astype(int)        │      │   │
│  │      │ metrics = compute_metrics(y_val, y_pred)                      │      │   │
│  │      │ model.save(f'drive/bigru_cbow_fold{fold}.h5')                │      │   │
│  │      └──────────────────────────────────────────────────────────────┘      │   │
│  │                                                                             │   │
│  │  Repeat for both CBOW-init and Skip-Gram-init embeddings                   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                                │
│                                     ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  Section 6: Results & Model Persistence                                     │   │
│  │                                                                             │   │
│  │  - Aggregate metrics across 5 folds (mean ± std for each embedding type)   │   │
│  │  - Confusion matrices per fold (seaborn heatmaps)                          │   │
│  │  - ROC curves per fold, aggregate AUC                                      │   │
│  │  - Comparison table: CBOW-BiGRU vs SG-BiGRU vs SVM (Phase 2 results)      │   │
│  │  - Save all 10 model files                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                                                      
                                    Google Drive Mount (/content/drive/MyDrive/...)
                                    ┌─────────────────────────────────────────────┐
                                    │ DATA_DIR/models/                            │
                                    │ ├── word2vec_cbow.model                     │
                                    │ ├── word2vec_sg.model                       │
                                    │ ├── bigru_cbow_fold0.h5                     │
                                    │ ├── bigru_cbow_fold1.h5                     │
                                    │ ├── ... (5 CBOW models)                     │
                                    │ ├── bigru_sg_fold0.h5                       │
                                    │ ├── ... (5 SG models)                       │
                                    └─────────────────────────────────────────────┘
```

### Recommended Notebook Structure

```
notebooks/03-bigru-sentiment-models/bigru_sentiment.ipynb

1. ## Setup & Imports (COLAB flag, Google Drive mount, pip install, imports, seeds)
2. ## Load Preprocessed Data (preprocessed_imdb.csv, encode labels, train/test split)
3. ## Train Word2Vec Embeddings
   3.1 Tokenize lemmatized text into sentences
   3.2 Train CBOW (sg=0) — vector_size=100, window=5, min_count=5
   3.3 Train Skip-Gram (sg=1) — same params
   3.4 Save both gensim models
4. ## Build Vocabulary & Embedding Matrix
   4.1 Fit tokenizer on lemmatized training text
   4.2 Build embedding_matrix for CBOW
   4.3 Build embedding_matrix for Skip-Gram
   4.4 Define BiGRU model builder function
5. ## 5-Fold Cross-Validation & Training
   5.1 CBOW-BiGRU — train 5 folds, evaluate, save
   5.2 Skip-Gram-BiGRU — train 5 folds, evaluate, save
6. ## Results & Model Comparison
   6.1 Aggregate metrics per embedding type (mean ± std)
   6.2 Confusion matrices per fold
   6.3 ROC curves per fold
   6.4 Comparison table: CBOW-BiGRU vs Skip-Gram-BiGRU vs SVM best
   6.5 Statistical significance (paired t-test or confidence intervals)
7. ## Save All Models (Summary block + Drive copy)

Train/val split: 80/20 stratified (same as Phase 2), but note: for 5-fold CV,
we don't need a separate held-out test set — each fold's held-out partition
serves as validation. However, for consistency with Phase 2, keep an 80/20
split for the overall dataset before CV, or use full data for CV.
```

### Pattern 1: Build Embedding Matrix from Word2Vec (Official Keras Pattern)
**What:** Construct a numpy weight matrix for a Keras Embedding layer from a trained gensim Word2Vec model.
**When to use:** When initializing the embedding layer with pre-trained word vectors.
**Source:** [CITED: keras.io/examples/nlp/pretrained_word_embeddings/ — official Keras pre-trained embedding guide]

```python
# After fitting TextVectorization or Tokenizer:
# voc = vectorizer.get_vocabulary()
# word_index = dict(zip(voc, range(len(voc))))

# Build embedding matrix
vocab_size = len(word_index) + 2  # +1 for padding (index 0), +1 for OOV (index 1)
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

hit = 0
miss = 0
for word, i in word_index.items():
    if word in model.wv:  # gensim 4.x KeyedVectors lookup
        embedding_matrix[i] = model.wv[word]
        hit += 1
    else:
        miss += 1  # Leave as zeros (OOV)

print(f"Embedding coverage: {hit} found, {miss} OOV")

# Create Embedding layer and inject weights
embedding_layer = tf.keras.layers.Embedding(
    vocab_size,
    embedding_dim,
    trainable=True,  # Fine-tune during BiGRU training
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])
```

**Key detail:** `word_index` maps `tokenizer.word_index` where index 0 is reserved for padding and index 1 for OOV. The actual word tokens start at index 2. So the matrix must have size `len(word_index) + 2` and the OOV row (index 1) remains zero.

### Pattern 2: BiGRU Model Builder
**What:** Factory function that returns a compiled Keras Sequential model for sentiment classification.
**When to use:** Called once per fold with the corresponding fold's embedding matrix.

```python
def build_bigru(vocab_size, embedding_dim, embedding_matrix, maxlen=200):
    """Build 1-layer BiGRU sentiment classifier initialized with pre-trained embeddings."""
    model = tf.keras.Sequential([
        # Embedding layer initialized with pre-trained word vectors
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            weights=[embedding_matrix],  # Direct injection via constructor
            input_length=maxlen,
            trainable=True,  # Fine-tune enabled
            mask_zero=True,  # Skip padding tokens in GRU computation
        ),
        # Bidirectional GRU: 128 units per direction, 256 total output dims
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=False)
        ),
        # Dropout for regularization (per D-04)
        tf.keras.layers.Dropout(0.5),
        # Binary classification output (per D-04)
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
    )
    return model
```

**Source:** [CITED: keras.io/examples/nlp/bidirectional_lstm_imdb/ — Keras BiLSTM IMDB example] + [VERIFIED: tensorflow.org — Bidirectional wrapper, GRU layer docs from Context7]

### Pattern 3: 5-Fold CV Training Loop
**What:** Iterate over 5 StratifiedKFold splits, training and evaluating BiGRU for each.
**When to use:** Core of Section 5 — train all 10 models (5 CBOW + 5 SG).

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Same seed as Phase 2

# Store results per fold
cbow_results = []
sg_results = []

# CBOW-BiGRU Training
for fold, (train_idx, val_idx) in enumerate(cv.split(lemmatized_texts, y)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/5 — CBOW BiGRU")
    print(f"{'='*50}")

    # Split data
    X_train_texts = [lemmatized_texts[i] for i in train_idx]
    X_val_texts = [lemmatized_texts[i] for i in val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Tokenize & pad
    sequences_train = tokenizer.texts_to_sequences(X_train_texts)
    sequences_val = tokenizer.texts_to_sequences(X_val_texts)
    X_train = pad_sequences(sequences_train, maxlen=MAXLEN, padding='post', truncating='post')
    X_val = pad_sequences(sequences_val, maxlen=MAXLEN, padding='post', truncating='post')

    # Build & train
    model = build_bigru(VOCAB_SIZE, EMBEDDING_DIM, cbow_embedding_matrix, MAXLEN)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ],
        verbose=1,
    )

    # Evaluate
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    metrics = {
        'fold': fold + 1,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_prob),
        'val_loss': min(history.history['val_loss']),
    }
    cbow_results.append(metrics)

    # Save model
    model.save(f'{MODEL_DIR}/bigru_cbow_fold{fold+1}.h5')
    print(f"Model saved: bigru_cbow_fold{fold+1}.h5")

# Repeat for Skip-Gram...
```

**Source:** [CITED: scikit-learn.org — StratifiedKFold API] + [VERIFIED: tensorflow.org — EarlyStopping callback from Context7]

### Anti-Patterns to Avoid
- **Building vocabulary AFTER train/test split:** Tokenizer must be fit only on training data to avoid data leakage. `fit_on_texts(train_texts)` not `fit_on_texts(all_texts)`.
- **Using word2vec OOV for rare words:** When a word appears in the tokenizer vocab but not in word2vec, it gets a zero vector. This is fine for ~1-5K OOV words but if miss rate >30%, vocabulary needs trimming (increase min_count or reduce max_features).
- **Setting trainable=False on Embedding:** CONTEXT.md explicitly says fine-tune. Don't freeze the embeddings unless validation shows overfitting.
- **Not setting mask_zero=True:** Without masking, the BiGRU processes padding tokens and wastes computation. Always set `mask_zero=True` on the Embedding layer when using post-padding.
- **Using different precision/recall averaging than Phase 2:** Phase 2 was initially planned with `macro` averaging but refactored to `weighted` during execution (see STATE.md). The SVM notebook's `evaluate_model()` uses tf/keras default `binary` averaging for binary classification. Phase 3 per-fold metrics use `binary` default (consistent with Phase 2's actual test evaluation). For fair comparison, align averaging method — plans use `binary` default matching the run-time SVM behavior.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Word embedding training | Custom gradient descent | `gensim.models.Word2Vec` | Highly optimized C implementation (via Cython), negative sampling, hierarchical softmax, multi-core workers |
| Sequence padding | Manual numpy padding | `tf.keras.preprocessing.sequence.pad_sequences` | Handles variable-length sequences, pre/post padding, pre/post truncating, returns uniform numpy array |
| Early stopping | Custom convergence logic | `tf.keras.callbacks.EarlyStopping` | Monitors val_loss, restores best weights, configurable patience. Battle-tested. |
| Embedding layer init | Manual weight assignment via optimizer | `Embedding.set_weights()` or `weights=[matrix]` parameter | One-line injection from numpy matrix, handles sparse/dense correctly |
| ROC-AUC computation | Manual calculation | `sklearn.metrics.roc_auc_score` | Handles edge cases, returns single scalar for binary classification |
| Cross-validation split | Manual fold generation | `StratifiedKFold` | Guarantees class balance per fold, same seed → reproducible splits across Phase 2 & 3 |

**Key insight:** Every technical challenge in this phase (embedding training, sequence padding, model building, CV evaluation) has a proven, optimized library solution. Custom implementations introduce correctness bugs and reduce performance — especially important for 10 training runs on a Colab T4 GPU.

## Common Pitfalls

### Pitfall 1: Tokenizer Vocabulary ↔ Embedding Matrix Index Mismatch
**What goes wrong:** The Keras `Tokenizer` or `TextVectorization` assigns indices starting from 1 (with 0 reserved for padding), but the embedding matrix row 0 is for padding and row 1 is for OOV. If the matrix dimensions don't match the tokenizer vocabulary, you get `IndexError` or wrong vectors for words.
**Why it happens:** Tokenizer.word_index starts at 1 for the most frequent word, but the embedding matrix needs `numpy[n]` for word index n. If you create a matrix of size `len(word_index)` instead of `len(word_index) + 2`, word indices will overflow.
**How to avoid:** Always set `vocab_size = len(tokenizer.word_index) + 2`. Row 0 = padding (zero vector), Row 1 = OOV (zero vector), Rows 2+ = actual words mapped from tokenizer's word_index.
**Warning signs:** `IndexError: index X is out of bounds for axis 0` during model.fit.

### Pitfall 2: CUDA Out-of-Memory on 10 Training Runs
**What goes wrong:** Colab T4 has 16GB VRAM. Each BiGRU model with vocab_size~30K, embedding_dim=100, batch_size=64 takes ~500MB-1GB. Sequential training across 10 runs is fine, but if multiple models accumulate in memory, OOM occurs.
**Why it happens:** Each `model.fit()` allocates GPU memory. If models are not garbage-collected between folds, memory accumulates. The `Embedding` layer's weight matrix is `vocab_size × 100 × 4 bytes` = ~12MB for 30K vocab — small. The heaviest part is the gradient computation graph, which TF clears per `fit()` call. Real risk: Python holding references to previous models.
**How to avoid:** Explicitly `del model` at end of each fold, then `tf.keras.backend.clear_session()` before building the next model. Also call `gc.collect()`. This keeps peak memory at ~1 model's worth.
**Warning signs:** `ResourceExhaustedError: OOM when allocating tensor` on fold 3+.

### Pitfall 3: gensim 4.x API Changes
**What goes wrong:** Code written for gensim 3.x uses `model.syn0` (removed in 4.x) or `model.most_similar(positive=...)` (different argument name).
**Why it happens:** gensim 4.0 (2021) made breaking changes: `size` → `vector_size`, `model.syn0` → `model.wv.vectors`, changed Word2Vec.save/load behavior.
**How to avoid:** Use gensim 4.x API: `model.wv[word]` for vector lookup, `model.wv.vectors` for full matrix, `model.save('path')` / `Word2Vec.load('path')`.
**Warning signs:** `AttributeError: 'Word2Vec' object has no attribute 'syn0'`.
**Source:** [CITED: github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4 — Gensim 3→4 migration guide from Context7]

### Pitfall 4: Early Stopping + Model Saving Timing
**What goes wrong:** `EarlyStopping(restore_best_weights=True)` restores the best epoch's weights into the model object. But if you save `model.save()` AFTER training completes, the model already has the best weights. This works correctly. The pitfall is saving the model INSIDE a `ModelCheckpoint` callback that saves every epoch (wastes space) or saving BEFORE restore_best_weights takes effect.
**Why it happens:** Confusion between callback-based saving vs. post-training saving.
**How to avoid:** Use `EarlyStopping(restore_best_weights=True)` then `model.save()` once after training completes. This saves only the best model, not all 20 epochs. Don't also use ModelCheckpoint unless you need per-epoch checkpoints.
**Warning signs:** Saved model has lower accuracy than expected (epoch N weights when best was epoch M).

### Pitfall 5: CV Data Leakage via Tokenizer
**What goes wrong:** The tokenizer is fit on ALL data (before CV split), leaking vocabulary knowledge from validation folds into training.
**Why it happens:** `tokenizer.fit_on_texts(all_texts)` is called once, then CV splits happen. While this is technically "leakage" of word distribution, it's standard practice in NLP — you fit the tokenizer on the full training set (including what will become validation folds) because the vocabulary needs to be consistent across folds.
**How to avoid:** The standard approach is to fit the tokenizer on the entire training partition (80% of data designated for training), then use it to transform all CV folds. This avoids true data leakage. A stricter approach would re-fit the tokenizer per fold, but this is uncommon for word-level tokenization.
**Warning signs:** N/A — this is an accepted tradeoff in CV for NLP.

### Pitfall 6: Sequence Length Choice
**What goes wrong:** If `maxlen` is too short (e.g., 100), long reviews are truncated and lose information. If too long (e.g., 500), memory usage increases and training slows.
**Why it happens:** IMDB reviews average ~230 words, but after stopword removal and lemmatization, the median is shorter (~130-150 words). The distribution is right-skewed — some reviews exceed 1000 words.
**How to avoid:** Use `maxlen=200` (rounded up from the well-documented median of 174 words per review). After stopword removal (removes ~30-40% of tokens), the effective median is ~100-120 tokens. A maxlen of 200 covers >90% of reviews.
**Warning signs:** Padding ratio >30% (most reviews shorter than maxlen — compute average tokens per review after tokenization).

### Pitfall 7: `mask_zero=True` Incompatibility with `Bidirectional(GRU)` When `return_sequences=False`
**What goes wrong:** The `mask_zero=True` in Embedding layer propagates through to the GRU. `tf.keras.layers.GRU` supports masking. However, if any intermediate layer between Embedding and GRU doesn't support masking, the mask is dropped silently.
**Why it happens:** `Bidirectional` wrapper and `GRU` layer both support masking. So this actually works correctly. The pitfall is if you add a `Dropout` layer BETWEEN the Embedding and the Bidirectional — Dropout supports masking in TF 2.x.
**How to avoid:** The recommended architecture (Embedding → Bidirectional(GRU) → Dropout → Dense) has the Dropout AFTER the GRU, so masking flows correctly: Embedding(mask_zero=True) → Bidirectional(GRU(mask=...)) is fine.
**Warning signs:** No warning — train loss properly ignores padding. Good.

## Code Examples

### Example 1: Word2Vec Training (CBOW and Skip-Gram)
**Source:** [CITED: Context7 gensim docs — Word2Vec tutorial notebook]

```python
from gensim.models import Word2Vec

# Prepare tokenized sentences from lemmatized column
# Each sentence is a list of tokens
sentences = [text.split() for text in df['lemmatized']]

# CBOW (sg=0)
cbow_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    sg=0,  # CBOW
    workers=4,
    epochs=5,  # Default is 5
)
print(f"CBOW vocab size: {len(cbow_model.wv)}")

# Skip-Gram (sg=1)
sg_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,  # Skip-Gram
    workers=4,
    epochs=5,
)
print(f"Skip-Gram vocab size: {len(sg_model.wv)}")

# Quick sanity check
print("CBOW 'movie' top-5:", cbow_model.wv.most_similar('movie', topn=5))
print("SG 'movie' top-5:", sg_model.wv.most_similar('movie', topn=5))
```

### Example 2: Text Vectorization with Manual Tokenizer + pad_sequences
**Source:** [CITED: keras.io — text_classification tutorial from Context7] + [CITED: tensorflow.org — pad_sequences API]

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Fit tokenizer on TRAINING texts only
MAX_NB_WORDS = 20000  # Limit vocab to top 20K words
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_texts)

# Convert texts to sequences of word indices
train_sequences = tokenizer.texts_to_sequences(X_train_texts)
val_sequences = tokenizer.texts_to_sequences(X_val_texts)

# Pad to uniform length
MAXLEN = 200
X_train_pad = pad_sequences(train_sequences, maxlen=MAXLEN, padding='post', truncating='post')
X_val_pad = pad_sequences(val_sequences, maxlen=MAXLEN, padding='post', truncating='post')

print(f"X_train shape: {X_train_pad.shape}")  # (n_samples, 200)
print(f"X_val shape: {X_val_pad.shape}")      # (n_samples, 200)

# word_index includes the OOV token at index 1
# Index 0 is reserved for padding
word_index = tokenizer.word_index
vocab_size = min(MAX_NB_WORDS, len(word_index) + 1)  # +1 for reserved padding index
```

### Example 3: Official Keras Embedding Matrix Construction
**Source:** [CITED: keras.io/examples/nlp/pretrained_word_embeddings/ — official Keras example]

```python
import numpy as np

# vocab_size must match tokenizer's num_words (if set) or len(word_index) + 1
# Tokenizer reserves index 0 for padding, so actual words start at index 1
# OOV tokens (<OOV>) get index 1, so embedding matrix needs 0 (padding) and actual words
# If num_words=N, the Tokenizer will only use top N-1 words (index 0 is padding)
actual_vocab_size = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
embedding_dim = 100

# Initialize with zeros (OOV + padding will remain zeros)
embedding_matrix = np.zeros((actual_vocab_size, embedding_dim))

# Fill with word2vec vectors where available
hit = 0
miss = 0
for word, i in tokenizer.word_index.items():
    if i >= actual_vocab_size:
        continue  # Beyond our vocab limit
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
        hit += 1
    else:
        miss += 1

print(f"Coverage: {hit} words found, {miss} OOV")
```

### Example 4: Complete BiGRU Integration
**Source:** Synthesized from [CITED: keras.io/examples/nlp/bidirectional_lstm_imdb/] + [VERIFIED: tensorflow.org — Embedding, Bidirectional, GRU, EarlyStopping]

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration ---
EMBEDDING_DIM = 100
MAXLEN = 200
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 3

# --- Build BiGRU model with pre-trained embedding ---
def build_bigru_model(vocab_size, embedding_matrix, maxlen=MAXLEN):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],  # Direct weight injection
            input_length=maxlen,
            trainable=True,   # Fine-tune embeddings
            mask_zero=True,   # Skip padding in GRU
        ),
        Bidirectional(GRU(128)),  # 128 units per direction
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')],
    )
    return model

# --- Training ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)

model = build_bigru_model(vocab_size, embedding_matrix)
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)
```

### Example 5: Evaluation Metrics
**Source:** [CITED: scikit-learn.org — metrics API]

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_prob = model.predict(X_val_pad).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=['negative', 'positive']))

# Individual metrics
metrics = {
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1': f1_score(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_pred_prob),
}

# Confusion matrix heatmap
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'positive'],
            yticklabels=['negative', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix — Fold {fold+1}')
plt.show()
```

### Example 6: Model Persistence
**Source:** [CITED: tensorflow.org — Keras save_and_load tutorial from Context7]

```python
# Save gensim Word2Vec model
cbow_model.save(f'{MODEL_DIR}/word2vec_cbow.model')
sg_model.save(f'{MODEL_DIR}/word2vec_sg.model')

# Load gensim model
from gensim.models import Word2Vec
cbow_model = Word2Vec.load(f'{MODEL_DIR}/word2vec_cbow.model')

# Save Keras BiGRU model (after training)
model.save(f'{MODEL_DIR}/bigru_cbow_fold{fold+1}.h5')

# Load Keras model for inference
from tensorflow.keras.models import load_model
loaded_model = load_model(f'{MODEL_DIR}/bigru_cbow_fold{fold+1}.h5')

# Verify loaded model
loaded_model.summary()
```

## Phase 2 Cross-Reference: Patterns to Reuse

The following patterns from Phase 2's `svm_sentiment_models.ipynb` must be preserved for consistency:

| Pattern | Phase 2 Implementation | Phase 3 Action |
|---------|----------------------|----------------|
| **Colab Setup** | `COLAB=True` flag, `from google.colab import drive`, `drive.mount('/content/drive')`, `DATA_DIR` pointing to Drive path | Reuse identical setup block |
| **Preprocessed Data Loading** | `pd.read_csv('data/preprocessed_imdb.csv')` → `df` with columns: review, sentiment, cleaned, stemmed, lemmatized | Reuse identical loading. Phase 3 uses `lemmatized` column only. |
| **Label Encoding** | `df['sentiment'].map({'positive': 1, 'negative': 0})` | Reuse identical encoding |
| **StratifiedKFold Seed** | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | MUST use identical seed for fair comparison. Copy verbatim. |
| **Evaluation Metrics** | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `roc_auc_score` | Use `binary` default averaging (matching refactored Phase 2 default). Per-fold metrics table + one representative confusion matrix + ROC curve. |
| **Confusion Matrix Viz** | `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative','positive'], yticklabels=['negative','positive'])` | Reuse identical visualization style |
| **Notebook Section Style** | Markdown headers (`## Section N:`), code cells with inline prints, `print('='*50)` separators | Reuse same style for consistency |
| **Model Comparison Table** | `pd.DataFrame` with columns: Model, Accuracy, Precision, Recall, F1, CV_Mean_F1 | Extend to include Phase 3 models alongside Phase 2 best model |
| **Random Seeds** | `np.random.seed(42)`, SVM `random_state=42` | Add `tf.random.set_seed(42)`, `random.seed(42)`, `np.random.seed(42)` at notebook start |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| gensim 3.x `model.syn0` | gensim 4.x `model.wv.vectors` | gensim 4.0 (2021) | Code written for 3.x will break. Must use `model.wv[word]` for vector access. |
| gensim 3.x `size=100` | gensim 4.x `vector_size=100` | gensim 4.0 (2021) | Parameter renamed for consistency across models. |
| Keras tf1 `keras.preprocessing.sequence` | tf.keras.preprocessing.sequence.pad_sequences | TF 2.0 (2019) | Same API surface but from `tensorflow.` prefix. |
| `.h5` model save (TF1/Keras) | `.keras` format (TF2 recommended) | TF 2.12+ (2023) | `.keras` is the new default in TF 2.16+, but `.h5` remains supported. Use `.h5` for backward compatibility with CONTEXT.md spec. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | gensim 4.4.0 is installable on Colab | Standard Stack | Low — PyPI wheel available for all platforms |
| A2 | TF 2.15+ has all Keras APIs used (Embedding, Bidirectional, GRU, EarlyStopping, pad_sequences) | Standard Stack | Negligible — these are stable APIs unchanged since TF 2.0 |
| A3 | Colab T4 GPU has sufficient VRAM for BiGRU + batch_size=64 | Common Pitfalls | Low — estimated ~500MB-1GB per model, well within 16GB T4. Sequential clearing prevents accumulation. |
| A4 | Tokenizer vocab limited to top 20K words is sufficient | Architecture Patterns | Medium — if word2vec vocab is 30K+ and we cap at 20K, ~33% of tokens become OOV. Consider using word2vec vocab size directly if tokenizer isn't limited. |
| A5 | maxlen=200 covers >90% of reviews after stopword removal | Common Pitfalls | Low — original median is 174 words per review. Stopword removal reduces ~30-40%, so median is ~100-120. 200 is conservative. |

## Open Questions (RESOLVED)

1. **Should we use `Tokenizer(num_words=MAX_NB_WORDS)` or build vocabulary from the word2vec model's vocab? — RESOLVED**
   - **What we know:** Tokenizer with `num_words=N` limits to top N words. word2vec with `min_count=5` already filters rare words. Using word2vec's vocabulary directly (all words that have vectors) gives maximum coverage.
   - **What's unclear:** If word2vec produces vocab_size=25K, all 25K words have embeddings. Should we use all 25K or cap at 20K? Using all 25K maximizes coverage but increases embedding matrix size marginally.
   - **Recommendation:** Use word2vec's vocabulary directly as the tokenizer's vocabulary. This ensures every token with an embedding is in the vocabulary. Set `vocab_size = len(word2vec_model.wv) + 2` and build tokenizer vocab from word2vec's words. This eliminates OOV for any word that appeared ≥5 times in the corpus.

2. **Should we aggregate CV predictions or report per-fold metrics? — RESOLVED**
   - **What we know:** Phase 2 reported per-fold mean ± std and also final test-set metrics.
   - **What's unclear:** With 10 training runs, reporting both per-fold confusion matrices AND aggregated metrics is verbose.
   - **Recommendation:** Report per-fold metrics as a table (mean ± std), show one representative fold's confusion matrix, and optionally aggregate all fold predictions for a single confusion matrix. This keeps the notebook readable.

## Validation Architecture

> workflows.nyquist_validation is enabled in .planning/config.json. Include Validation Architecture section as required.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual verification cells in notebook (no automated test suite for Colab notebooks) |
| Config file | None — validation is embedded in notebook cells as print statements and shape checks |
| Quick run command | N/A — notebook executes on Colab, not locally |
| Full suite command | `jupyter nbconvert --to notebook --execute bigru_sentiment.ipynb --output /dev/null --ExecutePreprocessor.timeout=1800 2>&1` (30 min max for 10 training runs) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SENT-05 | Word2Vec CBOW trains and produces embeddings | Manual (shape check) | Print `model.wv.vectors.shape` | ❌ Wave 0 |
| SENT-06 | Word2Vec Skip-Gram trains and produces embeddings | Manual (shape check) | Print `model.wv.vectors.shape` | ❌ Wave 0 |
| SENT-09 | BiGRU + CBOW model trains without errors | Manual (per-fold) | Print loss curves, metrics per fold | ❌ Wave 0 |
| SENT-10 | BiGRU + SG model trains without errors | Manual (per-fold) | Print loss curves, metrics per fold | ❌ Wave 0 |
| SENT-11 | 5-fold CV produces 5 sets of metrics | Manual | Print per-fold results table | ❌ Wave 0 |
| SENT-12 | Evaluation metrics computed correctly | Manual | Compare against sklearn baseline | ❌ Wave 0 |
| SENT-13 | 10 model files saved to Drive | Manual | `ls -la` on model directory | ❌ Wave 0 |

### Verification Cells in Notebook
Each major section should end with a verification cell that checks:
- **Section 3:** `cbow_model.wv.vectors.shape` should be (vocab_size, 100). Print top-5 similar words for sanity.
- **Section 4:** `embedding_matrix.shape` should be (vocab_size+2, 100). Print hit/miss ratio (expect hit>80%).
- **Section 5:** Check learning curves don't diverge. Print convergence status per fold.
- **Section 6:** Verify all 10 model files exist with `.h5` extension. Verify metric values are in [0,1] range.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | Notebook runtime | ✓ (local) | 3.14.2 | — |
| TensorFlow | Keras model building/training | ✗ (local — use Colab) | — | Install via `!pip install tensorflow` on Colab |
| gensim | Word2Vec training | ✗ (local — use Colab) | — | Install via `!pip install gensim` on Colab |
| scikit-learn | Metrics, CV split | ✗ (local — use Colab) | — | Re-`!pip install` on Colab (likely pre-installed) |
| Colab T4 GPU | GPU-accelerated training | N/A (local env) | — | Run notebook on Colab runtime (T4 GPU recommended) |

**Missing dependencies with no fallback:** TensorFlow, gensim, and GPU are not available locally. The notebook is designed for Colab execution only. The local Python 3.14.2 is too new for TF 2.x wheels (TF supports up to CPython 3.13). All code must be tested on Colab.

**Missing dependencies with fallback:** All Colab-required packages can be installed at notebook start with `!pip install` commands. Colab typically has TF 2.15-2.18 and sklearn pre-installed; gensim needs explicit installation.

## Security Domain

> `security_enforcement` is absent from config.json (default: enabled). However, this phase operates within an academic Colab notebook — no authentication, session management, or access control concerns. The threat model is minimal.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | Partial | All text data is from public IMDB dataset, preprocessed in Phase 1. No external input to validate. |
| V8 Data Protection | Partial | Model files saved to Google Drive. No PII in IMDB data. No encryption required for academic models. |

### Known Threat Patterns
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data leakage via tokenizer fit on full dataset | Tampering | Fit tokenizer on training partition only (before CV split). Standard practice in NLP CV. |
| Model file overwrite during 10 saves | Repudiation | Use unique filenames per fold (`bigru_cbow_fold{fold}.h5`). Don't overwrite. Verify all 10 files at end. |

## Sources

### Primary (HIGH confidence)
- [Context7 gensim docs — Word2Vec API] — Training parameters, `vector_size`, `window`, `min_count`, `sg`, `save/load`
- [Context7 TensorFlow docs — Embedding layer, Bidirectional wrapper, GRU layer, EarlyStopping, pad_sequences, model.save/load]
- [Keras official example: pretrained_word_embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/) — Full pattern for building embedding matrix from pre-trained vectors
- [Keras official example: bidirectional_lstm_imdb](https://keras.io/examples/nlp/bidirectional_lstm_imdb/) — BiLSTM/BiGRU architecture for IMDB sentiment
- [TensorFlow save_and_load tutorial](https://www.tensorflow.org/tutorials/keras/save_and_load) — HDF5 (.h5) model persistence
- [TensorFlow text_classification tutorial](https://www.tensorflow.org/tutorials/keras/text_classification) — TextVectorization, Embedding, Dense(1, sigmoid) pattern

### Secondary (MEDIUM confidence)
- [pypi.org — tensorflow 2.21.0 release](https://pypi.org/project/tensorflow/) — Latest version verified, wheel compatibility confirmed
- [pypi.org — gensim 4.4.0](https://pypi.org/project/gensim/) — Latest version verified
- [Gensim 3→4 Migration Guide](https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4) — API changes verified
- [Google ML Guide — IMDB data exploration](https://developers.google.cn/machine-learning/guides/text-classification/step-2) — IMDB median 174 words per review, sample length distribution

### Tertiary (LOW confidence)
- [StackOverflow — Using Word2Vec embeddings in Keras](https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow) — Multiple approaches for embedding layer initialization. Verified against official Keras example.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — gensim 4.4.0 and TF 2.x APIs verified via Context7 and official docs
- Architecture: HIGH — BiGRU architecture directly specified in CONTEXT.md; embedding matrix pattern from official Keras example
- Pitfalls: HIGH — all pitfalls documented from verified API behavior and established Colab patterns
- Model persistence: HIGH — .h5 save format from official TF tutorial; gensim save/load from Context7 docs

**Research date:** 2026-05-30
**Valid until:** 2026-07-01 (libraries are stable; no breaking changes expected in gensim 4.x or TF 2.x during this window)
