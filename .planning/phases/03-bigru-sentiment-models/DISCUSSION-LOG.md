# Discussion Log — Phase 03: BiGRU Sentiment Models

## Overview

Discussions held on 2026-05-30 covering 5 gray areas for the BiGRU sentiment model phase. All decisions finalized; no remaining open questions.

## Area 1: Word Embedding Configuration

**Question**: What vector size for Word2Vec?
- Options: 100, 200, 300
- **Selected**: 100
- **Rationale**: Standard dim for IMDB-sized corpus; keeps BiGRU params manageable; matches typical academic baseline

**Question**: What window size?
- Options: 5, 10
- **Selected**: 5
- **Rationale**: Sentiment signals are local; 5 is standard sentiment default

**Question**: What min_count?
- Options: 5, 2, 10
- **Selected**: 5
- **Rationale**: Filters noisy rare tokens while keeping meaningful low-frequency vocabulary

**Question**: Which preprocessing output to use?
- Options: Lemmatized, Both, Only the best one
- **Selected**: Lemmatized only
- **Rationale**: Single clean pipeline; avoids duplicating work; lemmatization captures semantic base forms best

## Area 2: BiGRU Architecture

**Question**: How many BiGRU layers?
- Options: 1 layer, 2 layers
- **Selected**: 1 layer
- **Rationale**: Sentiment classification is sentence-level — deeper layers add params without proportional gain; reduces overfit risk on 50K

**Question**: Hidden units per direction?
- Options: 128, 64, 256
- **Selected**: 128
- **Rationale**: 256 total (bidirectional) is well-proven for sentiment; 64 underfits, 256 per direction overfits

**Question**: Dropout rate?
- Options: 0.5, 0.3, Variable per layer
- **Selected**: 0.5
- **Rationale**: Standard dropout for RNN classifiers; pairs with fine-tuned embeddings for regularization

**Question**: Classifier head?
- Options: Dense(1, sigmoid), Dense(64,ReLU)→Dense(1,sigmoid)
- **Selected**: Dense(1, sigmoid)
- **Rationale**: Extra hidden layer adds unnecessary params for binary sentiment — direct projection from BiGRU output is standard and sufficient

**Question**: Embedding training strategy?
- Options: Fine-tune, Freeze
- **Selected**: Fine-tune
- **Rationale**: Word2Vec embeddings are task-agnostic; fine-tuning adapts them to sentiment during training, yielding higher accuracy

## Area 3: Training Strategy

**Question**: Training configuration?
- Options: Conservative (20 epochs, Adam 1e-3, batch 64, patience 3), Cautious (15 epochs, AdamW 1e-4, batch 32, patience 4), Fast (10 epochs, Adam 1e-3, batch 64, patience 2)
- **Selected**: Conservative
- **Rationale**: Early stopping prevents overrun; 20 epochs provides safety margin; Adam 1e-3 + batch 64 is well-tested for BiGRU sentiment

## Area 4: CV Strategy

**Question**: Cross-validation strategy?
- Options: 5-fold StratifiedKFold, 3-fold StratifiedKFold, 80/20 single split
- **Selected**: 5-fold StratifiedKFold
- **Rationale**: Matches Phase 2 SVM CV exactly — enables direct model comparison; maintains label distribution per fold

## Area 5: GPU Execution

**Question**: GPU execution strategy?
- Options: Colab GPU, Mixed (local subset, Colab full), Local GPU only
- **Selected**: Colab GPU
- **Rationale**: Free T4 GPU; matches Phase 2 infrastructure (COLAB=True, Drive mount); no local GPU dependency for reproducible academic submission
