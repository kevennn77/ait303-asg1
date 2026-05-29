---
phase: 03
slug: bigru-sentiment-models
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-30
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Jupyter Notebook (nbconvert + nbval) |
| **Config file** | none — Wave 0 installs dependencies |
| **Quick run command** | `jupyter nbconvert --to notebook --execute --inplace bigru_sentiment.ipynb` |
| **Full suite command** | `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 bigru_sentiment.ipynb` |
| **Estimated runtime** | ~6000 seconds (10 training runs × ~10 min on T4) |

---

## Sampling Rate

- **After every task commit:** Run `jupyter nbconvert --to notebook --execute --inplace bigru_sentiment.ipynb --ExecutePreprocessor.timeout=600` (limited cells if notebook is sectioned)
- **After every plan wave:** Run `jupyter nbconvert --to notebook --execute bigru_sentiment.ipynb --ExecutePreprocessor.timeout=7200` (full execution)
- **Before `/gsd-verify-work`:** Full Colab re-execution with all outputs verified
- **Max feedback latency:** 7200 seconds (full run on Colab)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 03-01-01 | 01 | 1 | SENT-05, SENT-06 | notebook cell | Cell block: "Train CBOW Embeddings" executes without error | ⬜ pending |
| 03-01-02 | 01 | 1 | SENT-05, SENT-06 | notebook cell | Cell block: "Train Skip-Gram Embeddings" executes without error | ⬜ pending |
| 03-01-03 | 01 | 1 | SENT-05, SENT-06 | file-exists | `models/w2v_cbow.model` and `models/w2v_sg.model` exist on disk | ⬜ pending |
| 03-02-01 | 02 | 2 | SENT-09, SENT-10 | notebook cell | Cell block: "Define BiGRU Architecture" executes without error | ⬜ pending |
| 03-02-02 | 02 | 2 | SENT-09, SENT-10, SENT-11 | nbconvert | 5-fold CV loop produces 10 trained models | ⬜ pending |
| 03-02-03 | 02 | 2 | SENT-12 | notebook cell | Cell block: "Evaluation Metrics" produces accuracy/precision/recall/F1/ROC-AUC table | ⬜ pending |
| 03-02-04 | 02 | 2 | SENT-13 | file-exists | 10 `.h5` files saved (`*_fold_0.h5` through `*_fold_4.h5`) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `requirements.txt` or Colab setup cell — installs `tensorflow`, `gensim`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- [ ] Google Drive mount cell with `COLAB=True` pattern matching Phase 2
- [ ] Seed initialization cell (`np.random.seed(42)`, `tf.random.set_seed(42)`, `random.seed(42)`)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Trained model quality | SENT-12 | Cannot assert metric thresholds automatically | Review per-fold metrics table — verify F1 > 0.80 across all folds |
| Embedding quality | SENT-05, SENT-06 | Word2Vec quality is qualitative | Review most_similar() output for semantic coherence |
| Confusion matrix | SENT-12 | Visual inspection needed | Verify TP/TN/FP/FN distribution looks reasonable |
| Colab end-to-end | SENT-13 | Requires GPU runtime | Run full notebook on Colab T4, verify all cells produce output |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all dependencies
- [ ] Notebook executes end-to-end on Colab T4
- [ ] All 10 models saved to Drive
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
