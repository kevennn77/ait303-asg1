# Phase 3: BiGRU Sentiment Models — User Setup Required

**Generated:** 2026-05-30
**Phase:** 03-bigru-sentiment-models
**Status:** Incomplete

Complete these items for the notebook to function. The agent automated everything possible; these items require human access to external services.

## Colab Execution

The Word2Vec, TensorFlow, and Keras libraries cannot be installed locally (Python 3.14 incompatibility with gensim's C extensions). All model training must run on Google Colab.

### Step 1: Upload data to Google Drive

- [ ] **Upload `data_asg/` folder to Google Drive**
  - Upload the entire `data_asg/` folder to your Google Drive root
  - Expected path: `/content/drive/MyDrive/data_asg/preprocessed_imdb.csv`
  - The `COLAB = True` flag in the notebook handles Drive mounting automatically

### Step 2: Open notebook in Colab

- [ ] **Open `bigru_sentiment_models.ipynb` in Google Colab**
  - Upload to Colab or open from Google Drive
  - Runtime → Change runtime type → Select **T4 GPU** (required for BiGRU training)

### Step 3: Run cells sequentially

Run all cells in order. Key sections:

| Section | Description | Est. Time |
|---------|-------------|-----------|
| 1-2 | Setup, imports, data loading | ~2 min |
| 3 | Train Word2Vec (CBOW + Skip-Gram) | ~5 min |
| 4 | Build vocabulary + embedding matrices + model builder | ~2 min |

## Expected Outputs

After Section 3 completes:
- `models/word2vec_cbow.model` (~400 MB)
- `models/word2vec_sg.model` (~400 MB)

## Verification

```python
from gensim.models import Word2Vec
cbow = Word2Vec.load('/content/drive/MyDrive/data_asg/models/word2vec_cbow.model')
print(f"CBOW vocab: {len(cbow.wv)}")
sg = Word2Vec.load('/content/drive/MyDrive/data_asg/models/word2vec_sg.model')
print(f"SG vocab: {len(sg.wv)}")
```

---

## Plan 03-02 (Wave 2) — Will Require Colab

Plan 03-02 adds the 5-fold cross-validation BiGRU training (both CBOW and SG embeddings). This section requires T4 GPU and takes ~60-90 minutes per embedding type.

- [ ] **After completing Plan 03-01 notebook cells**, Proceed to the next phase plan (03-02) for CV + evaluation

---

**Once all items complete:** Mark status as "Complete" at top of file.
