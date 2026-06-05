# AIT303 Assignment 1 — Aspect-Based Sentiment Analysis

An academic NLP assignment implementing a complete aspect-based sentiment analysis pipeline: train 4 sentiment models on IMDB 50K movie reviews, scrape Best Buy speaker reviews via TLS-impersonated API, extract aspects using LDA/BERTopic/CorEx, and rank products by aspect-level sentiment.

**GitHub:** `https://github.com/kevennn77/ait303-asg1`

---

## Pipeline

```
IMDB 50K Dataset ──▶ Preprocessing ──▶ SVM + BiGRU (4 models) ──▶ Best Model (TFIDF+Lemmatized)
                                                                             │
Best Buy (37 speakers) ──▶ curl_cffi Hidden API ──▶ LDA + BERTopic + CorEx ──┤
                                                                             │
                                                Aspect Labeling → Product Ranking → Best Product
```

**Four models compared:** SVM × CountVectorizer (stemmed/lemmatized) + SVM × TfidfVectorizer (stemmed/lemmatized) + BiGRU × CBOW embeddings + BiGRU × Skip-Gram embeddings. Best: TFIDF+Lemmatized SVM (F1=0.9091).

**Three topic models:** LDA (grid search K=6-10, best C_v=0.441), BERTopic (48 topics), CorEx (8 anchored aspects, TC=13.19). Best products ranked by composite score (60% positive ratio + 20% coverage + 20% rating).

---

## Notebooks (run in order on Colab)

| # | Notebook                                   | What It Does                                                                  | ⏱          |
| - | ------------------------------------------ | ----------------------------------------------------------------------------- | ----------- |
| 1 | `sentiment_analysis_preprocessing.ipynb` | Load IMDB 50K, clean (lowercase/HTML/punct/stopwords), stem + lemmatize       | ~2 min      |
| 2 | `svm_sentiment_models.ipynb`             | 4 SVM variants, 5-fold Stratified K-Fold CV, confusion matrices, top features | ~15–30 min |
| 3 | `bigru_sentiment_models.ipynb`           | Word2Vec (CBOW + Skip-Gram), BiGRU 5-fold CV with early stopping              | ~30–60 min |
| 4 | `notebooks/aspect_extraction.ipynb`      | LDA grid search, BERTopic, CorEx anchored topics, model persistence           | ~10–15 min |
| 5 | `notebooks/product_ranking.ipynb`        | SVM sentiment labeling, per-aspect scoring, rankings, 3 bar charts            | ~3 min      |

---

## Quick Start

### Google Colab (recommended)

1. Upload `data_asg/` folder to `MyDrive/data_asg/` on Google Drive
2. Open each notebook in Colab (File → Upload Notebook)
3. Set `COLAB = True` at the top (default) — mounts Drive automatically
4. Run all cells in order 1→5 (sequential dependencies)
5. Outputs (CSVs, models, charts) save to `MyDrive/data_asg/bestbuy/`

**Note:** Run on Colab's **Python 3.10** runtime (default) — gensim 4.4.0 and TensorFlow do not support Python 3.12+.

### Local (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install scikit-learn pandas matplotlib seaborn nltk tensorflow gensim spacy bertopic corextopic pyLDAvis curl-cffi
python -m spacy download en_core_web_sm
```

Set `COLAB = False` in each notebook before running. The scraper (`curl-cffi`) works on macOS/Linux but may have issues on Windows.

---

## Key Results

| Category                    | Result                                                                           |
| --------------------------- | -------------------------------------------------------------------------------- |
| **Best Model**        | SVM + TfidfVectorizer + Lemmatized — Test F1 =**0.9091**                  |
| **Runner-up**         | BiGRU + Skip-Gram — Mean F1 =**0.8994**                                   |
| **Aspect Extraction** | CorEx: 7/8 coherent aspects, TC = 13.19; LDA: best K=9, C_v = 0.441              |
| **Best Product**      | **Altec Lansing Jolt Mini Lifejacket** — Composite Score: **0.900** |
| **Scraped Dataset**   | 37 products, 3,199 reviews from Best Buy via hidden `/ugc/v2/reviews` API      |

---

## Data Structure

```
data_asg/
├── IMDB Dataset.csv                  ← 50K movie reviews (excluded from git, ~63 MB)
├── preprocessed_imdb.csv             ← Phase 1 output (excluded from git, ~194 MB)
└── bestbuy/
    ├── products.json                 ← Curated product list (38 SKUs)
    ├── all_reviews.csv               ← Scraped reviews (3,199)
    ├── aspect_labeled_reviews.csv    ← Phase 4: CorEx aspect labels
    ├── labeled_reviews.csv           ← Phase 5: aspect + sentiment labels
    ├── aspect_scores.csv             ← Per-product per-aspect metrics
    ├── product_rankings.csv          ← Top 5 per aspect
    ├── best_products.csv             ← Top 5 composite scores
    ├── product_rankings_chart.png    ← 7-aspect facet grid
    ├── sentiment_by_aspect.png       ← Stacked sentiment distribution
    └── best_product_profile.png      ← Best product aspect breakdown
```

Model files (`.h5`, `.safetensors`, `.pkl`, Word2Vec) are generated on Colab and saved to `models/`.

---

## Deliverables

| Artifact            | Location                             |
| ------------------- | ------------------------------------ |
| 5 Jupyter Notebooks | Root +`notebooks/`                 |
| Scraper script      | `bestbuy_scraper.py`               |
| Scraped data        | `data_asg/bestbuy/`                |
| Trained models      | `models/` (Colab-generated)        |
| PDF Report          | Submitted to Moodle                  |
| Cloud backup        | OneDrive / Google Drive              |
| Source code         | `github.com/kevennn77/ait303-asg1` |

---

*Built with Python 3.12 · scikit-learn · TensorFlow/Keras · gensim · BERTopic · CorEx · curl-cffi*
