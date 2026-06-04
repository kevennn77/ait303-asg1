# AIT303 Assignment 1 — Aspect-Based Sentiment Analysis

An academic NLP project implementing a complete aspect-based sentiment analysis pipeline on IMDB 50K movie reviews and scraped Best Buy speaker reviews.

## Pipeline

```
IMDB 50K Dataset ──> Preprocessing ──> SVM + BiGRU Sentiment Models ──> Best Model
                                                                             │
Best Buy Speakers ──> curl_cffi Scraper ──> LDA / BERTopic / CorEx ─────────┤
                                                                             │
                          ┌──────────────────────────────────────────────────┘
                          ▼
            Aspect Labeling → Product Ranking → Best Product Selection
```

## Notebooks (run in order)

| # | Notebook | Description | ⏱ |
|---|----------|-------------|----|
| 1 | `sentiment_analysis_preprocessing.ipynb` | Load IMDB 50K, text cleaning pipeline | ~2 min |
| 2 | `svm_sentiment_models.ipynb` | 4 SVM variants (CountVectorizer/Tfidf × stemmed/lemmatized) with 5-fold CV | ~15-30 min |
| 3 | `bigru_sentiment_models.ipynb` | Word2Vec (CBOW + Skip-Gram) + BiGRU models with 5-fold CV | ~30-60 min |
| 4 | `notebooks/aspect_extraction.ipynb` | LDA, BERTopic, CorEx aspect extraction on scraped Best Buy reviews | ~10-15 min |
| 5 | `notebooks/product_ranking.ipynb` | Sentiment labeling, aspect scoring, product rankings, charts | ~3 min |

## Setup

### Local (VS Code)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install scikit-learn pandas matplotlib seaborn nltk tensorflow gensim spacy bertopic corextopic pyLDAvis curl-cffi
python -m spacy download en_core_web_sm
```

### Google Colab
Each notebook has a `COLAB = True` flag at the top. Upload `data_asg/` to `MyDrive/data_asg/`, then run all cells.

## Key Results

**Best Sentiment Model:** SVM + TfidfVectorizer + lemmatized (Test F1 = 0.9091)

**Aspect Extraction:** 8 aspects (Design, Sound Quality, Battery, Price, Build Quality, Features, Connectivity, Comfort/Portability) extracted via CorEx anchored topic model.

**Best Product:** Altec Lansing Jolt Mini Lifejacket (Composite Score: 0.900)

## Data

- `data_asg/IMDB Dataset.csv` — IMDB 50K movie reviews (excluded from git, ~63 MB)
- `data_asg/preprocessed_imdb.csv` — Preprocessed IMDB data (excluded from git, ~194 MB)
- `data_asg/bestbuy/` — Scraped Best Buy speaker reviews (37 products, 3,199 reviews)

## Deliverables

| Artifact | Location |
|----------|----------|
| Jupyter Notebooks | Root + `notebooks/` |
| Scraper script | `bestbuy_scraper.py` |
| Scraped data | `data_asg/bestbuy/` |
| Saved models | `models/` (generated on Colab) |
| PDF Report | Submitted to Moodle |
| Cloud backup | Google Drive / OneDrive |

## Requirements

See `.planning/REQUIREMENTS.md` for the full requirement traceability matrix (35 requirements across 6 phases).
