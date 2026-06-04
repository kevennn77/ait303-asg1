# Phase 4: Web Scraping & Aspect Extraction — Pattern Map

**Mapped:** 2026-05-31
**Files analyzed:** 9 new/modified
**Analogs found:** 5 / 7 (2 test files are Wave 0 with no existing analogs)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `bestbuy_scraper.py` | utility | file-I/O | `_create_notebook.py` | role-match |
| `notebooks/aspect_extraction.ipynb` | notebook | transform | `svm_sentiment_models.ipynb` | exact |
| `data_asg/bestbuy/products/<sku>.json` | data | file-I/O | _(generated, no source analog)_ | — |
| `data_asg/bestbuy/products.json` | data | file-I/O | _(generated, no source analog)_ | — |
| `data_asg/bestbuy/all_reviews.csv` | data | file-I/O | _(generated, no source analog)_ | — |
| `data_asg/bestbuy/aspect_labeled_reviews.csv` | data | file-I/O | _(generated, no source analog)_ | — |
| `tests/conftest.py` | utility | config | _(no existing test infrastructure)_ | — |
| `tests/test_scraper.py` | test | request-response | _(no existing test infrastructure)_ | — |
| `tests/test_lda.py` | test | batch | _(no existing test infrastructure)_ | — |
| `tests/test_bertopic.py` | test | batch | _(no existing test infrastructure)_ | — |
| `tests/test_aspects.py` | test | batch | _(no existing test infrastructure)_ | — |
| `tests/test_corex.py` | test | batch | _(no existing test infrastructure)_ | — |
| `tests/test_persistence.py` | test | batch | _(no existing test infrastructure)_ | — |

## Pattern Assignments

### `bestbuy_scraper.py` (utility, file-I/O)

**Analog:** `_create_notebook.py` — the only standalone Python script in the project. Provides shebang, module docstring, and top-level execution structure.

**Imports pattern** (`_create_notebook.py` lines 1-5):
```python
#!/usr/bin/env python3
"""Create the SVM Sentiment Models notebook (svm_sentiment_models.ipynb)"""

import nbformat as nbf
```

**Script structure — recommend this pattern for the scraper:**
```python
#!/usr/bin/env python3
"""Scrape Best Buy Bluetooth & Wireless Speaker reviews for aspect-based sentiment analysis."""

import requests
from bs4 import BeautifulSoup
import time
import random
import json
import csv
import os
import sys

# ... function definitions ...

if __name__ == '__main__':
    main()
```

**No existing project analog for scraper-specific patterns.** The scraper is the first web-scraping script in this project. Use the following patterns from RESEARCH.md (Pattern 1, lines 267-300):

**Fetch with polite delays** (RESEARCH.md lines 286-293):
```python
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

session = requests.Session()

def fetch_page(url):
    """Fetch a page with randomized user-agent and delay."""
    delay = random.uniform(1.0, 3.0)
    time.sleep(delay)
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    resp = session.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text
```

**Expected URL patterns** (RESEARCH.md lines 296-299):
```python
# Product listing: https://www.bestbuy.com/site/portable-audio/portable-speakers-docks-radios/pcmcat310200050004.c?id=pcmcat310200050004&intl=nosplash
# Product page:    https://www.bestbuy.com/site/<product-name>/<sku>.p?skuId=<sku>
# Reviews page:    https://www.bestbuy.com/site/reviews/<product-name>/<sku>?page=<N>
```

**JSON output per product** (RESEARCH.md lines 560-580, example diagram lines 185-194):
```python
# Output structure:
# data_asg/bestbuy/
#   ├── products/
#   │   ├── <sku>.json          # Per-product: name, price, specs, all reviews
#   │   └── ...
#   ├── products.json           # Consolidated catalog of all products
#   └── all_reviews.csv         # Flat CSV: product_name, review_text, rating, date
```

---

### `notebooks/aspect_extraction.ipynb` (notebook, transform)

**Primary analog:** `svm_sentiment_models.ipynb` — exact role match. Both are Colab-deliverable Jupyter Notebooks with structured sections, pip installs, Drive mount, model training, evaluation, and save.

**Secondary analog:** `bigru_sentiment_models.ipynb` — for model persistence and MODEL_DIR configuration patterns.

#### Colab Configuration Pattern

**Source:** `svm_sentiment_models.ipynb` lines 50-66
```python
# ============================================
# CONFIGURATION
# ============================================
# Set to True when running on Google Colab
COLAB = True

# Data directory
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/data_asg'
else:
    DATA_DIR = 'data_asg'

print(f"Running in {'COLAB' if COLAB else 'LOCAL'} mode")
print(f"Data directory: {DATA_DIR}")
```

**Extended with MODEL_DIR (from `bigru_sentiment_models.ipynb` lines 48-67):**
```python
# Data and model directories
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/data_asg'
    MODEL_DIR = f'{DATA_DIR}/models'
else:
    DATA_DIR = 'data_asg'
    MODEL_DIR = 'models'

print(f"Running in {'COLAB' if COLAB else 'LOCAL'} mode")
print(f"Data directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}")
```

#### Imports Section Pattern

**Source:** `svm_sentiment_models.ipynb` lines 90-119
```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# [Domain-specific imports]

# Reproducibility
np.random.seed(42)

print("All imports loaded successfully")
```

#### pip Install Pattern (Colab)

**Source:** `bigru_sentiment_models.ipynb` lines 100-102
```python
!pip install gensim
```

**For Phase 4 — from RESEARCH.md lines 118-123:**
```python
!pip install gensim==4.4.0 spacy==3.8.0 bertopic==0.17.4 corextopic==1.1 pyLDAvis==3.4.1 umap-learn hdbscan
!python -m spacy download en_core_web_sm
```

#### Section Structure Pattern

**Source:** `svm_sentiment_models.ipynb` (throughout — markdown heading + code cells per section)

All 3 existing notebooks follow this strict pattern:
```
## N. Section Name

### N.M Subsection Name

[code cells with outputs]
```

**For Phase 4, from RESEARCH.md (architecture diagram lines 198-243):**
```
Section 1: Colab Setup & Data Loading
Section 2: Review Preprocessing (reuse clean_text from Phase 1)
Section 3: SpaCy Keyphrase Extraction (noun chunks + NER)
Section 4: LDA Topic Modeling (grid search 6-10 topics, C_v coherence)
Section 5: BERTopic Topic Modeling (all-MiniLM-L6-v2)
Section 6: Topic Analysis & Aspect Restructuring (keyword-to-aspect mapping)
Section 7: CorEx Semi-Supervised Modeling (anchored topics)
Section 8: Save Models & Export CSV for Phase 5
```

#### Data Loading from Drive Pattern

**Source:** `bigru_sentiment_models.ipynb` lines 361-368
```python
# Load preprocessed IMDB data
df = pd.read_csv(f'{DATA_DIR}/preprocessed_imdb.csv')
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nClass distribution:")
print(df['sentiment'].value_counts())
print(f"\nMissing values:\n{df.isnull().sum()}")
df.head(3)
```

#### clean_text() Reuse Pattern

**Source:** `sentiment_analysis_preprocessing.ipynb` lines 410-417
```python
# Define text cleaning function: lowercase, remove HTML, remove non-alpha, normalize whitespace
def clean_text(text):
    """Clean raw review text: lowercase, remove HTML tags, remove non-alpha characters, normalize whitespace."""
    text = text.lower()                                     # Convert to lowercase
    text = re.sub(r'<.*?>', '', text)                       # Remove HTML tags like <br />
    text = re.sub(r'[^a-zA-Z]', ' ', text)                 # Remove numbers, punctuation, special characters
    text = re.sub(r'\s+', ' ', text).strip()              # Normalize multiple spaces to single space
    return text
```

#### LDA Topic Modeling Pattern

**Source (code pattern):** RESEARCH.md lines 333-369 (Pattern 3)
```python
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

# Build dictionary and corpus from tokenized reviews
dictionary = Dictionary(tokenized_reviews)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_reviews]

# Grid search over topic counts
coherence_scores = {}
for num_topics in range(6, 11):
    lda = LdaMulticore(
        corpus=corpus, id2word=dictionary, num_topics=num_topics,
        chunksize=2000, passes=10, iterations=400,
        alpha='auto', eta='auto', random_state=42, workers=3
    )
    cm = CoherenceModel(model=lda, texts=tokenized_reviews, dictionary=dictionary, coherence='c_v')
    coherence_scores[num_topics] = cm.get_coherence()

best_k = max(coherence_scores, key=coherence_scores.get)
```

#### BERTopic with Pre-calculated Embeddings Pattern

**Source (code pattern):** RESEARCH.md lines 376-411 (Pattern 4)
```python
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

# Pre-calculate embeddings (80MB model download first time)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(cleaned_reviews, show_progress_bar=True)

# Configure UMAP for reproducibility
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric='cosine', random_state=42)

# Configure HDBSCAN for review data
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean',
                         cluster_selection_method='eom', prediction_data=True)

# Filter stopwords + use bigrams
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# Train
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10, verbose=True
)
topics, probs = topic_model.fit_transform(cleaned_reviews, embeddings)
```

#### CorEX Semi-Supervised Anchored Topic Pattern

**Source (code pattern):** RESEARCH.md lines 418-452 (Pattern 5)
```python
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer

# Build document-term matrix (binary BoW)
vectorizer = CountVectorizer(binary=True, max_features=5000)
X = vectorizer.fit_transform(cleaned_reviews)
words = vectorizer.get_feature_names_out()

# Anchors from top LDA/BERTopic keywords per predefined aspect
anchors = [
    ['design', 'look', 'style', 'aesthetic', 'appearance'],     # Design
    ['sound', 'audio', 'quality', 'clear', 'loud'],             # Sound Quality
    ['battery', 'life', 'charge', 'lasting', 'power'],           # Battery
    ['price', 'value', 'cost', 'cheap', 'expensive'],           # Price
    ['build', 'quality', 'solid', 'durable', 'well'],           # Build Quality
    ['features', 'app', 'connect', 'bluetooth', 'pair'],        # Features
    ['connectivity', 'range', 'signal', 'connection', 'stable'], # Connectivity
    ['comfort', 'portable', 'light', 'carry', 'size'],          # Comfort/Portability
]

topic_model = ct.Corex(n_hidden=8, seed=42, max_iter=500, verbose=True)
topic_model.fit(X, words=words, anchors=anchors, anchor_strength=3)

# Get topic assignments
topic_labels = topic_model.labels        # (n_docs,) with topic assignments or -1
topic_probs = topic_model.p_y_given_x     # (n_docs, n_hidden) topic probabilities
```

#### Model Persistence Pattern (LDA + BERTopic + CorEx)

**Source (model file listing pattern):** `bigru_sentiment_models.ipynb` lines 1486-1509
```python
# Verify and list all trained model files
print(f"\n{'='*60}")
print("Trained Model Files")
print(f"{'='*60}")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_files = sorted(os.listdir(MODEL_DIR))
model_count = 0
for f in model_files:
    # [check appropriate extensions]
    fsize = os.path.getsize(f'{MODEL_DIR}/{f}')
    print(f"  {f:45s} {fsize/1024/1024:.1f} MB")
    model_count += 1

print(f"\n  Total model files: {model_count}")
```

**BERTopic safetensors save/load** (RESEARCH.md lines 650-662):
```python
# Save (safetensors — ~20MB vs >500MB pickle)
embedding_model_ref = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save(
    "models/bertopic_model",
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model=embedding_model_ref
)

# Load
from bertopic import BERTopic
loaded_model = BERTopic.load("models/bertopic_model")
```

**LDA save:**
```python
lda_model.save(f'{MODEL_DIR}/lda_model/lda.model')
```

**CorEx save (pickle):**
```python
import pickle
with open(f'{MODEL_DIR}/corex_model.pkl', 'wb') as f:
    pickle.dump(topic_model, f)
```

#### Aspect Label CSV Export Pattern (Phase 5 Contract)

**Source (code pattern):** RESEARCH.md lines 666-684 (Example 4)
```python
import pandas as pd

output_df = pd.DataFrame({
    'review_id': range(len(cleaned_reviews)),
    'product_name': product_names,
    'review_text': cleaned_reviews,
    'aspect_label': aspect_labels,
    'aspect_confidence': aspect_confidences,
    'review_rating': original_ratings,
    'review_date': review_dates,
})

output_df.to_csv('data_asg/bestbuy/aspect_labeled_reviews.csv', index=False)
print(f"Saved {len(output_df)} labeled reviews for Phase 5")
```

#### Results Summary Pattern

**Source:** `svm_sentiment_models.ipynb` lines 1553-1583
```python
print("="*70)
print("  RESULTS SUMMARY — [MODEL NAME]")
print("="*70)

# Key results, observations, and takeaways
print(f"\n  Key Takeaways:")
print(f"  • Bullet-point findings")
print(f"  • Comparison highlights")
print("="*70)
```

---

### Output Data Files (generated, no source analog)

These files are created by the scraper or notebook and consumed downstream. Schema definitions from RESEARCH.md:

**`data_asg/bestbuy/products/<sku>.json`** — Per-product JSON (D-22):
```json
{
    "sku": "5496203",
    "name": "Portable Bluetooth Speaker",
    "price": 39.99,
    "rating": 4.5,
    "review_count": 9670,
    "specs": {"connectivity": "Bluetooth 5.0", "battery": "12 hours", ...},
    "reviews": [
        {"title": "Great sound", "text": "This speaker has amazing...", "rating": 5, "date": "2025-12-01"},
        ...
    ]
}
```

**`data_asg/bestbuy/all_reviews.csv`** — Consolidated flat CSV:
| product_name | review_text | rating | date |
|---|---|---|---|
| Portable Bluetooth Speaker | This speaker has amazing... | 5 | 2025-12-01 |

**`data_asg/bestbuy/aspect_labeled_reviews.csv`** — Phase 5 contract (D-35):
| review_id | product_name | review_text | aspect_label | aspect_confidence | review_rating | review_date |
|---|---|---|---|---|---|---|
| 0 | Portable BT Speaker | ... | Sound Quality | 0.87 | 5 | 2025-12-01 |

---

### Test Files (Wave 0 — no existing analogs in codebase)

No test infrastructure exists in the project yet. These are new files. Use standard pytest patterns with the following structure from RESEARCH.md (Validation Architecture, lines 714-748):

**`tests/conftest.py`** — Shared fixtures:
```python
import pytest
import pandas as pd

@pytest.fixture
def sample_reviews():
    return ["Great sound quality and battery life",
            "Poor build quality, stops charging after 3 months",
            "Excellent value for money, clear audio"]

@pytest.fixture
def tokenized_reviews(sample_reviews):
    return [r.lower().split() for r in sample_reviews]
```

**`tests/test_scraper.py`** — ABSA-01 coverage:
```python
def test_min_products():
    """Verify scraper collected 30+ products with 100+ reviews."""
    ...

def test_delay_between_requests():
    """Verify polite delays between requests."""
    ...

def test_json_output_format():
    """Verify JSON output has expected structure."""
    ...

def test_csv_schema():
    """Verify CSV has expected columns: product_name, review_text, rating, date."""
    ...

def test_no_pii_in_scraped_data():
    """Verify no PII leaked in scraped content."""
    ...
```

**`tests/test_lda.py`** — ABSA-02/04 coverage:
```python
def test_lda_trains():
    """Verify LDA model trains without error."""
    ...

def test_coherence_computed():
    """Verify C_v coherence is computed."""
    ...
```

**`tests/test_bertopic.py`** — ABSA-03 coverage:
```python
def test_bertopic_trains():
    """Verify BERTopic model trains without error."""
    ...
```

**`tests/test_aspects.py`** — ABSA-05 coverage:
```python
def test_min_aspects():
    """Verify 6+ aspects are defined with keywords."""
    ...
```

**`tests/test_corex.py`** — ABSA-06 coverage:
```python
def test_corex_trains():
    """Verify CorEx model trains with anchored topics."""
    ...
```

**`tests/test_persistence.py`** — ABSA-07 coverage:
```python
def test_models_saved():
    """Verify all models (LDA, BERTopic, CorEx) are saved to disk."""
    ...
```

---

## Shared Patterns

### Colab + Drive Mount
**Source:** `svm_sentiment_models.ipynb` lines 50-66, `bigru_sentiment_models.ipynb` lines 48-67
**Apply to:** `notebooks/aspect_extraction.ipynb` (all modeling sections)
```python
COLAB = True
if COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/data_asg'
else:
    DATA_DIR = 'data_asg'
```

### Review Text Preprocessing (Phase 1 `clean_text()` reuse)
**Source:** `sentiment_analysis_preprocessing.ipynb` lines 410-417
**Apply to:** `notebooks/aspect_extraction.ipynb` Section 2 (review preprocessing)
```python
def clean_text(text):
    """Clean raw review text: lowercase, remove HTML tags, remove non-alpha characters, normalize whitespace."""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### Random Seed for Reproducibility
**Source:** `svm_sentiment_models.ipynb` line 117, `bigru_sentiment_models.ipynb` lines 149-151
**Apply to:** All modeling sections in `aspect_extraction.ipynb`
```python
np.random.seed(42)
random.seed(42)
```

### Evaluation Visualization Pattern
**Source:** `svm_sentiment_models.ipynb` (confusion matrix heatmap pattern at lines 306-314), `bigru_sentiment_models.ipynb` (ROC curves at lines 1370-1391)
**Apply to:** LDA coherence plots, BERTopic topic visualizations, pyLDAvis charts
```python
plt.figure(figsize=(8, 5))
# ... plot elements ...
plt.title('Chart Title')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Text Data Loading from Drive with Validation
**Source:** `svm_sentiment_models.ipynb` lines 82-106
**Apply to:** Loading `all_reviews.csv` in Section 2 of notebook
```python
# Load from Drive path
df = pd.read_csv(f'{DATA_DIR}/bestbuy/all_reviews.csv')
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")
```

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `bestbuy_scraper.py` (scraping-specific) | utility | file-I/O | Only standalone py script is `_create_notebook.py` which is notebook builder, not scraper. Use RESEARCH.md Pattern 1 + Example 1 for scraper structure. |
| `tests/conftest.py` | utility | config | No test infrastructure exists in project. Standard pytest conftest + fixtures. |
| `tests/test_scraper.py` | test | request-response | No test files exist yet. Standard pytest test file. |
| `tests/test_lda.py` | test | batch | Same — Wave 0 test file. |
| `tests/test_bertopic.py` | test | batch | Same — Wave 0 test file. |
| `tests/test_aspects.py` | test | batch | Same — Wave 0 test file. |
| `tests/test_corex.py` | test | batch | Same — Wave 0 test file. |
| `tests/test_persistence.py` | test | batch | Same — Wave 0 test file. |
| `data_asg/bestbuy/products/<sku>.json` | data | file-I/O | Generated output — schema defined in CONTEXT.md D-22. |
| `data_asg/bestbuy/products.json` | data | file-I/O | Generated output — catalog of all products. |
| `data_asg/bestbuy/all_reviews.csv` | data | file-I/O | Generated output — flat CSV. |
| `data_asg/bestbuy/aspect_labeled_reviews.csv` | data | file-I/O | Generated output — Phase 5 contract. |

## Metadata

**Analog search scope:** Project root (`/Users/keven/Documents/degree_study_material/y3s3/Adv_NLP/ASG1/`)
**Files scanned:** 4 (sentiment_analysis_preprocessing.ipynb, svm_sentiment_models.ipynb, bigru_sentiment_models.ipynb, _create_notebook.py)
**Pattern extraction date:** 2026-05-31
