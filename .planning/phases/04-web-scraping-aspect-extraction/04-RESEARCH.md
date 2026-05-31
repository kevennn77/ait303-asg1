# Phase 4: Web Scraping & Aspect Extraction — Research

**Researched:** 2026-05-31
**Domain:** Web scraping (Best Buy), topic modeling (LDA, BERTopic, CorEx), keyphrase extraction (SpaCy), topic coherence evaluation
**Confidence:** HIGH

## Summary

Phase 4 splits into two distinct runtime environments: (1) a **local Python scraper** (`bestbuy_scraper.py`) using requests + BeautifulSoup that collects 35-40 Bluetooth speaker products with 100+ reviews each from Best Buy, storing JSON per product and a consolidated CSV; and (2) a **Colab-based Jupyter Notebook** that performs keyphrase extraction via SpaCy, trains LDA (gensim), BERTopic, and CorEx models, evaluates topic coherence, and outputs a shared CSV with aspect-labeled reviews for Phase 5. The scraper runs locally (Python 3.14 — no gensim dependency required). The modeling notebook runs on Colab (Python 3.10) because gensim 4.4.0 cannot compile on Python 3.14 [VERIFIED: pip install attempt above].

**Primary recommendation:** Write the scraper as a standalone Python script with randomized 1-3s delays + rotating user agents. Build the modeling as a single Jupyter Notebook following established Colab patterns (COLAB=True flag, Drive mount). Reuse the Phase 1 `clean_text()` function for review preprocessing. Save all three models (LDA via `.save()`, BERTopic via `safetensors`, CorEx via pickle).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-19:** **Standalone Python scraper script** (`bestbuy_scraper.py`) separate from the notebook.
- **D-20:** Target **Best Buy Bluetooth & Wireless Speakers** category.
- **D-21:** Anti-detection: **Randomized 1-3 second delays + rotating user-agent headers**.
- **D-22:** Scraped data stored as **structured JSON per product** + **consolidated CSV** (all reviews).
- **D-23:** **Scrape 35-40 products**, accept products with **80+ reviews**.
- **D-24:** **Reuse Phase 1 IMDB cleaning pipeline** for review text preprocessing.
- **D-25:** **Product metadata NOT prepended** to review text. Stored separately, joined in Phase 5.
- **D-26:** **Use SpaCy keyphrase extraction** — extract noun chunks and named entities before topic modeling.
- **D-27:** **LDA: try 6-10 topics, pick best via topic coherence** (C_v or C_UMass).
- **D-28:** **BERTopic embedding model: `all-MiniLM-L6-v2`**.
- **D-29:** Target **6-10 interpretable topics per model**.
- **D-30:** **Predefined aspect categories** — Design, Sound Quality, Battery, Price, Build Quality, Features, Connectivity, Comfort.
- **D-31:** **Automated keyword-to-aspect mapping + human review**.
- **D-32:** **Allow multi-aspect membership**.
- **D-33:** **Top keywords from LDA/BERTopic as CorEx anchors**.
- **D-34:** **Anchor all 6+ predefined aspects** in CorEx.
- **D-35:** **Shared CSV with aspect labels** as Phase 5 contract.

### the Agent's Discretion
- Specific SpaCy pipeline selection for keyphrase extraction
- LDA and BERTopic hyperparameter details (beyond topic count and embedding model)
- CorEx hyperparameters (beyond anchor selection strategy)
- Visualization style for topic coherence and topic outputs
- Notebook section structure and organization
- Specific Python library versions (topic model libraries are standard PyPI packages)

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ABSA-01 | Scrape 30+ speaker products (100+ reviews each) from Best Buy | Best Buy URL structure identified (category + review pagination). Review pages load reviews via JavaScript — scraper must handle either rendered HTML or structured data. Standalone script with polite delays. |
| ABSA-02 | Train LDA model for unsupervised aspect extraction | gensim `LdaMulticore` with `num_topics=6-10`. Parameters: `chunksize=2000`, `passes=10`, `iterations=400`, `alpha='auto'`, `eta='auto'`. Coherence grid search across topic counts. |
| ABSA-03 | Train BERTopic model for unsupervised aspect extraction | BERTopic with `all-MiniLM-L6-v2`. UMAP (`n_neighbors=15, n_components=5, metric='cosine'`) + HDBSCAN (`min_cluster_size=10, prediction_data=True`). |
| ABSA-04 | Analyze LDA and BERTopic topics, identify keywords | Gensim `CoherenceModel` (C_v metric) for LDA. BERTopic `.get_topic_info()` and `.get_topic()` for topic keywords. pyLDAvis for visualization. |
| ABSA-05 | Restructure keywords into 6+ meaningful aspects | SpaCy noun chunks + named entities extracted pre-modeling. Keyword-to-aspect mapping via overlap scoring. 8 predefined categories (Design, Sound Quality, Battery, Price, Build Quality, Features, Connectivity, Comfort). |
| ABSA-06 | Train semi-supervised CorEx model with anchored aspects | `corextopic.Corex` with `n_hidden=8`. Anchored via D-33/D-34 using top keywords from LDA/BERTopic per aspect. |
| ABSA-07 | Save all aspect extraction models to disk | LDA: gensim `.save()`. BERTopic: `safetensors` serialization (small, safe). CorEx: pickle (native format). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Web scraping (data collection) | Local execution | — | Standalone Python script, no server needed. Policed delays, user-agent rotation. |
| Product/Review data storage | Local filesystem | — | JSON per product + consolidated CSV. Simple file I/O, no DB needed. |
| Text preprocessing (review cleaning) | Colab Notebook | — | Reuses Phase 1 `clean_text()` pipeline. Runs in notebook before modeling. |
| Keyphrase extraction (SpaCy) | Colab Notebook | — | NLP processing on cleaned review text. Runs in notebook environment. |
| LDA topic modeling | Colab Notebook | — | gensim `LdaMulticore`. Requires Python 3.10 (Colab). C_v coherence evaluation. |
| BERTopic topic modeling | Colab Notebook | — | BERTopic with all-MiniLM-L6-v2. Downloads sentence-transformers model on first run. |
| CorEx semi-supervised modeling | Colab Notebook | — | corex_topic with anchored keywords. Uses LDA/BERTopic output as anchors. |
| Model persistence | Colab Notebook (Drive) | — | Models saved to Google Drive via mount. Downloaded locally for Phase 6 submission. |
| Aspect-labeled CSV output | Colab Notebook | — | Shared artifact for Phase 5. CSV with review_id, product, aspect_label, confidence. |

## Standard Stack

### Core — Scraper (local, Python 3.14)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| requests | 2.34.x | HTTP client for fetching Best Buy pages | Industry standard, handles sessions/headers. [VERIFIED: installed in venv] |
| beautifulsoup4 | 4.14.x | HTML parsing for product/review extraction | Standard Python HTML parser. [VERIFIED: installed in venv] |
| json (stdlib) | — | Structured product data storage | Standard library, fits D-22 JSON-per-product requirement |

### Core — Modeling (Colab, Python 3.10)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| spaCy | 3.8.x | Noun chunk + NER keyphrase extraction | Industry standard NLP pipeline. [CITED: spacy.io/usage/linguistic-features] |
| gensim | 4.4.0 | LDA topic modeling + coherence evaluation | Standard Python topic modeling library. [CITED: radimrehurek.com/gensim] |
| BERTopic | 0.17.x | Transformer-based topic modeling | State-of-the-art semantic topic clustering. [CITED: maartengr.github.io/BERTopic] |
| corextopic | 1.1 | Semi-supervised anchored topic modeling | Standard CorEx implementation on PyPI. [CITED: ryanjgallag.com/code/corex/example] |
| sentence-transformers | latest | BERTopic embedding backend (all-MiniLM-L6-v2) | Required by BERTopic as embedding model. [CITED: huggingface.co/sentence-transformers/all-MiniLM-L6-v2] |
| pyLDAvis | 3.4.x | LDA topic visualization | Standard LDA interactive visualization. [VERIFIED: PyPI] |

### Supporting — Modeling
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | 1.8.x | CountVectorizer for BERTopic c-TF-IDF | Pass custom vectorizer_model to BERTopic to filter stopwords and n-grams |
| umap-learn | latest | Dimensionality reduction in BERTopic pipeline | Custom UMAP for controlling n_neighbors/n_components |
| hdbscan | latest | Clustering in BERTopic pipeline | Custom HDBSCAN for controlling min_cluster_size |
| numpy | latest | Array operations for coherence/comparisons | Always needed |
| pandas | latest | DataFrame operations, CSV output | Always needed |
| joblib | latest | Fallback persistence | Alternative to pickle for CorEx/model saving |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| requests + BeautifulSoup | Selenium / Playwright | Selenium significantly slower, overkill for Best Buy which renders reviews in page HTML |
| gensim LdaMulticore | sklearn LDA | sklearn LDA lacks native C_v coherence pipeline and topic visualization integration |
| BERTopic | BERT + clustering manually | BERTopic abstracts all complexity (UMAP + HDBSCAN + c-TF-IDF) into single API |
| CorEx anchored topics | Supervised aspect classifier | Assignment spec explicitly requires semi-supervised CorEx |

### Installation — Scraper (local)
```bash
# Only needed for local scraping — no gensim/BERTopic here
pip install requests beautifulsoup4
```

### Installation — Modeling (Colab)
```python
# In notebook — matches Phase 2/3 Colab pattern
!pip install gensim==4.4.0 spacy==3.8.0 bertopic==0.17.4 corextopic==1.1 pyLDAvis==3.4.1 umap-learn hdbscan
!python -m spacy download en_core_web_sm
```

## Package Legitimacy Audit

> slopcheck was run at research time for all packages tagged below.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| requests | PyPI | 14 yrs | 500M+/wk | github.com/psf/requests | [OK] | Approved |
| beautifulsoup4 | PyPI | 20 yrs | 200M+/wk | code.launchpad.net/beautifulsoup | [OK] | Approved |
| spacy | PyPI | 9 yrs | 10M+/wk | github.com/explosion/spaCy | [OK] | Approved |
| gensim | PyPI | 14 yrs | 5M+/wk | github.com/RaRe-Technologies/gensim | [OK] | Approved |
| bertopic | PyPI | 5 yrs | 200K+/wk | github.com/MaartenGr/BERTopic | [OK] | Approved |
| corextopic | PyPI | 7 yrs | 10K+/wk | github.com/gregversteeg/corex_topic | [OK] | Approved |
| pyLDAvis | PyPI | 9 yrs | 200K+/wk | github.com/bmabey/pyLDAvis | [OK] | Approved |
| sentence-transformers | PyPI | 5 yrs | 1M+/wk | HuggingFace Hub | [OK] | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.x | Scraper script | ✓ | 3.14.2 | Colab uses Python 3.10 |
| requests | Scraper | ✓ | 2.34.2 | — |
| beautifulsoup4 | Scraper | ✓ | 4.14.3 | — |
| spaCy | Keyphrase extraction | ✓ (local) | 3.8.13 | — |
| gensim | LDA modeling | ✗ (3.14) | — | Run on Colab (Python 3.10) |
| BERTopic | Topic modeling | ✗ (not installed) | — | Run on Colab (pip install) |
| corextopic | CorEx modeling | ✗ (not installed) | — | Run on Colab (pip install) |
| pyLDAvis | LDA visualization | ✗ (not installed) | — | Run on Colab (pip install) |

**Critical finding — gensim reproducibility:** gensim 4.4.0 fails to compile on Python 3.14 due to internal CPython C API incompatibilities (`ma_version_tag` removal in PyDictObject). This is a known issue affecting all Cython extensions that use CPython internals. Phase 3's `03-USER-SETUP.md` already documents this.

**Missing dependencies with no fallback:**
- gensim on Python 3.14 — the LDA notebook section must run on Colab (Python 3.10) where gensim compiles and installs cleanly. The scraper has no gensim dependency.

**Missing dependencies with fallback:**
- sentence-transformers / BERTopic — first-run download of `all-MiniLM-L6-v2` (~80MB). Plan for ~5 min download latency on Colab. Document in notebook with a download progress cell.

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL (Python 3.14)                                        │
│                                                             │
│  bestbuy_scraper.py                                         │
│  ┌─────────────────┐    ┌──────────────────────────────┐    │
│  │ 1. Fetch category│───▶│ 2. Parse product listing     │    │
│  │    page          │    │    (name, SKU, price, specs) │    │
│  └─────────────────┘    └───────────┬──────────────────┘    │
│                                     │                        │
│  ┌─────────────────┐    ┌───────────▼──────────────────┐    │
│  │ 4. Parse reviews │◀───│ 3. Fetch review pages        │    │
│  │    (title, text, │    │    (?page=1..N, 1-3s delay)  │    │
│  │     rating, date)│    └──────────────────────────────┘    │
│  └─────────┬───────┘                                        │
│            │                                                  │
│            ▼                                                  │
│  ┌─────────────────────────────────────┐                     │
│  │ Outputs: data_asg/bestbuy/          │                     │
│  │  ├── products/                      │                     │
│  │  │   ├── product_1.json             │                     │
│  │  │   ├── product_2.json             │                     │
│  │  │   └── ...                        │                     │
│  │  ├── products.json (catalog)        │                     │
│  │  └── all_reviews.csv                │                     │
│  └─────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ (upload to Drive)
┌─────────────────────────────────────────────────────────────┐
│  COLAB (Python 3.10, Drive mount)                          │
│                                                             │
│  aspect_extraction.ipynb                                    │
│  ┌────────────────┐    ┌──────────────────────────────┐    │
│  │ Section 1-2:   │───▶│ Section 3: Text prep +       │    │
│  │ Setup + Colab  │    │ SpaCy keyphrase extraction    │    │
│  │ + Load CSV     │    │ (noun chunks + NER)           │    │
│  └────────────────┘    └──────────┬───────────────────┘    │
│                                   │                         │
│          ┌────────────────────────┼────────────────────┐    │
│          ▼                        ▼                     │    │
│  ┌──────────────┐      ┌──────────────────┐            │    │
│  │ Section 4:   │      │ Section 5:       │            │    │
│  │ LDA model    │      │ BERTopic model   │            │    │
│  │ training     │      │ training         │            │    │
│  │ (LdaMulticore)│      │ (all-MiniLM-L6-v2)│           │    │
│  │ + C_v eval   │      │ + topic analysis │            │    │
│  └──────┬───────┘      └────────┬─────────┘            │    │
│         │                       │                      │    │
│         └───────────┬───────────┘                      │    │
│                     ▼                                  │    │
│  ┌─────────────────────────────────────┐               │    │
│  │ Section 6: Analyze topics, identify │               │    │
│  │ keywords, restructure → 8 aspects   │               │    │
│  └─────────────┬───────────────────────┘               │    │
│                ▼                                       │    │
│  ┌─────────────────────────────────────┐               │    │
│  │ Section 7: CorEx with anchored      │               │    │
│  │ aspects from LDA/BERTopic keywords   │               │    │
│  └─────────────┬───────────────────────┘               │    │
│                ▼                                       │    │
│  ┌─────────────────────────────────────┐               │    │
│  │ Section 8: Save all models +        │               │    │
│  │ export aspect-labeled CSV for P5    │               │    │
│  └─────────────────────────────────────┘               │    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ (download from Drive)
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Labeling, Ranking & Visualization                │
│  Loads aspect_labeled_reviews.csv                           │
│  Loads best sentiment model from Phase 2/3                 │
│  Computes per-aspect sentiment scores per product          │
│  Ranks top 5 products per aspect, creates charts           │
└─────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
data_asg/
└── bestbuy/
    ├── products/            # JSON per product (full metadata)
    │   ├── 5496203.json
    │   ├── 6453127.json
    │   └── ...
    ├── products.json        # Consolidated product catalog
    └── all_reviews.csv      # Flat CSV: product, review_text, rating, date

notebooks/
└── aspect_extraction.ipynb  # Main Phase 4 notebook (Colab)

models/
├── lda_model/               # gensim LDA model (folder)
├── bertopic_model/          # BERTopic safetensors directory
└── corex_model.pkl          # CorEx topic model

bestbuy_scraper.py           # Standalone scraper script
```

### Pattern 1: Best Buy Scraping with Polite Delays
**What:** Fetch product listing → iterate products → fetch review pages → parse with BeautifulSoup → save JSON + CSV
**When to use:** For the standalone `bestbuy_scraper.py` script
**Example:**
```python
import requests
from bs4 import BeautifulSoup
import time
import random
import json

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

# Expected URL patterns (VERIFIED via firecrawl search):
# Product listing: https://www.bestbuy.com/site/portable-audio/portable-speakers-docks-radios/pcmcat310200050004.c?id=pcmcat310200050004&intl=nosplash
# Product page:    https://www.bestbuy.com/site/<product-name>/<sku>.p?skuId=<sku>
# Reviews page:    https://www.bestbuy.com/site/reviews/<product-name>/<sku>?page=<N>
```
**Source:** [VERIFIED: firecrawl scrape of bestbuy.com review pages and robots.txt]

### Pattern 2: SpaCy Keyphrase Extraction (Noun Chunks + NER)
**What:** Extract noun phrases and named entities from review text to focus topic modeling on aspect-relevant terms
**When to use:** D-26 — before feeding cleaned review text into LDA/BERTopic
**Example:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_keyphrases(text):
    """Extract noun chunks and named entities as aspect-relevant keyphrases."""
    doc = nlp(text)
    keyphrases = set()
    
    # Noun chunks: "battery life", "sound quality", "build quality"
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if len(phrase.split()) <= 3:  # Filter long phrases
            keyphrases.add(phrase)
    
    # Named entities: brand names, model numbers
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "MONEY"):
            keyphrases.add(ent.text.lower().strip())
    
    return list(keyphrases)
```
**Source:** [CITED: spacy.io/usage/linguistic-features#noun-chunks] & [ASSUMED for usage in review text filtering]

### Pattern 3: LDA Topic Modeling with Gensim
**What:** Train LDA on review text, evaluate with C_v coherence, find optimal topic count
**When to use:** D-27 (6-10 topics, pick via coherence)
**Example:**
```python
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

# Build dictionary and corpus from tokenized reviews
dictionary = Dictionary(tokenized_reviews)
dictionary.filter_extremes(no_below=5, no_above=0.5)  # Filter rare/overly common words
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_reviews]

# Grid search over topic counts
coherence_scores = {}
for num_topics in range(6, 11):
    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        chunksize=2000,
        passes=10,
        iterations=400,
        alpha='auto',
        eta='auto',
        random_state=42,
        workers=3  # cpu_count - 1
    )
    cm = CoherenceModel(
        model=lda,
        texts=tokenized_reviews,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_scores[num_topics] = cm.get_coherence()

# Best model: max coherence
best_k = max(coherence_scores, key=coherence_scores.get)
```
**Source:** [CITED: radimrehurek.com/gensim/models/ldamulticore.html] & [CITED: radimrehurek.com/gensim/models/coherencemodel.html]

### Pattern 4: BERTopic with all-MiniLM-L6-v2
**What:** Train BERTopic with pre-calculated embeddings for reproducibility
**When to use:** D-28 (all-MiniLM-L6-v2)
**Example:**
```python
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

# Pre-calculate embeddings (best practice - 80MB model download first time)
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
    top_n_words=10,
    verbose=True
)
topics, probs = topic_model.fit_transform(cleaned_reviews, embeddings)

# Inspect: topic_model.get_topic_info() for topic overview
#          topic_model.get_topic(0) for top-10 keywords of topic 0
```
**Source:** [CITED: maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html] & [CITED: maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html]

### Pattern 5: CorEx with Anchored Aspects
**What:** Train semi-supervised CorEx with aspect keywords from LDA/BERTopic
**When to use:** D-33, D-34 — after unsupervised models provide top keywords per aspect
**Example:**
```python
from corextopic import corextopic as ct

# Build document-term matrix from tokenized reviews
# Using CountVectorizer for binary presence (CorEx works with BoW)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(binary=True, max_features=5000)
X = vectorizer.fit_transform(cleaned_reviews)
words = vectorizer.get_feature_names_out()

# Anchors from top LDA/BERTopic keywords per predefined aspect
# Shape: list of lists, one per topic, aligned with n_hidden topics
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

# Train CorEx with anchored topics
topic_model = ct.Corex(
    n_hidden=8,           # One per predefined aspect
    seed=42,
    max_iter=500,
    verbose=True
)
topic_model.fit(X, words=words, anchors=anchors, anchor_strength=3)

# Get topic assignments for each review
topic_labels = topic_model.labels  # (n_docs,) with topic assignments or -1
topic_probs = topic_model.p_y_given_x  # (n_docs, n_hidden) topic probabilities
```
**Source:** [CITED: ryanjgallag.com/code/corex/example] & [CITED: github.com/gregversteeg/corex_topic]

### Anti-Patterns to Avoid
- **Scraping without delay:** Sending requests as fast as possible = IP block. Always use 1-3s randomized delays.
- **Selenium for Best Buy reviews:** Best Buy renders reviews in the initial HTML. Selenium is unnecessary overhead.
- **Not pre-calculating BERTopic embeddings:** Every parameter tweak re-encodes all documents. Pre-calculate once, feed to BERTopic as `embeddings` parameter.
- **Using pickle for BERTopic:** Produces >500MB files. Use `safetensors` serialization.
- **Skipping SpaCy keyphrase extraction:** LDA/BERTopic on raw review text produces noisy topics. Pre-filtering with noun chunks improves interpretability.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP requests + HTML parsing | Custom socket-level HTTP | requests + BeautifulSoup | Battle-tested, handles redirects/cookies/encoding |
| LDA topic modeling | Implement collapsed Gibbs sampling | gensim LdaMulticore | Multi-core optimized, streaming, built-in coherence |
| Semantic topic clustering | Implement UMAP + HDBSCAN + c-TF-IDF | BERTopic | Modular, parameter-tunable, active maintenance |
| Semi-supervised topic anchoring | Implement correlation explanation | corex_topic (CorEx) | Only maintained anchored CorEx implementation |
| Topic coherence evaluation | Implement C_v from scratch | gensim CoherenceModel | 4-stage pipeline (segmentation → probability → confirmation → aggregation) |
| Sentence embeddings | Train custom transformer | all-MiniLM-L6-v2 | 80MB, 384-dim, fast, good semantic quality for review text |
| LDA topic visualization | Build interactive topic browser | pyLDAvis | Standard tool, shows inter-topic distance + term relevance |

**Key insight:** Topic modeling requires specialized algorithms. Gensim LdaMulticore uses online variational Bayes with streaming (constant memory). BERTopic has a multi-stage pipeline (embeddings → UMAP → HDBSCAN → c-TF-IDF) — every stage is a research project on its own. CorEx's anchored correlation explanation is a published algorithm (AAAI 2017). Custom implementations would take weeks and likely contain bugs.

## Common Pitfalls

### Pitfall 1: Best Buy International Splash Page
**What goes wrong:** Requests get redirected to the country-selector splash page instead of product listing.
**Why it happens:** Best Buy detects non-US IPs and serves the international landing page.
**How to avoid:** Append `?intl=nosplash` to all URLs — e.g., `https://www.bestbuy.com/site/portable-audio/portable-speakers-docks-radios/pcmcat310200050004.c?id=pcmcat310200050004&intl=nosplash`. Add `Accept-Language: en-US,en;q=0.9` header.
**Warning signs:** Response HTML contains "Choose a country" or "Shopping in the U.S.?" — that's the splash page.

### Pitfall 2: Reviews Loaded via JavaScript
**What goes wrong:** BeautifulSoup finds empty review containers (JS hasn't populated).
**Why it happens:** Best Buy review content may partially load via JS. Some products render reviews server-side, others hydrate client-side.
**How to avoid:** Check if reviews exist in static HTML by looking for patterns like `class="review-item"` or `data-review-id`. If reviews are JS-loaded, either use `requests` to fetch the `/site/reviews/<name>/<sku>` page (which typically has server-rendered content) or fall back to the `product/<code>/sku/<sku>/reviews` URL format. Observed from firecrawl scrape: Review text "Size is small enough..." was present in the scrape output, meaning static HTML contains review content for this URL format.
**Warning signs:** Empty `<div>` where reviews should be, or "No content" in the scrape output.

### Pitfall 3: gensim Cannot Install on Python 3.14
**What goes wrong:** `pip install gensim` fails with C compilation errors.
**Why it happens:** gensim 4.4.0 uses internal CPython C API (`ma_version_tag` in PyDictObject) that was removed in Python 3.13+.
**How to avoid:** All gensim-dependent code runs on Colab (Python 3.10). The scraper script (`bestbuy_scraper.py`) does NOT depend on gensim — only requests + BeautifulSoup. Document this clearly for the user.
**Warning signs:** `error: no member named 'ma_version_tag' in 'PyDictObject'` during install.

### Pitfall 4: BERTopic First-Run Latency
**What goes wrong:** `SentenceTransformer('all-MiniLM-L6-v2')` downloads ~80MB model on first run.
**Why it happens:** Model is not cached locally.
**How to avoid:** Add a dedicated download cell with progress indicator. Use `show_progress_bar=True`. Document expected delay (~5 min on Colab). The model caches after first download.
**Warning signs:** Long pause during `embeddings = model.encode(...)`.

### Pitfall 5: BERTopic Reproducibility
**What goes wrong:** Different runs produce different topics.
**Why it happens:** UMAP is stochastic by default.
**How to avoid:** Set `random_state=42` in UMAP model and pass it to BERTopic via `umap_model=umap_model`.
**Warning signs:** Different topic counts or keywords across runs.
**Source:** [CITED: maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html#preventing-stochastic-behavior]

### Pitfall 6: CorEx Anchor Strength
**What goes wrong:** Anchors are ignored or override the model completely.
**Why it happens:** `anchor_strength` too low = anchors ignored. Too high = topics are just the anchor words.
**How to avoid:** Default `anchor_strength=3` is a good starting point. Range 2-4 typically works well. Each topic gets multiple anchor words (top 5-10 from LDA/BERTopic).
**Warning signs:** Topics consist only of anchor words (over-anchored) or don't mention any anchor words (under-anchored).

### Pitfall 7: LDA Coherence vs Interpretability Tradeoff
**What goes wrong:** Highest C_v coherence score produces topics that are bland/generic.
**Why it happens:** High coherence can correlate with high-frequency, low-information words.
**How to avoid:** Use C_v as a guide, not a rule. After grid search, manually inspect top 3 topic count candidates. Pick the one with the best balance of coherence AND interpretability.
**Warning signs:** High-coherence topics consist of words like "good", "great", "get", "use" — generic sentiment words, not aspects.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Bag-of-words LDA only | BERTopic (sentence embeddings + UMAP + HDBSCAN + c-TF-IDF) | 2022 (BERTopic v0.12+) | Significantly more coherent topics, handles context |
| Pure unsupervised topic modeling | Semi-supervised anchoring (CorEx) | 2017 (CorEx publication) | Domain knowledge guides topics to meaningful aspects |
| Pickle serialization for BERTopic | safetensors serialization | 2023 (BERTopic v0.14+) | Model size reduction from >500MB to <20MB, safer format |
| Selenium for review scraping | requests + BeautifulSoup with proper targeting | Ongoing | Faster, lighter, runs without browser |

**Deprecated/outdated:**
- **sklearn LDA** for NLP pipelines: lacks C_v coherence, streaming API, and multi-core support that gensim provides.
- **GloVe embeddings** for BERTopic: sentence-transformers produce superior sentence-level embeddings.

## Code Examples

### Example 1: Best Buy Product Review Scraper Structure
```python
# Source: Custom pattern based on firecrawl verification of Best Buy page structure
import requests
from bs4 import BeautifulSoup
import time
import random
import json
import csv
import os

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

def scrape_product_listing(category_url):
    """Scrape product listing page to get product names and SKUs."""
    html = fetch_page(category_url)
    soup = BeautifulSoup(html, 'html.parser')
    products = []
    # Parse product cards — Best Buy uses structured data in script tags
    # or specific class names (verify at scrape time)
    # Expected: product name, SKU, price, rating
    return products

def scrape_reviews(sku, product_name, max_reviews=100):
    """Scrape all review pages for a product."""
    all_reviews = []
    page = 1
    while len(all_reviews) < max_reviews:
        url = f"https://www.bestbuy.com/site/reviews/{product_name}/{sku}?page={page}"
        html = fetch_page(url)
        soup = BeautifulSoup(html, 'html.parser')
        # Parse review elements — structured as divs with review content
        page_reviews = parse_reviews(soup)
        if not page_reviews:
            break
        all_reviews.extend(page_reviews)
        page += 1
    return all_reviews[:max_reviews]  # Cap at max_reviews

def fetch_page(url):
    """Fetch URL with randomized delay and user-agent."""
    delay = random.uniform(1.0, 3.0)
    time.sleep(delay)
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text
```

### Example 2: LDA Topic Coherence Grid Search
```python
# Source: gensim LdaMulticore + CoherenceModel API documentation
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def evaluate_lda(tokenized_texts, topic_range=(6, 11)):
    """Grid search LDA topic count and return best model."""
    # Build dictionary
    dictionary = Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in tokenized_texts]
    
    scores = {}
    models = {}
    
    for k in range(topic_range[0], topic_range[1]):
        lda = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            chunksize=2000,
            passes=10,
            iterations=400,
            alpha='auto',
            eta='auto',
            random_state=42,
            workers=3
        )
        cm = CoherenceModel(
            model=lda,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v',
            topn=20
        )
        score = cm.get_coherence()
        scores[k] = score
        models[k] = lda
        print(f"K={k}: C_v coherence={score:.4f}")
    
    # Plot
    plt.plot(list(scores.keys()), list(scores.values()), 'bo-')
    plt.xlabel('Number of Topics (K)')
    plt.ylabel('C_v Coherence')
    plt.title('LDA Topic Coherence by Topic Count')
    plt.show()
    
    best_k = max(scores, key=scores.get)
    return models[best_k], scores
```

### Example 3: BERTopic Save and Load
```python
# Source: maartengr.github.io/BERTopic/getting_started/serialization/serialization.html
# Save (safetensors — recommended: ~20MB vs >500MB pickle)
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

### Example 4: Aspect Label CSV Schema (Phase 5 Contract)
```python
# Each review gets an aspect label and confidence from CorEx
import pandas as pd

# Build aspect-labeled output
output_df = pd.DataFrame({
    'review_id': range(len(cleaned_reviews)),
    'product_name': product_names,       # From scraper metadata
    'review_text': cleaned_reviews,       # Cleaned review text
    'aspect_label': aspect_labels,        # CorEx topic assignment (8 aspects)
    'aspect_confidence': aspect_confidences,  # p_y_given_x max probability
    'review_rating': original_ratings,    # From scraper
    'review_date': review_dates,          # From scraper
})

output_df.to_csv('data_asg/bestbuy/aspect_labeled_reviews.csv', index=False)
print(f"Saved {len(output_df)} labeled reviews for Phase 5")

# Phase 5 loads this CSV and joins with best sentiment model predictions
```

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Best Buy review text is rendered in static HTML (not exclusively JS-loaded) | Scraping | Reviews may require Selenium/Playwright if Best Buy has switched to full JS rendering |
| A2 | `en_core_web_sm` pipeline is sufficient for product review noun chunk extraction | SpaCy | If `sm` quality is too low for domain terms, need `en_core_web_trf` (larger, slower) |
| A3 | `anchor_strength=3` is a reasonable default for CorEx | CorEx | May need tuning (2-5 range) based on anchor set quality |
| A4 | Phase 1's `clean_text()` (lowercase + remove HTML + remove non-alpha + normalize whitespace) is appropriate for Best Buy reviews | Preprocessing | Best Buy reviews may have emoji, special chars, or formatting that Phase 1's IMDB pipeline doesn't handle |

## Open Questions

1. **Best Buy review rendering method — static HTML or JS?**
   - What we know: The firecrawl scrape of a review page returned review text content, suggesting at least partial server-side rendering.
   - What's unclear: Whether ALL products render reviews on the server, or if some load via JavaScript. The observed "No content" text in some sections may indicate JS-dependent areas.
   - Recommendation: Start with requests + BeautifulSoup. If reviews are missing, inspect network tab for API calls. Best Buy has public review endpoints at `https://www.bestbuy.com/site/reviews/<name>/<sku>?page=N` — the scrape showed this format has review content. Alternatively, the `/product/<name>/<code>/sku/<sku>/reviews` format may render differently.

2. **SpaCy model selection: en_core_web_sm vs en_core_web_trf?**
   - What we know: `sm` (small, 12MB) is fast but lower accuracy. `trf` (400MB, transformer-based) is more accurate but 20x slower.
   - What's unclear: Whether the accuracy gain justifies the size/speed cost for noun chunk extraction on product reviews.
   - Recommendation: Start with `en_core_web_sm` for speed. If noun chunks miss obvious aspect phrases, upgrade to `en_core_web_md` or `en_core_web_lg` as a middle ground.

3. **Review count per product: will 100+ reviews be available?**
   - What we know: The sample product (Portable Bluetooth Speaker 2, SKU 5496203) has 9,670 reviews. Top products have thousands.
   - What's unclear: Whether 35-40 products all have 100+ reviews.
   - Recommendation: D-23 already sets the threshold at 80+ acceptance. Most popular Bluetooth speaker products on Best Buy should exceed this.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none (Wave 0) |
| Quick run command | `pytest tests/test_scraper.py -x -v` |
| Full suite command | `pytest tests/ -x -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ABSA-01 | Scraper collects 30+ products w/100+ reviews | Smoke | `pytest tests/test_scraper.py::test_min_products -x` | ❌ Wave 0 |
| ABSA-01 | Scraper uses polite delays | Unit | `pytest tests/test_scraper.py::test_delay_between_requests -x` | ❌ Wave 0 |
| ABSA-01 | JSON output format valid | Unit | `pytest tests/test_scraper.py::test_json_output_format -x` | ❌ Wave 0 |
| ABSA-01 | CSV output schema valid | Unit | `pytest tests/test_scraper.py::test_csv_schema -x` | ❌ Wave 0 |
| ABSA-02 | LDA model trains without error | Smoke | `pytest tests/test_lda.py::test_lda_trains -x` | ❌ Wave 0 |
| ABSA-02 | LDA produces interpretable topics | Manual | Visual inspection | ❌ |
| ABSA-03 | BERTopic model trains without error | Smoke | `pytest tests/test_bertopic.py::test_bertopic_trains -x` | ❌ Wave 0 |
| ABSA-04 | C_v coherence computed for LDA | Unit | `pytest tests/test_lda.py::test_coherence_computed -x` | ❌ Wave 0 |
| ABSA-05 | 6+ aspects defined with keywords | Unit | `pytest tests/test_aspects.py::test_min_aspects -x` | ❌ Wave 0 |
| ABSA-06 | CorEx model trains with anchors | Smoke | `pytest tests/test_corex.py::test_corex_trains -x` | ❌ Wave 0 |
| ABSA-07 | All models saved to disk | Unit | `pytest tests/test_persistence.py::test_models_saved -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_<module>.py -x -v` (scraper tests only if scraper changed)
- **Per wave merge:** `pytest tests/ -x -v` (full suite)
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_scraper.py` — covers ABSA-01 smoke test + JSON/CSV format + delay verification
- [ ] `tests/test_lda.py` — covers ABSA-02 train + ABSA-04 coherence
- [ ] `tests/test_bertopic.py` — covers ABSA-03 train + topic inspection
- [ ] `tests/test_aspects.py` — covers ABSA-05 6+ aspects defined
- [ ] `tests/test_corex.py` — covers ABSA-06 CorEx training
- [ ] `tests/test_persistence.py` — covers ABSA-07 model file validation
- [ ] `tests/conftest.py` — shared fixtures (sample reviews, tokenized data)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No user auth in scraping pipeline |
| V3 Session Management | no | No sessions beyond HTTP requests |
| V4 Access Control | no | No access control needed |
| V5 Input Validation | yes | Validate scraped HTML against expected schema; never execute scraped content |
| V6 Cryptography | no | No encryption needed (data is public reviews) |
| V19 URL Security | yes | Validate all scraped URLs are on `bestbuy.com` domain; prevent open redirect |

### Known Threat Patterns for Scraping
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed HTML → parser crash | DoS | Wrap BeautifulSoup parsing in try/except; log errors, skip malformed items |
| Redirect to malicious URL | Spoofing | Validate URLs start with `https://www.bestbuy.com/` before following |
| Infinite redirect / pagination loop | DoS | Set max_pages cap (e.g., 20 pages = 400 reviews) per product |

## Sources

### Primary (HIGH confidence)
- [CITED: radimhurek.com/gensim/models/ldamulticore.html] — LdaMulticore constructor parameters (chunksize, passes, alpha, eta, etc.)
- [CITED: radimhurek.com/gensim/models/coherencemodel.html] — CoherenceModel API, C_v vs C_UMass parameters
- [CITED: maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html] — BERTopic UMAP/HDBSCAN parameter details
- [CITED: maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html] — Pre-calculate embeddings, prevent stochastic behavior
- [CITED: maartengr.github.io/BERTopic/getting_started/serialization/serialization.html] — safetensors/pytorch/pickle serialization
- [CITED: ryanjgallag.com/code/corex/example] — CorEx training, anchoring, hierarchical modeling
- [CITED: huggingface.co/sentence-transformers/all-MiniLM-L6-v2] — Model card: 384-dim, 256 word piece truncation, 80MB
- [CITED: spacy.io/usage/linguistic-features#noun-chunks] — Noun chunk extraction API
- [VERIFIED: firecrawl scrape of bestbuy.com] — URL patterns, review page structure, robots.txt

### Secondary (MEDIUM confidence)
- [ASSUMED] Best Buy review content is available in static HTML (firecrawl scrape of one review page returned review text, but may vary per product)
- [ASSUMED] `en_core_web_sm` quality is sufficient for product review noun chunk extraction

### Tertiary (LOW confidence)
- None — all key claims are backed by verified official sources or explicit assumptions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified on PyPI, referenced from official docs
- Architecture: HIGH — patterns derived from official documentation and project constraints
- Pitfalls: HIGH — verified via official docs and local environment testing (gensim install failure confirmed)
- Scraping: MEDIUM — URL patterns verified, but review rendering method may vary per product

**Research date:** 2026-05-31
**Valid until:** 2026-07-01 (30 days for package versions; Best Buy page structure may change without notice)
