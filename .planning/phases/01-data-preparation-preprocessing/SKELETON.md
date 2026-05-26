# Walking Skeleton — AIT303 Assignment 1: Aspect-Based Sentiment Analysis

**Phase:** 1 — Data Preparation & Preprocessing
**Generated:** 2026-05-26

## Capability Proven End-to-End

A Jupyter Notebook loads the IMDB 50K movie reviews CSV, inspects its structure and class balance, applies a standard text cleaning pipeline (lowercase, HTML removal, punctuation removal, whitespace normalization), and produces two parallel preprocessing outputs — Porter-stemmed text and POS-aware WordNet-lemmatized text — stored as DataFrame columns ready for downstream model training.

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Notebook Runtime** | Jupyter Notebook (.ipynb) | Required by assignment spec (DLVR-01). All code in notebook per D-05. |
| **Data Loading** | pandas `read_csv` from local `data/` directory | Dataset acquired manually via Kaggle web UI per D-08. CSV read into DataFrame with vectorized operations. |
| **Text Cleaning** | Python `re` module (regex) | No BeautifulSoup needed — IMDB reviews only contain `<br />` tags. `re.sub()` is faster and zero-dependency. |
| **Tokenization** | NLTK `word_tokenize` (Punkt tokenizer) | Handles contractions, punctuation boundaries, and sentence splitting. Lighter than spaCy (~500MB download avoided per D-02). |
| **Stopword Removal** | NLTK `stopwords.words('english')` (~179 words) | Academic standard. Domain-specific additions explicitly excluded per D-07. |
| **Stemming** | NLTK `PorterStemmer` | Algorithmic suffix stripping. Standard/academic default per D-04. |
| **Lemmatization** | NLTK `WordNetLemmatizer` with `pos_tag` | Dictionary-based lemmatization. POS tagging enabled per D-03 for verb/adjective accuracy. Not spaCy (avoids ~500MB model per D-02). |
| **Output Format** | Space-joined strings in DataFrame columns | Compatible with both sklearn `CountVectorizer`/`TfidfVectorizer` (Phase 2) and gensim word embeddings (Phase 3). |
| **Data Directory Layout** | `data/IMDB Dataset.csv` → DataFrame in-memory → 3 new columns | Raw file never modified. Preprocessing results stored in-memory for this phase, no intermediate files. |
| **Version Control** | `.gitignore` excludes `data/` large files | 66MB CSV excluded from git; manual download required. |

## Stack Touched in Phase 1

- [x] **Notebook runtime** — `sentiment_analysis_preprocessing.ipynb` created with 5 sections
- [x] **Data loading** — `data/IMDB Dataset.csv` read via pandas into DataFrame
- [x] **Data validation** — shape, missing values, class balance verified
- [x] **Text cleaning** — regex pipeline applied to all 50K reviews
- [x] **Stemming** — PorterStemmer via NLTK
- [x] **Lemmatization** — WordNetLemmatizer + POS tagging via NLTK
- [x] **Output columns** — `cleaned`, `stemmed`, `lemmatized` added to DataFrame
- [ ] *Manual download required:* Kaggle CSV must be placed at `data/IMDB Dataset.csv`

## Out of Scope (Deferred to Later Phases)

- Feature extraction (CountVectorizer/TfidfVectorizer) → Phase 2
- Word embedding training (CBOW/Skip-Gram) → Phase 3
- Model training (SVM, BiGRU) → Phases 2, 3
- Cross-validation and evaluation → Phases 2, 3
- Web scraping (Best Buy product reviews) → Phase 4
- Aspect extraction (LDA, BERTopic, CorEx) → Phase 4
- Product ranking and visualization → Phase 5
- Report writing → Phase 6

## Subsequent Slice Plan

Each later phase adds one vertical slice on top of this skeleton:

- **Phase 2:** SVM sentiment models — consumes `cleaned`/`stemmed`/`lemmatized` columns for CountVectorizer and TfidfVectorizer feature extraction, trains both SVM variants with cross-validation.
- **Phase 3:** BiGRU sentiment models — consumes same columns for CBOW and Skip-Gram word embedding training, builds and evaluates both BiGRU variants.
- **Phase 4:** Web scraping & aspect extraction — scrapes Best Buy speaker reviews, trains LDA, BERTopic, and CorEx aspect models.
- **Phase 5:** Labeling, ranking & visualization — applies best sentiment model + CorEx to scraped reviews, ranks products, creates bar charts.
- **Phase 6:** Report & deliverables — compiles everything into the final academic PDF and cloud submission.

## Local Run Command

```bash
# Install dependencies (one-time)
pip install nltk pandas numpy jupyter

# Ensure CSV is in place
ls -la data/IMDB Dataset.csv

# Run the notebook
jupyter nbconvert --to notebook --execute sentiment_analysis_preprocessing.ipynb --output outputs/preprocessing_results.ipynb

# Or open interactively
jupyter notebook sentiment_analysis_preprocessing.ipynb
```
