# Phase 4: Web Scraping & Aspect Extraction — Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Scrape 30+ Bluetooth & Wireless speaker products from Best Buy (100+ reviews each) using a Python scraper with polite delays, then perform unsupervised and semi-supervised aspect extraction using LDA, BERTopic, and CorEx to produce 6+ meaningful product aspects for downstream ranking.

**Requirements:** ABSA-01, ABSA-02, ABSA-03, ABSA-04, ABSA-05, ABSA-06, ABSA-07

**Dependencies:** None — standalone data collection phase. Produces scraped review data consumed by Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Scraping Strategy
- **D-19:** **Standalone Python scraper script** (`bestbuy_scraper.py`) separate from the notebook. Clean separation for debugging, retry, and polite delay management. Scraped data loaded into the notebook from cached files.
- **D-20:** Target **Best Buy Bluetooth & Wireless Speakers** category — largest category with broadest price range and sufficient product/review volume.
- **D-21:** Anti-detection: **Randomized 1-3 second delays + rotating user-agent headers**. Sufficient for Best Buy which isn't aggressively anti-scraper.
- **D-22:** Scraped data stored as **structured JSON per product** (full metadata: name, rating, price, specs) + **consolidated CSV** (all reviews: product, review_text, rating, date) for easy notebook loading.
- **D-23:** **Scrape 35-40 products**, accept products with **80+ reviews** if some pages have fewer. Ensures 30+ products with 100+ reviews even with some failures.

### Preprocessing for Aspect Extraction
- **D-24:** **Reuse Phase 1 IMDB cleaning pipeline** — lowercasing, HTML removal, punctuation removal, stopword removal, lemmatization. Keeps preprocessing consistent across the project.
- **D-25:** **Product metadata NOT prepended** to review text. Review text only for topic modeling. Product metadata stored separately in JSON and joined later in Phase 5.
- **D-26:** **Use SpaCy keyphrase extraction** — extract noun chunks and named entities before topic modeling. Filters noise and focuses LDA/BERTopic on meaningful aspect candidates (e.g., "battery life", "sound quality").

### Topic Model Configuration
- **D-27:** **LDA: try 6-10 topics, pick best via topic coherence** (C_v or C_UMass). Balances exploration with interpretability.
- **D-28:** **BERTopic embedding model: `all-MiniLM-L6-v2`** — lightweight (80MB), fast inference, good semantic quality for review text.
- **D-29:** Target **6-10 interpretable topics per model** to cover Design, Sound Quality, Price, Build, Features, Connectivity, Comfort, Battery Life.

### Aspect Restructuring Process
- **D-30:** **Predefined aspect categories** — define 6-10 candidate aspects upfront (Design, Sound Quality, Battery, Price, Build Quality, Features, Connectivity, Comfort). LDA/BERTopic keywords mapped to these categories for consistent, interpretable outputs.
- **D-31:** **Automated keyword-to-aspect mapping + human review** — automatically assign keywords to best-matching predefined aspect via word/semantic similarity. Human reviews and corrects obvious errors.
- **D-32:** **Allow multi-aspect membership** — keywords that fit multiple aspects are listed under all applicable categories. CorEx handles multi-aspect assignment at the review level.

### CorEx Anchoring & Phase 5 Integration
- **D-33:** **Top keywords from LDA/BERTopic as CorEx anchors** — take top 5-10 keywords per topic for each predefined aspect. Automates anchor creation from unsupervised outputs.
- **D-34:** **Anchor all 6+ predefined aspects** in CorEx. Gives clear guidance for semi-supervised extraction.
- **D-35:** **Shared CSV with aspect labels** as Phase 5 contract — CorEx assigns each review to an aspect cluster. Save aspect assignments + review IDs in CSV. Phase 5 loads this, joins with best sentiment model predictions (Phase 2/3), and computes per-aspect sentiment scores.

### the Agent's Discretion
- Specific SpaCy pipeline selection for keyphrase extraction
- LDA and BERTopic hyperparameter details (beyond topic count and embedding model)
- CorEx hyperparameters (beyond anchor selection strategy)
- Visualization style for topic coherence and topic outputs
- Notebook section structure and organization
- Specific Python library versions (topic model libraries are standard PyPI packages)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Assignment Spec
- `assignment.md` — Full assignment brief defining all requirements for all tasks

### Project Planning
- `.planning/PROJECT.md` — Project definition, key decisions, constraints
- `.planning/REQUIREMENTS.md` — Full requirements traceability matrix (ABSA-01 through ABSA-07)
- `.planning/ROADMAP.md` — Phase breakdown, dependencies, success criteria

### Phase 1-3 Artifacts
- `.planning/phases/03-bigru-sentiment-models/03-CONTEXT.md` — Phase 3 locked decisions
- `sentiment_analysis_preprocessing.ipynb` — Phase 1 preprocessing pipeline (reused for review cleaning)
- `_create_notebook.py` — Utility script for programmatic notebook creation using nbformat

### Prior Patterns to Follow
- `.planning/phases/02-svm-sentiment-models/02-CONTEXT.md` — Phase 2 decisions (Colab patterns, data structure)
- `svm_sentiment_models.ipynb` — Phase 2 notebook (Colab setup pattern, Drive mount, section structure)
- `bigru_sentiment_models.ipynb` — Phase 3 notebook (model persistence, evaluation patterns)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_create_notebook.py` — Python script for programmatic Jupyter notebook creation using `nbformat`. Can be adapted for Phase 4 notebook scaffolding.

### Established Patterns
- **Notebook-first approach** — All deliverables are Jupyter Notebooks with structured markdown sections and code cells
- **Colab execution** — `COLAB=True` flag, Google Drive mount, structured `data_asg/` data directory. Phase 4 may run locally (scraping) or on Colab (BERTopic downloads)
- **`_create_notebook.py` construction** — Scripted notebook creation via `nbformat` used for Phase 2. Can reuse for Phase 4.

### Creative Options
- Phase 4 could be a single notebook (scraping→preprocessing→topic models) or split into data collection + modeling notebooks
- SpaCy keyphrase extraction could run as a preprocessing step or inline in the modeling notebook
- Topic coherence evaluation could use gensim's `CoherenceModel` for interpretable metrics

</code_context>

<specifics>
## Specific Ideas

- Use `gensim.models.LdaMulticore` for LDA training (multi-core parallelization)
- Use `bertopic.BERTopic` with `all-MiniLM-L6-v2` sentence-transformer backend
- Use `corex_topic.CorEx` for semi-supervised aspect extraction with anchored keywords
- Evaluate LDA/BERTopic with gensim's `CoherenceModel` (C_v metric)
- Save all aspect models to disk: LDA (.pkl), BERTopic (.pkl or via native save), CorEx (.pkl)
- Aspect assignment CSV schema: `review_id, product_name, review_text, aspect_label, aspect_confidence`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 4-Web Scraping & Aspect Extraction*
*Context gathered: 2026-05-31*
