# Phase 4: Web Scraping & Aspect Extraction - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-31
**Phase:** 4-Web Scraping & Aspect Extraction
**Areas discussed:** Scraping approach, Preprocessing for aspect extraction, Topic model configuration, Aspect restructuring process, CorEx anchoring & integration

---

## Scraping Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone script + notebook | Separate Python scraper script for clean separation | ✓ |
| All in notebook | Scraping code lives directly in notebook cells | |
| Bluetooth & Wireless Speakers | Largest category with broadest price range | ✓ |
| Home Speakers | Fewer products but detailed reviews | |
| Multi-room / Smart speakers | Fewer distinct products, higher per-product review count | |
| Randomized delays + user-agent rotation | 1-3s random delays + rotating headers | ✓ |
| Full stealth mode | Delays + rotation + residential proxies + request randomization | |
| Structured JSON per product + CSV | Each product as JSON file + consolidated CSV | ✓ |
| Single JSON file | All products + reviews in one JSON file | |
| Scrape 35-40, keep min 80 reviews | Accept 80+ if some pages have fewer | ✓ |
| Strict cutoff — drop below 100 | Only keep products meeting exact threshold | |

**User's choice:** Standalone script + notebook, Bluetooth & Wireless Speakers, Randomized delays + user-agent rotation, JSON per product + CSV, Scrape 35-40 with 80+ minimum
**Notes:** User confirmed all recommended options without additional clarifications. Pragmatic approach — sufficient for the assignment's review volume needs.

---

## Preprocessing for Aspect Extraction

| Option | Description | Selected |
|--------|-------------|----------|
| Same pipeline as IMDB | Reuse Phase 1: lowercasing, HTML removal, punctuation, stopwords, lemmatization | ✓ |
| Lighter cleaning | Skip stopword removal and lemmatization | |
| No — reviews only | Topic model on review text alone | ✓ |
| Prepend product name | Add "Product: [Name]" to review text | |
| Yes — extract noun phrases | Use SpaCy NER/noun chunks before topic models | ✓ |
| No — raw text | Feed cleaned review text directly | |

**User's choice:** Same pipeline as IMDB, Reviews only (no metadata), Yes to SpaCy keyphrase extraction
**Notes:** User wants preprocessing consistency with earlier phases. Clean separation between review text and metadata. Keyphrase extraction to improve topic quality.

---

## Topic Model Configuration

| Option | Description | Selected |
|--------|-------------|----------|
| Try 6-10 topics, pick best via coherence | Run LDA with 6-10 topics, evaluate coherence | ✓ |
| Fixed at 8 topics | Single run at 8 topics | |
| all-MiniLM-L6-v2 (BERTopic) | Lightweight sentence-transformer (80MB) | ✓ |
| all-mpnet-base-v2 (BERTopic) | Higher quality but slower (420MB) | |
| 6-10 interpretable topics per model | Cover Design, Quality, Price, etc. | ✓ |
| 10-15 topics, then filter | Generate more, filter during restructuring | |

**User's choice:** Coherence-based topic selection, all-MiniLM-L6-v2 for BERTopic, 6-10 interpretable topics target
**Notes:** Pragmatic balance of quality and speed for all models.

---

## Aspect Restructuring Process

| Option | Description | Selected |
|--------|-------------|----------|
| Predefined aspect categories | Define 6-10 aspects upfront | ✓ |
| Data-driven — topics define aspects | Let topics determine categories organically | |
| Automated mapping + human review | Word/semantic similarity + human correction | ✓ |
| Fully manual mapping | Hand-inspect and assign all keywords | |
| Allow multi-aspect membership | Keywords under all applicable aspects | ✓ |
| Strict one-aspect-per-keyword | Single best-match assignment | |

**User's choice:** Predefined categories, Automated + human review, Multi-aspect membership allowed
**Notes:** Structured approach with efficiency of automation and safety of human oversight. Predefined categories ensure consistent Phase 5 output.

---

## CorEx Anchoring & Integration

| Option | Description | Selected |
|--------|-------------|----------|
| Top keywords from LDA/BERTopic as anchors | Top 5-10 keywords per topic as CorEx anchors | ✓ |
| Manually curated anchor words | Human-picked 3-5 anchos per aspect | |
| All 6+ predefined aspects | Anchor all aspects in CorEx | ✓ |
| Subset — let CorEx discover others | Anchor 4 core, let CorEx discover the rest | |
| Shared CSV with aspect labels | Save aspect assignments + review IDs for Phase 5 | ✓ |
| Phase 5 re-runs CorEx predictions | Load CorEx model and re-predict in Phase 5 | |

**User's choice:** Keywords from LDA/BERTopic as anchors, All predefined aspects, Shared CSV contract with Phase 5
**Notes:** Clean pipeline from unsupervised → semi-supervised → Phase 5 labeling via intermediate CSV.

---

## the agent's Discretion

- Specific SpaCy pipeline selection for keyphrase extraction
- LDA and BERTopic hyperparameter details (beyond topic count and embedding model)
- CorEx hyperparameters (beyond anchor selection strategy)
- Visualization style for topic coherence and topic outputs
- Notebook section structure and organization
- Specific Python library versions
- Whether to split into separate data collection vs modeling notebooks

## Deferred Ideas

None — discussion stayed within phase scope.
