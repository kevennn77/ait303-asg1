---
phase: "04"
slug: web-scraping-aspect-extraction
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-31
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Manual notebook execution + file existence checks |
| **Config file** | none — notebook-driven validation |
| **Quick run command** | `python bestbuy_scraper.py --dry-run` |
| **Full suite command** | Execute `04-aspect-extraction.ipynb` end-to-end in Colab |
| **Estimated runtime** | ~15-20 min (BERTopic first-run includes sentence-transformers download) |

---

## Sampling Rate

- **After scraper script:** Validate CSV output — verify 30+ products, 80+ reviews/product
- **After LDA training:** Validate coherence score and topic interpretability
- **After BERTopic training:** Validate topic count and topic coherence
- **After CorEx training:** Validate aspect assignments CSV
- **Before phase sign-off:** Full notebook re-execution check

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Validation Method | Verified |
|---------|------|------|-------------|-------------------|----------|
| scraper-script | 01 | 1 | ABSA-01 | File existence: `bestbuy_scraper.py`, `data/products/*.json`, `data/reviews.csv` | ⬜ pending |
| scraper-execution | 01 | 1 | ABSA-01 | CSV row count >= 2350 (30×80), unique products >= 30 | ⬜ pending |
| notebook-scaffold | 02 | 1 | — | Notebook cells execute without import errors | ⬜ pending |
| preprocessing | 02 | 2 | ABSA-01 | Cleaned review text has no HTML tags, under 50 chars after cleaning | ⬜ pending |
| keyphrase-extraction | 02 | 2 | ABSA-01 | SpaCy noun chunks extracted; at least 3 noun phrases per review | ⬜ pending |
| lda-training | 03 | 2 | ABSA-02 | LDA model saved; coherence >= 0.3 C_v | ⬜ pending |
| lda-visualization | 03 | 2 | ABSA-04 | Topic word tables and pyLDAvis rendered | ⬜ pending |
| bertopic-training | 04 | 3 | ABSA-03 | BERTopic model saved; topic count between 6-10 | ⬜ pending |
| bertopic-visualization | 04 | 3 | ABSA-04 | Topic word tables and inter-topic distance map | ⬜ pending |
| keyword-analysis | 05 | 3 | ABSA-04, ABSA-05 | 5+ keywords extracted per topic from both models | ⬜ pending |
| aspect-mapping | 05 | 3 | ABSA-05 | Keyword-to-aspect mapping table documented | ⬜ pending |
| corex-training | 06 | 4 | ABSA-06 | CorEx model saved; at least 6 of 8 anchored aspects present | ⬜ pending |
| aspect-csv-export | 06 | 4 | ABSA-07 | `data/aspect_labels.csv` exists with required columns | ⬜ pending |
| model-persistence | 06 | 4 | ABSA-07 | LDA (.pkl), BERTopic (.safetensors), CorEx (.pkl) all saved | ⬜ pending |

---

## Wave 0 Requirements

- [ ] `bestbuy_scraper.py` — standalone Python scraper script
- [ ] `04-aspect-extraction.ipynb` — Jupyter Notebook for all modeling

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Topic interpretability | ABSA-04 | Requires human judgment on topic coherence and meaningfulness | Read each topic's top 10 keywords — do they form a coherent theme? |
| Aspect mapping quality | ABSA-05 | Automated mapping requires human review per D-31 | Review automated keyword→aspect assignments; correct obvious errors |
| Review count adequacy | ABSA-01 | Depends on Best Buy page availability | Count unique products and reviews after scrape — must meet 30+ / 80+ thresholds |

---

## Validation Sign-Off

- [ ] All tasks have explicit verification criteria
- [ ] Scraper CSV output meets count thresholds
- [ ] LDA coherence >= 0.3 C_v
- [ ] BERTopic produces 6-10 interpretable topics
- [ ] CorEx produces 6+ anchored aspects
- [ ] Aspect labels CSV ready for Phase 5
- [ ] All 3 models saved to disk

**Approval:** pending
