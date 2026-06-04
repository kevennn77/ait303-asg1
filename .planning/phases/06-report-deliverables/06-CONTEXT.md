# Phase 6: Report & Deliverables — Context

**Gathered:** 2026-06-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Write the academic PDF report (10–15 pages), finalize notebooks for submission, organize cloud deliverables, and submit to Moodle.

**Requirements:** RPT-01 to RPT-08, CODE-01 to CODE-03, DLVR-01 to DLVR-04

**Dependencies:** Phase 2 (SVM results), Phase 3 (BiGRU results), Phase 4 (aspect extraction results), Phase 5 (ranking results)

</domain>

<decisions>
## Implementation Decisions

### Report Structure & Section Depth
- **D-36:** **10–15 page academic report**, Microsoft Word format, exported to PDF for Moodle submission
- **D-37:** Report sections (mapped to RPT requirements):
  1. **Introduction** (~1 page) — Overview, task description, datasets
  2. **Methodology** (~3 pages) — Preprocessing, SVM/BiGRU, scraping, aspect extraction, labeling/ranking pipeline
  3. **Results — Sentiment Models** (~2 pages) — 4-model comparison + BiGRU results + confusion matrices
  4. **Results — Aspect Extraction** (~2 pages) — LDA coherence, BERTopic topics, CorEx anchored aspects, keyword mapping
  5. **Results — Product Rankings** (~2 pages) — Top 5 per aspect tables, best product profile, bar charts
  6. **Discussion** (~2 pages) — Model performance analysis, aspect extraction quality, challenges (domain shift, aspect imbalance, small denominators)
  7. **Conclusion** (~1 page) — Summary, limitations, future work
  8. **References** (~1 page) — Dataset citations, library references

### Tables & Charts to Include
- **D-38:** **Full set of result tables** — all selected:
  - 4-model SVM comparison table (Test Accuracy, F1, CV Mean F1)
  - 2-model BiGRU comparison (CBOW vs Skip-Gram)
  - Confusion matrix heatmaps for best SVM and best BiGRU
  - LDA coherence plot (K=6 to K=10, best K=9 with C_v=0.441)
  - BERTopic summary (48 topics, 1,051 outliers)
  - CorEx results table (8 anchored aspects, TC=13.19)
  - Product ranking tables (Top 5 per aspect across 7 aspects)
  - Facet grid bar charts (7 aspects × 5 products)
  - Best product profile chart (Altec Lansing Jolt Mini Lifejacket)

### Deliverables & Submission
- **D-39:** **Report written in Microsoft Word**, exported to PDF for Moodle
- **D-40:** **Cloud upload priority**: OneDrive (primary) → Google Drive (secondary)
- **D-41:** **GitHub repo link** included in the report
- **D-42:** Cloud folder contains: 5 notebooks, model files, CSVs (labeled_reviews, aspect_scores, product_rankings, best_products), PNG charts (3), bestbuy_scraper.py
- **D-43:** **Submission to Moodle**: PDF report with cloud storage link highlighted

### the Agent's Discretion
- Exact page layout, font choice, and formatting details in Word
- Specific chart styling and color scheme for report figures
- Table formatting (font size, column widths, borders)
- Reference citation style (APA/IEEE — student's choice)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Assignment Spec
- `assignment.md` — Full assignment brief defining all requirements

### Project Planning
- `.planning/PROJECT.md` — Project definition, key decisions, constraints
- `.planning/REQUIREMENTS.md` — Full requirements traceability matrix (RPT-01 through DLVR-04)
- `.planning/ROADMAP.md` — Phase breakdown, dependencies, success criteria
- `.planning/phases/06-report-deliverables/06-PLAN.md` — Existing plan to be updated

### Phase Results (Data Sources for Report)
- `svm_sentiment_models.ipynb` — SVM model comparison table, confusion matrices (Cells 27, 32-35)
- `bigru_sentiment_models.ipynb` — BiGRU aggregate metrics, confusion matrices (Cells 38, 46)
- `notebooks/aspect_extraction.ipynb` — LDA coherence scores, BERTopic summary, CorEx results (Cells 19, 37, 44, 52)
- `notebooks/product_ranking.ipynb` — Rankings, composite scores, best product profile (Cells 21, 24, 31, 32)

</canonical_refs>

<code_context>
## Existing Code Insights

### Data Sources for Report
All notebooks have been executed on Colab with outputs. Key cell locations for report data:

**SVM Results** (`svm_sentiment_models.ipynb`):
- Cell 32: Full 4-model comparison table with Accuracy/Precision/Recall/F1/CV Mean F1
- Cell 33: Best model identification (TFIDF + Lemmatized, F1=0.9091)
- Cell 27: Confusion matrix heatmaps for all 4 variants
- Cells 29-30: Top informative features per variant

**BiGRU Results** (`bigru_sentiment_models.ipynb`):
- Cell 38: Aggregate metrics table (CBOW vs Skip-Gram)
- Cell 46: Best BiGRU variant: Skip-Gram-BiGRU (Mean F1=0.8994)
- Cells 31-35: Per-fold metrics

**Aspect Extraction** (`notebooks/aspect_extraction.ipynb`):
- Cell 19: LDA coherence scores (best K=9, C_v=0.441)
- Cell 37: BERTopic topics (48 topics, 1,051 outliers)
- Cell 44: CorEx total correlation (TC=13.19)
- Cell 52: Phase summary with all model stats

**Product Rankings** (`notebooks/product_ranking.ipynb`):
- Cell 21: Top 5 per aspect with positive ratios
- Cell 24: Ranking summary table
- Cell 31: Composite scores (Best: Altec Lansing Jolt Mini Lifejacket, 0.900)
- Cell 32: Best product detailed profile

</code_context>

<specifics>
## Specific Ideas

- Use the exact numbers from notebook outputs (not screenshots) for report tables
- Include bar charts as embedded images exported from the notebook
- GitHub repo: `https://github.com/kevennn77/ait303-asg1`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 6-Report & Deliverables*
*Context gathered: 2026-06-04*
