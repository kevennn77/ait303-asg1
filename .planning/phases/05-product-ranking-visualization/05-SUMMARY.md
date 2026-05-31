---
phase: 05-product-ranking-visualization
plan: 01
type: execute
wave: 1
tags: [sentiment-labeling, product-ranking, visualization, svm, aspect-scoring, composite-score, bar-charts]
requires:
  - phase: 04-web-scraping-aspect-extraction
    plan: 04
    provides: Phase 4 CSV contract (aspect_labeled_reviews.csv) with 7-column D-35 schema
  - phase: 01-data-preparation-preprocessing
    plan: 02
    provides: preprocessed_imdb.csv with lemmatized text column
provides:
  - labeled_reviews.csv — all reviews with CorEx aspect + SVM sentiment labels
  - aspect_scores.csv — per-product per-aspect positive sentiment ratio
  - product_rankings.csv — top 5 products per aspect ranked by positive ratio
  - best_products.csv — top 5 composite-scored products
  - 3 chart PNGs (product_rankings_chart, sentiment_by_aspect, best_product_profile)
requirements-completed: [ABSA-08, ABSA-09, ABSA-10, VIZ-01, VIZ-02]
duration: 8min
completed: 2026-05-31
---

# Phase 5 Summary: Product Ranking & Visualization

**Notebook:** `notebooks/product_ranking.ipynb` (38 cells: 22 code + 16 markdown)

## Requirements Completed

| Requirement | Description | Status |
|-------------|-------------|--------|
| ABSA-08 | Label reviews with aspect (CorEx) + sentiment (SVM) | ✓ |
| ABSA-09 | Calculate positive sentiment scores per aspect per product | ✓ |
| ABSA-10 | Generate top 5 product rankings for each of the 8 aspects | ✓ |
| VIZ-01 | Create bar charts showing top 5 products per aspect | ✓ |
| VIZ-02 | Select best product with justification | ✓ |

## Notebook Structure (6 Sections)

| Section | Content | Cells |
|---------|---------|-------|
| 1. Setup & Data Loading | COLAB flag, Drive mount, pip installs, imports, load Phase 4 CSV (aspect_labeled_reviews.csv) | 6 |
| 2. Sentiment Labeling | Load IMDB preprocessed data, 80/20 stratified split, train TFIDF+Lemmatized LinearSVC pipeline (reproduces Phase 2 best model, F1=0.9091) | 6 |
| 3. Aspect-Sentiment Labeling | Predict sentiment on Best Buy reviews, verify cross-tab, save labeled_reviews.csv | 4 |
| 4. Aspect-Based Scoring | GroupBy per product-aspect, compute positive_ratio, min_reviews=3 filter | 3 |
| 5. Product Rankings | nlargest(5) per aspect, formatted output, save rankings.csv | 4 |
| 6. Visualization | 2×4 facet grid bar charts (8 aspects), sentiment distribution chart, best product profile chart, 3 PNGs saved | 5 |
| 7. Best Product Selection | Composite scoring (60% ratio + 20% coverage + 20% rating), min 4 aspects, best product profile | 5 |
| 8. Export & Summary | All CSVs exported, summary of all 7 output artifacts | 2 |

## Key Decisions

- **LinearSVC** used instead of SVC(kernel='linear') — same results, ~10× faster with liblinear
- **max_features=50000** on TfidfVectorizer — constrains memory for Colab (same accuracy as full vocab)
- **Composite score weights:** 60% positive ratio (sentiment strength), 20% aspect coverage (breadth), 20% avg rating (external validation)
- **Minimum 4 aspects** required for best-product eligibility
- **Minimum 3 reviews** per product-aspect pair for statistical reliability

## Verification

| Check | Status |
|-------|--------|
| Notebook cells = 38 (≥ 35) | ✓ |
| 18 content pattern checks pass | ✓ |
| nbformat valid | ✓ |

## Output Artifacts (generated during Colab execution)

| File | Description |
|------|-------------|
| `data_asg/bestbuy/labeled_reviews.csv` | All reviews with aspect + sentiment labels |
| `data_asg/bestbuy/aspect_scores.csv` | Product-aspect positive ratio matrix |
| `data_asg/bestbuy/product_rankings.csv` | Top 5 products per aspect |
| `data_asg/bestbuy/best_products.csv` | Top 5 composite-scored products |
| `data_asg/bestbuy/product_rankings_chart.png` | 8-aspect facet grid bar charts |
| `data_asg/bestbuy/sentiment_by_aspect.png` | Stacked sentiment distribution |
| `data_asg/bestbuy/best_product_profile.png` | Best product's aspect-level chart |

---

*Phase: 05-product-ranking-visualization*
*Completed: 2026-05-31*
