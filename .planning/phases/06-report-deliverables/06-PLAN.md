---
phase: 06-report-deliverables
plan: 01
type: execute
requirements: [RPT-01 to RPT-08, CODE-01 to CODE-03, DLVR-01 to DLVR-04]
---

# Phase 6 Plan: Report & Deliverables

## Objective

Finalize all notebooks, produce the submission-ready PDF report, and organize cloud deliverables. Phase 6 is a **writing and cleanup phase** — no new ML models or analysis.

## Deliverables

| # | Artifact | Description |
|---|----------|-------------|
| 1 | 5 Jupyter Notebooks | Clean, well-commented, section-numbered, no stale errors |
| 2 | PDF Report | Academic PDF (student writes, guided by outline below) |
| 3 | Cloud Upload | Notebooks + models + CSVs + charts uploaded to Google Drive/OneDrive |
| 4 | Submission Package | All files organized for Moodle submission |

## Task Breakdown

### Task A: Notebook Cleanup (CODE-01, CODE-02, CODE-03)

**A1 — Fix missing Section 3 (SpaCy Keyphrase Extraction) in aspect_extraction.ipynb**
- Add a new Section 3 between the current Sections 2 and 4
- Implement the keyphrase extraction function using SpaCy `noun_chunks` and `ents`
- Insert 3 cells: 1 markdown header + 1 code cell (define & apply function) + 1 code cell (show sample output)
- Content matches the pattern from PATTERNS.md lines 306-327

**A2 — Fix section numbering in bigru_sentiment_models.ipynb**
- Rename `## Summary` → `## 8. Summary`

**A3 — Add student name to all notebooks**
- Verify all 5 notebooks have `[Student Name]` placeholder in Cell 0
- Add if missing

**A4 — Clear stale outputs in all notebooks**
- For `product_ranking.ipynb`: output-heavy (embedded PNGs, ~6MB). Clear all outputs before final save.
- For `aspect_extraction.ipynb`: same, clear outputs.
- For `svm_sentiment_models.ipynb` and `bigru_sentiment_models.ipynb`: clear outputs.
- Keep `sentiment_analysis_preprocessing.ipynb` clean.

### Task B: Report Writing (student task — guided by outline)

The student writes a PDF report (~10-15 pages) with the following structure:

```
Title Page
  - AIT303 Assignment 1
  - Aspect-Based Sentiment Analysis
  - Student Name, Student ID, Date

1. Introduction
   - Overview of sentiment analysis and ABSA
   - Task description: 4 sentiment models + aspect extraction on speaker reviews
   - Dataset: IMDB 50K (sentiment), Best Buy speakers (aspects)

2. Methodology
   2.1 Data Preparation & Preprocessing
       - IMDB loading, class balance check
       - Pipeline: lowercase → HTML removal → punctuation removal → stopword removal → tokenization → lemmatization
   2.2 Sentiment Models (Phase 2 & 3)
       - SVM variants: CountVectorizer + TfidfVectorizer × (stemmed + lemmatized)
       - BiGRU variants: CBOW + Skip-Gram embeddings
       - 5-fold Stratified K-Fold CV
   2.3 Web Scraping (Phase 4)
       - curl_cffi with TLS impersonation bypasses Akamai
       - Hidden API: /ugc/v2/reviews
       - 37 products × 100 reviews = 3,199 reviews
   2.4 Aspect Extraction (Phase 4)
       - LDA: grid search K=6-10, C_v coherence
       - BERTopic: all-MiniLM-L6-v2, UMAP + HDBSCAN
       - CorEx: anchored semi-supervised with 8 aspect categories
   2.5 Labeling & Ranking (Phase 5)
       - Best SVM (TFIDF + Lemmatized) labels sentiment on Best Buy reviews
       - Per-aspect positive ratio scoring
       - Composite score: 60% positive ratio + 20% aspect coverage + 20% rating

3. Results
   3.1 Sentiment Model Comparison
       - Table: 4 models with accuracy, precision, recall, F1
       - Confusion matrices
       - Best model: TfidfVectorizer + Lemmatized (F1=0.9091)
   3.2 Aspect Extraction Results
       - LDA: coherence scores, top keywords, pyLDAvis visualization
       - BERTopic: topic clusters, inter-topic distance
       - CorEx: 8 anchored aspects, TC=13.19, topic quality per aspect
       - Keyword-to-aspect mapping table
   3.3 Product Rankings
       - Top 5 per aspect table (7 aspects)
       - Best product: Altec Lansing Jolt Mini Lifejacket
         - Composite score: 0.900
         - 100% positive on Build Quality and Sound Quality

4. Discussion
   4.1 Model Performance
       - Why SVM outperformed BiGRU on this dataset
       - Domain shift (IMDB → Best Buy) caveat
   4.2 Aspect Extraction Quality
       - LDA vs BERTopic vs CorEx
       - Strengths of semi-supervised (anchored) approach
   4.3 Challenges
       - Web scraping with Akamai bot protection
       - CorEx aspect imbalance (Comfort/Portability dominates)
       - Small denominators for some products

5. Conclusion
   - Summary of findings
   - Limitations and future work

References
   - IMDB dataset (Maas et al., 2011)
   - gensim, BERTopic, CorEx library citations
   - scikit-learn
```

### Task C: Deliverables Organization (DLVR-01 to DLVR-04)

**C1 — Create cloud upload structure**
```
cloud-upload/
├── notebooks/
│   ├── sentiment_analysis_preprocessing.ipynb
│   ├── svm_sentiment_models.ipynb
│   ├── bigru_sentiment_models.ipynb
│   ├── notebooks/aspect_extraction.ipynb
│   └── notebooks/product_ranking.ipynb
├── models/
│   ├── lda_model/              (LDA gensim .save)
│   ├── bertopic_model/          (safetensors)
│   ├── corex_model.pkl          (CorEx pickle)
│   ├── word2vec_cbow.model
│   ├── word2vec_sg.model
│   ├── bigru_cbow_fold1-5.h5   (5 folds)
│   └── bigru_sg_fold1-5.h5     (5 folds)
├── data/
│   ├── aspect_labeled_reviews.csv
│   ├── labeled_reviews.csv
│   ├── product_rankings.csv
│   ├── aspect_scores.csv
│   └── best_products.csv
├── charts/
│   ├── product_rankings_chart.png
│   ├── sentiment_by_aspect.png
│   └── best_product_profile.png
└── bestbuy_scraper.py
```

**C2 — Submission checklist**
- [ ] All 5 notebooks run end-to-end on Colab
- [ ] PDF report uploaded to Moodle
- [ ] Cloud storage folder shared (view-only link)
- [ ] Link to cloud folder highlighted in report
- [ ] Jupyter Notebook has clear comments and section headers
- [ ] Code has proper variable names and readable structure

## Verification

| Check | Method |
|-------|--------|
| All notebooks have 0 syntax errors | `ast.parse()` on each notebook |
| Section numbering is sequential (1..N) | Manual check |
| Student name placeholder present | grep for `[Student Name]` |
| Report covers all RPT-01 to RPT-08 | Manual review |
| Deliverable files exist | `ls` review |

## Success Criteria

1. All 5 notebooks have clean, consistent section numbering (CODE-02 ✓)
2. All notebooks have clear comments (CODE-01 ✓) and proper variable names (CODE-03 ✓)
3. Report outline covers all RPT requirements
4. Deliverable checklist complete
5. All files ready for cloud upload + Moodle submission
