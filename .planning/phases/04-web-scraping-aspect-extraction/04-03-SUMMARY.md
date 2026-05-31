# 04-03 Summary

**Plan:** LDA + BERTopic Unsupervised Topic Modeling  
**Phase:** 04-web-scraping-aspect-extraction  
**Wave:** 2  
**Duration:** ~8 min (subagent)

## Deliverables

| Artifact | Description | Status |
|----------|-------------|--------|
| `notebooks/aspect_extraction.ipynb` | Sections 4-6 appended (19 new cells) | ✅ |

## Sections Added

### Section 4 — LDA Topic Modeling (Cells A-G)
- gensim Dictionary and corpus from tokenized reviews
- Grid search K=6-10 with `LdaMulticore` (chunksize=2000, passes=10, iterations=400, alpha='auto', eta='auto')
- C_v coherence optimization → selects best K (`best_k`, `best_lda`)
- Coherence score line plot (matplotlib)
- pyLDAvis inter-topic distance map
- Model saved via `best_lda.save()`

### Section 5 — BERTopic (Cells H-N)
- `SentenceTransformer("all-MiniLM-L6-v2")` pre-calculated embeddings
- UMAP (n_neighbors=15, n_components=5, cosine, random_state=42)
- HDBSCAN (min_cluster_size=10, eom, prediction_data=True)
- CountVectorizer (stop_words="english", ngram_range=(1,2))
- `.fit_transform()` with pre-calculated embeddings
- Barchart + inter-topic distance visualization
- Model saved via `.save(serialization="safetensors")`

### Section 6 — Topic Analysis (Cells O-S)
- LDA top 10 keywords per topic via `best_lda.show_topic()`
- BERTopic top 10 keywords per topic via `topic_model.get_topic()`
- Model comparison (coherence score, topic count, outlier count)
- Keyword inventory saved to `{MODEL_DIR}/topic_keywords.json`

## D-Constraints

- D-27: LDA grid search 6-10, C_v coherence
- D-28: all-MiniLM-L6-v2 sentence embeddings
- D-29: 6-10 interpretable topics

## Commit

- `96d9a6f` — feat(04-scraping): add LDA + BERTopic topic modeling (Sections 4-6)
- `7e61309` — fix(04-web-scraping): fix literal newlines in f-strings
