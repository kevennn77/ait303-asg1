# Phase 1: Data Preparation & Preprocessing — Context

**Gathered:** 2025-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Load the IMDB 50K movie reviews dataset and build a complete text preprocessing pipeline. This phase covers dataset acquisition, inspection, and all text cleaning steps required before feature extraction. Two parallel preprocessing outputs (stemmed and lemmatized) are produced to support model comparison in later phases.

**Requirements:** SENT-01, SENT-02, CODE-01, CODE-02, CODE-03
</domain>

<decisions>
## Implementation Decisions

### Lemmatization vs Stemming
- **D-01:** Use **both** stemming (Porter) and lemmatization (NLTK WordNet) for comparison — both preprocessing outputs flow into downstream model training (Phases 2-3) so accuracy differences can be compared
- **D-02:** Library: NLTK WordNet Lemmatizer (not spaCy — avoids ~500MB model download)
- **D-03:** POS tagging enabled for lemmatizer accuracy (noun/verb/adjective detection via NLTK POS tagger)
- **D-04:** Stemmer: NLTK PorterStemmer (standard, built-in, no extra downloads)

### Notebook Architecture
- **D-05:** Notebook-only approach — all preprocessing code lives in the Jupyter Notebook (no separate .py utility modules), with clear markdown section headers
- **D-06:** Notebook sections: 1. Setup/Imports, 2. Load & Inspect, 3. Text Cleaning Pipeline, 4. Stemming Output, 5. Lemmatization Output

### Stopword Handling
- **D-07:** Use NLTK default stopword list (~179 words) — no domain-specific additions

### Dataset Acquisition
- **D-08:** Manual download from Kaggle web UI → save CSV to `data/` directory → notebook reads from local path. Kaggle API not used.
- **D-09:** Data directory structure: `data/IMDB Dataset.csv` (raw), then preprocessed outputs saved separately

### the Agent's Discretion
- Preprocessing order within the pipeline (lowercasing → HTML removal → punctuation removal → tokenization → stopword removal → stemming/lemmatization is standard NLP order — agent can follow this)
- Specific NLTK download management (WordNet, stopwords, averaged_perceptron_tagger for POS)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Assignment Spec
- `assignment.md` — Full assignment brief defining all requirements for all tasks

### Project Planning
- `.planning/PROJECT.md` — Project definition, key decisions, constraints
- `.planning/REQUIREMENTS.md` — Full requirements traceability matrix
- `.planning/ROADMAP.md` — Phase breakdown, dependencies, success criteria

### Dataset
- https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews — IMDB 50K dataset source
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- No existing code — this is a greenfield project. First code written here serves as foundation for all later phases.

### Established Patterns
- No existing patterns — this phase establishes the preprocessing conventions that Phases 2 and 3 will consume.

### Integration Points
- Phase 1 output (preprocessed text) feeds Phase 2 (SVM feature extraction) and Phase 3 (word embeddings)
- Two parallel outputs expected: `text_cleaned_stemmed` and `text_cleaned_lemmatized`
</code_context>

<specifics>
## Specific Ideas

- Both stemmed and lemmatized outputs should be saved alongside the raw cleaned text so downstream models can be compared across preprocessing strategies
- Preprocessing sections should show before/after samples for the report discussion
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.
</deferred>

---

*Phase: 1-Data Preparation & Preprocessing*
*Context gathered: 2025-05-26*
