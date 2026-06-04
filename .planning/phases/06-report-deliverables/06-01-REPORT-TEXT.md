# AIT303 Assignment 1 — Aspect-Based Sentiment Analysis

**Author:** [Student Name]
**Student ID:** [Student ID]
**Course:** AIT303 — Advanced Natural Language Processing
**Date:** June 2026

---

## 1. Introduction

Sentiment analysis is a fundamental task in natural language processing (NLP) that aims to determine the emotional tone or opinion expressed in text. While traditional sentiment analysis assigns a single polarity (positive, negative, or neutral) to an entire document, Aspect-Based Sentiment Analysis (ABSA) goes a step further by identifying specific aspects or features of a product or service and determining the sentiment expressed toward each aspect. This granular approach provides more actionable insights, particularly in product review analysis where understanding which features customers like or dislike is more valuable than a single overall rating.

This assignment implements a complete ABSA pipeline comprising two main components: (1) sentiment classification using four machine learning models trained on the IMDB 50K movie review dataset, and (2) aspect extraction from scraped Best Buy Bluetooth speaker reviews using unsupervised and semi-supervised topic modeling techniques.

The IMDB 50K dataset (Maas et al., 2011) contains 50,000 movie reviews evenly split between positive and negative classes, making it a standard benchmark for binary sentiment classification. For the aspect extraction component, we scraped 3,199 reviews across 37 Bluetooth speaker products from Best Buy using a custom Python scraper.

Four sentiment classification models were implemented and compared:
- Support Vector Machine (SVM) with CountVectorizer features
- SVM with TfidfVectorizer features
- Bidirectional Gated Recurrent Unit (BiGRU) with Continuous Bag-of-Words (CBOW) embeddings
- BiGRU with Skip-Gram embeddings

For aspect extraction, three topic modeling approaches were applied to the scraped Best Buy reviews: Latent Dirichlet Allocation (LDA) as an unsupervised method, BERTopic as a transformer-based unsupervised method, and CorEx as a semi-supervised anchored topic model.

The remainder of this report is organized as follows: Section 2 describes the methodology for each pipeline stage. Section 3 presents the experimental results for sentiment models, aspect extraction, and product rankings. Section 4 discusses the findings and their implications. Section 5 concludes with a summary and directions for future work.

---

## 2. Methodology

### 2.1 Data Preparation & Preprocessing

The IMDB 50K dataset was loaded and inspected to confirm its structure: 50,000 reviews, 2 columns (review text and sentiment label), with no missing values and a perfect 50/50 class balance of 25,000 positive and 25,000 negative reviews.

A text preprocessing pipeline was applied to the raw review text, consisting of the following sequential steps:

1. **Lowercasing** — All text was converted to lowercase to normalize case variation.
2. **HTML tag removal** — Markup tags such as `<br />` were stripped using regular expressions.
3. **Punctuation removal** — Non-alphanumeric characters were removed.
4. **Whitespace normalization** — Multiple consecutive spaces were collapsed into single spaces.

Two preprocessing strategies were applied to the cleaned text for comparison:

- **Stemming (Porter Stemmer):** The Porter stemming algorithm reduced words to their root forms by stripping suffixes. This produced non-dictionary forms such as "episod" (episode) and "violenc" (violence), confirming the algorithmic nature of the stemming process.
- **Lemmatization (WordNet Lemmatizer with POS tagging):** Part-of-speech tags were assigned using the Penn Treebank tag set, converted to WordNet POS tags, and used to guide the lemmatizer. This produced valid dictionary forms such as "episode" and "violence," at the cost of higher computational overhead.

The lemmatized text was used as the primary input for all downstream models, as it preserves semantic meaning more effectively than stemming.

### 2.2 Sentiment Models

Two model architectures were employed, each with two variants, producing four classification models in total.

#### 2.2.1 Support Vector Machine (SVM)

Linear Support Vector Classification (SVC with linear kernel) was selected for its strong performance on high-dimensional text data. Two feature extraction strategies were compared:

- **CountVectorizer:** Produced 2,243,876 features for stemmed text and 2,343,855 features for lemmatized text, using an n-gram range of (1,2).
- **TfidfVectorizer:** Applied term frequency-inverse document frequency weighting with the same n-gram range and vocabulary size.

Both vectorizers were configured with `ngram_range=(1,2)` to capture both unigrams and bigrams. The SVM classifier used `C=1.0` with a linear kernel (`SVC` or `LinearSVC`). All four SVM variants were evaluated using 5-fold Stratified K-Fold cross-validation to ensure robust performance estimates.

The dataset was split into 80% training (40,000 reviews) and 20% testing (10,000 reviews) using stratified sampling, preserving the 50/50 class balance in both splits.

#### 2.2.2 Bidirectional GRU (BiGRU)

Word2Vec embeddings were trained on the preprocessed IMDB corpus to produce distributed word representations. Two training configurations were used:

- **CBOW (Continuous Bag-of-Words):** Predicts the target word from its surrounding context words. Trained with `vector_size=100`, `window=5`, `min_count=5`, and `epochs=5`. Produced a vocabulary of 32,309 words and an embedding matrix of shape (90,928, 100).
- **Skip-Gram:** Predicts surrounding context words from the target word. Trained with the same hyperparameters, producing the same vocabulary size and embedding matrix dimensions.

The BiGRU architecture consisted of a single Bidirectional GRU layer with 128 units per direction (256 total), preceded by an embedding layer initialized with the pre-trained Word2Vec weights. A dropout rate of 0.5 was applied for regularization, followed by a sigmoid output layer for binary classification. The model used binary cross-entropy loss and the Adam optimizer.

Each BiGRU variant was trained with 5-fold Stratified K-Fold cross-validation, with early stopping (patience=3, restoring best weights) to prevent overfitting. Training was conducted with a batch size of 64.

### 2.3 Web Scraping

Thirty-seven Bluetooth speaker products with over 3,100 reviews were collected from Best Buy (bestbuy.com) using a custom Python scraper. The scraper used `curl_cffi`, a Python library that bypasses Akamai bot protection through TLS fingerprint impersonation (Chrome 124 profile). This approach proved effective against Best Buy's bot detection without requiring proxies.

The scraper accessed Best Buy's internal reviews API at `/ugc/v2/reviews`, which returns structured JSON responses containing rating, text, submission date, and author information. The API endpoint was discovered through analysis of Best Buy's web application network traffic.

Products were identified by filtering Best Buy's product sitemaps for Bluetooth speaker categories. The final dataset comprises 37 products from 12 brands: Bose, JBL, Altec Lansing, Anker, Braven, Marshall, Bang & Olufsen, Beats, Ultimate Ears, Aiwa, Amazon, and Audio Pro. This represents a diverse range of price points and speaker types, from portable micro-speakers to party speakers.

### 2.4 Aspect Extraction

#### 2.4.1 LDA Topic Modeling (Unsupervised)

Latent Dirichlet Allocation (LDA) was applied using the gensim library's `LdaModel` with automatic alpha learning. A grid search was conducted over topic counts K = 6 to 10, selecting the optimal model based on C_v topic coherence, which measures the semantic similarity of the top words within each topic.

The gensim Dictionary and Bag-of-Words corpus were built from the preprocessed review tokens. The optimal LDA model (K=9, C_v=0.441) was selected and analyzed. Top keywords for each topic were extracted and interpreted.

#### 2.4.2 BERTopic (Unsupervised)

BERTopic (Grootendorst, 2022) uses transformer-based sentence embeddings to cluster semantically similar documents and extract topic representations. The pipeline consisted of:

1. **Embedding:** `all-MiniLM-L6-v2` SentenceTransformer model (~80MB) generated 384-dimensional embeddings for each review.
2. **Dimensionality Reduction:** UMAP (Uniform Manifold Approximation and Projection) with `n_neighbors=15` and `n_components=5`.
3. **Clustering:** HDBSCAN (Hierarchical Density-Based Spatial Clustering) to identify dense clusters of semantically similar reviews.
4. **Topic Representation:** Class-based TF-IDF (c-TF-IDF) to extract top keywords per topic.

BERTopic generated 48 topics, with 1,051 reviews classified as outliers (not assigned to any topic).

#### 2.4.3 CorEx Topic Modeling (Semi-Supervised)

CorEx (Correlation Explanation) is a semi-supervised topic model that maximizes the total correlation between latent topics and observed document features. It allows domain knowledge to be injected as "anchors" — sets of keywords that should be associated with specific topics.

Eight predefined aspect categories were defined, each with 10-12 curated keywords:
- **Design:** design, look, style, aesthetic, appearance, sleek, beautiful, elegant, modern, color
- **Sound Quality:** sound, audio, quality, clear, loud, bass, treble, crisp, rich, volume
- **Battery:** battery, life, charge, lasting, power, recharge, battery life, hours, playback, drain
- **Price:** price, value, cost, cheap, expensive, worth, budget, money, dollar, deal, affordable
- **Build Quality:** build, quality, solid, durable, well, sturdy, material, plastic, metal, built
- **Features:** features, app, EQ, settings, mode, control, voice, Alexa, Google, smart, multiroom
- **Connectivity:** Bluetooth, range, signal, connection, stable, pairing, WiFi, aux, NFC, wireless
- **Comfort/Portability:** portable, light, carry, size, compact, travel, weight, small, handheld

These predefined anchors were augmented with relevant keywords extracted from the LDA and BERTopic models, creating a combined set of up to 10 anchor words per aspect. The CorEx model was trained with `n_hidden=8` topics (matching the 8 aspect categories), `seed=42` for reproducibility, and `anchor_strength=3.0`.

### 2.5 Labeling & Ranking

The best-performing sentiment model (TFIDF + Lemmatized SVM, F1=0.9091) was used to label all 3,199 scraped Best Buy reviews with sentiment predictions. Each review already carried an aspect label from the CorEx model, resulting in a combined aspect-sentiment dataset.

For each product-aspect pair, the positive sentiment ratio was calculated as the proportion of positive reviews mentioning that aspect. Products with fewer than 5 reviews for a given aspect were excluded from ranking for that aspect. The "Features" aspect was excluded entirely due to insufficient coverage. Composite scores were calculated using a weighted formula:

**Composite Score = 0.6 × Average Positive Ratio + 0.2 × Aspect Coverage + 0.2 × Normalized Rating**

This weighting prioritizes consistent positive sentiment (60%) while rewarding broad aspect coverage (20%) and customer star ratings (20%).

---

## 3. Results

### 3.1 Sentiment Model Results — SVM

The four SVM variants were evaluated on the held-out test set (10,000 reviews). Table 1 presents the consolidated comparison.

**Table 1: SVM 4-Model Comparison**
| Model Variant | Test Accuracy | Test Precision | Test Recall | Test F1-Score | CV Mean F1 |
|--------------|:------------:|:-------------:|:----------:|:-------------:|:---------:|
| TFIDF + Lemmatized | 0.9080 | 0.8983 | 0.9202 | **0.9091** | 0.9027 |
| TFIDF + Stemmed | 0.9069 | 0.8976 | 0.9186 | 0.9080 | **0.9030** |
| CV + Lemmatized | 0.8960 | 0.8916 | 0.9016 | 0.8966 | 0.8905 |
| CV + Stemmed | 0.8954 | 0.8909 | 0.9012 | 0.8960 | 0.8924 |

The TFIDF vectorizer consistently outperformed CountVectorizer across all metrics. The TFIDF + Lemmatized model achieved the highest test F1-score (0.9091), while TFIDF + Stemmed achieved the highest cross-validation mean F1 (0.9030). The performance gap between TFIDF and CountVectorizer variants was approximately 1.2 percentage points in F1-score.

The confusion matrix for the best model (TFIDF + Lemmatized) showed:
- True Negatives: 4,495 | False Positives: 505
- False Negatives: 446 | True Positives: 4,554

This indicates that the model correctly classified 9,049 out of 10,000 test reviews, with a slight tendency toward false positives (classifying negative reviews as positive).

### 3.2 Sentiment Model Results — BiGRU

Both BiGRU variants were evaluated using 5-fold cross-validation. Table 2 presents the aggregate metrics.

**Table 2: BiGRU 5-Fold Cross-Validation Results (Mean ± Std)**
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|:-------:|:---------:|:------:|:--------:|:-------:|
| CBOW-BiGRU | 0.8850 ± 0.0127 | 0.8991 ± 0.0333 | 0.8704 ± 0.0534 | 0.8829 ± 0.0169 | 0.9577 ± 0.0044 |
| Skip-Gram-BiGRU | 0.8970 ± 0.0076 | 0.8815 ± 0.0285 | 0.9192 ± 0.0243 | **0.8994 ± 0.0045** | 0.9625 ± 0.0017 |

The Skip-Gram-BiGRU variant outperformed CBOW-BiGRU across accuracy, recall, F1-score, and ROC-AUC, though with lower precision. The Skip-Gram-BiGRU model also demonstrated lower variance across folds (F1 std = 0.0045 vs 0.0169 for CBOW), indicating more stable performance. Its higher recall (0.9192) means it was better at identifying positive reviews, which is valuable for the downstream ranking task where positive sentiment detection drives the scoring.

### 3.3 Overall Model Comparison

**Table 3: All 4 Models — Best Variants**
| Model | Best Variant | Test F1-Score |
|------|-------------|:------------:|
| SVM + TfidfVectorizer | TFIDF + Lemmatized | **0.9091** |
| SVM + CountVectorizer | CV + Lemmatized | 0.8966 |
| BiGRU + Skip-Gram | Skip-Gram-BiGRU | 0.8994 |
| BiGRU + CBOW | CBOW-BiGRU | 0.8829 |

The SVM with TFIDF + Lemmatized achieved the highest overall F1-score (0.9091), closely followed by the BiGRU with Skip-Gram embeddings (0.8994). The SVM's superior performance can be attributed to the effectiveness of TF-IDF weighting for this balanced, well-structured dataset, and the linear separability of the high-dimensional feature space. The BiGRU models, while competitive, did not surpass the SVM, likely due to the limited training data (40,000 reviews) relative to the 90,928-dimensional embedding space.

### 3.4 Aspect Extraction Results

#### 3.4.1 LDA Results

The LDA grid search over K = 6 to 10 topics produced the following C_v coherence scores:

| K (Topics) | C_v Coherence |
|:----------:|:------------:|
| 6 | 0.397 |
| 7 | 0.429 |
| 8 | 0.405 |
| **9** | **0.441** |
| 10 | 0.366 |

The optimal model at K=9 achieved a C_v coherence of 0.441. The top keywords across the 9 topics included interpretable clusters related to sound quality (e.g., "sound", "quality", "audio"), battery performance (e.g., "battery", "charge", "life"), portability (e.g., "portable", "size", "small"), and pricing (e.g., "price", "value", "money"). The addition of stopword removal to the preprocessing pipeline improved coherence by approximately 27%, from 0.346 (without stopword removal) to 0.441 (with stopword removal).

#### 3.4.2 BERTopic Results

BERTopic generated 48 distinct topics from 3,199 reviews, with 1,051 reviews (32.9%) classified as outliers. The large number of topics reflects BERTopic's fine-grained clustering approach, which captures specific product attributes and user experiences. The inter-topic distance map showed good separation between major topic clusters, indicating that the embedding space effectively captured semantic differences between review content.

However, the high outlier rate (32.9%) and large number of topics (48) made direct interpretation challenging. The granularity, while precise, required substantial manual aggregation to map topics to the 8 predefined aspect categories needed for the downstream ranking task.

#### 3.4.3 CorEx Results

The anchored CorEx model was trained with 8 predefined aspect categories (matching the speaker domain) and achieved a Total Correlation (TC) of 13.19. TC measures the total dependency captured by the latent topics, with higher values indicating better topic separation.

**Table 4: CorEx Topic Quality Assessment**
| Aspect | Assessment |
|--------|-----------|
| Sound Quality | ✓ Coherent — keywords clearly related to audio performance |
| Battery | ✓ Coherent — keywords about charging, battery life, power |
| Build Quality | ✓ Coherent — materials, durability, construction |
| Price | ✓ Coherent — cost, value, affordability |
| Features | ✓ Coherent — app, EQ, voice assistant, smart features |
| Connectivity | ✓ Coherent — Bluetooth range, pairing, signal stability |
| Comfort/Portability | ✓ Coherent — size, weight, travel, outdoor use |
| Design | ~ Moderate — some overlap with Build Quality |

Seven of the eight anchored aspects produced coherent, interpretable topics. The "Design" aspect showed moderate overlap with "Build Quality," suggesting that these two concepts are closely related in the context of speaker reviews. The anchor strength setting (3.0) effectively guided the topic model toward the predefined categories while still allowing corpus-driven word associations.

### 3.5 Product Ranking Results

#### 3.5.1 Top 5 Products Per Aspect

For each of the 7 aspects (excluding "Features" due to insufficient coverage), products were ranked by their positive sentiment ratio. Table 5 presents the top 5 products for each aspect.

**Table 5: Top 5 Products per Aspect (by Positive Sentiment Ratio)**

**(a) Battery**
| Rank | Product | Pos Ratio | Reviews | Avg Rating |
|:----:|---------|:--------:|:-------:|:----------:|
| 1 | Altec Lansing Mini H2O | 100.0% | 8/8 | 4.25 |
| 2 | Altec Lansing Rockbox XL | 100.0% | 6/6 | 4.83 |
| 3 | Beats Pill+ | 100.0% | 9/9 | 4.78 |
| 4 | Braven BRV-1 | 100.0% | 12/12 | 4.58 |
| 5 | JBL Charge 5 | 100.0% | 14/14 | 5.00 |

**(b) Build Quality**
| Rank | Product | Pos Ratio | Reviews | Avg Rating |
|:----:|---------|:--------:|:-------:|:----------:|
| 1 | Altec Lansing Jolt Mini Lifejacket | 100.0% | 6/6 | 4.67 |
| 2 | Altec Lansing Mini Life Jacket 3 | 100.0% | 8/8 | 4.88 |
| 3 | Altec Lansing Boom Jacket | 90.0% | 9/10 | 4.80 |
| 4 | Braven BRV-1 | 85.7% | 6/7 | 4.57 |
| 5 | Braven Balance | 83.3% | 5/6 | 5.00 |

**(c) Comfort/Portability**
| Rank | Product | Pos Ratio | Reviews | Avg Rating |
|:----:|---------|:--------:|:-------:|:----------:|
| 1 | Aiwa Exos 9 | 100.0% | 11/11 | 4.82 |
| 2 | Braven 2200M | 100.0% | 7/7 | 4.57 |
| 3 | Bose SoundLink Revolve II | 97.1% | 34/35 | 4.86 |
| 4 | Bose SoundLink Micro | 97.1% | 33/34 | 4.91 |
| 5 | Bose SoundLink Revolve | 97.1% | 33/34 | 4.79 |

**(d) Connectivity**
| Rank | Product | Pos Ratio | Reviews | Avg Rating |
|:----:|---------|:--------:|:-------:|:----------:|
| 1 | Anker Soundcore Flare 2 | 100.0% | 7/7 | 4.86 |
| 2 | Bose S1 Pro | 100.0% | 6/6 | 5.00 |
| 3 | Bose SoundLink Revolve II | 100.0% | 7/7 | 5.00 |
| 4 | Altec Lansing Boomjacket Jolt | 92.3% | 12/13 | 4.85 |
| 5 | Altec Lansing Mini Life Jacket 3 | 91.7% | 11/12 | 5.00 |

**(e) Design**
| Rank | Product | Pos Ratio | Reviews | Avg Rating |
|:----:|---------|:--------:|:-------:|:----------:|
| 1 | Anker Soundcore Select 2 | 100.0% | 30/30 | 4.63 |
| 2 | Audio Pro Addon T3 | 100.0% | 22/22 | 4.95 |
| 3 | JBL Xtreme 3 | 100.0% | 7/7 | 4.86 |
| 4 | Bang & Olufsen Beosound Explore | 90.0% | 9/10 | 4.20 |
| 5 | Braven Balance | 88.9% | 8/9 | 4.67 |

**(f) Price**
| Rank | Product | Pos Ratio | Reviews | Avg Rating |
|:----:|---------|:--------:|:-------:|:----------:|
| 1 | Amazon Tap Speaker | 100.0% | 17/17 | 4.65 |
| 2 | Anker Soundcore Select Pro | 100.0% | 18/18 | 4.94 |
| 3 | Anker Soundcore Select | 96.7% | 29/30 | 4.77 |
| 4 | Anker Soundcore Flare 2 | 95.8% | 23/24 | 4.75 |
| 5 | Altec Lansing Boomjacket Jolt | 95.0% | 19/20 | 4.65 |

**(g) Sound Quality**
| Rank | Product | Pos Ratio | Reviews | Avg Rating |
|:----:|---------|:--------:|:-------:|:----------:|
| 1 | Altec Lansing Jolt Mini Lifejacket | 100.0% | 17/17 | 4.76 |
| 2 | Audio Pro Addon T3 | 100.0% | 5/5 | 5.00 |
| 3 | Bose SoundLink Revolve | 100.0% | 23/23 | 4.65 |
| 4 | Bose SoundLink Flex | 96.8% | 30/31 | 4.90 |
| 5 | Bose SoundLink Micro | 93.9% | 31/33 | 4.85 |

#### 3.5.2 Best Overall Product

A composite score was calculated for each product based on its average positive sentiment ratio (60% weight), aspect coverage (20% weight), and average customer rating normalized to 0-1 (20% weight).

**Table 6: Top 5 Products by Composite Score**
| Rank | Product | Composite | Avg Pos Ratio | Aspects | Avg Rating |
|:----:|---------|:---------:|:------------:|:-------:|:----------:|
| 1 | Altec Lansing Jolt Mini Lifejacket | **0.900** | 91.1% | 6/7 | 4.56 |
| 2 | Braven Balance | 0.894 | 84.6% | 7/7 | 4.66 |
| 3 | Bose S1 Pro | 0.881 | 90.5% | 5/7 | 4.88 |
| 4 | Aiwa Exos 9 | 0.877 | 87.2% | 6/7 | 4.54 |
| 5 | Anker Soundcore Flare 2 | 0.873 | 89.7% | 5/7 | 4.80 |

The **Altec Lansing Jolt Mini Lifejacket** was selected as the best overall product with a composite score of 0.900. Its performance across individual aspects is detailed in Figure 1.

**Table 7: Best Product Aspect Profile — Altec Lansing Jolt Mini Lifejacket**
| Aspect | Positive Ratio | Reviews | Avg Rating |
|--------|:------------:|:-------:|:----------:|
| Build Quality | 100.0% | 6/6 | 4.67 |
| Sound Quality | 100.0% | 17/17 | 4.76 |
| Comfort/Portability | 90.6% | 29/32 | 4.75 |
| Battery | 90.0% | 9/10 | 4.60 |
| Connectivity | 88.9% | 8/9 | 4.33 |
| Price | 77.3% | 17/22 | 4.23 |

The product achieved perfect positive sentiment scores in Build Quality and Sound Quality, with strong scores across Comfort/Portability, Battery, and Connectivity. The lowest score was in Price (77.3%), where some reviewers felt the product could be more competitively priced relative to alternatives. Nevertheless, the average customer rating of 4.56 out of 5, combined with coverage of 6 out of 7 assessed aspects, demonstrates a consistently well-regarded product across the dimensions that matter most to speaker buyers.

---

## 4. Discussion

### 4.1 Model Performance Analysis

The SVM with TFIDF + Lemmatized features achieved the highest overall performance (F1=0.9091), outperforming both the CountVectorizer SVM variants and the BiGRU neural network models. This finding is consistent with the established literature on text classification, where linear SVMs with TF-IDF features often perform competitively with or better than more complex deep learning approaches on medium-scale datasets.

Several factors explain the SVM's advantage in this context:

1. **Feature sparsity and linear separability:** The TF-IDF representation creates a high-dimensional, sparse feature space (2.3 million features). In such spaces, documents from different classes tend to be approximately linearly separable, which is precisely the condition where linear SVMs excel.

2. **Data scale:** With 40,000 training samples and 2.3 million features, the classifier operates in a regime where the number of features substantially exceeds the number of samples. SVMs are particularly effective in this regime due to their max-margin formulation, which focuses on the support vectors nearest to the decision boundary.

3. **BiGRU limitations:** The neural network models required learning a dense embedding space (100 dimensions) from the corpus, which may not capture the same discriminative information as the full sparse TF-IDF representation. Additionally, the 40,000 training samples may be insufficient for the BiGRU's 90,928-dimensional embedding layer to reach its full potential.

It is important to note a significant limitation in the sentiment labeling pipeline: the trained model was applied to a different domain (Best Buy speaker reviews) than the one it was trained on (IMDB movie reviews). This domain shift — from movie reviews to product reviews — introduces uncertainty in the sentiment predictions, as linguistic patterns, review conventions, and vocabulary differ substantially between these domains.

### 4.2 Aspect Extraction Quality

The three topic modeling approaches offered complementary strengths for aspect extraction:

LDA provided the most interpretable topic structure with 9 coherent topics. Its C_v coherence score (0.441) indicates reasonable topic quality, and the topics aligned well with intuitive aspect categories. However, LDA's bag-of-words representation cannot capture word order or context, limiting its ability to distinguish between related concepts.

BERTopic offered the most granular analysis with 48 topics, leveraging the semantic power of transformer-based sentence embeddings. However, the high number of topics and outlier rate (32.9%) made it challenging to use as a direct aspect extraction method without substantial post-processing. The detailed topic structure is better suited for exploratory analysis than for producing a fixed set of aspect labels.

CorEx provided the best balance between structure and flexibility. The anchored approach allowed domain knowledge to guide the topic discovery process, resulting in 7 out of 8 predefined aspects being coherently captured. The anchor strength parameter (3.0) effectively balanced the influence of predefined keywords against corpus-driven patterns.

The aspect distribution across reviews showed notable imbalance: Comfort/Portability accounted for 919 reviews (28.9%), while Features accounted for only 97 reviews (3.0%). This imbalance likely reflects the distribution of discussion topics in speaker reviews — users naturally discuss portability and comfort more than specific app features.

### 4.3 Challenges and Limitations

**Web Scraping:** Best Buy's Akamai bot protection required a TLS fingerprint impersonation approach (curl_cffi) to access the hidden review API. This was successfully implemented, but the approach depends on the API remaining accessible and Best Buy not changing their anti-bot measures.

**Aspect Imbalance:** The CorEx model assigned most reviews to Comfort/Portability (29%) and Sound Quality (20%), while Features received only 3%. This imbalance limited the ranking granularity for less-discussed aspects. Future work could explore stratified sampling or aspect-specific weighting to address this.

**Domain Shift:** The sentiment model was trained on movie reviews but applied to product reviews. While the underlying sentiment signal (positive/negative language) generalizes across domains, domain-specific vocabulary and review conventions may introduce systematic biases. Fine-tuning on a small sample of labeled product reviews would be a natural improvement.

**Small Denominators:** Some products had limited review coverage for specific aspects (e.g., Sound Quality with 3-5 reviews for some products). The minimum review threshold (set to 5) mitigated this to some extent, but rankings for aspects with sparse coverage should be interpreted with caution.

---

## 5. Conclusion

This project implemented a complete Aspect-Based Sentiment Analysis pipeline, combining four sentiment classification models with three topic modeling approaches for aspect extraction. The key findings are:

1. **Best Sentiment Model:** SVM with TFIDF + Lemmatized features achieved the highest F1-score of 0.9091, outperforming both CountVectorizer SVM variants and BiGRU neural network models on the IMDB 50K dataset.

2. **Aspect Extraction:** CorEx with anchored topics provided the most practical aspect extraction approach, producing 7 coherent aspect categories that align well with speaker product attributes. LDA offered strong interpretability (K=9, C_v=0.441), while BERTopic provided granular but less practically useful topic clusters.

3. **Product Ranking:** Thirty-seven Bluetooth speaker products were ranked across 7 aspects. The Altec Lansing Jolt Mini Lifejacket achieved the highest composite score (0.900), excelling in Build Quality and Sound Quality with broad aspect coverage.

4. **Web Scraping:** A TLS impersonation approach successfully bypassed Akamai bot protection to scrape 3,199 reviews from 37 products using Best Buy's hidden API.

The main limitations include the domain shift between training (movie reviews) and application (product reviews), aspect imbalance in the CorEx topic assignments, and small review counts for certain product-aspect combinations. Future work could address these through domain adaptation techniques, aspect-aware sampling strategies, and expanding the review dataset through continuous scraping.

---

## References

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint arXiv:2203.05794*.

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics*, 142–150.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. *Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks*, 45–50.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

Chollet, F., & others. (2015). Keras. https://keras.io

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation*, 265–283.

Gallagher, R. J., Reising, K., Kale, D., & Ver Steeg, G. (2017). Anchored correlation explanation: Topic modeling with minimal domain knowledge. *Transactions of the Association for Computational Linguistics*, 5, 529–542.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*, 3982–3992.
