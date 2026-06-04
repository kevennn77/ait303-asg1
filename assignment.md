# AIT303 Assignment 1 – Advanced Issues of Artificial Intelligence (Natural Language Processing)

## Course Information
- **Course Code:** AIT303
- **Course Name:** Advanced Issues of Artificial Intelligence (Natural Language Processing)
- **Lecturer:** Chua Chong Chai
- **Academic Session:** 2026/04
- **Assessment Title:** Assignment 1
- **Submission Deadline:** 5 June 2025 (Friday), 5:00 PM

---

# Assignment Overview

This assignment evaluates the full cycle of **aspect-based sentiment analysis** using sentiment analysis and topic modeling techniques.

You are required to apply the following NLP techniques:

- Text preprocessing
- Feature extraction
- Word embedding
- Machine learning (supervised, unsupervised, semi-supervised)
- Deep neural networks

---

# Task 1: Sentiment Analyzer

## Objective

Train prediction models to classify sentiment labels (**positive** or **negative**) for text reviews.

## Dataset

Use the **IMDB Movie Dataset** from Kaggle:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Required Models

### 1. Support Vector Machine (SVM)

Train **two different SVM models** using:

- CountVectorizer
- TfidfVectorizer

### 2. Bidirectional Gated Recurrent Unit (BiGRU)

Train **two different BiGRU models** using word embeddings:

- CBOW (Continuous Bag of Words)
- Skip-Gram

---

## Required Workflow

### 1. Data Preparation
- Load and inspect IMDB dataset
- Check missing values
- Balance dataset if necessary

### 2. Text Preprocessing
- Lowercasing
- Remove HTML tags
- Remove punctuation
- Remove stopwords
- Tokenization
- Lemmatization or stemming

### 3. Feature Extraction / Word Embedding

For SVM:
- CountVectorizer
- TfidfVectorizer

For BiGRU:
- CBOW embeddings
- Skip-Gram embeddings

### 4. Model Training
Train the following four models:

1. SVM + CountVectorizer
2. SVM + TfidfVectorizer
3. BiGRU + CBOW
4. BiGRU + Skip-Gram

### 5. Cross-Validation
Use cross-validation techniques to improve and validate training.

Examples:
- K-Fold Cross Validation
- Stratified K-Fold

### 6. Model Evaluation

Evaluate each model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 7. Save Trained Models

Save all trained models for later use.

Examples:
- `.pkl`
- `.joblib`
- `.h5`

---

# Task 2: Aspect-Based Sentiment Analysis

## Part A: Aspect Extraction

### Step 1: Scrape Product Reviews

Scrape product reviews from **Best Buy**:

https://www.bestbuy.com

### Requirements

- At least **30 different products**
- Products must be from the **same category**
- Each product must have **at least 100 reviews**

---

### Step 2: Unsupervised Aspect Extraction

Train aspect extraction models using:

#### LDA (Latent Dirichlet Allocation)

Purpose:
- Discover hidden topics from review text

#### BERTopic

Purpose:
- Generate semantic topic clusters using transformer embeddings

---

### Step 3: Semi-Supervised Aspect Extraction (CorEx)

#### 1. Analyze Topics and Key Terms

Review outputs from:

- LDA
- BERTopic

Identify meaningful keywords.

#### 2. Restructure Keywords into at Least 6 Aspects

Examples of aspects:

- Design
- Quality
- Comfortability
- Performance
- Price
- Battery Life

These should reflect major purchasing considerations.

#### 3. Train CorEx Model

Use the manually restructured aspects as anchor terms.

#### 4. Save Aspect Models

Save all trained aspect extraction models.

---

## Part B: Label Product Reviews

Use:

- Trained **CorEx** model
- **Best sentiment analysis model** from Task 1

Process all downloaded reviews and assign:

- Aspect labels
- Sentiment labels

---

## Part C: Product Ranking

For **each of the 6 aspects**:

- Calculate total positive sentiment score
- Rank products based on sentiment score
- Select top 5 products

### Required Output

You should produce:

- Top 5 products for Aspect 1
- Top 5 products for Aspect 2
- Top 5 products for Aspect 3
- Top 5 products for Aspect 4
- Top 5 products for Aspect 5
- Top 5 products for Aspect 6

Total: **6 ranking lists**

---

## Part D: Best Product Selection

Choose **one best product** according to your personal preferences.

Explain:

- Why you selected it
- Which aspects influenced your decision
- Why it is better than other products

---

# Task 3: Report Requirements

## A. Report Organization

Organize the report into proper academic sections.

Recommended sections:

1. Introduction
2. Methodology
3. Sentiment Analysis
4. Aspect-Based Sentiment Analysis
5. Results and Discussion
6. Conclusion

---

## B. Sentiment Analysis Section

Include:

### 1. Workflow Discussion

Explain:

- Data preparation
- Text preprocessing
- Feature extraction
- Model training
- Cross-validation
- Evaluation

### 2. Compare the Four Models

Compare:

- SVM + CountVectorizer
- SVM + TfidfVectorizer
- BiGRU + CBOW
- BiGRU + Skip-Gram

Include:

- Accuracy table
- Evaluation metrics
- Confusion matrices

### 3. Justify the Best Model

Explain why one model performed best.

Possible discussion points:

- Better feature representation
- Better contextual understanding
- Better generalization

---

## C. Aspect-Based Sentiment Analysis Section

Include:

### 1. Discuss Unsupervised Outputs

Analyze results from:

- LDA
- BERTopic

Discuss topic coherence and interpretability.

### 2. Discuss Restructured Aspects

Explain how you grouped keywords into meaningful aspects.

### 3. Discuss CorEx Outputs

Show and analyze aspect extraction performance.

### 4. Compare Unsupervised vs Semi-Supervised Results

Compare:

- Topic quality
- Aspect relevance
- Interpretability

### 5. Discuss Top 5 Products for Each Aspect

Explain rankings and product strengths.

### 6. Discuss Your Selected Best Product

Provide detailed reasoning.

### 7. Create Bar Charts

For every aspect:

- Plot top 5 products
- Visualize sentiment scores

Total charts required: **6 bar charts**

---

# Task 4: Code Requirements

## Jupyter Notebook

Complete all programming tasks using:

- Python
- Jupyter Notebook

## Documentation

Code must include:

- Clear comments
- Section headers
- Proper variable names
- Readable structure

---

# Submission Requirements

## Submit to Moodle

Upload:

- Assignment report (**PDF format**)

---

## Upload to Cloud Storage (OneDrive / Google Drive / Box)

Upload:

- Jupyter notebooks (`.ipynb`)
- Generated Excel files
- Trained model files
- ZIP file if necessary

---

## Include Shared Folder Link

Highlight the cloud storage link inside your report.

---

# Marking Breakdown

| Component | Marks |
|----------|-------|
| Sentiment Analysis | 6 |
| Aspect-Based Sentiment Analysis | 9 |
| Code & Report Quality | 5 |
| **Total** | **20** |

---

# Important Note

Include the **marking rubric** when submitting your coursework.

---

