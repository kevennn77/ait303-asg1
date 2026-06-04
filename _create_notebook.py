#!/usr/bin/env python3
"""Create the SVM Sentiment Models notebook (svm_sentiment_models.ipynb)"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (venv)",
        "language": "python",
        "name": "venv"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.14.2"
    }
}

cells = []

# ========================
# Cell 1 — Title (markdown)
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "# AIT303 Assignment 1 \u2014 SVM Sentiment Models (CountVectorizer)\n\n"
    "**Author:** [Student Name]\n\n"
    "This notebook trains and evaluates SVM sentiment classification models using "
    "CountVectorizer feature extraction on both stemmed and lemmatized IMDB 50K movie reviews.\n\n"
    "**Models trained (this notebook):**\n"
    "- SVM + CountVectorizer (Stemmed)\n"
    "- SVM + CountVectorizer (Lemmatized)\n\n"
    "**Evaluation:** 5-fold Stratified K-Fold cross-validation + held-out test set (80/20 split).\n"
    "**Interpretability:** Top positive/negative features from linear SVM coefficients."
))

# ========================
# Cell 2 — Imports (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# ===== Section 1: Setup & Imports =====\n"
    "import os\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "%matplotlib inline\n\n"
    "# sklearn \u2014 feature extraction\n"
    "from sklearn.feature_extraction.text import CountVectorizer\n\n"
    "# sklearn \u2014 model\n"
    "from sklearn.svm import SVC\n"
    "from sklearn.pipeline import Pipeline\n\n"
    "# sklearn \u2014 cross-validation\n"
    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate\n\n"
    "# sklearn \u2014 metrics\n"
    "from sklearn.metrics import (\n"
    "    accuracy_score, precision_score, recall_score, f1_score,\n"
    "    confusion_matrix, classification_report\n"
    ")\n\n"
    "# Fix random seeds for reproducibility (per T-02-03)\n"
    "np.random.seed(42)\n\n"
    "print(\"All imports successful.\")\n"
    "print(f\"pandas {pd.__version__}, numpy {np.__version__}\")\n"
    "print(\"scikit-learn imported\")"
))

# ========================
# Section 2
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "## 2. Load Preprocessed Data"
))

# Cell 4 — Load data
cells.append(nbf.v4.new_code_cell(
    "# Load preprocessed DataFrame \u2014 from CSV cache or Phase 1 notebook\n"
    "csv_path = 'data/preprocessed_imdb.csv'\n\n"
    "if os.path.exists(csv_path):\n"
    "    print(f\"Loading cached preprocessed data from {csv_path}...\")\n"
    "    df = pd.read_csv(csv_path)\n"
    "else:\n"
    "    print(f\"{csv_path} not found. Executing Phase 1 preprocessing notebook...\")\n"
    "    %run sentiment_analysis_preprocessing.ipynb\n"
    "    # Cache to CSV for future reuse\n"
    "    os.makedirs('data', exist_ok=True)\n"
    "    df.to_csv(csv_path, index=False)\n"
    "    print(f\"Preprocessed data cached to {csv_path}\")\n\n"
    "# Validate expected shape and columns (per T-02-01 mitigation)\n"
    "expected_cols = ['review', 'sentiment', 'cleaned', 'stemmed', 'lemmatized']\n"
    "expected_shape = (50000, 5)\n"
    "assert df.shape == expected_shape, f\"Expected shape {expected_shape}, got {df.shape}\"\n"
    "assert all(c in df.columns for c in expected_cols), \"Missing expected columns\"\n\n"
    "print(f\"\\nDataFrame shape: {df.shape}\")\n"
    "print(f\"Columns: {df.columns.tolist()}\")\n"
    "print(f\"Missing values:\\n{df.isnull().sum()}\")\n"
    "print(f\"\\nFirst 2 rows (stemmed[0] preview):\")\n"
    "print(f\"  {df['stemmed'].iloc[0][:100]}...\")\n"
    "print(f\"\\nFirst 2 rows (lemmatized[0] preview):\")\n"
    "print(f\"  {df['lemmatized'].iloc[0][:100]}...\")"
))

# Cell 5 — Encode labels
cells.append(nbf.v4.new_code_cell(
    "# Encode sentiment labels: positive -> 1, negative -> 0\n"
    "df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'negative': 0})\n\n"
    "# Verify encoding and balance\n"
    "print(\"Class distribution (encoded):\")\n"
    "print(df['sentiment_encoded'].value_counts())\n"
    "print(f\"\\nPositive (1): {df['sentiment_encoded'].sum()} reviews\")\n"
    "print(f\"Negative (0): {(1 - df['sentiment_encoded']).sum()} reviews\")\n\n"
    "# Define target vector\n"
    "y = df['sentiment_encoded'].values\n"
    "print(f\"\\nTarget vector shape: {y.shape}\")"
))

# Cell 6 — Train/test split subsection
cells.append(nbf.v4.new_markdown_cell(
    "### Train/Test Split (80/20, stratified)"
))

# Cell 7 — train_test_split
cells.append(nbf.v4.new_code_cell(
    "# Stratified train/test split \u2014 80/20 per D-16\n"
    "X_train_stemmed, X_test_stemmed, y_train, y_test = train_test_split(\n"
    "    df['stemmed'].values, y,\n"
    "    test_size=0.2, random_state=42, stratify=y\n"
    ")\n\n"
    "X_train_lemmatized, X_test_lemmatized = train_test_split(\n"
    "    df['lemmatized'].values, y,\n"
    "    test_size=0.2, random_state=42, stratify=y\n"
    ")[:2]\n\n"
    "print(f\"Training set size:   {len(X_train_stemmed):,}\")\n"
    "print(f\"Test set size:       {len(X_test_stemmed):,}\")\n"
    "print(f\"\\nTraining class distribution:\")\n"
    "print(f\"  Positive: {y_train.sum():,} ({y_train.mean()*100:.1f}%)\")\n"
    "print(f\"  Negative: {(1-y_train).sum():,} ({(1-y_train.mean())*100:.1f}%)\")\n"
    "print(f\"\\nTest class distribution:\")\n"
    "print(f\"  Positive: {y_test.sum():,} ({y_test.mean()*100:.1f}%)\")\n"
    "print(f\"  Negative: {(1-y_test).sum():,} ({(1-y_test.mean())*100:.1f}%)\")"
))

# ========================
# Section 3
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "## 3. Feature Extraction with CountVectorizer\n\n"
    "Using `CountVectorizer(ngram_range=(1,2))` per D-11 \u2014 unigrams and bigrams to capture "
    "both individual words and common sentiment phrases (e.g., \"not good\")."
))

# Cell 9 — CountVectorizer feature extraction
cells.append(nbf.v4.new_code_cell(
    "# ===== CountVectorizer Feature Extraction =====\n"
    "cv_stemmed = CountVectorizer(ngram_range=(1,2))\n"
    "X_train_cv_stemmed = cv_stemmed.fit_transform(X_train_stemmed)\n"
    "X_test_cv_stemmed = cv_stemmed.transform(X_test_stemmed)\n\n"
    "cv_lemmatized = CountVectorizer(ngram_range=(1,2))\n"
    "X_train_cv_lemmatized = cv_lemmatized.fit_transform(X_train_lemmatized)\n"
    "X_test_cv_lemmatized = cv_lemmatized.transform(X_test_lemmatized)\n\n"
    "print(\"=== CountVectorizer Vocabulary Sizes ===\")\n"
    "vocab_stemmed = len(cv_stemmed.get_feature_names_out())\n"
    "vocab_lemmatized = len(cv_lemmatized.get_feature_names_out())\n"
    "print(f\"Stemmed:   {vocab_stemmed:,} terms\")\n"
    "print(f\"Lemmatized: {vocab_lemmatized:,} terms\")\n"
    "print(f\"\\nStemmed matrix shape:    {X_train_cv_stemmed.shape}\")\n"
    "print(f\"Lemmatized matrix shape: {X_train_cv_lemmatized.shape}\")"
))

# ========================
# Section 4
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "## 4. SVM Model Training\n\n"
    "### 4.1 SVM + CountVectorizer (Stemmed)\n"
    "### 4.2 SVM + CountVectorizer (Lemmatized)\n\n"
    "Using `SVC(kernel='linear', C=1.0, random_state=42)` per D-12 / D-13.\n\n"
    "Pipelines wrap vectorizer + classifier so cross-validation operates on the full "
    "feature extraction + training pipeline."
))

# Cell 11 — Define Pipelines
cells.append(nbf.v4.new_code_cell(
    "# Define both SVM Pipelines (CountVectorizer + Linear SVM)\n"
    "pipeline_cv_stemmed = Pipeline([\n"
    "    ('vectorizer', CountVectorizer(ngram_range=(1,2))),\n"
    "    ('svm', SVC(kernel='linear', C=1.0, random_state=42))\n"
    "])\n\n"
    "pipeline_cv_lemmatized = Pipeline([\n"
    "    ('vectorizer', CountVectorizer(ngram_range=(1,2))),\n"
    "    ('svm', SVC(kernel='linear', C=1.0, random_state=42))\n"
    "])\n\n"
    "print(\"Pipeline CV Stemmed:\")\n"
    "print(pipeline_cv_stemmed)\n"
    "print(f\"\\nPipeline CV Lemmatized:\")\n"
    "print(pipeline_cv_lemmatized)\n\n"
    "print(\"Both Pipelines defined (not yet fitted).\")"
))

# ========================
# Section 5
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "## 5. Cross-Validation & Evaluation\n\n"
    "### 5.1 CountVectorizer Models \u2014 5-Fold Stratified K-Fold CV\n\n"
    "Per D-14: 5-fold Stratified K-Fold preserves class distribution in each fold."
))

# Cell 13 — cross_validate
cells.append(nbf.v4.new_code_cell(
    "# ===== 5-Fold Stratified K-Fold Cross-Validation =====\n"
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
    "scoring = ['accuracy', 'precision', 'recall', 'f1']\n\n"
    "print(\"Running 5-fold Stratified K-Fold CV for SVM + CountVectorizer...\")\n"
    "print(\"This may take several minutes...\")\n\n"
    "# CV \u2014 Stemmed (Pipeline handles vectorization inside CV)\n"
    "print(\"\\n[1/2] Stemmed variant...\")\n"
    "cv_results_stemmed = cross_validate(\n"
    "    pipeline_cv_stemmed, X_train_stemmed, y_train,\n"
    "    cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1\n"
    ")\n\n"
    "print(\"\\n=== SVM + CountVectorizer (Stemmed) \u2014 CV Results ===\")\n"
    "for metric in scoring:\n"
    "    tr_scores = cv_results_stemmed['train_' + metric]\n"
    "    te_scores = cv_results_stemmed['test_' + metric]\n"
    "    tr_mean = tr_scores.mean()\n"
    "    tr_std = tr_scores.std()\n"
    "    te_mean = te_scores.mean()\n"
    "    te_std = te_scores.std()\n"
    "    print(f\"  {metric.capitalize():12s}  Train: {tr_mean:.4f} +/- {tr_std:.4f}  |  Test: {te_mean:.4f} +/- {te_std:.4f}\")\n\n"
    "# CV \u2014 Lemmatized\n"
    "print(\"\\n[2/2] Lemmatized variant...\")\n"
    "cv_results_lemmatized = cross_validate(\n"
    "    pipeline_cv_lemmatized, X_train_lemmatized, y_train,\n"
    "    cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1\n"
    ")\n\n"
    "print(\"\\n=== SVM + CountVectorizer (Lemmatized) \u2014 CV Results ===\")\n"
    "for metric in scoring:\n"
    "    tr_scores = cv_results_lemmatized['train_' + metric]\n"
    "    te_scores = cv_results_lemmatized['test_' + metric]\n"
    "    tr_mean = tr_scores.mean()\n"
    "    tr_std = tr_scores.std()\n"
    "    te_mean = te_scores.mean()\n"
    "    te_std = te_scores.std()\n"
    "    print(f\"  {metric.capitalize():12s}  Train: {tr_mean:.4f} +/- {tr_std:.4f}  |  Test: {te_mean:.4f} +/- {te_std:.4f}\")\n\n"
    "# Comparison table\n"
    "print(\"\\n\" + \"=\" * 70)\n"
    "print(\"  CV Comparison: Stemmed vs Lemmatized (Test Set)\")\n"
    "print(\"=\" * 70)\n"
    "header = f\"  {'Metric':12s} | {'Stemmed':21s} | {'Lemmatized':21s}\"\n"
    "print(header)\n"
    "print(\"  \" + \"-\" * 12 + \"-+-\" + \"-\" * 21 + \"-+-\" + \"-\" * 21)\n"
    "for metric in scoring:\n"
    "    s_mean = cv_results_stemmed['test_' + metric].mean()\n"
    "    s_std  = cv_results_stemmed['test_' + metric].std()\n"
    "    l_mean = cv_results_lemmatized['test_' + metric].mean()\n"
    "    l_std  = cv_results_lemmatized['test_' + metric].std()\n"
    "    print(f\"  {metric.capitalize():12s} | {s_mean:.4f} +/- {s_std:.4f}   | {l_mean:.4f} +/- {l_std:.4f}\")"
))

# Cell 14 — Fit on full training data
cells.append(nbf.v4.new_code_cell(
    "# Fit both Pipelines on full training data for test-set evaluation\n"
    "print(\"Fitting Pipeline CV Stemmed on full training set (40K reviews)...\")\n"
    "pipeline_cv_stemmed.fit(X_train_stemmed, y_train)\n"
    "print(\"  [OK] Stemmed Pipeline fitted.\")\n\n"
    "print(\"Fitting Pipeline CV Lemmatized on full training set (40K reviews)...\")\n"
    "pipeline_cv_lemmatized.fit(X_train_lemmatized, y_train)\n"
    "print(\"  [OK] Lemmatized Pipeline fitted.\")"
))

# ========================
# Test-set Evaluation
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "### 5.2 Test-Set Evaluation \u2014 CountVectorizer Models\n\n"
    "Evaluating both fitted Pipelines on the held-out test set (10,000 reviews)."
))

# Cell 16 — Test-set metrics + confusion matrices
cells.append(nbf.v4.new_code_cell(
    "# ===== Test-Set Evaluation =====\n"
    "def evaluate_model(pipeline, X_test, y_test, model_name):\n"
    "    \"\"\"Predict, compute metrics, and display results.\"\"\"\n"
    "    y_pred = pipeline.predict(X_test)\n\n"
    "    acc  = accuracy_score(y_test, y_pred)\n"
    "    prec = precision_score(y_test, y_pred, average='macro')\n"
    "    rec  = recall_score(y_test, y_pred, average='macro')\n"
    "    f1   = f1_score(y_test, y_pred, average='macro')\n\n"
    "    print(\"\\n\" + \"=\" * 55)\n"
    "    print(\"  \" + model_name)\n"
    "    print(\"=\" * 55)\n"
    "    print(f\"  Accuracy:   {acc:.4f}\")\n"
    "    print(f\"  Precision:  {prec:.4f}\")\n"
    "    print(f\"  Recall:     {rec:.4f}\")\n"
    "    print(f\"  F1-Score:   {f1:.4f}\")\n\n"
    "    print(\"\\n  Classification Report:\")\n"
    "    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))\n\n"
    "    cm = confusion_matrix(y_test, y_pred)\n"
    "    print(f\"  Confusion Matrix (raw):\\n{cm}\")\n\n"
    "    plt.figure(figsize=(5, 4))\n"
    "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
    "                xticklabels=['negative', 'positive'],\n"
    "                yticklabels=['negative', 'positive'])\n"
    "    plt.title(f'Confusion Matrix \\u2014 {model_name}')\n"
    "    plt.xlabel('Predicted')\n"
    "    plt.ylabel('Actual')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n\n"
    "    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}\n\n"
    "print(\"Evaluating on test set...\")\n"
    "metrics_stemmed = evaluate_model(\n"
    "    pipeline_cv_stemmed, X_test_stemmed, y_test,\n"
    '    "SVM + CountVectorizer (Stemmed)")\n\n'
    "metrics_lemmatized = evaluate_model(\n"
    "    pipeline_cv_lemmatized, X_test_lemmatized, y_test,\n"
    '    "SVM + CountVectorizer (Lemmatized)")'
))

# Cell 17 — Metrics comparison table
cells.append(nbf.v4.new_code_cell(
    "# Side-by-side comparison of both CountVectorizer variants\n"
    "comparison = pd.DataFrame({\n"
    "    'Metric': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)'],\n"
    "    'Stemmed': [\n"
    "        f\"{metrics_stemmed['accuracy']:.4f}\",\n"
    "        f\"{metrics_stemmed['precision']:.4f}\",\n"
    "        f\"{metrics_stemmed['recall']:.4f}\",\n"
    "        f\"{metrics_stemmed['f1']:.4f}\"\n"
    "    ],\n"
    "    'Lemmatized': [\n"
    "        f\"{metrics_lemmatized['accuracy']:.4f}\",\n"
    "        f\"{metrics_lemmatized['precision']:.4f}\",\n"
    "        f\"{metrics_lemmatized['recall']:.4f}\",\n"
    "        f\"{metrics_lemmatized['f1']:.4f}\"\n"
    "    ]\n"
    "})\n\n"
    "print(\"=\" * 55)\n"
    "print(\"  Test-Set Metrics Comparison\")\n"
    "print(\"=\" * 55)\n"
    "print(comparison.to_string(index=False))\n\n"
    "sf1 = metrics_stemmed['f1']\n"
    "lf1 = metrics_lemmatized['f1']\n"
    "if sf1 > lf1:\n"
    "    print(f\"\\n  [BEST] Stemmed variant has higher F1 ({sf1:.4f} vs {lf1:.4f})\")\n"
    "elif lf1 > sf1:\n"
    "    print(f\"\\n  [BEST] Lemmatized variant has higher F1 ({lf1:.4f} vs {sf1:.4f})\")\n"
    "else:\n"
    "    print(f\"\\n  Both variants have equal F1 ({sf1:.4f})\")"
))

# ========================
# Top Features
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "### 5.3 Most Informative Features\n\n"
    "Linear SVM coefficients indicate which features are most indicative of each class. "
    "Positive coefficients -> positive sentiment, negative coefficients -> negative sentiment."
))

# Cell 19 — Top features from coefficients
cells.append(nbf.v4.new_code_cell(
    "# ===== Most Informative Features =====\n"
    "def display_top_features(pipeline, model_name, n=10):\n"
    "    \"\"\"Extract and display top N positive and negative features from linear SVM.\"\"\"\n"
    "    feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()\n"
    "    coefficients = pipeline.named_steps['svm'].coef_[0]\n\n"
    "    top_pos_idx = np.argsort(coefficients)[-n:][::-1]\n"
    "    top_neg_idx = np.argsort(coefficients)[:n]\n\n"
    "    print(\"\\n\" + \"=\" * 70)\n"
    "    print(\"  \" + model_name)\n"
    "    print(\"=\" * 70)\n\n"
    "    print(\"  Top {} Positive Features (--> positive sentiment):\".format(n))\n"
    "    print(\"  {:<3s} {:<28s} {:<15s} {:s}\".format('#', 'Feature', 'Coefficient', 'Association'))\n"
    "    print(\"  {:-<3s} {:-<28s} {:-<15s} {:-<15s}\".format('', '', '', ''))\n"
    "    for i, idx in enumerate(top_pos_idx, 1):\n"
    "        feat = feature_names[idx]\n"
    "        coef = coefficients[idx]\n"
    "        print(\"  {:3d} {:<28s} {:+.6f}      {:s}\".format(i, feat, coef, 'positive'))\n\n"
    "    print(\"\\n  Top {} Negative Features (--> negative sentiment):\".format(n))\n"
    "    print(\"  {:<3s} {:<28s} {:<15s} {:s}\".format('#', 'Feature', 'Coefficient', 'Association'))\n"
    "    print(\"  {:-<3s} {:-<28s} {:-<15s} {:-<15s}\".format('', '', '', ''))\n"
    "    for i, idx in enumerate(top_neg_idx, 1):\n"
    "        feat = feature_names[idx]\n"
    "        coef = coefficients[idx]\n"
    "        print(\"  {:3d} {:<28s} {:+.6f}      {:s}\".format(i, feat, coef, 'negative'))\n\n"
    "display_top_features(pipeline_cv_stemmed, \"SVM + CountVectorizer (Stemmed)\", n=10)\n"
    "display_top_features(pipeline_cv_lemmatized, \"SVM + CountVectorizer (Lemmatized)\", n=10)"
))

# ========================
# Section 6
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "## 6. Results Summary\n\n"
    "### 6.1 CountVectorizer SVM Results"
))

# Cell 21 — Results summary
cells.append(nbf.v4.new_code_cell(
    "# ===== Final Results Summary =====\n"
    "sf1 = metrics_stemmed['f1']\n"
    "lf1 = metrics_lemmatized['f1']\n\n"
    "cv_s_f1_mean = cv_results_stemmed['test_f1'].mean()\n"
    "cv_s_f1_std  = cv_results_stemmed['test_f1'].std()\n"
    "cv_l_f1_mean = cv_results_lemmatized['test_f1'].mean()\n"
    "cv_l_f1_std  = cv_results_lemmatized['test_f1'].std()\n\n"
    "best_var = 'Stemmed' if sf1 >= lf1 else 'Lemmatized'\n\n"
    "print(\"=\" * 70)\n"
    "print(\"  SECTION 6: RESULTS SUMMARY \\u2014 COUNT VECTORIZER SVM\")\n"
    "print(\"=\" * 70)\n"
    "print(f\"\\n  Best CountVectorizer variant:  {best_var}\")\n"
    "print(f\"  (based on test-set F1-score)\\n\")\n"
    "print(\"  {:<20s} | {:<22s} | {:<22s}\".format('Metric', 'Stemmed', 'Lemmatized'))\n"
    "print(\"  \" + \"-\" * 20 + \"-+-\" + \"-\" * 22 + \"-+-\" + \"-\" * 22)\n"
    "print(\"  {:<20s} | {:.4f} +/- {:.4f}  | {:.4f} +/- {:.4f}\".format(\n"
    "    'CV F1 (mean +/- std)', cv_s_f1_mean, cv_s_f1_std, cv_l_f1_mean, cv_l_f1_std))\n"
    "print(\"  {:<20s} | {:.4f}               | {:.4f}\".format('Test F1', sf1, lf1))\n"
    "print(\"  {:<20s} | {:.4f}               | {:.4f}\".format('Test Accuracy', metrics_stemmed['accuracy'], metrics_lemmatized['accuracy']))\n"
    "print(\"  {:<20s} | {:.4f}               | {:.4f}\".format('Test Precision', metrics_stemmed['precision'], metrics_lemmatized['precision']))\n"
    "print(\"  {:<20s} | {:.4f}               | {:.4f}\".format('Test Recall', metrics_stemmed['recall'], metrics_lemmatized['recall']))\n\n"
    "print(\"  ---\")\n\n"
    "f1_diff = abs(sf1 - lf1)\n"
    "if f1_diff < 0.005:\n"
    "    print(\"  Observation: Both preprocessing strategies achieve nearly identical\")\n"
    "    print(\"  performance (F1 diff: {:.4f}). Preprocessing choice has minimal\".format(f1_diff))\n"
    "    print(\"  impact on SVM + CountVectorizer for this dataset.\")\n"
    "else:\n"
    "    better = 'stemmed' if sf1 > lf1 else 'lemmatized'\n"
    "    print(\"  Observation: The {} preprocessing strategy performs better\".format(better))\n"
    "    print(\"  for SVM + CountVectorizer (F1 diff: {:.4f}).\".format(f1_diff))\n\n"
    "avg_f1 = (sf1 + lf1) / 2\n"
    "print(\"  Key Takeaway: Linear SVM with CountVectorizer (ngram_range=(1,2))\")\n"
    "print(\"  achieves strong sentiment classification on the IMDB 50K dataset\")\n"
    "print(\"  with avg test F1-score around {:.3f}. This establishes a strong\".format(avg_f1))\n"
    "print(\"  baseline for comparison with TfidfVectorizer variants and BiGRU models.\")\n"
    "print(\"=\" * 70)"
))

nb.cells = cells

# Write notebook
output_path = "/Users/keven/Documents/degree_study_material/y3s3/Adv_NLP/ASG1/svm_sentiment_models.ipynb"
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(cells)}")
