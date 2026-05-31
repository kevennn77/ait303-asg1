#!/usr/bin/env python3
"""Create the Aspect Extraction notebook (aspect_extraction.ipynb) for Phase 4."""

import nbformat as nbf
import os

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
    "# AIT303 Assignment 1 — Aspect Extraction & Topic Modeling\n\n"
    "**Author:** [Student Name]\n\n"
    "This notebook performs aspect extraction on scraped Best Buy Bluetooth & Wireless Speaker reviews\n"
    "using three topic modeling approaches:\n"
    "- **LDA** (gensim LdaMulticore) — unsupervised\n"
    "- **BERTopic** (all-MiniLM-L6-v2 embeddings) — unsupervised\n"
    "- **CorEx** (anchored semi-supervised) — semi-supervised\n\n"
    "**Pipeline:** Preprocessing → SpaCy Keyphrase Extraction → LDA/BERTopic → Aspect Restructuring → CorEx"
))

# ========================
# Cell 2 — Colab Instructions (markdown)
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "## 1. Colab Setup & Data Loading\n\n"
    "### ⚡ Colab Instructions\n"
    "If running on Google Colab:\n"
    "1. Upload the `data_asg/bestbuy/` folder to your Google Drive (as `data_asg/bestbuy/`)\n"
    "2. Set `COLAB = True` in the config cell below\n"
    "3. Run all cells — the notebook will mount your Drive\n\n"
    "**First-run note:** The pip install cell downloads ~200MB of packages. Sentence-transformers\n"
    "model (all-MiniLM-L6-v2, 80MB) downloads on first BERTopic use. Both cache after first run."
))

# ========================
# Cell 3 — Configuration (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# ============================================\n"
    "# CONFIGURATION\n"
    "# ============================================\n"
    "COLAB = True\n\n"
    "if COLAB:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    DATA_DIR = '/content/drive/MyDrive/data_asg'\n"
    "    BESTBUY_DIR = f'{DATA_DIR}/bestbuy'\n"
    "    MODEL_DIR = f'{DATA_DIR}/models'\n"
    "else:\n"
    "    DATA_DIR = 'data_asg'\n"
    "    BESTBUY_DIR = f'{DATA_DIR}/bestbuy'\n"
    "    MODEL_DIR = 'models'\n\n"
    "print(f\"Running in {'COLAB' if COLAB else 'LOCAL'} mode\")\n"
    "print(f\"Data directory: {DATA_DIR}\")\n"
    "print(f\"Best Buy data:  {BESTBUY_DIR}\")\n"
    "print(f\"Model directory: {MODEL_DIR}\")"
))

# ========================
# Cell 4 — pip install (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# Install all required packages (Colab)\n"
    "# gensim 4.4.0 cannot install on Python 3.14 - this notebook runs on Colab (Python 3.10)\n"
    "!pip install gensim==4.4.0 spacy==3.8.0 bertopic==0.17.4 corextopic==1.1 pyLDAvis==3.4.1 umap-learn hdbscan\n"
    "!python -m spacy download en_core_web_sm\n\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "print(\"Package installation complete\")"
))

# ========================
# Cell 5 — All imports (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# ============================================\n"
    "# IMPORTS\n"
    "# ============================================\n"
    "import os\n"
    "import re\n"
    "import json\n"
    "import random\n"
    "import warnings\n"
    "import pickle\n"
    "warnings.filterwarnings('ignore')\n\n"
    "# Data manipulation\n"
    "import pandas as pd\n"
    "import numpy as np\n\n"
    "# Visualization\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "%matplotlib inline\n\n"
    "# NLP\n"
    "import spacy\n\n"
    "# LDA\n"
    "from gensim.corpora import Dictionary\n"
    "from gensim.models import LdaMulticore\n"
    "from gensim.models.coherencemodel import CoherenceModel\n\n"
    "# BERTopic\n"
    "from sentence_transformers import SentenceTransformer\n"
    "from umap import UMAP\n"
    "from hdbscan import HDBSCAN\n"
    "from sklearn.feature_extraction.text import CountVectorizer\n"
    "from bertopic import BERTopic\n\n"
    "# CorEx\n"
    "from corextopic import corextopic as ct\n\n"
    "# LDA visualization\n"
    "import pyLDAvis\n"
    "import pyLDAvis.gensim_models as gensimvis\n\n"
    "# Scikit-learn utilities\n"
    "from sklearn.feature_extraction.text import CountVectorizer as SkCountVectorizer\n\n"
    "# Reproducibility\n"
    "np.random.seed(42)\n"
    "random.seed(42)\n\n"
    "print(\"All imports loaded successfully\")\n"
    "print(f\"pandas {pd.__version__}, numpy {np.__version__}, spaCy {spacy.__version__}\")"
))

# ========================
# Cell 6 — Section 2 header (markdown)
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "## 2. Load & Preprocess Reviews"
))

# ========================
# Cell 7 — Load scraped data from Drive (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# Load the consolidated reviews CSV\n"
    "csv_path = f'{BESTBUY_DIR}/all_reviews.csv'\n"
    "df = pd.read_csv(csv_path)\n\n"
    "print(f\"DataFrame shape: {df.shape}\")\n"
    "print(f\"Columns: {list(df.columns)}\")\n"
    "print(f\"Missing values:\\n{df.isnull().sum()}\")\n"
    "print(f\"\\nUnique products: {df['product_name'].nunique()}\")\n"
    "print(f\"Total reviews:   {len(df)}\")\n"
    "print(f\"\\nFirst 3 rows:\")\n"
    "df.head(3)"
))

# ========================
# Cell 8 — Load product catalog (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# Load product metadata catalog\n"
    "catalog_path = f'{BESTBUY_DIR}/products.json'\n"
    "with open(catalog_path, 'r') as f:\n"
    "    products = json.load(f)\n"
    "products_df = pd.DataFrame(products)\n\n"
    "print(f\"Products in catalog: {len(products_df)}\")\n"
    "print(f\"Columns: {list(products_df.columns)}\")\n"
    "products_df.head(5)"
))

# ========================
# Cell 9 — Review distribution plot (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# Reviews per product\n"
    "review_counts = df['product_name'].value_counts()\n\n"
    "plt.figure(figsize=(12, 5))\n"
    "review_counts.head(20).plot(kind='bar')\n"
    "plt.title('Reviews per Product (Top 20)')\n"
    "plt.xlabel('Product')\n"
    "plt.ylabel('Review Count')\n"
    "plt.xticks(rotation=45, ha='right')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "print(f\"Min reviews per product: {review_counts.min()}\")\n"
    "print(f\"Max reviews per product: {review_counts.max()}\")\n"
    "print(f\"Mean reviews per product: {review_counts.mean():.1f}\")\n"
    "print(f\"Products with >= 80 reviews: {(review_counts >= 80).sum()}\")"
))

# ========================
# Cell 10 — Preprocessing subsection (markdown)
# ========================
cells.append(nbf.v4.new_markdown_cell(
    "### 2.1 Text Preprocessing — Clean Review Text\n\n"
    "Reusing the `clean_text()` pipeline from Phase 1 (sentiment_analysis_preprocessing.ipynb):\n"
    "lowercase → remove HTML → remove non-alpha → normalize whitespace (per D-24).\n"
    "Product metadata is stored separately and NOT prepended to review text (per D-25)."
))

# ========================
# Cell 11 — Define clean_text (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "def clean_text(text):\n"
    "    \"\"\"Clean raw review text: lowercase, remove HTML tags, remove non-alpha characters, normalize whitespace.\"\"\"\n"
    "    if not isinstance(text, str):\n"
    "        return \"\"\n"
    "    text = text.lower()\n"
    "    text = re.sub(r'<.*?>', '', text)\n"
    "    text = re.sub(r'[^a-zA-Z]', ' ', text)\n"
    "    text = re.sub(r'\\s+', ' ', text).strip()\n"
    "    return text"
))

# ========================
# Cell 12 — Apply clean_text (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# Apply Phase 1 preprocessing to review text\n"
    "df['cleaned_review'] = df['review_text'].apply(clean_text)\n\n"
    "# Verify cleaning\n"
    "print(\"Before vs After Cleaning (sample):\")\n"
    "for i in range(3):\n"
    "    print(f\"\\n--- Sample {i+1} ---\")\n"
    "    print(f\"BEFORE: {df['review_text'].iloc[i][:150]}\")\n"
    "    print(f\"AFTER:  {df['cleaned_review'].iloc[i][:150]}\")\n\n"
    "print(f\"\\nCleaned review length stats:\")\n"
    "print(df['cleaned_review'].str.len().describe())\n\n"
    "# Filter out empty reviews after cleaning\n"
    "empty_count = (df['cleaned_review'].str.len() == 0).sum()\n"
    "print(f\"\\nEmpty reviews after cleaning: {empty_count} ({empty_count/len(df)*100:.1f}%)\")"
))

# ========================
# Cell 13 — Tokenize for LDA (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# Tokenize cleaned reviews for gensim LDA input\n"
    "df['tokens'] = df['cleaned_review'].str.split()\n\n"
    "# Filter short reviews (minimum 3 tokens for meaningful topic contribution)\n"
    "df_valid = df[df['tokens'].apply(len) >= 3].copy()\n\n"
    "print(f\"Reviews with >= 3 tokens: {len(df_valid)} / {len(df)}\")\n"
    "print(f\"Removed {len(df) - len(df_valid)} very short reviews\")"
))

# ========================
# Cell 14 — Save preprocessed data (code)
# ========================
cells.append(nbf.v4.new_code_cell(
    "# Save preprocessed DataFrame for downstream use\n"
    "preprocessed_path = f'{BESTBUY_DIR}/preprocessed_reviews.csv'\n"
    "df_valid.to_csv(preprocessed_path, index=False)\n"
    "print(f\"Preprocessed data saved to: {preprocessed_path}\")\n"
    "print(f\"Shape: {df_valid.shape}\")"
))

# Set cells and write notebook
nb.cells = cells

# Ensure notebooks directory exists
os.makedirs('notebooks', exist_ok=True)

outpath = 'notebooks/aspect_extraction.ipynb'
with open(outpath, 'w') as f:
    nbf.write(nb, f)

code_count = sum(1 for c in cells if c.cell_type == 'code')
md_count = sum(1 for c in cells if c.cell_type == 'markdown')
print(f"Notebook written to: {outpath}")
print(f"Cells: {len(cells)} total ({code_count} code, {md_count} markdown)")
