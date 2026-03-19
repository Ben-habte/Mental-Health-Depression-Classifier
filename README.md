# 🧠 Depression Text Classifier — Machine Learning

A binary text classifier that detects depression-related social media posts using classical machine learning. Built with scikit-learn, TF-IDF vectorization, and three model types compared head-to-head.

---

##  Overview

Social media posts can carry signals of mental health struggles. This project builds and compares three ML classifiers on a labelled Reddit dataset to detect depression-related language — with a focus on **recall**, since missing a positive case carries higher risk than a false alarm.

> **Disclaimer:** This project is for educational and research purposes only. It is not intended for clinical use or diagnosis.

---

## 📊 Dataset

- **Source:** [mrjunos/depression-reddit-cleaned](https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned) on Hugging Face
- **Size:** ~28,000 labelled social media posts
- **Labels:** `0` = Non-Depressed, `1` = Depressed
- **Split:** 70% train / 15% validation / 15% test (stratified)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| scikit-learn | Model training & evaluation |
| NLTK | Stopword removal |
| pandas & NumPy | Data manipulation |
| Matplotlib & seaborn | Visualizations |
| Hugging Face `datasets` | Dataset loading |

---

## Project Structure

```
depression-text-classifier/
│
├── depression_detection.ipynb   # Full Jupyter notebook (recommended)
├── depression_detection_ml.py   # Plain Python script version
├── requirements.txt             # Dependencies
└── README.md
```

---

## Models Compared

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear | Strong baseline for sparse TF-IDF features |
| Linear SVM | Linear | Often state-of-the-art on text classification |
| Decision Tree | Non-linear | Depth tuned via validation F1 to prevent overfitting |

---

## Pipeline

1. **Preprocessing** — Lowercase, remove URLs & punctuation, strip stopwords
2. **TF-IDF Vectorization** — Unigrams + bigrams, 20K features, fit on train only
3. **Hyperparameter Tuning** — Validation-based `max_depth` search for Decision Tree
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrices
5. **Interpretability** — Top TF-IDF features by Logistic Regression coefficient

---

## Visualizations Produced

- Confusion matrices (raw + normalized) for all three models
- Model comparison bar chart across all metrics
- Decision Tree depth tuning curve (Val F1 vs max_depth)
- Top 20 TF-IDF features for depressed vs non-depressed predictions

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Upload `depression_detection.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Uncomment the `!pip install` cell and run it
3. Run all cells in order

### Option 2 — Local
```bash
git clone https://github.com/YOUR_USERNAME/depression-text-classifier.git
cd depression-text-classifier
pip install -r requirements.txt
python depression_detection_ml.py
```

---

## Requirements

```
datasets
scikit-learn
nltk
pandas
numpy
matplotlib
seaborn
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## Evaluation Focus

> **Recall** is the primary metric. A false negative (missing a depressed post) is considered more harmful than a false positive in this context. Model selection and tuning decisions prioritize recall for the positive class.

---

## Future Work

- Fine-tuned transformer models (DistilBERT, RoBERTa) for higher performance
- Cross-dataset generalization evaluation
- SHAP values for deeper feature-level explainability
- Decision threshold tuning to further maximize recall
