# AI Customer Review Sentiment Analysis System

An end-to-end machine learning pipeline that analyzes customer reviews from the **Women's Clothing E-Commerce Reviews** dataset to predict customer recommendation status, uncover complaint themes through clustering, and generate actionable business insights.

Built as a graduation project for the Artificial Intelligence Professional Diploma.

## Project Overview

Organizations receive thousands of customer reviews and struggle to manually extract sentiment, recurring issues, and actionable patterns. This project builds a system that:

- Classifies whether a customer recommends a product based on their review text and metadata
- Identifies common complaint categories using unsupervised clustering
- Explains *why* the model makes its predictions using multiple interpretability techniques
- Surfaces business-relevant insights from the combined analysis

## Dataset

**Women's Clothing E-Commerce Reviews** (23,486 reviews)

| Feature | Description |
|---|---|
| Review Text | Free-text customer review |
| Title | Optional review title |
| Rating | 1–5 star rating |
| Recommended IND | Target variable — 1 = recommended, 0 = not recommended |
| Age | Customer age |
| Positive Feedback Count | Number of users who found the review helpful |
| Division / Department / Class Name | Product category hierarchy |

The target class is imbalanced (~82% recommended / 18% not recommended), which shaped several modeling decisions below.

## Pipeline

### 1. Data Cleaning
- Dropped rows with missing `Review Text`, `Division Name`, `Department Name`, `Class Name`
- Removed duplicate rows
- Text cleaning: lowercasing, URL/HTML/number/punctuation/emoji removal, whitespace normalization
- Tokenization, stopword removal, lemmatization (NLTK)
- Dropped reviews with fewer than 3 tokens after cleaning
- Final dataset: 22,611 reviews

### 2. Feature Engineering
- **Text embeddings**: `all-MiniLM-L6-v2` (Sentence-Transformers) — 384-dimensional semantic embeddings, used instead of plain TF-IDF for the primary classifiers to better capture review meaning
- **Structured features**: scaled `Rating`, `Age`, `Positive Feedback Count` + one-hot encoded `Division Name`, `Department Name`, `Class Name`
- Combined feature vector: 416 dimensions total

### 3. Classification Models
Trained and compared, with `class_weight='balanced'` / `scale_pos_weight` throughout to address class imbalance:

| Model | F1 (class 0) | ROC-AUC |
|---|---|---|
| **XGBoost (tuned)** | **0.85** | **0.974** |
| Logistic Regression (tuned threshold) | 0.85 | 0.973 |
| XGBoost (default) | 0.84 | 0.972 |
| Voting Ensemble | 0.84 | 0.973 |
| Stacking Ensemble | 0.84 | 0.973 |
| Random Forest (tuned) | 0.81 | 0.960 |

Hyperparameters tuned via `RandomizedSearchCV` (scored on F1 to account for imbalance); decision threshold additionally tuned via precision-recall curve analysis for Logistic Regression.

### 4. Unsupervised Clustering
- K-Means on PCA-reduced embeddings (50 components, ~65% variance retained)
- k=6 selected via elbow method + silhouette score
- Each cluster labeled by its most frequent terms and analyzed against recommendation rate to surface distinct complaint themes (e.g. sizing/fit, fabric quality, styling)

### 5. Interpretability (XAI)
- **SHAP** on a parallel TF-IDF + Logistic Regression model — word-level explanations for global model behavior
- **Permutation Importance** — both individual-feature and grouped (embeddings vs. structured features) analysis on the tuned XGBoost model
- **LIME** — instance-level explanation for individual review predictions

### 6. Visualizations
- Class/recommendation distribution
- Review rating distribution
- Most common words (word cloud)
- PCA-reduced cluster scatter plot
- Recommendation rate by cluster
- Model comparison bar chart
- Confusion matrix heatmaps

## Key Findings

- Star rating is the single strongest individual predictor of recommendation status, but text embeddings contribute substantial additional signal collectively even though no single embedding dimension is individually important
- Adding structured metadata (rating, age, category) to text embeddings improved minority-class F1 substantially over an embeddings-only baseline
- Six distinct complaint/review themes emerge from clustering, with clearly different recommendation rates across clusters

- ##📈 Power BI Dashboard

To make the dataset explorable for non-technical stakeholders, the cleaned dataset was also modeled and visualized in Power BI, turning the raw review data into a business-facing analytics dashboard.
<img width="1600" height="822" alt="WhatsApp Image 2026-07-19 at 7 42 51 PM" src="https://github.com/user-attachments/assets/7b122f85-3315-48af-9add-e8181128b4fd" />

What the dashboard covers:

KPI cards — total review count (22.6K), average customer age (43.28), and average rating (4.18) at a glance
Sum of Rating by Rating — distribution showing ratings are heavily skewed toward 4★ and 5★
Recommended IND breakdown — a donut chart confirming the ~82%/18% class split (81.88% recommended)
Category breakdown table — ratings, age, and recommendation counts sliced by Class Name and Department Name, with Dresses and Knits/Tops driving the largest share of reviews
Average Rating by Department Name — comparative bar view across Bottoms, Intimate, Jackets, and Tops
Sum of Rating by Positive Feedback Count — a donut chart showing how "helpfulness" votes are distributed across reviews

## Tech Stack

- **Language**: Python (Jupyter/Colab)
- **NLP**: NLTK, Sentence-Transformers, TF-IDF (scikit-learn)
- **ML**: scikit-learn, XGBoost
- **Interpretability**: SHAP, LIME, scikit-learn permutation importance
- **Visualization**: Matplotlib, Seaborn, WordCloud

## Repository Structure

```
├── final_Project_ITI.ipynb        # Full notebook: cleaning → modeling → clustering → XAI
├── Womens_Clothing_Reviews.csv    # Raw dataset
├── report.pdf / report.docx       # 2–4 page written report
├── presentation.pptx              # Slide deck
└── README.md
```

## How to Run

1. Open the notebook in Google Colab or Jupyter
2. Install dependencies: `pip install sentence-transformers xgboost shap lime wordcloud nltk`
3. Run cells top to bottom (`Runtime → Run all` in Colab)
4. Outputs (models, plots, cluster assignments) generate inline

## Authors

[Add team member names here]
