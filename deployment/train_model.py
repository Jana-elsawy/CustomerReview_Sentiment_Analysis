"""
Training script for the Women's Clothing Review "Recommended?" classifier.
Rebuilds the cleaning + feature pipeline from the notebook, fixes the
lower-casing bug, and trains a TF-IDF + Logistic Regression model that is
fully self-contained (no external embedding downloads needed) so it can be
deployed easily in Streamlit.
"""
import re
import string
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Full cleaning pipeline: lowercase -> remove urls/html/numbers/punct/emoji
    -> tokenize -> remove stopwords -> lemmatize -> join."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" "]+",
        flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = " ".join(text.split())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    df.dropna(subset=['Review Text', 'Division Name', 'Department Name', 'Class Name'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Clean_Review'] = df['Review Text'].apply(clean_text)
    df = df[df['Clean_Review'].str.split().str.len() >= 3].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    print("Loading & cleaning data...")
    df = load_and_clean("data/Womens Clothing E-Commerce Reviews.csv")
    print("Rows after cleaning:", len(df))

    y = df['Recommended IND']

    text_train, text_test, cat_train, cat_test, num_train, num_test, y_train, y_test = train_test_split(
        df['Clean_Review'],
        df[['Division Name', 'Department Name', 'Class Name']],
        df[['Rating', 'Age', 'Positive Feedback Count']],
        y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=3000, min_df=3, ngram_range=(1, 2))
    X_text_train = tfidf.fit_transform(text_train)
    X_text_test = tfidf.transform(text_test)

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(num_train)
    X_num_test = scaler.transform(num_test)

    ohe = OneHotEncoder(handle_unknown='ignore')
    X_cat_train = ohe.fit_transform(cat_train)
    X_cat_test = ohe.transform(cat_test)

    X_train = hstack([X_text_train, csr_matrix(X_num_train), X_cat_train]).tocsr()
    X_test = hstack([X_text_test, csr_matrix(X_num_test), X_cat_test]).tocsr()

    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(tfidf, "artifacts/tfidf.joblib")
    joblib.dump(scaler, "artifacts/scaler.joblib")
    joblib.dump(ohe, "artifacts/ohe.joblib")
    joblib.dump(clf, "artifacts/model.joblib")

    cats = {
        'Division Name': sorted(df['Division Name'].dropna().unique().tolist()),
        'Department Name': sorted(df['Department Name'].dropna().unique().tolist()),
        'Class Name': sorted(df['Class Name'].dropna().unique().tolist()),
    }
    joblib.dump(cats, "artifacts/category_options.joblib")
    print("Saved artifacts to artifacts/")


if __name__ == "__main__":
    main()
