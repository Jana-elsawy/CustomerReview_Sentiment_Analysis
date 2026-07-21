# Recommendation Predictor — Streamlit App

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The `artifacts/` folder already contains the trained model (TF-IDF vectorizer,
scaler, one-hot encoder, and logistic regression model), so no retraining is
needed to run the app.

## How to retrain the model
If you update the dataset, run `train_model.py` (in the parent folder) again —
it will regenerate the files inside `artifacts/`.

## Deploying online (free)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, connect your GitHub, and pick `app.py`
   as the entry point.
3. Streamlit Cloud will install `requirements.txt` and launch the app — you'll
   get a public link to share.
