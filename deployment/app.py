import re
import string
import joblib
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Fashion Review Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DOWNLOAD NLTK
# ==========================================

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        _ = nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ==========================================
# LOAD ARTIFACTS
# ==========================================

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("artifacts/tfidf.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    ohe = joblib.load("artifacts/ohe.joblib")
    model = joblib.load("artifacts/model.joblib")
    cats = joblib.load("artifacts/category_options.joblib")

    # --- scikit-learn version compatibility shim ---
    # If the model was pickled with a different scikit-learn version,
    # some internal attributes may be missing after unpickling.
    # Patch in safe defaults so predict_proba() doesn't crash.
    for attr, default in [
        ("multi_class", "auto"),
        ("n_jobs", None),
        ("l1_ratio", None),
    ]:
        if not hasattr(model, attr):
            setattr(model, attr, default)

    return tfidf, scaler, ohe, model, cats


tfidf, scaler, ohe, model, cats = load_artifacts()

# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS
    ]
    return " ".join(words)

# ==========================================
# PREDICTION
# ==========================================

def predict(review_text, rating, age, feedback, division, department, class_name):
    clean = clean_text(review_text)

    X_text = tfidf.transform([clean])
    X_num = scaler.transform([[rating, age, feedback]])
    X_cat = ohe.transform([[division, department, class_name]])

    X = hstack([X_text, csr_matrix(X_num), X_cat]).tocsr()

    probability = model.predict_proba(X)[0, 1]
    prediction = int(probability >= 0.5)

    return prediction, probability

# ==========================================
# CUSTOM CSS  —  PROFESSIONAL DESIGN SYSTEM
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

:root{
    --ink:#0F172A;
    --muted:#64748B;
    --line:#E7EAF3;
    --card:#FFFFFF;
    --bg:#F4F6FB;
    --primary:#4338CA;
    --primary-2:#7C3AED;
    --accent:#06B6D4;
    --good:#059669;
    --good-bg:#ECFDF5;
    --bad:#DC2626;
    --bad-bg:#FEF2F2;
    --radius:20px;
}

html, body, [class*="css"]{
    font-family:'Plus Jakarta Sans', sans-serif;
    color:var(--ink) !important;
}

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
/* Recolor the native header from Streamlit's default black to a clean
   white bar. The sidebar toggle arrow lives inside it and is left
   completely untouched so it keeps working. */
header[data-testid="stHeader"]{
    background:var(--card) !important;
    box-shadow:0 1px 0 var(--line) !important;
}

/* Force a consistent light theme regardless of the visitor's OS/browser
   dark-mode setting — this is what was washing the page out */
.stApp{
    background:
        radial-gradient(1200px 500px at 100% -10%, rgba(124,58,237,.08), transparent),
        radial-gradient(1000px 500px at -10% 10%, rgba(6,182,212,.06), transparent),
        var(--bg) !important;
    color:var(--ink) !important;
}

.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp div[data-testid="stMarkdownContainer"] *{
    color:var(--ink) !important;
}

/* Kill any residual fade/opacity transitions that could leave content
   looking washed-out or semi-transparent */
.stApp, .stApp *,
div[data-testid="element-container"],
div[data-testid="stMarkdownContainer"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
[data-testid="stMetric"], [data-testid="stMetric"] *{
    opacity:1 !important;
    animation:none !important;
    transition:background-color .2s ease, border-color .2s ease, box-shadow .2s ease, transform .2s ease !important;
}
section[data-testid="stSidebar"] *{
    color:#E9E7FF !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"]{
    color:#FFFFFF !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"]{
    color:#B9B4EA !important;
}

/* Hero text stays white on its own gradient */
.hero, .hero *{
    color:#FFFFFF !important;
}
.hero-eyebrow{ color:#E4E1FF !important; }
.hero-sub{ color:#DCD8FF !important; }

/* Result card text color is set inline per-result, don't override it */
.result-card, .result-card *{
    color:inherit !important;
}

.block-container{
    padding-top:2.8rem;
    padding-left:3.2rem;
    padding-right:3.2rem;
    padding-bottom:3rem;
    max-width:1280px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#12123A 0%, #1E1B4B 55%, #2D1B69 100%);
    padding-top:1rem;
}
section[data-testid="stSidebar"] > div{
    padding-top:1.5rem;
}
section[data-testid="stSidebar"] *{
    color:#E9E7FF !important;
}
section[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,.12);
}
[data-testid="stMetric"]{
    background:rgba(255,255,255,.06);
    border:1px solid rgba(255,255,255,.10);
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:10px;
}
[data-testid="stMetricLabel"]{
    font-size:12.5px !important;
    color:#B9B4EA !important;
    text-transform:uppercase;
    letter-spacing:.06em;
}
[data-testid="stMetricValue"]{
    font-size:22px !important;
    font-weight:800 !important;
    color:#FFFFFF !important;
}

/* ---------- Hero ---------- */
.hero{
    background:linear-gradient(120deg, #4338CA 0%, #6D28D9 55%, #7C3AED 100%);
    padding:44px 46px;
    border-radius:26px;
    box-shadow:0 20px 45px rgba(67,56,202,.28);
    margin-bottom:32px;
    position:relative;
    overflow:hidden;
}
.hero::after{
    content:"";
    position:absolute;
    right:-60px; top:-60px;
    width:260px; height:260px;
    background:rgba(255,255,255,.10);
    border-radius:50%;
}
.hero-eyebrow{
    display:inline-block;
    font-size:12.5px;
    font-weight:700;
    letter-spacing:.10em;
    text-transform:uppercase;
    color:#E4E1FF;
    background:rgba(255,255,255,.14);
    padding:6px 14px;
    border-radius:999px;
    margin-bottom:16px;
}
.hero-title{
    font-family:'Sora', sans-serif;
    font-size:40px;
    font-weight:800;
    color:#FFFFFF;
    line-height:1.15;
    letter-spacing:-.01em;
}
.hero-sub{
    font-size:16.5px;
    color:#DCD8FF;
    margin-top:12px;
    max-width:640px;
    line-height:1.6;
}

/* ---------- Section headings ---------- */
.section-head{
    font-family:'Sora', sans-serif;
    font-size:15px;
    font-weight:700;
    color:var(--ink);
    text-transform:uppercase;
    letter-spacing:.05em;
    margin:8px 0 16px 0;
    display:flex;
    align-items:center;
    gap:8px;
}
.section-head .bar{
    width:5px; height:18px;
    border-radius:4px;
    background:linear-gradient(180deg, var(--primary), var(--accent));
    display:inline-block;
}

/* ---------- Bordered containers as cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--card);
    border-radius:var(--radius) !important;
    border:1px solid var(--line) !important;
    box-shadow:0 6px 20px rgba(15,23,42,.05);
}

/* ---------- Inputs ---------- */
.stTextArea textarea{
    border-radius:14px !important;
    border:1.5px solid var(--line) !important;
    font-size:15px !important;
    background:#FFFFFF !important;
    color:var(--ink) !important;
}
.stTextArea textarea::placeholder{
    color:#94A3B8 !important;
    opacity:1 !important;
}
.stTextArea textarea:focus{
    border-color:var(--primary) !important;
    box-shadow:0 0 0 3px rgba(67,56,202,.12) !important;
}
.stNumberInput input, .stSelectbox div[data-baseweb="select"]{
    border-radius:12px !important;
    background:#FFFFFF !important;
    color:var(--ink) !important;
}
.stNumberInput input{
    color:var(--ink) !important;
}
div[data-baseweb="select"] *{
    color:var(--ink) !important;
}
.stSlider label, .stNumberInput label, .stSelectbox label, .stTextArea label{
    color:var(--ink) !important;
    font-weight:600 !important;
}
.stSlider [data-baseweb="slider"]{
    margin-top:6px;
}

/* ---------- Button ---------- */
.stButton>button{
    width:100%;
    height:56px;
    border:none;
    border-radius:14px;
    background:linear-gradient(120deg, var(--primary), var(--primary-2));
    color:white;
    font-size:16.5px;
    font-weight:700;
    letter-spacing:.01em;
    box-shadow:0 10px 24px rgba(67,56,202,.30);
    transition:.25s ease;
}
.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 14px 30px rgba(67,56,202,.38);
}
.stButton>button:active{
    transform:translateY(0px);
}

/* ---------- Result card ---------- */
.result-card{
    padding:36px 38px;
    border-radius:24px;
    position:relative;
    overflow:hidden;
}
.result-badge{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:7px 16px;
    border-radius:999px;
    font-weight:700;
    font-size:13px;
    letter-spacing:.03em;
    text-transform:uppercase;
}
.result-score{
    font-family:'Sora', sans-serif;
    font-size:56px;
    font-weight:800;
    margin:14px 0 4px 0;
    letter-spacing:-.02em;
}
.result-msg{
    font-size:15.5px;
    color:var(--muted);
    max-width:520px;
    line-height:1.6;
}

/* ---------- Feature pills ---------- */
.pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    background:#EEF2FF;
    color:#3730A3;
    border:1px solid #E0E4FF;
    padding:10px 16px;
    border-radius:12px;
    font-size:14.5px;
    font-weight:600;
    margin:5px 6px 5px 0;
}

hr{
    margin-top:26px;
    margin-bottom:26px;
    border-color:var(--line);
}

/* Final safety net: guarantee sidebar text is always light,
   no matter what else in this stylesheet might conflict */
html body section[data-testid="stSidebar"] [data-testid="stMetricValue"]{
    color:#FFFFFF !important;
}
html body section[data-testid="stSidebar"] [data-testid="stMetricLabel"]{
    color:#C9C4F5 !important;
}
/* ===== SIDEBAR TEXT FIX ===== */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *{
    color:#FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">NLP · Machine Learning · Fashion Analytics</div>
    <div class="hero-title">🛍️ Fashion Review Intelligence</div>
    <div class="hero-sub">
        Predict whether a customer will recommend a clothing product from their
        review text, rating, and profile — powered by TF-IDF and a
        Logistic Regression classifier.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("""
<h3 style="
color:white;
font-weight:700;
margin-bottom:8px;
">
📊 Project Snapshot
</h3>
""", unsafe_allow_html=True)
    st.markdown("---")

    st.metric("Accuracy", "94%")
    st.metric("ROC AUC", "0.98")
    st.metric("Algorithm", "Logistic Regression")
    st.metric("Dataset", "23K Reviews")

    st.markdown("---")
    st.success("NLP + Machine Learning")

# ==========================================
# INPUT SECTION
# ==========================================

st.markdown('<div class="section-head"><span class="bar"></span>Customer Information</div>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### 👤 Customer Details")
        age = st.number_input("Age", min_value=13, max_value=99, value=35)
        rating = st.slider("⭐ Rating", min_value=1, max_value=5, value=4)
        feedback = st.number_input("👍 Positive Feedback Count", min_value=0, value=0)

with right:
    with st.container(border=True):
        st.markdown("#### 👗 Product Details")
        division = st.selectbox("Division", cats["Division Name"])
        department = st.selectbox("Department", cats["Department Name"])
        class_name = st.selectbox("Class", cats["Class Name"])

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# REVIEW
# ==========================================

st.markdown('<div class="section-head"><span class="bar"></span>Customer Review</div>', unsafe_allow_html=True)

with st.container(border=True):
    review_text = st.text_area(
        "Customer review",
        placeholder="Example:\n\nI absolutely loved this dress. The quality is excellent, "
                    "the fabric feels premium, and I would definitely recommend it.",
        height=200,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button("🚀  Analyze Review")

# ==========================================
# PREDICTION OUTPUT
# ==========================================

if predict_clicked:

    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review first.")

    else:
        with st.spinner("🤖 AI is analyzing your review..."):
            prediction, probability = predict(
                review_text, rating, age, feedback, division, department, class_name
            )

        confidence = probability if prediction == 1 else 1 - probability

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            bg, ink, badge_bg, badge, message = (
                "linear-gradient(120deg,#ECFDF5,#F0FDFA)",
                "#059669",
                "#059669",
                "✓  HIGH POSSIBILITY",
                "This customer is very likely to recommend the product."
            )
        else:
            bg, ink, badge_bg, badge, message = (
                "linear-gradient(120deg,#FEF2F2,#FFF7ED)",
                "#DC2626",
                "#DC2626",
                "✕  LOW POSSIBILITY",
                "This customer is unlikely to recommend the product."
            )

        st.markdown(
            f"""
            <div class="result-card" style="background:{bg}; border:1px solid {ink}22;">
                <span class="result-badge" style="background:{badge_bg}22; color:{ink};">{badge}</span>
                <div class="result-score" style="color:{ink};">{confidence:.1%}</div>
                <div class="result-msg">{message}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### Confidence")
        st.progress(float(confidence))
        st.caption(f"{confidence:.1%} confidence")

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Prediction", "Recommend" if prediction else "Not Recommend")
        with c2:
            st.metric("Confidence", f"{confidence:.1%}")
        with c3:
            st.metric("Rating", rating)

        st.markdown("---")

        with st.expander("📄 Processed Review (model input)"):
            st.code(clean_text(review_text), language=None)

# ==========================================
# ABOUT THE MODEL
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

st.markdown('<div class="section-head"><span class="bar"></span>About the AI Model</div>', unsafe_allow_html=True)

left, right = st.columns([2, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("""
#### 🧠 Machine Learning Pipeline

1. Clean the review text
2. Remove stop words
3. Lemmatize words
4. Generate TF-IDF features
5. Scale numerical features
6. Encode categorical features
7. Predict using Logistic Regression

This hybrid pipeline combines textual, numerical, and categorical
information to produce a recommendation prediction.
""")

with right:
    with st.container(border=True):
        st.metric("Dataset", "23K Reviews")
        st.metric("Vectorizer", "TF-IDF")
        st.metric("Classifier", "Logistic Regression")
        st.metric("Language", "English")

# ==========================================
# FEATURES
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-head"><span class="bar"></span>Application Features</div>', unsafe_allow_html=True)

st.markdown("""
<span class="pill">✔ NLP Text Cleaning</span>
<span class="pill">✔ TF-IDF Vectorization</span>
<span class="pill">✔ One-Hot Encoding</span>
<span class="pill">✔ Feature Scaling</span>
<span class="pill">✔ Machine Learning Prediction</span>
<span class="pill">✔ Confidence Score</span>
<span class="pill">✔ Interactive Dashboard</span>
<span class="pill">✔ Streamlit Deployment</span>
""", unsafe_allow_html=True)

# ==========================================
# PROJECT INFO
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📖 Project Description"):
    st.write("""
This project predicts whether a customer will recommend a women's clothing item.

The application combines:

• NLP
• Machine Learning
• TF-IDF
• Logistic Regression

to classify customer recommendations based on review text and structured features.
""")

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:10px 0 20px 0;">
    <h2 style="color:#4338CA; font-family:'Sora',sans-serif; font-weight:800;">
        🛍 Fashion Review Intelligence
    </h2>
    <p style="font-size:16px;color:#64748B;">AI-Powered Recommendation Prediction</p>
    <br>
    <p style="color:#334155;">
        Built with <b>Python</b> • <b>Scikit-Learn</b> • <b>NLTK</b> • <b>Streamlit</b>
    </p>
    <br>
</div>
""", unsafe_allow_html=True)