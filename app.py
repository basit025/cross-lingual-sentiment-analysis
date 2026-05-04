"""
Cross-Lingual Sentiment Analysis — Live Demo (Cloud Version)
Course: Computational Intelligence (CIS-423)
Model: XLM-RoBERTa (3-Class Zero-Shot, Run 9)
Deployed via: Streamlit Community Cloud + Hugging Face Hub
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import os
import time
import urllib.request

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cross-Lingual Sentiment Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Dark Premium Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }

    .result-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(248, 113, 113, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-neutral {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15), rgba(250, 204, 21, 0.08));
        border: 1px solid rgba(234, 179, 8, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .result-label {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-positive .result-label { color: #10b981; }
    .result-negative .result-label { color: #ef4444; }
    .result-neutral .result-label { color: #eab308; }

    .result-confidence {
        font-size: 1rem;
        color: #cbd5e1;
        font-weight: 400;
    }

    .result-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.3rem;
    }

    .conf-container { margin-top: 1rem; }
    .conf-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.7rem;
    }
    .conf-label {
        width: 90px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .conf-bar-bg {
        flex: 1;
        height: 28px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        overflow: hidden;
        margin: 0 12px;
    }
    .conf-bar-fill-neg {
        height: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #ef4444, #f87171);
        transition: width 0.8s ease;
    }
    .conf-bar-fill-neu {
        height: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #eab308, #facc15);
        transition: width 0.8s ease;
    }
    .conf-bar-fill-pos {
        height: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #10b981, #34d399);
        transition: width 0.8s ease;
    }
    .conf-pct {
        width: 55px;
        text-align: right;
        font-size: 0.9rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b, #312e81) !important;
    }
    [data-testid="stSidebar"] * {
        color: #c7d2fe !important;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .status-online { background: #10b981; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .stat-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "xlm-roberta-base"
HF_MODEL_URL = "https://huggingface.co/basit025/xlmroberta_amazon_crosslingual_sentiment/resolve/main/best_xlmr_sentiment_model_run9.pt"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "xlmr_sentiment")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "best_xlmr_sentiment_model_run9.pt")
NUM_LABELS = 3
MAX_LENGTH = 256
LABELS = ["Negative", "Neutral", "Positive"]
EMOJIS = ["😠", "😐", "😊"]
COLORS = ["negative", "neutral", "positive"]


@st.cache_resource(show_spinner=False)
def load_model():
    """Download weights from Hugging Face (if not cached) and load the model."""
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    # Download weights if not already cached
    if not os.path.exists(MODEL_CACHE_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with st.spinner("📥 Downloading model weights from Hugging Face (~1.1 GB)... This only happens once."):
            urllib.request.urlretrieve(HF_MODEL_URL, MODEL_CACHE_PATH)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    state_dict = torch.load(MODEL_CACHE_PATH, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return tokenizer, model


def predict_sentiment(text, tokenizer, model):
    """Run inference on a single text input."""
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        ).logits
    probabilities = F.softmax(logits, dim=1).squeeze().tolist()
    predicted_class = probabilities.index(max(probabilities))
    return predicted_class, probabilities


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 About This Project")
    st.markdown(
        "This app demonstrates **Cross-Lingual Sentiment Analysis** using "
        "a fine-tuned **XLM-RoBERTa** model. The model was trained exclusively "
        "on **English** Amazon reviews and can predict sentiment in **6 languages** "
        "it has never seen — a technique called **Zero-Shot Transfer Learning**."
    )

    st.markdown("---")
    st.markdown("### 🔬 Model Details")
    st.markdown("""
    - **Architecture:** XLM-RoBERTa Base
    - **Tokenizer:** SentencePiece (250K vocab)
    - **Training Run:** Run 9 (3-Class Zero-Shot)
    - **Training Data:** 120K English reviews
    - **Classes:** Negative · Neutral · Positive
    - **Best Val Accuracy:** 80.88%
    """)

    st.markdown("---")
    st.markdown("### 🏆 Zero-Shot Accuracy")
    st.markdown("""
    | Language | Accuracy |
    |---|---|
    | 🇬🇧 English | **81.00%** |
    | 🇩🇪 German | **79.74%** |
    | 🇫🇷 French | **77.32%** |
    | 🇪🇸 Spanish | **76.48%** |
    | 🇯🇵 Japanese | **72.36%** |
    | 🇨🇳 Chinese | **67.54%** |
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.75rem; color:#6366f1;'>"
        "CIS-423 · Computational Intelligence</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Main Content — Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌍 Cross-Lingual Sentiment Analyzer</h1>
    <p>Type a review in any language. The model predicts sentiment — even for languages it was never trained on.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Stats Row
# ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-value">6</div><div class="stat-label">Languages Supported</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-value">81%</div><div class="stat-label">English Accuracy</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-value">120K</div><div class="stat-label">Training Reviews</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><div class="stat-value">Zero</div><div class="stat-label">Target Language Data</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
with st.spinner("🔄 Loading XLM-RoBERTa model..."):
    tokenizer, model = load_model()

st.markdown(
    '<p style="text-align:center; color:#10b981; font-size:0.85rem;">'
    '<span class="status-dot status-online"></span>Model loaded and ready</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Input Section
# ──────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("#### ✍️ Enter a Review")

EXAMPLES = {
    "🇬🇧 English (Positive)": "This product is absolutely amazing! Best purchase I've ever made. The quality is outstanding and it arrived so quickly.",
    "🇬🇧 English (Negative)": "Terrible product. It broke after two days. Complete waste of money. I want a full refund immediately.",
    "🇫🇷 French (Positive)": "Produit excellent ! La qualité est remarquable et la livraison très rapide. Je recommande vivement.",
    "🇩🇪 German (Negative)": "Schreckliches Produkt. Es ist nach zwei Tagen kaputt gegangen. Totale Geldverschwendung.",
    "🇪🇸 Spanish (Neutral)": "El producto está bien, nada especial. Cumple su función pero esperaba algo mejor por el precio.",
    "🇯🇵 Japanese (Positive)": "この製品は素晴らしいです！品質が非常に高く、大満足しています。絶対にまた購入します。",
    "🇯🇵 Japanese (Negative)": "この製品は最悪です。二度と買いません。品質がひどくて、すぐに壊れました。",
    "🇨🇳 Chinese (Positive)": "非常好的产品！质量很高，物流也很快。强烈推荐给大家！",
    "🇨🇳 Chinese (Negative)": "太差了，质量很差，用了两天就坏了。浪费钱，不推荐购买。",
}

st.markdown(
    '<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.5rem;">'
    "Click an example or type your own review below:</p>",
    unsafe_allow_html=True,
)

example_cols = st.columns(3)
selected_example = None
example_keys = list(EXAMPLES.keys())
for i, key in enumerate(example_keys):
    col_idx = i % 3
    with example_cols[col_idx]:
        if st.button(key, key=f"example_{i}", use_container_width=True):
            selected_example = EXAMPLES[key]

default_text = selected_example if selected_example else ""
user_input = st.text_area(
    "Review Text",
    value=default_text,
    height=120,
    placeholder="Type or paste a review in English, French, German, Spanish, Japanese, or Chinese...",
    label_visibility="collapsed",
)

predict_btn = st.button("🔍  Analyze Sentiment", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Prediction & Results
# ──────────────────────────────────────────────
if predict_btn and user_input.strip():
    start_time = time.time()
    pred_class, probs = predict_sentiment(user_input.strip(), tokenizer, model)
    inference_time = time.time() - start_time

    label = LABELS[pred_class]
    emoji = EMOJIS[pred_class]
    color = COLORS[pred_class]
    confidence = probs[pred_class] * 100

    st.markdown(f"""
    <div class="result-{color}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">{label}</div>
        <div class="result-confidence">Confidence: {confidence:.1f}% · Inference: {inference_time:.2f}s</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Confidence Breakdown")

    neg_pct = probs[0] * 100
    neu_pct = probs[1] * 100
    pos_pct = probs[2] * 100

    st.markdown(f"""
    <div class="conf-container">
        <div class="conf-row">
            <div class="conf-label">😠 Negative</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill-neg" style="width:{neg_pct}%"></div></div>
            <div class="conf-pct">{neg_pct:.1f}%</div>
        </div>
        <div class="conf-row">
            <div class="conf-label">😐 Neutral</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill-neu" style="width:{neu_pct}%"></div></div>
            <div class="conf-pct">{neu_pct:.1f}%</div>
        </div>
        <div class="conf-row">
            <div class="conf-label">😊 Positive</div>
            <div class="conf-bar-bg"><div class="conf-bar-fill-pos" style="width:{pos_pct}%"></div></div>
            <div class="conf-pct">{pos_pct:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif predict_btn and not user_input.strip():
    st.warning("⚠️ Please enter some text before analyzing.")
