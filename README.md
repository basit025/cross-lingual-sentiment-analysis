# Cross-Lingual Sentiment Analyzer

A live demo of **Zero-Shot Cross-Lingual Sentiment Analysis** using XLM-RoBERTa.

## What This App Does

Type a product review in **any of 6 languages** (English, French, German, Spanish, Japanese, Chinese), and the model predicts whether the sentiment is **Negative**, **Neutral**, or **Positive** — even for languages it was **never trained on**.

## How It Works

The model (`xlm-roberta-base`) was fine-tuned on **120,000 English-only** Amazon reviews mapped to 3 sentiment classes. Thanks to XLM-RoBERTa's cross-lingual SentencePiece embeddings, it transfers this knowledge to 5 other languages with zero additional training.

## Zero-Shot Accuracy

| Language | Accuracy |
|---|---|
| 🇬🇧 English | **81.00%** |
| 🇩🇪 German | **79.74%** |
| 🇫🇷 French | **77.32%** |
| 🇪🇸 Spanish | **76.48%** |
| 🇯🇵 Japanese | **72.36%** |
| 🇨🇳 Chinese | **67.54%** |

## Tech Stack

- **Model:** XLM-RoBERTa Base (278M parameters)
- **Framework:** PyTorch + Hugging Face Transformers
- **UI:** Streamlit
- **Weights:** Hosted on Hugging Face Hub

## Course

Computational Intelligence (CIS-423)
