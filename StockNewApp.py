import streamlit as st
import feedparser
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Set device
device = torch.device("cpu")  # Force CPU usage on Streamlit Cloud

# Initialize summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# Load sentiment model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
labels = ['negative', 'neutral', 'positive']

# Sentiment function
def get_sentiment_score(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)[0]
        probs = torch.nn.functional.softmax(outputs, dim=-1)[0].cpu().detach().numpy()
        score = probs[2] - probs[0]  # positive - negative
        return dict(zip(labels, probs)), score
    except Exception as e:
        return {"error": str(e)}, 0

# Summarizer function
def summarize_headlines_locally(headlines):
    if not headlines:
        return None
    text = " ".join(headlines)
    try:
        input_length = len(text.split())
        max_len = min(130, input_length)
        summary = summarizer(text, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Summarizer Error: {e}"

# Streamlit UI
st.set_page_config(page_title="Stock News Summarizer", layout="centered")
st.title("📈 Stock News Summarizer & Sentiment Analyzer")

stock = st.text_input("Enter a Stock Symbol (e.g., AAPL, TSLA, MSFT)", "AAPL")

if st.button("Analyze"):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock}&region=US&lang=en-US"
    feed = feedparser.parse(url)

    if not feed.entries:
        st.warning("No headlines found.")
    else:
        headlines = [entry.title for entry in feed.entries[:10]]
        st.subheader("Latest Headlines")
        for h in headlines:
            st.markdown(f"- {h}")

        summary = summarize_headlines_locally(headlines)
        if summary:
            st.subheader("📝 Summary")
            st.write(summary)

            sentiment, score = get_sentiment_score(summary)
            st.subheader("📊 Sentiment Analysis")
            st.write(sentiment)
            st.metric(label="Sentiment Score (Positive - Negative)", value=f"{score:.2f}")
        else:
            st.error("Failed to summarize the headlines.")
