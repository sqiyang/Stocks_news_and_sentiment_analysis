import streamlit as st
import feedparser
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

st.set_page_config(page_title="Stock News Summarizer", layout="centered")

# Force CPU usage on Streamlit Cloud
device = torch.device("cpu")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=False  # Avoid meta tensor issue
    ).to(device)
    return tokenizer, model

# Load models
summarizer = load_summarizer()
tokenizer, model = load_sentiment_model()

labels = ['negative', 'neutral', 'positive']

# Sentiment scoring
def get_sentiment_score(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)[0]
        probs = torch.nn.functional.softmax(outputs, dim=-1)[0].cpu().detach().numpy()
        score = probs[2] - probs[0]  # positive - negative
        return dict(zip(labels, probs)), score
    except Exception as e:
        return {"error": str(e)}, 0

# Summarizer logic
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
            #st.write(sentiment)
            st.metric(label="Sentiment Score (-1 to 1)", value=f"{score:.2f}")
        else:
            st.error("Failed to summarize the headlines.")
