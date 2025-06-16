# Stock News Summarizer & Sentiment Analyzer

A simple and powerful Streamlit web app that fetches the latest financial news for any stock symbol, summarizes the headlines using NLP, and performs sentiment analysis on the summarized content.

**Try it here:** [Stock News Summarizer App](https://stocksanalysis-5mork6kxedzapplt2pwwh3p.streamlit.app/)

---

## Features

- 🔎 Enter any **stock ticker symbol** (e.g., `AAPL`, `TSLA`, `MSFT`)
- 📰 Fetch the **latest 10 headlines** from Yahoo Finance RSS feeds
- 📝 Generate a **summary of the headlines** using a pre-trained transformer model
- 📊 Perform **sentiment analysis** on the summary using a RoBERTa-based sentiment classifier
- 📉 View a **sentiment score** between -1 (negative) and +1 (positive)

---

## Models Used

- **Summarization**  
  [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) — A distilled version of BART fine-tuned on the CNN/DailyMail dataset for abstractive summarization.

- **Sentiment Analysis**  
  [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — A RoBERTa model fine-tuned on Twitter data for sentiment classification into `positive`, `neutral`, and `negative`.

---

##  Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/stock-news-summarizer.git
   cd stock-news-summarizer
