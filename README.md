📈 Stock News Summarizer & Sentiment Analyzer
This is a simple and interactive Streamlit web application that fetches the latest stock news headlines, summarizes them using a pre-trained transformer model, and analyzes the overall sentiment of the summarized news.


🚀 Features
🔎 Enter any stock symbol (e.g., AAPL, TSLA, MSFT)

📰 Fetch latest headlines from Yahoo Finance RSS feeds

📝 Summarize multiple news headlines using a fine-tuned BART model

📊 Perform sentiment analysis on the summarized news using Twitter RoBERTa

⚖️ Sentiment score range from -1 (negative) to +1 (positive)

🧠 Models Used
Summarization: sshleifer/distilbart-cnn-12-6 — a distilled BART model fine-tuned for CNN/DailyMail summarization.

Sentiment Analysis: cardiffnlp/twitter-roberta-base-sentiment-latest — a RoBERTa model trained on Twitter data for classifying text as positive, neutral, or negative.

🛠️ Installation
Clone the repo

bash
Copy
Edit
git clone https://github.com/yourusername/stock-news-summarizer.git
cd stock-news-summarizer
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
📦 Dependencies
streamlit

feedparser

torch

transformers

numpy

You can install them all using:

bash
Copy
Edit
pip install streamlit feedparser torch transformers numpy
📸 Example Output
text
Copy
Edit
Stock Symbol: TSLA

Latest Headlines:
- Tesla stock surges after earnings
- Elon Musk teases new product launch
...

📝 Summary:
Tesla reported strong earnings with higher-than-expected revenue. Elon Musk also hinted at a new product launch, boosting investor confidence...

📊 Sentiment Score: +0.67 (Positive)
⚙️ Deployment
The app is designed to work on Streamlit Cloud, using CPU-only inference for resource compatibility.

📄 License
This project is open-source under the MIT License.

🙌 Acknowledgements
Hugging Face Transformers

Streamlit

Yahoo Finance RSS Feeds

💡 Future Improvements
Add multi-language support for global stocks

Visualize sentiment over time

Enable user-defined date ranges

Integrate with stock price data for correlation insights

