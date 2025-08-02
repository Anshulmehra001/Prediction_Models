import requests
from bs4 import BeautifulSoup
from transformers import pipeline


class IndianNewsScraper:
    def __init__(self):
        self.sentiment_analyzer = pipeline("text-classification", model="yiyanghkust/finbert-tone")

    def get_moneycontrol_news(self):
        url = "https://www.moneycontrol.com/news/business/stocks/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.select('.newslist li a')]
        return self.analyze_sentiment(headlines)

    def analyze_sentiment(self, headlines):
        results = self.sentiment_analyzer(headlines)
        return sum(1 for r in results if r['label'] == 'Positive') / len(results)