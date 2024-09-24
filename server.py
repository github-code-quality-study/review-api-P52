import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

for review in reviews:
    review['Timestamp'] = datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S')

valid_locations = {
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona"
}

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """
        path = environ["PATH_INFO"]
        method = environ["REQUEST_METHOD"]

        if method == "GET":
            query_params = parse_qs(environ["QUERY_STRING"])
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            filtered_reviews = reviews

            if location:
                if location in valid_locations:
                    filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]
                else:
                    start_response("400 Bad Request", [("Content-Type", "text/plain")])
                    return [b"Invalid Location"]

            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if review['Timestamp'] >= start_date]

            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if review['Timestamp'] <= end_date]

            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

            response = []
            for review in sorted_reviews:
                response.append({
                    "ReviewId": review['ReviewId'],
                    "ReviewBody": review['ReviewBody'],
                    "Location": review['Location'],
                    "Timestamp": review['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    "sentiment": review['sentiment']
                })

            response_body = json.dumps(response, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if method == "POST":
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
                post_params = parse_qs(post_data)

                review_body = post_params.get('ReviewBody', [None])[0]
                location = post_params.get('Location', [None])[0]

                if not review_body or not location:
                    start_response("400 Bad Request", [("Content-Type", "text/plain")])
                    return [b"ReviewBody and Location are required"]

                if location not in valid_locations:
                    start_response("400 Bad Request", [("Content-Type", "text/plain")])
                    return [b"Invalid Location"]

                review_id = str(uuid.uuid4())
                timestamp = datetime.now()

                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp,
                    "sentiment": self.analyze_sentiment(review_body)
                }

                reviews.append(new_review)

                response = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')
                }

                response_body = json.dumps(response, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
                return [str(e).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()