# Import necessary libraries
import os
import re
import pickle
import pandas as pd
import tensorflow as tf
import nltk

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and vectorizer
model = load_model('sentiment_model.h5')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    count_vectorizer = pickle.load(f)

# Initialize lemmatizer and stop words (excluding negations)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'}

# Preprocess user input for prediction
def preprocess_text(text, skip_language_detection=False):
    try:
         # Detect and translate non-English text (if detection not skipped)
        if not skip_language_detection:
            detected_lang = detect(text)
            print(f"Detected Language: {detected_lang}")
            if detected_lang != 'en':
                text = GoogleTranslator(source='auto', target='en').translate(text)
                print(f"Translated text: {text}")
        else:
            print("Language detection skipped")

        # Normalize and clean text
        text = text.lower()
        text = re.sub(r"\b(can't|won't|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|n't)\b", "not", text)
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text

    except Exception as e:
        print(f"Error in preprocessing text: {str(e)}")
        return text


# Convert sentiment score to class label
def classify_sentiment(score):
    if score >= 0.7:
        return 'Positive'
    elif score < 0.4:
        return 'Negative'   
    else:
        return 'Neutral'

# Make predictions for one or more reviews
def predict_sentiment(texts, batch_size=512, skip_language_detection=False):

    cleaned_texts = [preprocess_text(text, skip_language_detection=skip_language_detection) for text in texts]
    text_vectors = count_vectorizer.transform(cleaned_texts)
    
    # Initialize lists for batched predictions
    sentiment_scores = []
    sentiment_labels = []
    
    # Process in batches to avoid memory overload
    for i in range(0, text_vectors.shape[0], batch_size):
        batch_vectors = text_vectors[i:i+batch_size].toarray()  # Convert batch to array
        batch_predictions = model.predict(batch_vectors)
        
        batch_scores = [round(float(p[0]), 2) for p in batch_predictions]
        batch_labels = [classify_sentiment(score) for score in batch_scores]
        
        sentiment_scores.extend(batch_scores)
        sentiment_labels.extend(batch_labels)
    
    return sentiment_scores, sentiment_labels

    
# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload page route
@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')


# API to handle sentiment prediction (single review)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    reviews = data.get('reviews', [])

    if not reviews:
        return jsonify({'error': 'No reviews found in the request'}), 400

    # Enable language detection and translation for single review
    sentiment_scores, sentiment_labels = predict_sentiment(
        reviews,
        skip_language_detection=False  
    )

    return jsonify({
        'sentiment_scores': sentiment_scores,
        'sentiment_labels': sentiment_labels
    })

# API to handle CSV file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        df = pd.read_csv(file, encoding='utf-8')

        # Attempt to find a relevant review column
        possible_review_columns = ['review', 'reviews', 'comment', 'comments', 'text', 'Text', 'reviews.text']
        review_column = next((col for col in possible_review_columns if col in df.columns), None)  

        if review_column is None:
            return jsonify({'error': 'No review column found in the CSV file'}), 400
        
        reviews = df[review_column].dropna().tolist()
        print(f"Loaded {len(reviews)} reviews from file.")

        if not reviews:
            return jsonify({'error': 'No valid reviews found in the file'}), 400
        
        # Paralel Preprocessing
        with ThreadPoolExecutor() as executor:
            cleaned_texts = list(executor.map(lambda r: preprocess_text(r, skip_language_detection=True), reviews))

        # Predict in batches
        batch_size = 512
        sentiment_scores = []

        def process_batch(batch_reviews):
            vectors = count_vectorizer.transform(batch_reviews).toarray()
            batch_predictions = model.predict(vectors)
            return batch_predictions

        with ThreadPoolExecutor() as executor:
            all_batches = [cleaned_texts[i:i + batch_size] for i in range(0, len(cleaned_texts), batch_size)]
            batch_predictions = list(executor.map(process_batch, all_batches))


        for batch in batch_predictions:
            sentiment_scores.extend([round(float(p[0]), 2) for p in batch])

        # Calculate overall sentiment distribution
        overall_score = round(float(sum(sentiment_scores) / len(sentiment_scores)), 2) if sentiment_scores else 0.5
        overall_sentiment = classify_sentiment(overall_score)

        print(f" Overall Sentiment Score: {overall_score} - Label: {overall_sentiment}")

        # Distribution counts
        count_positive = sum(1 for s in sentiment_scores if classify_sentiment(s) == 'Positive')
        count_neutral  = sum(1 for s in sentiment_scores if classify_sentiment(s) == 'Neutral')
        count_negative = sum(1 for s in sentiment_scores if classify_sentiment(s) == 'Negative')
        total = len(sentiment_scores)
        
        distribution_percentage = {
            'Positive': round(count_positive / total * 100),
            'Neutral': round(count_neutral / total * 100),
            'Negative': round(count_negative / total * 100)
        }

        distribution_counts = {
            'Positive': count_positive,
            'Neutral': count_neutral,
            'Negative': count_negative
        }

        print(f"Distribution Summary (percent): {distribution_percentage}")
        print(f"Distribution Summary (counts): {distribution_counts}")

        return jsonify({
            'overall_score': overall_score,
            'overall_sentiment': overall_sentiment,
            "distribution": {
                "percent": distribution_percentage,
                "counts": distribution_counts
            }
        })

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({'error': f"Error processing file: {str(e)}"}), 500

# API to detect language
@app.route('/detect_language', methods=['POST'])
def detect_language():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text found in the request'}), 400

        detected_lang = detect(text)

        print(f" Detected Language: {detected_lang}")
        
        return jsonify({'language': detected_lang})
        
    except Exception as e:
        print(f"Error detecting language: {str(e)}")
        return jsonify({'error': f"Error detecting language: {str(e)}"}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True, threaded=True)