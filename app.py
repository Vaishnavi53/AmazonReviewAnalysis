from flask import Flask, render_template, request
from joblib import load
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the model and vectorizer
model = load('sentiment_model.pkl')
vectorizer = load('tfidf_vectorizer.pkl')

# Ensure stopwords are downloaded
nltk.download('stopwords')
stp_words = set(stopwords.words('english'))

def clean_review(review):
    # Clean the review by removing stopwords
    return " ".join(word for word in review.split() if word not in stp_words)

def predict_sentiment(review_text):
    # Clean the review text
    cleaned_review = clean_review(review_text)
    # Transform the review using the TF-IDF vectorizer
    transformed_review = vectorizer.transform([cleaned_review]).toarray()
    # Predict the sentiment using the model
    prediction = model.predict(transformed_review)
    # Return the result
    return "Negative" if prediction[0] == 0 else "Positive"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['review']
    prediction = predict_sentiment(review_text)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
