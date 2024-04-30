from flask import Flask, render_template, request
from joblib import load
import numpy as np

# Load the trained models and vectorizers
lr_bow_model = load('lr_model_bow.joblib')
bow_vectorizer = load('lr_vectorizer_bow.joblib')

def preprocess_review(review, vectorizer, model_type='tfidf'):
    if model_type == 'tfidf' or model_type == 'bow':
        # Transform the review using the loaded vectorizer (TF-IDF or BoW)
        review_transformed = vectorizer.transform([review])
    elif model_type == 'word2vec':
        # Assuming 'w2v_model' is your trained Word2Vec model
        review_transformed = np.array([document_vector(w2v_model, review.split())])
    return review_transformed

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form['review']
        processed_review = preprocess_review(review, bow_vectorizer, 'bow')
        prediction = lr_bow_model.predict(processed_review)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        return render_template('index.html', original_input={'Review': review},
                               result=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)