from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Load models and vectorizers
naive_bayes_count_model = load('naive_bayes_count_model.joblib')
naive_bayes_tfidf_model = load('naive_bayes_tfidf_model.joblib')
logistic_count_model = load('logistic_count_model.joblib')
logistic_tfidf_model = load('logistic_tfidf_model.joblib')
count_vectorizer = load('count_vectorizer.joblib')
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the sentiment of a given text based on the specified model and vectorizer types.

    Parameters:
        None

    Returns:
        A JSON response containing the following fields:
        - text (str): The input text.
        - model_type (str): The type of model used for prediction.
        - vectorizer_type (str): The type of vectorizer used for feature extraction.
        - sentiment (str): The predicted sentiment of the input text ('Positive' or 'Negative').

    Raises:
        HTTPException: If the specified model or vectorizer type is invalid.
    """
    data = request.get_json()
    text = data['text']
    model_type = data['model_type']
    vectorizer_type = data['vectorizer_type']

    # Select the appropriate model and vectorizer
    if model_type == 'naive_bayes' and vectorizer_type == 'count':
        vectorizer = count_vectorizer
        model = naive_bayes_count_model
    elif model_type == 'naive_bayes' and vectorizer_type == 'tfidf':
        vectorizer = tfidf_vectorizer
        model = naive_bayes_tfidf_model
    elif model_type == 'logistic' and vectorizer_type == 'count':
        vectorizer = count_vectorizer
        model = logistic_count_model
    elif model_type == 'logistic' and vectorizer_type == 'tfidf':
        vectorizer = tfidf_vectorizer
        model = logistic_tfidf_model
    else:
        return jsonify({"error": "Invalid model or vectorizer type"}), 400

    # Process the input text
    input_vec = vectorizer.transform([text])
    prediction = model.predict(input_vec)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'

    return jsonify({
        "text": text,
        "model_type": model_type,
        "vectorizer_type": vectorizer_type,
        "sentiment": sentiment
    })

if __name__ == '__main__':
    app.run(debug=True)
