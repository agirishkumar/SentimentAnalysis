import numpy as np
import torch
import nltk
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
eng_stopwords = stopwords.words('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(features, labels, model_type, vectorizer_type):
    """
    Trains a machine learning model using the given features and labels.

    Parameters:
        features (array-like): The input features for training the model.
        labels (array-like): The corresponding labels for the features.
        model_type (str): The type of model to train. Must be either 'naive_bayes' or 'logistic'.
        vectorizer_type (str): The type of vectorizer to use for feature extraction. Must be either 'count' or 'tfidf'.

    Returns:
        None

    Raises:
        ValueError: If the model_type or vectorizer_type is invalid.

    Prints:
        The accuracy of the trained model for the given model_type and vectorizer_type.

    Saves:
        The trained model as a joblib file with the name '{model_type}_{vectorizer_type}_model.joblib'.
        The vectorizer as a joblib file with the name '{vectorizer_type}_vectorizer.joblib'.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = None

    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(stop_words=eng_stopwords)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=eng_stopwords)
    
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Accuracy of {model_type} with {vectorizer_type}: {accuracy_score(y_test, preds)}")
    dump(model, f'{model_type}_{vectorizer_type}_model.joblib')
    dump(vectorizer, f'{vectorizer_type}_vectorizer.joblib')


def load_data():
    """
    Load the IMDB dataset from the 'stanfordnlp/imdb' dataset and split it into training and testing sets.

    Returns:
        train_data (pandas.DataFrame): The training data as a pandas DataFrame.
        test_data (pandas.DataFrame): The testing data as a pandas DataFrame.
    """
    imdb_dataset = load_dataset('stanfordnlp/imdb', split={'train': 'train', 'test': 'test'})
    train_data = imdb_dataset['train'].to_pandas()
    test_data = imdb_dataset['test'].to_pandas()
    return train_data, test_data

def main():
    """
    The main function that loads the train and test data, checks the class distribution, and trains models using CountVectorizer and TF-IDF.

    Returns:
        None
    """
    train_data, test_data = load_data()

    # Use the full dataset instead of a slice
    texts = train_data['text'].values  # Convert to numpy array
    labels = train_data['label'].values

    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))

    # Train models with CountVectorizer and TF-IDF
    for model_type in ['naive_bayes', 'logistic']:
        for vectorizer_type in ['count', 'tfidf']:
            train_model(texts, labels, model_type, vectorizer_type)

if __name__ == '__main__':
    main()

