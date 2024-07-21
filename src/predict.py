import os
import joblib
from src.utils import preprocess_text
from .train import train_model

"""def load_model(model_path='models/best_classifier.joblib'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)
"""
    

def load_model(model_path='models/best_classifier.joblib'):
    if not os.path.exists(model_path):
        print(f"Model not found. Training new model...")
        train_model()  # This will create and save the model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

def predict_sentiment(text, model=None):
    if model is None:
        model = load_model()
    processed_text = preprocess_text(text)
    sentiment = model.predict([processed_text])[0]
    return 'positive' if sentiment == 1 else 'negative'

if __name__ == '__main__':
    model = load_model()
    while True:
        text = input("Enter text to classify or 'exit' to quit: ")
        if text.lower() == 'exit':
            break
        print(f"Sentiment: {predict_sentiment(text, model)}")