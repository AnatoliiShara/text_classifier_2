import joblib
from utils import preprocess_text

def load_model():
    return joblib.load('models/best_classifier.joblib')

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