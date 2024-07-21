from flask import Flask, request, jsonify
from src.predict import predict_sentiment, load_model

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    sentiment = predict_sentiment(text, model)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)