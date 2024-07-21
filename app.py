import os
from flask import Flask, request, jsonify
from src.predict import load_model, predict

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    result = predict(data, model)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
