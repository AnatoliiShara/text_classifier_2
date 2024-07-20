import unittest
import pandas as pd
from src.train import train_model
import os

class TestTrain(unittest.TestCase):
    def setUp(self):
        # Create test directories if they don't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)

    def test_train_model(self):
        # create a dummy CSV file for testing
        df = pd.DataFrame({
            'text': ["Це чудово", "Це жахливо"],
            'sentiment': [1, 0]
        })
        test_csv_path = os.path.join('data', 'test_sentiment_data.csv')
        df.to_csv(test_csv_path, index=False)

        train_model()

        test_model_path = os.path.join('models', 'best_classifier.joblib')
        self.assertTrue(os.path.exists(test_model_path))

    def tearDown(self):
        # Clean up test files
        test_csv_path = os.path.join('data', 'test_sentiment_data.csv')
        test_model_path = os.path.join('models', 'best_classifier.joblib')
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)
        if os.path.exists(test_model_path):
            os.remove(test_model_path)

if __name__ == '__main__':
    unittest.main()