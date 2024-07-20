import unittest
import pandas as pd
from src.train import train_model
import os 

class TestTrain(unittest.TestCase):
    def test_train_model(self):
        # create a dummy CSV file for testing
        df = pd.DataFrame({
            'text': ["Це чудово", "Це жахливо"],
            'sentiment': [1, 0]
        })
        df.to_csv('data/sentiment_data.csv', index=False)
        
        train_model()
        self.assertTrue(os.path.exists('models/best_classifier.joblib'))
        os.remove('data/sentiment_data.csv')
        os.remove('models/best_classifier.joblib')
        
if __name__ == '__main__':
    unittest.main()
        