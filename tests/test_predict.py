import unittest
from src.predict import predict_sentiment, load_model
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test directories if they don't exist
        os.makedirs('models', exist_ok=True)

        # Create a dummy model for testing
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        X = ['Це чудово', 'Це жахливо']
        y = [1, 0]
        pipeline.fit(X, y)
        
        cls.test_model_path = os.path.join('models', 'test_best_classifier.joblib')
        joblib.dump(pipeline, cls.test_model_path)

    def test_predict_sentiment(self):
        model = load_model(self.test_model_path)
        self.assertEqual(predict_sentiment("Це чудово", model), 'positive')
        self.assertEqual(predict_sentiment("Це жахливо", model), 'negative')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_model_path):
            os.remove(cls.test_model_path)

if __name__ == '__main__':
    unittest.main()