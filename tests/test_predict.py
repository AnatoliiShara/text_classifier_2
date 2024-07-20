import unittest
from src.predict import predict_sentiment, load_model
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # crate a dummy model for testing
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        X = ['Це чудово', 'Це жахливо']
        y = [1, 0]
        pipeline.fit(X, y)
        joblib.dump(pipeline, 'models/best_classifier.joblib')
        
    def test_predict_sentiment(self):
        model = load_model()
        text = 'Це чудово'
        self.assertEqual(predict_sentiment("Це чудово", model), 'positive')
        self.assertEqual(predict_sentiment("Це жахливо", model), 'negative')
        
    @classmethod
    def tearDownClass(cls) -> None:
        import os
        os.remove('models/best_classifier.joblib')
        
if __name__ == '__main__':
    unittest.main()
