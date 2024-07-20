import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from .utils import preprocess_text

def train_model():
    # Load data
    df = pd.read_csv('/home/anatolii-shara/Documents/text_classifier_2/data/sentiment_data.csv')
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42
    )

    # create pipeline for each classifier
    pipelines = {
        'nb': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ]),
        'rf': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier())
        ]),
        'lr': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
    }

    # set parameters for grid search for each classifier
    param_grid = {
        'nb': {'tfidf__max_features': [1000, 2000, 3000], 'clf__alpha': [0.1, 0.5, 1.0]},
        'rf': {'tfidf__max_features': [1000, 2000, 3000], 'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [10, 20, 30]},
        'lr': {'tfidf__max_features': [1000, 2000, 3000], 'clf__C': [0.1, 0.5, 1.0]}
    }

    best_clf = None
    best_score = 0

    for name, pipeline in pipelines.items():
        grid_search = GridSearchCV(pipeline, param_grid[name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_score:
            best_clf = grid_search.best_estimator_
            best_score = grid_search.best_score_

    print(f'Best classifier: {best_clf}')
    print(f'Best score: {best_score}')

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the 'models' directory in the parent directory of the current script
    models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(models_dir, 'best_classifier.joblib')
    joblib.dump(best_clf, model_path)

if __name__ == '__main__':
    train_model()