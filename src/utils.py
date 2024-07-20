import re 
import spacy
import spacy.cli

try:
    nlp = spacy.load('uk_core_news_sm')
except OSError:
    print("Downloading the Ukrainian model...")
    spacy.cli.download('uk_core_news_sm')
    nlp = spacy.load('uk_core_news_sm')
    
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([token.lemma_ for token in nlp(text) if not token.is_stop])
    return text