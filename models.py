import pickle
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

def load_classifier():
    with open('classifier/model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return clf

def predict_class(email, clf):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    email_embedding = embedder.encode([email])
    return clf.predict(email_embedding)[0]
