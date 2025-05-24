from pii_masking.mask import mask_text
from models import load_classifier, predict_class

# Load the classifier model
clf = load_classifier()

def classify_email(email_text: str):
    # Mask the PII
    masked_email, _ = mask_text(email_text)
    
    # Classify the masked email
    classification = predict_class(masked_email, clf)
    
    return masked_email, classification
