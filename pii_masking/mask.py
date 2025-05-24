from pii_masking.analyzer_engine import analyze_text
from pii_masking.anonymizer_engine import mask_entities

def mask_text(text):
    entities = analyze_text(text)
    masked = mask_entities(text, entities)
    return masked, entities
