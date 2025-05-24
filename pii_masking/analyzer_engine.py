import re
from config import REGEX_PATTERNS
from pii_masking.gliner_recognizer import GLiNERRecognizer

def regex_detect(text):
    entities = []
    for label, pattern in REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text):
            start, end = match.span()
            entities.append({
                "position": [start, end],
                "label": label,
                "entity": match.group()
            })
    return entities

def deduplicate(entities):
    seen = set()
    result = []
    for e in sorted(entities, key=lambda x: x["position"][0]):
        key = tuple(e["position"])
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result

def analyze_text(text):
    gliner = GLiNERRecognizer()
    regex_entities = regex_detect(text)
    gliner_entities = gliner.recognize(text)
    return deduplicate(regex_entities + gliner_entities)
