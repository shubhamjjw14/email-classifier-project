# recognizers/gliner_recognizer.py

from gliner import GLiNER
from config import TARGET_ENTITIES

class GLiNERRecognizer:
    def __init__(self):
        self.model = GLiNER.from_pretrained("urchade/gliner_base")  # Load the GLiNER model

    def recognize(self, text):
        entities = self.model.predict_entities(text, labels=TARGET_ENTITIES)  # Get the entities
        return [
            {
                "position": [ent["start"], ent["end"]],
                "label": ent["label"],
                "entity": text[ent["start"]:ent["end"]]
            }
            for ent in entities
        ]
