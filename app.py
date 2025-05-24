from fastapi import FastAPI
from pydantic import BaseModel
from api import classify_email  # Function from api.py that handles classification

app = FastAPI()

class EmailInput(BaseModel):
    text: str

@app.post("/mask_and_classify")
def mask_and_classify(input: EmailInput):
    masked_email, classification = classify_email(input.text)
    return {"masked_email": masked_email, "classification": classification}
