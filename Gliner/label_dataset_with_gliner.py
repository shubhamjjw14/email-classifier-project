import pandas as pd
from gliner import GLiNER
from tqdm import tqdm

# Load GLiNER model
model = GLiNER.from_pretrained("urchade/gliner_base")

# Define target PII entities
TARGET_ENTITIES = [
    "full_name", "email", "phone_number", "dob",
    "aadhar_num", "credit_debit_no", "cvv_no", "expiry_no"
]

# Load dataset
df = pd.read_csv("combined_emails_with_natural_pii.csv")

# Ensure the text column exists
assert "email" in df.columns, "Dataset must contain a 'email' column"

# Detect entities using GLiNER
def detect_entities(text):
    try:
        entities = model.predict_entities(text, labels=TARGET_ENTITIES)
        return [
            {
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"],
                "entity": text[ent["start"]:ent["end"]]
            }
            for ent in entities
        ]
    except Exception as e:
        print(f"Error on text: {text[:30]}... -> {e}")
        return []

# Apply entity detection
tqdm.pandas()
df["detected_entities"] = df["email"].progress_apply(detect_entities)

# Save labeled dataset
df.to_csv("labeled_emails_with_gliner.csv", index=False)

print("✅ Dataset labeled and saved as 'labeled_emails_with_gliner.csv'")
