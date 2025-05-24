import pandas as pd
import json

df = pd.read_csv("labeled_emails_with_gliner.csv")

gliner_format = []
for _, row in df.iterrows():
    if isinstance(row["detected_entities"], str):
        try:
            entities = json.loads(row["detected_entities"].replace("'", '"'))
        except json.JSONDecodeError:
            continue
        gliner_format.append({
            "text": row["email"],
            "entities": entities
        })

with open("gliner_train_data.json", "w") as f:
    for item in gliner_format:
        f.write(json.dumps(item) + "\n")

print("✅ Dataset saved in GLiNER format to gliner_train_data.json")
