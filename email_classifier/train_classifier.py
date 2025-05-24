import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle
import os

# Step 1: Load dataset
df = pd.read_csv('emails_masked_sample.csv')  # Must contain 'email' and 'type' columns

# Step 2: Split into train (90%) and temp (10%)
train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['type'])

# Step 3: Split temp into eval (5%) and test (5%)
eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['type'])


# Step 4: Save CSVs
os.makedirs("data", exist_ok=True)
train_df.to_csv('data/train.csv', index=False)
eval_df.to_csv('data/eval.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print("[INFO] Dataset split complete. Files saved to data/train.csv, data/eval.csv, data/test.csv")

# Step 5: Prepare data
texts = train_df['email'].tolist()
labels = train_df['type'].tolist()
eval_texts = eval_df['email'].tolist()
eval_labels = eval_df['type'].tolist()

# Step 6: Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
eval_labels = label_encoder.transform(eval_labels)

# Step 7: Generate embeddings
print("[INFO] Generating sentence embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_train = embedder.encode(texts, show_progress_bar=True)
X_eval = embedder.encode(eval_texts, show_progress_bar=True)

# Step 8: Train classifier
print("[INFO] Training XGBoost classifier...")
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, labels)

# Step 9: Evaluate on validation set
eval_preds = clf.predict(X_eval)
print("\n[RESULTS] Evaluation Report (on validation set):\n")
print(classification_report(eval_labels, eval_preds, target_names=label_encoder.classes_))

# Step 10: Save model
os.makedirs('classifier', exist_ok=True)
with open('classifier/model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("[INFO] Model saved to classifier/model.pkl")
