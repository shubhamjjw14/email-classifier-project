import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
import numpy as np
import torch

# Load data
train_df = pd.read_csv('data/train.csv')
eval_df = pd.read_csv('data/eval.csv')
test_df = pd.read_csv('data/test.csv')

# Encode labels
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['type'])
eval_df['label'] = label_encoder.transform(eval_df['type'])
test_df['label'] = label_encoder.transform(test_df['type'])

# Save label classes for inference
label_classes = label_encoder.classes_
with open("classifier/label_classes.txt", "w") as f:
    for label in label_classes:
        f.write(label + "\n")

# Convert to Hugging Face Dataset
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df[['email', 'label']]),
    'eval': Dataset.from_pandas(eval_df[['email', 'label']]),
    'test': Dataset.from_pandas(test_df[['email', 'label']])
})

# Tokenizer and preprocessing
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(example):
    return tokenizer(example['email'], padding='max_length', truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(['email'])

# Model
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": np.mean(preds == labels),
    }

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./classifier/distilbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate on validation set
print("\n[RESULTS] Evaluation on Validation Set:\n")
preds_output = trainer.predict(dataset['eval'])
pred_labels = np.argmax(preds_output.predictions, axis=1)
print(classification_report(dataset['eval']['label'], pred_labels, target_names=label_encoder.classes_))

# Save model + tokenizer
trainer.save_model("classifier/distilbert")
tokenizer.save_pretrained("classifier/distilbert")

print("\n[INFO] DistilBERT model and tokenizer saved to classifier/distilbert")
