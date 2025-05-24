from datasets import load_dataset, Dataset
from transformers import GLiNERForTokenClassification, GLiNERTokenizer, Trainer, TrainingArguments
import evaluate
import json

# Load the dataset
def load_json_dataset(filepath):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

dataset = load_json_dataset("gliner_train_data.json")
dataset = dataset.train_test_split(test_size=0.1)

# Load tokenizer and model
tokenizer = GLiNERTokenizer.from_pretrained("urchade/gliner_base")
model = GLiNERForTokenClassification.from_pretrained("urchade/gliner_base")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

tokenized_datasets = dataset.map(tokenize, batched=True)

# Define metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)
    return metric.compute(predictions=predictions, references=labels)

# Training args
args = TrainingArguments(
    output_dir="./gliner_finetuned",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
