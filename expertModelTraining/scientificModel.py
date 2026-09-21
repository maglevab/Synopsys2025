from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import roc_auc_score
import torch
import os

train_data = pd.read_csv('/Users/arahan/Downloads/ai-ga-dataset.csv')
print(train_data.head())
train_data["label"].value_counts()

train0 = train_data[train_data["label"] == 0]
train1 = train_data[train_data["label"] == 1]

train0 = train0.iloc[:3000]
train1 = train1.iloc[:3000]

trainNew = pd.concat([train0, train1]).reset_index(drop=True).sample(frac = 1, random_state = 50)

train = trainNew.sample(frac=0.8)
validation = trainNew.drop(train.index)

print(len(train))
print(len(validation))

model_checkpoint = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["abstract"], truncation = True)

train = Dataset.from_pandas(train)
validation = Dataset.from_pandas(validation)
train_token = train.map(preprocess_function, batched=True)
validation_token = validation.map(preprocess_function, batched=True)

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

#Hyperparameters
metric_name = "roc_auc"
model_name = "deberta-v3-xsmall"
train_batch_size = 2
eval_batch_size = 8
grad_acc = 4

num_steps = len(train) // (train_batch_size * grad_acc)
print(num_steps)
args = TrainingArguments(
    "deberta-v3-xsmall-finetuned_1",
    evaluation_strategy = "steps",
    save_strategy = "steps",
    eval_steps = num_steps,
    save_steps = num_steps,
    learning_rate=2e-5,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=grad_acc,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model=metric_name,
    report_to='none'
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    auc = roc_auc_score(labels, probs[:,1], multi_class='ovr')
    return {"roc_auc": auc}

trainer = Trainer(
    model,
    args,
    train_dataset = train_token,
    eval_dataset = validation_token,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)
trainer.train()

# Save model
save_dir = "/Users/arahan/Desktop/Synopsys 2025/Models/ScientificModel"
trainer.save_model(save_dir)
