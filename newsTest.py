from keras._tf_keras.keras.models import load_model
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import Dataset
from sklearn.metrics import roc_auc_score
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

train_data = pd.read_csv("/Users/arahan/Desktop/Synopsys2025/Data/newsv4.csv")

train_data = train_data.sample(n=1000, random_state = 1, ignore_index = True)
train = train_data.sample(frac = 0.8)
validation = train_data.drop(train.index)

print(len(train))
print(len(validation))

model_checkpoint = "/Users/arahan/PycharmProjects/Synopsys2025/expert Trainings/deberta"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train = Dataset.from_pandas(train)
validation = Dataset.from_pandas(validation)
train_token = train.map(preprocess_function, batched=True)
validation_token = validation.map(preprocess_function, batched=True)

print("creating model")
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, force_download = True)
# Hyperparameters
metric_name = "roc_auc"
model_name = "deberta"
train_batch_size = 2
eval_batch_size = 8
grad_acc = 4

num_steps = len(train) // (train_batch_size * grad_acc)

args = TrainingArguments(
    "deberta_expert_news",
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

print("making trainer")
trainer = Trainer(
    model,
    args,
    train_dataset = train_token,
    eval_dataset = validation_token,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

if __name__ == '__main__':
    print("training")
    trainer.train()

    print('saving')
    trainer.save_model("/Users/arahan/Desktop/Synopsys2025/Models/newsExpertModel")
