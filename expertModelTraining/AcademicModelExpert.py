from keras._tf_keras.keras.models import load_model
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from datasets import Dataset
from sklearn.metrics import roc_auc_score

#train_data = pd.read_csv("/Users/arahan/Downloads/llm-detect-ai-generated-text (1)/train_essays.csv")
#print(train_data.head())

#print(train_data['generated'].value_counts())

train_daigt = pd.read_csv("/Users/arahan/Downloads/train_v2_drcat_02.csv")
print(train_daigt.head())
print(train_daigt.source.value_counts())


human = train_daigt[train_daigt['source'] == 'persuade_corpus'].sample(n=6000)
non_human = train_daigt[train_daigt['source'] != 'persuade_corpus']

char_human = set(list(''.join(human.text.to_list())))
char_non_human = set(list(''.join(non_human.text.to_list())))

chars_to_remove = ''.join([x for x in char_non_human if x not in char_human])

translation_table = str.maketrans('', '', chars_to_remove)
def remove_chars(s):
    return s.translate(translation_table)
non_human['text'] = non_human['text'].apply(remove_chars)

train_daigt = pd.concat([human, non_human]).reset_index(drop=True)
print(train_daigt.source.value_counts())

train = train_daigt.sample(frac=0.8)
validation = train_daigt.drop(train.index)

print(len(train))
print(len(validation))

model_checkpoint = "microsoft/deberta-v3-xsmall"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train = Dataset.from_pandas(train)
validation = Dataset.from_pandas(validation)
train_token = train.map(preprocess_function, batched=True)
validation_token = validation.map(preprocess_function, batched=True)

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
# Hyperparameters
metric_name = "roc_auc"
model_name = "deberta-v3-xsmall"
train_batch_size = 2
eval_batch_size = 8
grad_acc = 4

num_steps = len(train) // (train_batch_size * grad_acc)

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

from sklearn.metrics import roc_auc_score

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

test = pd.read_csv("/Users/arahan/Downloads/llm-detect-ai-generated-text (1)/test.csv")
test = Dataset.from_pandas(test)
test_token = test.map(preprocess_function, batched=True)

test_preds = trainer.predict(test_token)
logits = test_preds.predictions
probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
sub = pd.DataFrame()
sub['id'] = test['id']
sub['generated'] = probs[:,1]
sub.to_csv('submission.csv', index=False)
sub.head()

# Save model
save_dir = "/content/drive/MyDrive/Deep learning/model_1"
trainer.save_model(save_dir)