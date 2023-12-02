# Libraries
import re
import unicodedata

import evaluate
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Variables
data_path = "dataset_items.csv"  # @param {type:"string"}
text_column_name = "text"  # @param {type:"string"}
label_column_name = "label"  # @param {type:"string"}

model_name = "distilbert-base-uncased"  # @param {type:"string"}
test_size = 0.3  # @param {type:"number"}
val_size = 0.5  # @param {type:"number"}
# num_labels = 3  # @param {type:"number"}
random_state = 4242  # @param {type:"number"}

num_train_epochs = 10  # @param {type:"number"}
output_dir = "./results/"  # @param {type:"string"}

# Process dataset
df = pd.read_csv("dataset_items.csv")
df.columns = [text_column_name, label_column_name + "_original"]
id2label = dict(enumerate(df[label_column_name + "_original"].unique()))
num_labels = len(id2label.keys())
label2id = {v: k for k, v in id2label.items()}

df.loc[:, [label_column_name]] = (
    df.copy().loc[:, [label_column_name + "_original"]].replace(label2id)
)


# Clean dataset
class Cleaner:
    def __init__(self):
        pass

    def put_line_breaks(self, text):
        text = text.replace("</p>", "</p>\n")
        return text

    def remove_html_tags(self, text):
        cleantext = BeautifulSoup(text, "lxml").text
        return cleantext

    def lower_case(self, text):
        return text.lower()

    def remove_accent(self, text):
        ascii_text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("utf-8")
            .strip()
            .replace("[^\w\s]", "")
        )
        return ascii_text

    def remove_special_characters(self, text):
        text = text.strip()
        PATTERN = r"[^a-zA-Z0-9]"  # only extract alpha-numeric characters
        text = re.sub(PATTERN, r" ", text)
        return text

    def clean(self, text):
        text = self.put_line_breaks(text)
        text = self.remove_html_tags(text)
        text = self.lower_case(text)
        text = self.remove_accent(text)
        text = self.remove_special_characters(text)
        return text


cleaner = Cleaner()
df["text_cleaned"] = df[text_column_name].apply(cleaner.clean)

# Train/ Test/ Validation split
df_train, test_val_data = train_test_split(
    df, test_size=test_size, random_state=random_state
)
df_test, df_val = train_test_split(
    test_val_data, test_size=val_size, random_state=random_state
)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)
val_dataset = Dataset.from_pandas(df_val)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples):
    return tokenizer(examples["text_cleaned"], truncation=True)


tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

# Train model
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


seed = random_state
set_seed(seed)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("label_classification_model")

# Classification report
# Metricas para multi-label: confusion matrix, precision, recall, f1-score

# Test
preds = trainer.predict(tokenized_test)
preds = np.argmax(preds[:3][0], axis=1)  # preds[:3][1]
GT = df_test["label"].tolist()
print(classification_report(GT, preds))

# Validation
preds = trainer.predict(tokenized_val)
preds = np.argmax(preds[:3][0], axis=1)  # preds[:3][1]
GT = df_val["label"].tolist()
print(classification_report(GT, preds))
