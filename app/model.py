"""
model.py - Module for training and saving a sequence classification model

Overview:
    This module provides a function for training a sequence classification model using the Hugging Face Transformers library.
    The model is trained on tokenized text data and evaluated on a separate tokenized test set.

Functions:
    - train_and_save_model(
        model_name,
        num_labels,
        tokenizer,
        tokenized_train,
        tokenized_test,
        num_train_epochs,
        output_dir,
        random_state
    ): Initializes, trains, and saves a sequence classification model.

Dependencies:
    - datetime
    - evaluate
    - numpy
    - transformers

Usage:
    from model import train_and_save_model

    # Example usage
    trained_model = train_and_save_model(
        model_name="bert-base-uncased",
        num_labels=2,
        tokenizer=tokenizer,
        tokenized_train=train_dataset,
        tokenized_test=test_dataset,
        num_train_epochs=5,
        output_dir="/path/to/save/model",
        random_state=42,
    )

Note:
    - Make sure to install the required dependencies (datetime, evaluate, numpy, transformers) before using this module.
    - Ensure that the tokenized datasets and tokenizer are prepared before calling this function.
"""

# Libraries
from datetime import datetime

import evaluate
import numpy as np
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def train_and_save_model(
    model_name,
    num_labels,
    tokenizer,
    tokenized_train,
    tokenized_test,
    tokenized_val,
    df_val,
    num_train_epochs,
    output_dir,
    random_state,
):
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Define metrics and data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    seed = random_state
    set_seed(seed)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Modify the output directory or model name with the timestamp
    output_dir_with_timestamp = f"{output_dir}{timestamp}_text_classification_model"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir_with_timestamp,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
    )

    # Train the model
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

    # Save the model with the modified name
    # Includes all the necessary information for continuing training or further evaluation
    trainer.save_model(output_dir_with_timestamp)

    # Save the model, tokenizer, and configuration in binary format
    # Ideal with dealing with the model outside of the training loop
    # You can load this model later using AutoModelForSequenceClassification.from_pretrained()
    model.save_pretrained(output_dir_with_timestamp)

    # Validation
    preds = trainer.predict(tokenized_val)
    preds = np.argmax(preds[:3][0], axis=1)  # preds[:3][1]
    GT = df_val["label"].tolist()
    print(classification_report(GT, preds))

    return model
