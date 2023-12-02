# Libraries
from datetime import datetime

import evaluate
import numpy as np
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
    num_train_epochs,
    output_dir,
    random_state,
):
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

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Modify the output directory or model name with the timestamp
    output_dir_with_timestamp = f"{output_dir}{timestamp}_text_classification_model"

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
    trainer.save_model()

    return model
