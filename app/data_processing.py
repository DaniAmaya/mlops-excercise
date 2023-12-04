"""
data_processing.py - Module for data processing and tokenization

Overview:
    This module provides functions for splitting datasets into training, testing, and validation sets,
    converting datasets to the Hugging Face Dataset format, and tokenizing text data using Hugging Face Transformers.

Functions:
    - split_data(df, test_size, val_size, random_state): Splits a DataFrame into training, testing, and validation sets.
    - convert_to_datasets(df_train, df_test, df_val): Converts DataFrames to Hugging Face Dataset format.
    - tokenize_data(model_name, train_dataset, test_dataset, val_dataset): Tokenizes text data using a specified model.

Dependencies:
    - datasets
    - sklearn
    - transformers

Usage:
    from data_processing import split_data, convert_to_datasets, tokenize_data

Note:
    - Make sure to install the required dependencies (datasets, sklearn, transformers) before using this module.
"""

# Libraries
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


# Train/ Test/ Validation split
def split_data(df, test_size, val_size, random_state):
    df_train, test_val_data = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    df_test, df_val = train_test_split(
        test_val_data, test_size=val_size, random_state=random_state
    )

    return df_train, df_test, df_val


# Convert to HuggingFace Dataset
def convert_to_datasets(df_train, df_test, df_val):
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)
    val_dataset = Dataset.from_pandas(df_val)

    return train_dataset, test_dataset, val_dataset


# Tokenizer
def tokenize_data(model_name, train_dataset, test_dataset, val_dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text_cleaned"], truncation=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)

    return tokenizer, tokenized_train, tokenized_test, tokenized_val
