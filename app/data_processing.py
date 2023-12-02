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
