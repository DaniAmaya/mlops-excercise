"""
main.py - Script for training a machine learning model using modularized components

Overview:
    This script facilitates the training of a machine learning model by leveraging modularized components
    for data processing, preprocessing, model training, and other tasks. It follows a structured workflow
    to ensure code organization and reusability.

Dependencies:
    - Python 3.x
    - dotenv
    - pandas
    - Hugging Face Transformers
    - (other dependencies as needed)

Usage:
    1. Ensure that the required dependencies are installed. You can install them using:
        pip install -r requirements.txt

    2. Set up a .env file with the necessary configuration parameters. Example:
        DATA_PATH=/path/to/your/dataset.csv
        TEXT_COLUMN_NAME=text
        LABEL_COLUMN_NAME=label
        MODEL_NAME=distilbert-base-uncased
        TEST_SIZE=0.3
        VAL_SIZE=0.5
        RANDOM_STATE=4242
        NUM_TRAIN_EPOCHS=10
        OUTPUT_DIR=/path/to/save/model

    3. Run the script:
        python main.py

Workflow:
    1. Load environment variables from the .env file.
    2. Clean the dataset using the Cleaner class from the preprocessing module.
    3. Split the dataset into training, testing, and validation sets.
    4. Convert datasets to the Hugging Face Dataset format.
    5. Tokenize the data using the specified model.
    6. Train and save the machine learning model.
"""

# Libraries
import os

from data_processing import convert_to_datasets, split_data, tokenize_data
from dotenv import load_dotenv
from model import train_and_save_model
from preprocessing import Cleaner

load_dotenv()

# Variables
data_path = os.getenv("DATA_PATH")
text_column_name = os.getenv("TEXT_COLUMN_NAME")
label_column_name = os.getenv("LABEL_COLUMN_NAME")
model_name = os.getenv("MODEL_NAME")
test_size = float(os.getenv("TEST_SIZE"))
val_size = float(os.getenv("VAL_SIZE"))
random_state = int(os.getenv("RANDOM_STATE"))
num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS"))
output_dir = os.getenv("OUTPUT_DIR")


# Clean the dataset
cleaner = Cleaner()
df, num_labels, id2label, label2id = cleaner.clean_dataset(
    data_path, text_column_name, label_column_name
)

# Train/ Test/ Validation split
df_train, df_test, df_val = split_data(
    df, test_size=test_size, val_size=val_size, random_state=random_state
)

# Convert to HuggingFace Dataset
train_dataset, test_dataset, val_dataset = convert_to_datasets(
    df_train, df_test, df_val
)

# Tokenize data
tokenizer, tokenized_train, tokenized_test, tokenized_val = tokenize_data(
    model_name, train_dataset, test_dataset, val_dataset
)

# Train and save the model
model = train_and_save_model(
    model_name,
    num_labels,
    tokenizer,
    tokenized_train,
    tokenized_test,
    num_train_epochs,
    output_dir,
    random_state,
)
