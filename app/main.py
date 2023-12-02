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
