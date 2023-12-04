"""
inference.py - Module for classifying new text using a trained sequence classification model

Overview:
    This module provides a method for classifying new text using a pre-trained sequence classification model. It loads the latest model
    from a specified directory, tokenizes the input text, performs inference, and returns the predicted class label.

Methods:
    - get_latest_subdirectory(directory): Method to order the models trained and get the latest model's subdirectory.
    - classify_new_text(output_dir, new_text, id2label): Method to classify unknown text using the latest pre-trained model.

Dependencies:
    - os
    - torch
    - transformers

Usage:
    from inference import classify_new_text, get_latest_subdirectory

    # Example usage
    output_dir = "/path/to/your/model" # ex: './results/'
    new_text = "Your new text here." # ex: 'novo creme facial'

    # Get the latest subdirectory containing the model
    latest_subdirectory = get_latest_subdirectory(output_dir)

    # Load the model and classify the new text
    id2label = {0: 'MLB-FACIAL_SKIN_CARE_PRODUCTS', 1: 'MLB-MAKEUP', 2: 'MLB-BEAUTY_AND_PERSONAL_CARE_SUPPLIES'}  # Map class indices to labels
    predicted_label = classify_new_text(latest_subdirectory, new_text, id2label)

    print(f"The predicted class label for the new text is: {predicted_label}")

Note:
    - Ensure that the necessary dependencies (os, torch, transformers) are installed before using this module.
    - The id2label dictionary is required to map the model's class indices to human-readable class labels.
"""

# Libraries
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Method to order the models trained and get the latest model
def get_latest_subdirectory(directory):
    # Get a list of all subdirectories in the specified directory
    subdirectories = [
        subdir
        for subdir in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, subdir))
    ]

    # If there are no subdirectories, return None
    if not subdirectories:
        return None

    # Get the full path of each subdirectory along with its last modification time
    subdirectory_paths = [
        (subdir, os.path.getmtime(os.path.join(directory, subdir)))
        for subdir in subdirectories
    ]

    # Find the subdirectory with the latest modification time
    latest_subdirectory = max(subdirectory_paths, key=lambda x: x[1])[0]

    # Return the full path to the latest subdirectory
    return os.path.join(directory, latest_subdirectory)


# Method to classify unknown text
def classify_new_text(output_dir, new_text, id2label):
    latest_subdirectory = get_latest_subdirectory(output_dir)

    # Load the saved model
    model = AutoModelForSequenceClassification.from_pretrained(latest_subdirectory)
    tokenizer = AutoTokenizer.from_pretrained(latest_subdirectory)

    # Tokenize the new text
    tokenized_text = tokenizer(new_text, return_tensors="pt")

    # Forward pass to get logits
    with torch.no_grad():
        logits = model(**tokenized_text).logits

    # Get predicted class index
    predicted_class = torch.argmax(logits).item()

    return id2label[predicted_class]
