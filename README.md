# Text Classification with Transformers
## Overview
This repository contains code for a text classification task using the Transformers library in PyTorch. The model is designed to classify text into three labels. It utilizes the Hugging Face Transformers library, which provides pre-trained transformer models for natural language processing tasks.

## Getting Started
### Prerequisites
* Python 3.x
* PyTorch
* Transformers library (transformers)

Install the required dependencies using:

```
pip install -r requirements.txt
```

## Training
To train the text classification model, run the following command:

```
docker build -t your_image_name .
docker run --env-file .env your_image_name
```
If you have an ARM Machine, run the following command instead:
```
docker buildx build --platform linux/arm64 -t your_image_name . 
```

Make sure to customize the training parameters, model architecture and HuggingFace Token on the .env file.

## Inference
After training, you can use the trained model for text classification. You can execute commands within the running container using:

```
docker exec -it your-container-name python predict.py "Your input text goes here."
```

This will output the predicted label for the given text.

## Model
The model architecture used in this project is based on the Hugging Face Transformers library. You can find more information about the model on the Hugging Face Model Hub.

## Data
The training data used for this project is not included in this repository. Remember to upload the .csv file to the 'data' folder.

## Results
The metrics for the validation test are the following:

<img width="383" alt="Screenshot 2023-12-04 at 6 52 42 PM" src="https://github.com/DaniAmaya/mlops-excercise/assets/20273279/ed2d2834-970e-4a3d-b17a-6ab95f36d060">

## Playground
If you want to play with the classification model, excecute the following on your docker container and a tab will appear on your browser:

```
docker run -p <local_port>:<container_port> <your_image_name> streamlit run stream_app.py
```

## Proposal
If you would like to read about a different approach to the text classification problem, proposed by the author, please refer to the Notebook: _Text_classifier_sentence_transformers.ipynb_

## Acknowledgments
Hugging Face Transformers: For providing state-of-the-art pre-trained transformer models.

## Author
Daniel Amaya
