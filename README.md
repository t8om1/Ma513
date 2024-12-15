# Named Entity Recognition (NER)

## Description

This project implements a Named Entity Recognition (NER) system. The model is trained to classify tokens into predefined categories using a dataset annotated in the BIO format. Additionally, predictions can be generated and saved in JSONLINES format for further evaluation or deployment.

## Features

- Token classification.
- Dataset preprocessing and tokenization using the Hugging Face Transformers library.
- Training and evaluation with the Hugging Face Trainer API.
- Support for GPU acceleration.
- Flexible prediction pipeline with results saved in JSONLINES format.

## Installation

To run this project, you need Python 3.7 or later and the following Python packages:

```bash
pip install torch transformers datasets pandas scikit-learn
```

## Dataset

The training, validation, and testing datasets should be in JSONLINES format with the following structure:

- Each line is a JSON object containing:
  - `tokens`: A list of tokens.
  - `ner_tags`: A list of corresponding NER tags for each token (optional for the test dataset).
  - `unique_id`: A unique identifier for each entry (only in the test dataset).

Example:

```json
{"tokens": ["The", "company", "Apple", "is", "based", "in", "California"], "ner_tags": ["O", "O", "B-Entity", "O", "O", "O", "B-Location"]}
```

## Usage

### 1. Training the Model

1. **Prepare the Dataset**
   Place your training and validation datasets in your directory with filenames:
   - `NER-TRAINING.jsonlines`
   - `NER-VALIDATION.jsonlines`

2. **Run the Training Script**
   Execute the training script (placed in the same directory):

   ```bash
   python train_model.py
   ```

3. **Model Output**

   - The trained model and tokenizer will be saved in the directory `YOUR RESULT DIRECTORY`.
   - Training logs and results are stored in the directories `./logs` and `./results`.

### 2. Generating Predictions

1. **Prepare the Test Dataset**
   Place your test dataset in the same directory with the filename:
   - `NER-TESTING.jsonlines`

2. **Run the Prediction Script**
   Execute the prediction script (placed in the same directory):

   ```bash
   python results_model.py
   ```

3. **Prediction Output**

   - The predictions will be saved in `NER-TESTING-PREDICTIONS.jsonlines` in JSONLINES format.
   - The script also evaluates the model on the validation dataset and prints the classification report.

## Key Files

- `train_model.py`: Main script for preprocessing, training, and saving the model.
- `results_model.py`: Script for generating predictions and evaluating the model.
- `NER-TRAINING.jsonlines`: Training dataset.
- `NER-VALIDATION.jsonlines`: Validation dataset.
- `NER-TESTING.jsonlines`: Test dataset.

## Model specification
You can use almost any pretrained model you want. Some exemple are:
- `google-bert/bert-large-uncased`
- `google-bert/bert-base-uncased`
- ...

## Model Details

- Number of Labels: 7
- Training Parameters:
  - Learning Rate: `1e-5`
  - Batch Size: `16`
  - Epochs: `30`
  - Weight Decay: `0.01`

## Output

- Final Model: `YOUR RESULT DIRECTORY/`
- Logs: `./logs/`
- Results: `./results/`
- Predictions: `NER-TESTING-PREDICTIONS.jsonlines`

## Contributing

If you'd like to contribute, please fork the repository and submit a pull request.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for the tokenization and model APIs.
- Dataset and task inspired by the BIO format for Named Entity Recognition.
- Mr. Atilla ALKAN for guiding us throughout this project


