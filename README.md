# Ma513
Hands-on : Machine Learning for cyber security

# BERT-Based Named Entity Recognition (NER)

## Description
This project implements a Named Entity Recognition (NER) system using a BERT-based architecture. The model is trained to classify tokens into predefined categories using a dataset annotated in the BIO format.

## Features
- Token classification using BERT (e.g., `bert-large-uncased`).
- Dataset preprocessing and tokenization using the Hugging Face Transformers library.
- Training and evaluation with the Hugging Face Trainer API.
- Support for GPU acceleration.

## Installation
To run this project, you need Python 3.7 or later and the following Python packages:

```bash
pip install torch transformers datasets pandas scikit-learn
```

## Dataset
The training and validation datasets should be in JSONLINES format with the following structure:
- Each line is a JSON object containing:
  - `tokens`: A list of tokens.
  - `ner_tags`: A list of corresponding NER tags for each token.

Example:
```json
{"tokens": ["The", "company", "Apple", "is", "based", "in", "California"], "ner_tags": ["O", "O", "B-Entity", "O", "O", "O", "B-Location"]}
```

## Usage

1. **Prepare the Dataset**
   Place your training and validation datasets in the directory `NER-TRAINING` with filenames:
   - `NER-TRAINING.jsonlines`
   - `NER-VALIDATION.jsonlines`

2. **Run the Code**
   Execute the main script:
   ```bash
   python bert_ner_training.py
   ```

3. **Model Output**
   - The trained model and tokenizer will be saved in the directory `BERT-LARGE-UNCASED`.
   - Training logs and results are stored in the directories `./logs` and `./results`.

## Key Files
- `bert_ner_training.py`: Main script for preprocessing, training, and saving the model.
- `NER-TRAINING.jsonlines`: Training dataset.
- `NER-VALIDATION.jsonlines`: Validation dataset.

## Model Details
- Pretrained Model: `google-bert/bert-large-uncased`
- Number of Labels: 7
- Training Parameters:
  - Learning Rate: `1e-5`
  - Batch Size: `16`
  - Epochs: `30`
  - Weight Decay: `0.01`

## Output
- Final Model: `BERT-LARGE-UNCASED/`
- Logs: `./logs/`
- Results: `./results/`

## Contributing
If you'd like to contribute, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) for the tokenization and model APIs.
- Dataset and task inspired by the BIO format for Named Entity Recognition.

