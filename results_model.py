import json
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from sklearn.metrics import classification_report

# Chemin vers le modèle préalablement entraîné
model_sauv = "./YOUR RESULT DIRECTORY"

# Chargement du modèle et du tokenizer sauvegardés après l'entraînement
model_reloaded = BertForTokenClassification.from_pretrained(model_sauv)
tokenizer_reloaded = BertTokenizerFast.from_pretrained(model_sauv)

# Fonction pour lire un fichier JSONLINES
# Chaque ligne doit contenir un objet JSON

def read_json(file):
    with open(file, 'r') as f:
        corpus_json = [json.loads(line) for line in f]
    return corpus_json

# Fonction pour mapper des valeurs dans une chaîne en fonction d'un dictionnaire
def replace(mot, dic):
    for key, value in dic.items():
        mot = mot.replace(str(key), str(value))
    return mot

# Dictionnaires de mappage pour les tags NER
dic_tag = {'O': 0, 'B-Entity': 1, 'B-Action': 2, 'I-Action': 3, 'I-Entity': 4, 'B-Modifier': 5, 'I-Modifier': 6}
dic_tag_inv = {v: k for k, v in dic_tag.items()}  # Inverse du dictionnaire

# Préparation des données pour la prédiction
def prepa_data_pred(file, tokenizer, labels=False):
    corpus = read_json(file)
    if labels:
        data = {
            "tokens": [[t if t != "\uf0b7" else "/uf0b7" for t in val["tokens"]] for val in corpus],
            "ner_tags": [val["ner_tags"] for val in corpus]
        }
    else:
        data = {
            "tokens": [[t if t != "\uf0b7" else "/uf0b7" for t in val["tokens"]] for val in corpus],
            "unique_id": [val["unique_id"] for val in corpus]
        }

    # Tokenisation des données
    tokenized_inputs = tokenizer(data['tokens'], truncation=True, padding=True, is_split_into_words=True)

    # Identification des tokens alignés
    is_labels = []
    for i in range(len(data["tokens"])):
        add = -1
        is_label = []
        for j in tokenized_inputs.word_ids(i):
            if j is not None and add != j:
                is_label.append(True)
                add = j
            else:
                is_label.append(False)
        is_labels.append(is_label)

    if labels:
        return tokenized_inputs, is_labels, data['ner_tags']
    else:
        return tokenized_inputs, is_labels, data

# Fonction pour obtenir les prédictions finales
def prediction_finale(model, data, islabel):
    with torch.no_grad():
        outputs = model(**{key: torch.tensor(val) for key, val in data.items()})
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    pred_words = []
    for i in range(len(predictions)):
        pred = []
        for j in range(len(predictions[i])):
            if islabel[i][j]:
                pred.append(replace(str(int(predictions[i][j])), dic_tag_inv))
        pred_words.append(pred)
    return pred_words

# Préparation des données de validation et de test
tokenize_val, isLabels_val, ner_tags_val = prepa_data_pred("data/NER-VALIDATION.jsonlines", tokenizer_reloaded, labels=True)
tokenize_test, isLabels_test, data_test = prepa_data_pred("data/NER-TESTING.jsonlines", tokenizer_reloaded, labels=False)

# Prédictions sur les ensembles de validation et de test
pred_val_final = prediction_finale(model_reloaded, tokenize_val, isLabels_val)
pred_test_final = prediction_finale(model_reloaded, tokenize_test, isLabels_test)

# Évaluation sur l'ensemble de validation
pred_val_flat = [item for sublist in pred_val_final for item in sublist]
ner_tags_val_flat = [item for sublist in ner_tags_val for item in sublist]
print(classification_report(ner_tags_val_flat, pred_val_flat))

# Sauvegarde des prédictions sur le jeu de test
output_file = "NER-TESTING-PREDICTIONS.jsonlines"
with open(output_file, "w") as json_file:
    for i in range(len(pred_test_final)):
        json.dump({
            "unique_id": data_test["unique_id"][i],
            "tokens": [t if t != "/uf0b7" else "\uf0b7" for t in data_test["tokens"][i]],
            "ner_tags": pred_test_final[i]
        }, json_file)
        json_file.write("\n")

print(f"Predictions saved to {output_file}")
