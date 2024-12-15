import json
import pandas as pd
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import BertTokenizerFast
from transformers import BertForTokenClassification, Trainer, TrainingArguments
import torch

# Configuration du modèle
model_name = "PRETRAINED MODEL NAME"  # Nom du modèle pré-entraîné
model = BertForTokenClassification.from_pretrained(model_name, num_labels=7)

# Configuration du dispositif (CPU ou GPU), si possible on utilise le GPU pour réduire le temps d'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model.device)  # Affiche le dispositif sur lequel le modèle est chargé (e.g., 'cuda:0')

# Chargement du tokenizer associé au modèle
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Fonction pour lire un fichier JSON
# Le fichier doit contenir une ligne par enregistrement au format JSON
# Retourne une liste de dictionnaires

def read_json(file):
    with open(file, 'r') as f:
        corpus_json = [json.loads(l) for l in list(f)]  
    return corpus_json

# Fonction pour remplacer des clés dans une chaîne en fonction d'un dictionnaire
def replace(mot, dic):
    for key, value in dic.items():
        mot = mot.replace(key, str(value))
    return mot

# Dictionnaires de mappage entre les étiquettes NER et leurs indices
# Format BIO (Begin-Inside-Outside)
dic_tag = {'O': 0, 'B-Entity': 1, 'B-Action': 2, 'I-Action': 3, 'I-Entity': 4, 'B-Modifier': 5, 'I-Modifier': 6}
dic_tag_inv = {v: k for k, v in dic_tag.items()}  # Inversion pour une récupération facile des étiquettes

# Préparation des données pour l'entraînement
def prepa_data(file, tokenizer):
    # Lecture du corpus au format JSON
    corpus = read_json(file)

    # Nettoyage et organisation des données
    data = {
        "tokens": [[t if t != "\uf0b7" else "/uf0b7" for t in val["tokens"]] for val in corpus],
        "ner_tags": [val["ner_tags"] for val in corpus]
    }

    # Tokenisation des entrées avec gestion des séquences et padding
    tokenized_inputs = tokenizer(data['tokens'], truncation=True, padding=True, is_split_into_words=True)

    # Préparation des étiquettes alignées avec les sous-tokens
    Labels = []
    for i in range(len(data["ner_tags"])):
        add = -1  # Indicateur pour éviter les doublons de tokens
        Label = []
        for j in tokenized_inputs.word_ids(i):
            if j is not None and add != j:
                Label.append(int(replace(data["ner_tags"][i][j], dic_tag)))
                add = j
            else:
                Label.append(-100)  # Ignorer les sous-tokens ajoutés par BERT
        Labels.append(Label)

    # Ajout des étiquettes alignées aux entrées tokenisées
    tokenized_inputs["labels"] = Labels
    return Dataset.from_dict(tokenized_inputs)

# Chargement des jeux de données
# Format attendu : fichier JSONLINES avec les champs "tokens" et "ner_tags"
data_train = prepa_data("NER-TRAINING/NER-TRAINING.jsonlines", tokenizer)
data_val = prepa_data("NER-TRAINING/NER-VALIDATION.jsonlines", tokenizer)

# Configuration des paramètres d'entraînement
training_args = TrainingArguments(
    output_dir='./results',  # Répertoire pour les résultats
    evaluation_strategy="epoch",  # Évaluation après chaque époque
    learning_rate=1e-5,  # Taux d'apprentissage
    per_device_train_batch_size=16,  # Taille du batch
    num_train_epochs=30,  # Nombre d'époques
    weight_decay=0.01,  # Pénalité pour régularisation L2
    logging_dir='./logs'  # Répertoire pour les logs
)

# Configuration du formateur (Trainer) pour simplifier l'entraînement
trainer = Trainer(
    model=model,  # Modèle BERT avec classification par tokens
    args=training_args,  # Arguments d'entraînement
    train_dataset=data_train,  # Jeu de données d'entraînement
    eval_dataset=data_val  # Jeu de validation
)

# Démarrage de l'entraînement
trainer.train()

# Sauvegarde du modèle et du tokenizer après l'entraînement
model.save_pretrained('YOUR RESULT DIRECTORY')
tokenizer.save_pretrained('YOUR RESULT DIRECTORY')
