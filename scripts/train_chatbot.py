# scripts/train_chatbot.py
import os
import json
import joblib
import spacy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ✅ Import du config.py à la racine
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_chatbot import data
from scripts.model import ChatBotMLP
from scripts.utils.plot_utils import plot_training_curve
from config import (
    HIDDEN_LAYERS, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    PATIENCE, STEP_SIZE, GAMMA, DROPOUT_RATE,
    SHOW_TRAINING_CURVE, SHOW_CONFUSION_MATRIX
)

# --- Répertoires et chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH_MODEL = os.path.join(MODEL_DIR, "chatbot_model.pth")
MODEL_PATH_VECTORIZER = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH_METRICS = os.path.join(MODEL_DIR, "metrics.json")
MODEL_PATH_BEST = os.path.join(MODEL_DIR, "chatbot_best.pth")

# --- Initialisation spaCy ---
print("🔤 Chargement du modèle linguistique français...")
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    from spacy.cli import download
    print("⚙️ Modèle 'fr_core_news_sm' introuvable. Téléchargement en cours...")
    download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

# --- Nettoyage + lemmatisation ---
def clean_and_lemmatize(text):
    """
    Nettoie et lemmatise le texte :
    - minuscules
    - garde les stopwords
    - conserve uniquement les mots alphabétiques
    """
    text = text.lower().strip()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha])

# --- Entraînement principal ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Entraînement sur : {device}")

    # --- Préparation des données ---
    corpus, y = [], []
    for i, entry in enumerate(data):
        qs = entry["question"]
        if isinstance(qs, list):
            corpus.extend(qs)
            y.extend([i] * len(qs))
        else:
            corpus.append(qs)
            y.append(i)

    # ✅ Nettoyage linguistique complet avec spaCy
    corpus = [clean_and_lemmatize(q) for q in corpus]

    # --- Split train/test ---
    corpus_train, corpus_test, y_train, y_test = train_test_split(
        corpus, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Vectorisation enrichie (ngram 1–4) ---
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 4),
        max_features=10000,
        sublinear_tf=True,
        stop_words=None
    )

    X_train = vectorizer.fit_transform(corpus_train).toarray()
    X_test = vectorizer.transform(corpus_test).toarray()

    input_size = X_train.shape[1]
    output_size = len(data)
    hidden_layers = HIDDEN_LAYERS

    model = ChatBotMLP(input_size, hidden_layers, output_size, dropout_rate=DROPOUT_RATE).to(device)

    # --- Tensorisation + DataLoader ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)  # ⬅️ batch size volontairement faible

    # --- Optimiseur / Scheduler / Early stopping ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    losses = []
    best_loss = float("inf")
    patience, wait = PATIENCE, 0

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("\n🚀 Début de l'entraînement (qualité maximale)...\n")

    for epoch in tqdm(range(NUM_EPOCHS), desc="Entraînement du modèle", ncols=90, colour="cyan"):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

        # Early stopping + sauvegarde best
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH_BEST)
        else:
            wait += 1
            if wait >= patience:
                print("🛑 Early stopping déclenché.")
                break

    # --- Sauvegardes ---
    torch.save(model.state_dict(), MODEL_PATH_MODEL)
    joblib.dump(vectorizer, MODEL_PATH_VECTORIZER)
    print(f"\n✅ Modèle sauvegardé dans : {MODEL_PATH_MODEL}")
    print(f"✅ Meilleur modèle : {MODEL_PATH_BEST}")
    print(f"✅ Vectorizer sauvegardé dans : {MODEL_PATH_VECTORIZER}")

    # --- Courbe de perte ---
    if SHOW_TRAINING_CURVE:
        plot_training_curve(losses, MODEL_DIR)

    # --- Évaluation finale ---
    model.load_state_dict(torch.load(MODEL_PATH_BEST, map_location=device))
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = torch.argmax(model(X_test_tensor), dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print(f"\n🎯 Accuracy : {acc*100:.2f}%")
    print(f"💡 F1-score pondéré : {f1*100:.2f}%")

    metrics = {"accuracy": acc, "f1_score": f1, "epochs_run": len(losses)}
    with open(MODEL_PATH_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # --- Matrice de confusion ---
    if SHOW_CONFUSION_MATRIX:
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{i}" for i in range(len(data))])
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Matrice de confusion - MobiliaBot (Qualité max)")
        plt.tight_layout()
        plt.show()

    return model, vectorizer, hidden_layers
