# scripts/train_chatbot.py
import os
import re
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ✅ Permet d’importer config.py situé à la racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_chatbot import data
from scripts.model import ChatBotMLP
from scripts.utils.plot_utils import plot_training_curve
from config import (
    HIDDEN_LAYERS, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    PATIENCE, STEP_SIZE, GAMMA, DROPOUT_RATE,
    SHOW_TRAINING_CURVE, SHOW_CONFUSION_MATRIX
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH_MODEL = os.path.join(MODEL_DIR, "chatbot_model.pth")
MODEL_PATH_VECTORIZER = os.path.join(MODEL_DIR, "vectorizer.pkl")


def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ'\s-]", "", t)
    return t.strip()


def train_model():
    """Entraîne le modèle de chatbot et retourne (model, vectorizer, hidden_layers)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Entraînement sur : {device}")

    corpus, y = [], []
    for i, entry in enumerate(data):
        questions = entry["question"]
        if isinstance(questions, list):
            corpus.extend(questions)
            y.extend([i] * len(questions))
        else:
            corpus.append(questions)
            y.append(i)

    corpus = [clean_text(q) for q in corpus]

    corpus_train, corpus_test, y_train, y_test = train_test_split(
        corpus, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(corpus_train).toarray()
    X_test = vectorizer.transform(corpus_test).toarray()

    input_size = X_train.shape[1]
    output_size = len(data)
    hidden_layers = HIDDEN_LAYERS

    model = ChatBotMLP(input_size, hidden_layers, output_size, dropout_rate=DROPOUT_RATE).to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_loss = float("inf")
    counter = 0
    losses = []

    print("\n🚀 Début de l'entraînement...\n")

    for epoch in tqdm(range(NUM_EPOCHS), desc="Entraînement du modèle", ncols=90, colour="cyan"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                tqdm.write("🛑 Early stopping déclenché.")
                break

        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {loss.item():.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH_MODEL)
    joblib.dump(vectorizer, MODEL_PATH_VECTORIZER)
    print(f"\n✅ Modèle sauvegardé dans : {MODEL_PATH_MODEL}")
    print(f"✅ Vectorizer sauvegardé dans : {MODEL_PATH_VECTORIZER}")

    if SHOW_TRAINING_CURVE:
        try:
            plot_training_curve(losses, MODEL_DIR)
        except Exception:
            plt.plot(losses)
            plt.title("Courbe de perte - ChatBot")
            plt.xlabel("Épochs")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"))
            plt.close()
            print(f"📊 Courbe sauvegardée dans : {os.path.join(MODEL_DIR, 'loss_curve.png')}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print("\n📈 ÉVALUATION DU MODÈLE")
    print(f"🎯 Accuracy : {acc*100:.2f}%")
    print(f"💡 F1-score pondéré : {f1*100:.2f}%")

    if SHOW_CONFUSION_MATRIX:
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{i}" for i in range(len(data))])
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Matrice de confusion - MobiliaBot")
        plt.tight_layout()
        plt.show()

    return model, vectorizer, hidden_layers
