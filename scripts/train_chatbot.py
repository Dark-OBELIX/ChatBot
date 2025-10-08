# scripts/train_chatbot.py
import os
import re
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

from scripts.data_chatbot import data
from scripts.model import ChatBotMLP
from scripts.utils.plot_utils import plot_training_curve  # ✅ Import du module utils

# --- Répertoires et chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH_MODEL = os.path.join(MODEL_DIR, "chatbot_model.pth")
MODEL_PATH_VECTORIZER = os.path.join(MODEL_DIR, "vectorizer.pkl")

# --- Fonction de nettoyage du texte ---
def clean_text(t):
    """
    Nettoie le texte :
    - met en minuscules
    - retire les caractères spéciaux et ponctuations
    - garde les lettres accentuées et les tirets
    """
    t = t.lower()
    t = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ'\s-]", "", t)
    return t.strip()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Entraînement sur : {device}")

    # --- Préparation du dataset ---
    corpus, y = [], []
    for i, entry in enumerate(data):
        questions = entry["question"]
        if isinstance(questions, list):
            corpus.extend(questions)
            y.extend([i] * len(questions))
        else:
            corpus.append(questions)
            y.append(i)

    # ✅ Nettoyage du texte
    corpus = [clean_text(q) for q in corpus]

    # --- Split train/test (80/20) ---
    corpus_train, corpus_test, y_train, y_test = train_test_split(
        corpus, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Vectorisation TF-IDF ---
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(corpus_train).toarray()
    X_test = vectorizer.transform(corpus_test).toarray()

    input_size = X_train.shape[1]
    output_size = len(data)
    hidden_layers = [64, 32, 16]

    model = ChatBotMLP(input_size, hidden_layers, output_size, dropout_rate=0.3).to(device)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)

    num_epochs = 1000
    losses = []

    print("\n🚀 Début de l'entraînement...\n")

    # --- Boucle d'entraînement avec barre de progression ---
    for epoch in tqdm(range(num_epochs), desc="Entraînement du modèle", ncols=90, colour="cyan"):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}")

    # --- Sauvegarde du modèle ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH_MODEL)
    joblib.dump(vectorizer, MODEL_PATH_VECTORIZER)
    print(f"\n✅ Modèle sauvegardé dans : {MODEL_PATH_MODEL}")
    print(f"✅ Vectorizer sauvegardé dans : {MODEL_PATH_VECTORIZER}")

    # --- Courbe d'entraînement via utils ---
    plot_training_curve(losses, MODEL_DIR)

    # --- Évaluation automatique ---
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    print("\n📈 ÉVALUATION DU MODÈLE")
    print(f"🎯 Précision globale (Accuracy) : {acc*100:.2f}%")
    print(f"💡 F1-score pondéré : {f1*100:.2f}%")

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{i}" for i in range(len(data))])
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Matrice de confusion - MobiliaBot")
    plt.tight_layout()
    plt.show()

    return model, vectorizer
