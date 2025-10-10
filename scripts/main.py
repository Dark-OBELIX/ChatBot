# scripts/main.py
import os
import torch
import joblib
import json
import sys

# ✅ Permet d’importer config.py situé à la racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_chatbot import train_model, MODEL_DIR
from scripts.use_chatbot import use_model
from scripts.model import ChatBotMLP
from scripts.data_chatbot import data
from config import HIDDEN_LAYERS  # architecture par défaut

MODEL_PATH_MODEL = os.path.join(MODEL_DIR, "chatbot_model.pth")
MODEL_PATH_VECTORIZER = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH_META = os.path.join(MODEL_DIR, "model_meta.json")


def save_model_metadata(hidden_layers):
    os.makedirs(MODEL_DIR, exist_ok=True)
    meta = {"hidden_layers": hidden_layers}
    with open(MODEL_PATH_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)


def load_model_metadata():
    if os.path.exists(MODEL_PATH_META):
        with open(MODEL_PATH_META, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    print("=== 🤖 Chatbot Mobilia - Version MLP ===")
    print("1️⃣ Utiliser le modèle existant")
    print("2️⃣ Réentraîner un nouveau modèle\n")

    choice = input("Choix (1/2) : ").strip()

    if choice == "2" or not (os.path.exists(MODEL_PATH_MODEL) and os.path.exists(MODEL_PATH_VECTORIZER)):
        model, vectorizer, hidden_layers = train_model()
        save_model_metadata(hidden_layers)

    else:
        print("📦 Chargement du modèle existant...")
        meta = load_model_metadata()
        hidden_layers = meta.get("hidden_layers", HIDDEN_LAYERS) if meta else HIDDEN_LAYERS

        vectorizer = joblib.load(MODEL_PATH_VECTORIZER)
        input_size = len(vectorizer.get_feature_names_out())
        output_size = len(data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ChatBotMLP(input_size, hidden_layers, output_size).to(device)

        try:
            model.load_state_dict(torch.load(MODEL_PATH_MODEL, map_location=device))
            print(f"✅ Modèle chargé avec succès (architecture : {hidden_layers})")
        except RuntimeError as e:
            print(f"⚠️ Erreur de compatibilité : {e}")
            print("🔁 Réentraînement automatique...")
            model, vectorizer, hidden_layers = train_model()
            save_model_metadata(hidden_layers)

    use_model(model, vectorizer)


if __name__ == "__main__":
    main()
