# scripts/main.py
import os
import torch
import joblib
from scripts.train_chatbot import train_model, MODEL_DIR
from scripts.use_chatbot import use_model
from scripts.model import ChatBotNN
from scripts.data_chatbot import data

MODEL_PATH_MODEL = os.path.join(MODEL_DIR, "chatbot_model.pth")
MODEL_PATH_VECTORIZER = os.path.join(MODEL_DIR, "vectorizer.pkl")

def main():
    print("=== Chatbot PyTorch ===")
    print("1️⃣ Utiliser le dernier modèle")
    print("2️⃣ Réentraîner un nouveau modèle")

    choice = input("Choix (1/2) : ").strip()

    if choice == "2" or not (os.path.exists(MODEL_PATH_MODEL) and os.path.exists(MODEL_PATH_VECTORIZER)):
        print("Entraînement du modèle en cours...")
        model, vectorizer = train_model()
    else:
        print("Chargement du modèle existant...")
        vectorizer = joblib.load(MODEL_PATH_VECTORIZER)
        input_size = len(vectorizer.get_feature_names_out())
        output_size = len(data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChatBotNN(input_size, 8, output_size).to(device)
        model.load_state_dict(torch.load(MODEL_PATH_MODEL, map_location=device))

    use_model(model, vectorizer)

if __name__ == "__main__":
    main()
