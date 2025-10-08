# scripts/main.py
import os
import torch
from scripts.train_chatbot import train_model, ChatBotNN, MODEL_PATH
from scripts.use_chatbot import use_model

def main():
    print("=== Chatbot PyTorch ===")
    print("1️⃣ Utiliser le dernier modèle")
    print("2️⃣ Réentraîner un nouveau modèle")

    choice = input("Choix (1/2) : ").strip()

    if choice == "2" or not os.path.exists(MODEL_PATH):
        print("🧠 Entraînement du modèle en cours...")
        model, vectorizer = train_model()
    else:
        print("📂 Chargement du modèle existant...")
        checkpoint = torch.load(MODEL_PATH)
        vectorizer = checkpoint["vectorizer"]

        input_size = len(vectorizer.get_feature_names_out())
        output_size = len(checkpoint["model_state"]["fc2.bias"])
        model = ChatBotNN(input_size, 8, output_size)
        model.load_state_dict(checkpoint["model_state"])

    use_model(model, vectorizer)

if __name__ == "__main__":
    main()
