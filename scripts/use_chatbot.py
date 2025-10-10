# scripts/use_chatbot.py
import os
import sys
import torch

# ✅ Permet d’importer config.py depuis la racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_chatbot import data
from scripts.monitoring import log_misclassified
from config import SHOW_PROBABILITIES, ASK_FEEDBACK, CONFIDENCE_THRESHOLD, ENABLE_MONITORING


def use_model(model, vectorizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("\n🤖 Chatbot prêt ! Tapez 'quit' pour quitter.\n")

    while True:
        user_input = input("Vous : ").strip()
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("🔚 Fin de la session.")
            break

        vec = vectorizer.transform([user_input]).toarray()
        x = torch.tensor(vec, dtype=torch.float32).to(device)
        output = model(x)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy().flatten()

        predicted_index = int(probs.argmax())
        confidence = float(probs[predicted_index])

        if SHOW_PROBABILITIES:
            print("\n--- Détails des probabilités ---")
            for i, p in enumerate(probs):
                print(f"{i}: {data[i]['answer']} -> {p*100:.2f}%")

        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Chatbot : Je ne suis pas sûr 🤔 (confiance={confidence*100:.1f}%)")
            if ENABLE_MONITORING:
                log_misclassified(user_input, predicted_index, confidence)
            continue

        answer = data[predicted_index]["answer"]
        action = data[predicted_index]["action_index"]

        print(f"Chatbot : {answer} (confiance={confidence*100:.2f}%)")

        if ASK_FEEDBACK:
            feedback = input("Était-ce la bonne réponse ? (o/n) : ").strip().lower()
            if feedback == "n" and ENABLE_MONITORING:
                correct_label = input("➡️  Quelle était la bonne intention (0–12) ? : ").strip()
                log_misclassified(user_input, predicted_index, confidence, correct_label)

        if action == 1:
            print("🔚 Action spéciale détectée : fermeture du programme.")
            break
