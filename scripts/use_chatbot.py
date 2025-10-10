# scripts/use_chatbot.py
import torch
from scripts.data_chatbot import data
from scripts.monitoring import log_misclassified

# 🔧 Paramètres globaux
SHOW_PROBABILITIES = False  # Afficher toutes les probabilités (True/False)
ASK_FEEDBACK = False        # Demander à l'utilisateur s'il valide la réponse (True/False)


def use_model(model, vectorizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("\n🤖 Chatbot prêt ! Tapez 'quit' pour quitter.\n")

    while True:
        user_input = input("Vous : ").strip()
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("🔚 Fin de la session.")
            break

        # --- Vectorisation ---
        vec = vectorizer.transform([user_input]).toarray()
        x = torch.tensor(vec, dtype=torch.float32).to(device)
        output = model(x)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy().flatten()

        predicted_index = int(probs.argmax())
        confidence = float(probs[predicted_index])

        # --- Affichage paramétrable des probabilités ---
        if SHOW_PROBABILITIES:
            print("\n--- Détails des probabilités ---")
            for i, p in enumerate(probs):
                print(f"{i}: {data[i]['answer']} -> {p*100:.2f}%")

        # --- Confiance trop faible ---
        if confidence < 0.9:
            print(f"Chatbot : Je ne suis pas sûr 🤔 (confiance={confidence*100:.1f}%)")
            log_misclassified(user_input, predicted_index, confidence)
            continue

        # --- Réponse principale ---
        answer = data[predicted_index]["answer"]
        action = data[predicted_index]["action_index"]

        print(f"Chatbot : {answer} (confiance={confidence*100:.2f}%)")

        # --- Feedback optionnel (paramétrable) ---
        if ASK_FEEDBACK:
            feedback = input("Était-ce la bonne réponse ? (o/n) : ").strip().lower()
            if feedback == "n":
                correct_label = input("➡️  Quelle était la bonne intention (0–12) ? : ").strip()
                log_misclassified(user_input, predicted_index, confidence, correct_label)

        # --- Action spéciale ---
        if action == 1:
            print("🔚 Action spéciale détectée : fermeture du programme.")
            break
