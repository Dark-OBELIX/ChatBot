# scripts/use_chatbot.py
import os
import sys
import torch
import spacy

# ✅ Permet d’importer config.py et autres modules depuis la racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_chatbot import data
from scripts.monitoring import log_misclassified
from config import SHOW_PROBABILITIES, ASK_FEEDBACK, CONFIDENCE_THRESHOLD, ENABLE_MONITORING

# --- Initialisation spaCy (même modèle que pour l'entraînement) ---
print("🔤 Initialisation du modèle linguistique français (spaCy)...")
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    from spacy.cli import download
    print("⚙️  Modèle 'fr_core_news_sm' introuvable. Téléchargement en cours...")
    download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")


# --- Fonction de prétraitement identique à l'entraînement ---
def clean_and_lemmatize(text):
    """
    Nettoie et lemmatise une phrase utilisateur :
    - convertit en minuscules
    - conserve les stopwords
    - garde uniquement les mots alphabétiques
    """
    text = text.lower().strip()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha])


# --- Boucle principale du chatbot ---
def use_model(model, vectorizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("\n🤖 Chatbot prêt ! Tapez 'quit' pour quitter.\n")

    while True:
        user_input = input("Vous : ").strip()
        if not user_input or len(user_input.replace(" ", "")) == 0:
            print("⚠️ Entrée vide ignorée.")
            continue

        # ✅ Prétraitement identique à l'entraînement
        processed_text = clean_and_lemmatize(user_input)

        # Vectorisation
        vec = vectorizer.transform([processed_text]).toarray()
        x = torch.tensor(vec, dtype=torch.float32).to(device)

        # Prédiction
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy().flatten()

        predicted_index = int(probs.argmax())
        confidence = float(probs[predicted_index])

        # Affichage des probabilités (optionnel)
        if SHOW_PROBABILITIES:
            print("\n--- Détails des probabilités ---")
            for i, p in enumerate(probs):
                print(f"{i}: {data[i]['answer']} -> {p*100:.2f}%")

        # Gestion des faibles confiances
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Chatbot : Je ne suis pas sûr 🤔 (confiance={confidence*100:.1f}%)")
            if ENABLE_MONITORING:
                log_misclassified(user_input, predicted_index, confidence)
            continue

        # Réponse principale
        answer = data[predicted_index]["answer"]
        action = data[predicted_index]["action_index"]
        print(f"Chatbot : {answer} (confiance={confidence*100:.2f}%)")

        # Feedback utilisateur (optionnel)
        if ASK_FEEDBACK:
            feedback = input("Était-ce la bonne réponse ? (o/n) : ").strip().lower()
            if feedback == "n" and ENABLE_MONITORING:
                correct_label = input("➡️  Quelle était la bonne intention (0–12) ? : ").strip()
                log_misclassified(user_input, predicted_index, confidence, correct_label)

        # Action spéciale (ex: fermeture)
        if action == 1:
            print("🔚 Action spéciale détectée : fermeture du programme.")
            break
