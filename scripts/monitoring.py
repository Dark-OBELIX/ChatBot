# scripts/monitoring.py
import json
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "misclassified.json")


def log_misclassified(user_input, predicted_label, confidence, correct_label=None):
    """
    Enregistre une phrase mal comprise ou incertaine.
    - user_input : texte saisi
    - predicted_label : catégorie prédite
    - confidence : score de confiance (0 à 1)
    - correct_label : (optionnel) correction manuelle de l’intention
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "input": user_input,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 3),
        "correct_label": correct_label,
    }

    # Lecture de l’existant
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Ajout et réécriture
    data.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"📝 Enregistré pour amélioration : '{user_input}' (confiance={confidence:.2f})")
