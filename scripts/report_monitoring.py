# scripts/report_monitoring.py
import json
from collections import Counter
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "logs" / "misclassified.json"

if not LOG_FILE.exists():
    print("Aucune donnée de monitoring pour le moment.")
else:
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📊 {len(data)} entrées enregistrées pour analyse.\n")

    # Statistiques globales
    predicted = [str(d["predicted_label"]) for d in data if "predicted_label" in d]
    corrected = [str(d["correct_label"]) for d in data if d.get("correct_label")]

    print("📈 Intentions mal prédites les plus fréquentes :")
    for label, freq in Counter(predicted).most_common():
        print(f"  - Intent {label} : {freq} erreurs")

    if corrected:
        print("\n✅ Intentions corrigées manuellement :")
        for label, freq in Counter(corrected).most_common():
            print(f"  - Intent {label} : {freq} corrections")
