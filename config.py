"""
Configuration centrale du Chatbot Mobilia 🤖
Mode : QUALITÉ MAXIMALE – GPU optimisé pour précision et stabilité.
"""

# ==============================
# 🧠 MODÈLE ET ENTRAÎNEMENT
# ==============================

# Architecture MLP — plus profonde et large
# (exploite la pleine capacité GPU et capture les dépendances fines entre expressions)
HIDDEN_LAYERS = [1024, 512, 256, 128, 64]

# Hyperparamètres d'entraînement
LEARNING_RATE = 0.001         # Apprentissage doux, convergence stable
WEIGHT_DECAY = 1e-5           # Régularisation minimale pour laisser le modèle s’exprimer
NUM_EPOCHS = 2000             # Long entraînement pour affiner jusqu’à la convergence totale
PATIENCE = 200                # Early stopping très tolérant
STEP_SIZE = 400               # Scheduler progressif
GAMMA = 0.8                   # Décroissance lente du LR (long polish final)
DROPOUT_RATE = 0.4            # Forte régularisation pour éviter l’overfit malgré le grand réseau

# ==============================
# 💬 CHATBOT (interaction)
# ==============================

SHOW_PROBABILITIES = False    # True = debug complet
ASK_FEEDBACK = False
CONFIDENCE_THRESHOLD = 0.9    # Confiance minimale requise pour répondre

# ==============================
# 📊 MONITORING & AMÉLIORATION
# ==============================

ENABLE_MONITORING = True
LOG_PATH = "../logs/misclassified.json"

# ==============================
# ⚙️ DIVERS
# ==============================

SHOW_TRAINING_CURVE = True
SHOW_CONFUSION_MATRIX = True
