# scripts/utils/plot_utils.py
import os
import matplotlib.pyplot as plt

def plot_training_curve(losses, model_dir, filename="loss_curve.png", title="Courbe d'entraînement du Chatbot (MLP)"):
    """
    Affiche et sauvegarde la courbe du loss pendant l'entraînement.
    - losses : liste des valeurs de perte par epoch
    - model_dir : répertoire où sauvegarder (ex: models/)
    - filename : nom du fichier image
    - title : titre du graphique
    """
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, filename)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss d'entraînement", color='royalblue')
    plt.title(title)
    plt.xlabel("Époque")
    plt.ylabel("Loss (Erreur)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()

    print(f"📊 Courbe d'entraînement sauvegardée dans : {save_path}")
