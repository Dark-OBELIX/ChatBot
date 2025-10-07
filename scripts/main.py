    # ========================
# Chatbot minimal PyTorch + CUDA
# ========================

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

# Importer les données depuis le fichier séparé
from data_chatbot import data

# -----------------------------
# Étape 0 : Préparer le device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

# -----------------------------
# Étape 1 : Préparer les données
# -----------------------------
corpus = [d["question"] for d in data]  # extraire les questions
y = [i for i in range(len(data))]  # indices des réponses

# -----------------------------
# Étape 2 : Transformer le texte en vecteurs (Bag of Words)
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()


# -----------------------------
# Étape 3 : Définir le modèle PyTorch
# -----------------------------
class ChatBotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = X.shape[1]
hidden_size = 8
output_size = len(data)

model = ChatBotNN(input_size, hidden_size, output_size).to(device)

# -----------------------------
# Étape 4 : Préparer les tenseurs PyTorch
# -----------------------------
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)

# -----------------------------
# Étape 5 : Définir la perte et l'optimiseur
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Étape 6 : Entraîner le modèle
# -----------------------------
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Entraînement terminé !")


# -----------------------------
# Étape 7 : Fonction de prédiction
# -----------------------------
def predict(question):
    vec = vectorizer.transform([question]).toarray()
    x = torch.tensor(vec, dtype=torch.float32).to(device)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return data[predicted.item()]["answer"]


# -----------------------------
# Étape 8 : Test interactif
# -----------------------------
print("\nChatbot prêt ! Tape 'quit' pour quitter.")

while True:
    user_input = input("Vous : ")
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot : À bientôt !")
        break
    response = predict(user_input)

    print(f"Chatbot : {response}")

