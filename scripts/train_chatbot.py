# scripts/train_chatbot.py
import os
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from scripts.data_chatbot import data  # import corrigé

# Dossier "models" au même niveau que "scripts"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pth")

class ChatBotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Entraînement sur : {device}")

    corpus, y = [], []
    for i, entry in enumerate(data):
        questions = entry["question"]
        if isinstance(questions, list):
            corpus.extend(questions)
            y.extend([i]*len(questions))
        else:
            corpus.append(questions)
            y.append(i)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()

    input_size = X.shape[1]
    hidden_size = 8
    output_size = len(data)

    model = ChatBotNN(input_size, hidden_size, output_size).to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Sauvegarde du modèle
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vectorizer": vectorizer
    }, MODEL_PATH)

    print(f"✅ Modèle sauvegardé dans : {MODEL_PATH}")

    return model, vectorizer
