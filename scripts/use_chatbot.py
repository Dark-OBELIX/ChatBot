# scripts/use_chatbot.py
import torch
import torch.nn as nn
from scripts.data_chatbot import data

class ChatBotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def use_model(model, vectorizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("\n🤖 Chatbot prêt !")

    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("👋 Fin de la conversation.")
            break

        vec = vectorizer.transform([user_input]).toarray()
        x = torch.tensor(vec, dtype=torch.float32).to(device)
        output = model(x)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy().flatten()

        predicted_index = int(probs.argmax())
        confidence = float(probs[predicted_index])

        print("\n--- Détails des probabilités ---")
        for i, p in enumerate(probs):
            print(f"{i}: {data[i]['answer']} -> {p*100:.2f}%")

        if confidence < 0.9:
            print("Chatbot : Je ne comprends pas 🤔")
            continue

        answer = data[predicted_index]["answer"]
        action = data[predicted_index]["action_index"]

        print(f"Chatbot : {answer} (confiance={confidence*100:.2f}%, action_index={action})")

        if action == 1:
            print("🔚 Action spéciale détectée : fermeture du programme.")
            break
