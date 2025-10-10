import torch
from scripts.data_chatbot import data
from scripts.train_chatbot import clean_and_lemmatize

def use_model(model, vectorizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print("\n🤖 Chatbot prêt !")

    while True:
        user_input = input("Vous : ").strip()
        if not user_input:
            continue

        text = clean_and_lemmatize(user_input)
        vec = vectorizer.transform([text]).toarray()
        x = torch.tensor(vec, dtype=torch.float32).to(device)
        output = model(x)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy().flatten()

        predicted_index = int(probs.argmax())
        confidence = float(probs[predicted_index])

        print("\n--- Détails des probabilités ---")
        for i, p in enumerate(probs):
            print(f"{i}: {data[i]['answer']} -> {p*100:.2f}%")

        if confidence < 0.75:
            print("Chatbot : Je ne comprends pas, veuillez etre plus claire dans votre demande !")
            continue

        answer = data[predicted_index]["answer"]
        action = data[predicted_index]["action_index"]

        print(f"Chatbot : {answer} (confiance={confidence*100:.2f}%, action_index={action})")

        if action == 1:
            print("🔚 Action spéciale détectée : fermeture du programme.")
            break
