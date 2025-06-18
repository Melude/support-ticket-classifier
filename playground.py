from logic.classifier import classify
from logic.responder import generate_response

def main():
    print("🔧 Support-Ticket Demo")
    print("Gib eine Kundenanfrage ein:")
    user_input = input("> ").strip()

    if not user_input:
        print("⚠️  Bitte gib einen Text ein.")
        return

    # Klassifikation + dynamisch verwendete Kategorien
    category, confidence, categories_used = classify(user_input)
    print(f"\n✅ Erkannte Kategorie: {category} ({confidence}%)")
    print("📦 Verwendete Kategorien:", ", ".join(categories_used))

    # GPT-Antwort generieren
    response = generate_response(category, user_input)
    print(f"\n🤖 GPT-Antwort:\n{response}")

if __name__ == "__main__":
    main()
