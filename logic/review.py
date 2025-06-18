from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def review_classification(user_input: str, current_category: str, categories: list[str]) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein Support-Experte, der Kategorien in einem Ticketsystem bewertet. "
                        "Du analysierst, ob die automatisch gewählte Kategorie zur Kundenanfrage passt."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Ein Support-System ordnet Kundenanfragen automatisch Kategorien zu.\n\n"
                        f"Aktuelle Kategorien:\n{', '.join(categories)}\n\n"
                        f"Beispiel-Anfrage:\n\"{user_input}\"\n\n"
                        f"Vom System zugewiesene Kategorie: \"{current_category}\"\n\n"
                        "Frage:\n"
                        "1. Passt diese Kategorie zur Anfrage?\n"
                        "2. Wenn nein, welche Kategorie wäre treffender?\n"
                        "3. Gibt es eine wichtige Kategorie, die fehlt und ergänzt werden sollte?\n"
                        "Antwort bitte in maximal 3 Sätzen."
                    ),
                },
            ],
            temperature=0.3,
        )

        message = response.choices[0].message.content
        return message.strip() if message else "[Fehler: Keine Antwort vom Modell erhalten]"

    except Exception as e:
        return f"[Fehler bei der Klassifikationsprüfung: {e}]"
