from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()  

def generate_response(category: str, user_input: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein freundlicher Support-Assistent. "
                        "Deine Aufgabe ist es, kurze, professionelle Antworten auf Kundenanfragen zu formulieren, "
                        "basierend auf der erkannten Kategorie. "
                        "Antworte sachlich, hilfsbereit und ohne überflüssige Floskeln."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Ein Kunde hat folgende Anfrage gestellt:\n\n'
                        f'"{user_input}"\n\n'
                        f'Die Anfrage wurde automatisch der Kategorie "{category}" zugeordnet.\n\n'
                        "Formuliere eine kurze, hilfreiche Antwort, wie sie ein Support-Team schreiben würde."
                    ),
                },
            ],
            temperature=0.7,
        )

        message = response.choices[0].message.content
        return message.strip() if message else "[Fehler: Keine Antwort vom Modell erhalten]"

    except Exception as e:
        return f"[Fehler bei der Antwortgenerierung: {e}]"
