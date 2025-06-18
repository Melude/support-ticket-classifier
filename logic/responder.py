from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()  

SYSTEM_PROMPT = """Du bist ein freundlicher Support-Assistent. 
Deine Aufgabe ist es, kurze, professionelle Antworten auf Kundenanfragen zu formulieren, basierend auf der erkannten Kategorie.
Antworte sachlich, hilfsbereit und ohne überflüssige Floskeln."""

def generate_response(category: str, user_input: str) -> str:
    prompt = f"""
    Ein Kunde hat folgende Anfrage gestellt:

    "{user_input}"

    Die Anfrage wurde automatisch der Kategorie "{category}" zugeordnet.

    Formuliere eine kurze, hilfreiche Antwort, wie sie ein Support-Team schreiben würde.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        message = response.choices[0].message.content
        return message.strip() if message else "[Fehler: Keine Antwort vom Modell erhalten]"

    except Exception as e:
        return f"[Fehler bei der Antwortgenerierung: {e}]"
