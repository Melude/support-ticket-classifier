from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def review_classification(user_input: str, current_category: str, categories: list[str]) -> str:
    prompt = f"""
Ein Support-System ordnet Kundenanfragen automatisch Kategorien zu.

Derzeitige Kategorien:
{', '.join(categories)}

Beispiel-Anfrage:
"{user_input}"

Vom System zugewiesene Kategorie: "{current_category}"

Frage:
1. Passt diese Kategorie zur Anfrage?
2. Wenn nein, welche Kategorie wäre treffender?
3. Gibt es eine wichtige Kategorie, die fehlt und ergänzt werden sollte?
Antwort bitte in maximal 3 Sätzen.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du bist ein Support-Experte, der Kategorien in einem Ticketsystem bewertet."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        message = response.choices[0].message.content
        return message.strip() if message else "[Fehler: Keine Antwort vom Modell erhalten]"

    except Exception as e:
        return f"[Fehler bei der Klassifikationsprüfung: {e}]"
