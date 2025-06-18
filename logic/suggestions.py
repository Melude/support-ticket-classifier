import json
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

LOG_FILE = "suggested_categories.jsonl"

def extract_clean_suggestions(text: str) -> list[str]:
    if text.upper().strip() == "KEINE":
        return []

    suggestions = []
    for line in text.splitlines():
        if "bessere passende kategorien" in line.lower():
            # Zeile enthält GPT-Floskel → extrahiere nur den Teil nach dem :
            parts = line.split(":", 1)
            if len(parts) == 2:
                cleaned = parts[1].strip()
                parts = cleaned.split(",")
            else:
                parts = []
        else:
            parts = line.split(",")

        for part in parts:
            candidate = part.strip("-•:. ").strip()
            if candidate:
                suggestions.append(candidate)

    return suggestions

def suggest_new_categories(text: str, current_category: str, categories: list[str]) -> list[str]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein Assistent für Support-Ticket-Klassifikation. "
                        "Du bekommst eine Kundenanfrage und eine vom System zugewiesene Kategorie. "
                        "Wenn die Kategorie nicht passt, schlage bitte 1–2 präzisere Kategorien vor – "
                        "als einfache, durch Kommas getrennte Begriffe (ohne Einleitung oder Erklärung). "
                        "Wenn die vorhandene Kategorie ausreichend ist, antworte mit: KEINE"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f'Anfrage: "{text}"\n'
                        f'Systemkategorie: "{current_category}"\n'
                        f'Vorhandene Kategorien: {", ".join(categories)}'
                    ),
                },
            ],
            temperature=0.3,
        )

        message = response.choices[0].message.content
        content = message.strip() if message else "[KEINE ANTWORT]"

        new_suggestions = extract_clean_suggestions(content)
        filtered = [c for c in new_suggestions if c not in categories]

        print("GPT-Rohantwort:", repr(content))
        print("Extrahierte Vorschläge:", new_suggestions)
        print("Neue Kategorien (nicht in YAML):", filtered)

        if filtered:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "input": text,
                "system_category": current_category,
                "suggested_categories": filtered,
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        else:
            print("Keine neuen Kategorien erkannt – Datei wird nicht geschrieben.")

        return filtered

    except Exception as e:
        print(f"Fehler bei der Kategorie-Vorschlagsanalyse: {e}")
        return []

