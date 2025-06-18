import json
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

LOG_FILE = "suggested_categories.jsonl"

def extract_clean_suggestions(text: str) -> list[str]:
    """
    Entfernt irrelevante GPT-Antwortteile und extrahiert saubere Kategoriemöglichkeiten.
    """
    text = text.strip()
    if text.upper() == "KEINE":
        return []

    lines = []
    for raw_line in text.splitlines():
        parts = raw_line.split(",")
        lines.extend(part.strip("-• ").strip() for part in parts if part.strip())

    clean = []
    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            candidate = parts[1].strip()
            if candidate:
                clean.append(candidate)
        else:
            clean.append(line)

    # Doppelte filtern und leere ignorieren
    return [c for c in clean if c and not c.lower().startswith("bessere")]

def suggest_new_categories(text: str, current_category: str, categories: list[str]) -> list[str]:
    system_prompt = (
        "Du bist ein Assistent für Support-Ticket-Klassifikation. "
        "Du bekommst eine Kundenanfrage, die vom System einer Kategorie zugewiesen wurde. "
        "Wenn diese Kategorie unpassend ist, schlage bitte 1–2 besser passende Kategorien vor. "
        "Wenn alle bestehenden Kategorien ausreichend sind, antworte mit 'KEINE'."
    )

    user_prompt = f"""
Anfrage: "{text}"
Systemkategorie: "{current_category}"
Vorhandene Kategorien: {', '.join(categories)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        message = response.choices[0].message.content
        content = message.strip() if message else "[KEINE ANTWORT]"

        new_suggestions = extract_clean_suggestions(content)

        # Nur neue Kategorien loggen
        filtered = [c for c in new_suggestions if c not in categories]
        if filtered:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "input": text,
                "system_category": current_category,
                "suggested_categories": filtered,
            }

            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return filtered

    except Exception as e:
        print(f"Fehler bei der Kategorie-Vorschlagsanalyse: {e}")
        return []
