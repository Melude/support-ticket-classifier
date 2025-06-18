import json
from pathlib import Path
from logic.category_store import load_categories, save_categories
from logic.review import review_classification

LOG_FILE = Path("suggested_categories.jsonl")
REVIEWED_FILE = Path("reviewed.jsonl")
IGNORED_FILE = Path("ignored_categories.jsonl")

def load_suggestions():
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def mark_as_reviewed(entry, response):
    with open(REVIEWED_FILE, "a", encoding="utf-8") as f:
        data = entry.copy()
        data["gpt_review"] = response
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def append_ignored(entry):
    with open(IGNORED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def run_review():
    print("🔍 Starte Kategorie-Review via GPT")
    entries = load_suggestions()
    existing = set(load_categories())

    for entry in entries:
        suggested = [c for c in entry["suggested_categories"] if c not in existing]
        if not suggested:
            continue

        print(f"\n📝 Anfrage: {entry['input']}")
        print(f"🏷️  Systemkategorie: {entry['system_category']}")
        print(f"💡 GPT-Vorschläge: {', '.join(suggested)}")

        review = review_classification(entry["input"], entry["system_category"], list(existing))
        print(f"\n🧠 GPT-Review:\n{review}\n")

        action = input("➕ [Enter] übernehmen, ⏭️ [s] skippen, ❌ [x] verwerfen: ").strip().lower()
        if action == "":
            updated = sorted(existing.union(suggested))
            save_categories(updated)
            mark_as_reviewed(entry, review)
            print("Kategorien übernommen und gespeichert.")
        elif action == "x":
            append_ignored(entry)
            print("Vorschlag verworfen.")
        else:
            print("Übersprungen.")

if __name__ == "__main__":
    run_review()
