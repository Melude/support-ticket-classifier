from transformers.pipelines import pipeline
from typing import Any, Tuple
from logic.suggestions import suggest_new_categories
from logic.category_store import load_categories, save_categories

# Klassifikator initialisieren
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Schwelle für GPT-Fallback
CONFIDENCE_THRESHOLD = 40.0


def classify(text: str) -> Tuple[str, float, list[str]]:
    categories = load_categories()

    # === 1. Klassifikation (Zero-Shot) ===
    result: dict[str, Any] = classifier(text, candidate_labels=categories) # type: ignore
    labels: list[str] = result["labels"]
    scores: list[float] = result["scores"]

    best_label = labels[0]
    score = round(scores[0] * 100, 2)

    print(f"Zero-Shot Ergebnis: {result}")
    print(f"Top Label: '{best_label}' mit Score: {score} %")

    # === 2. Ergebnis ausreichend sicher? ===
    if score >= CONFIDENCE_THRESHOLD:
        print("Score ausreichend hoch, kein Fallback nötig.")
        return best_label, score, categories

    print("Low confidence – GPT-Fallback wird aktiviert.")

    # === 3. GPT-Vorschläge einholen ===
    new_suggestions = suggest_new_categories(text, best_label, categories)

    if new_suggestions:
        updated_categories = categories + [c for c in new_suggestions if c not in categories]
    
        if updated_categories != categories:
            save_categories(updated_categories)
            print("Neue Kategorien gespeichert:", new_suggestions)

            # Neue Klassifikation mit erweiterter Liste
            result_updated: dict[str, Any] = classifier(text, candidate_labels=updated_categories) # type: ignore
            labels_updated: list[str] = result_updated["labels"]
            scores_updated: list[float] = result_updated["scores"]

            best_label_updated = labels_updated[0]
            score_updated = round(scores_updated[0] * 100, 2)

            print(f"Neue Klassifikation: '{best_label_updated}' mit Score: {score_updated} %")

            return best_label_updated, score_updated, updated_categories

        else:
            print("Vorschläge, aber keine neuen Kategorien – behalte ursprüngliches Label")
            return best_label, score, categories

    # === 5. Kein brauchbarer Vorschlag → Label unklar ===
    print("Keine besseren Vorschläge – setze Label auf 'unklar'")
    return "unklar", score, categories
