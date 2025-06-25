from transformers.pipelines import pipeline
from typing import Tuple, Any
from logic.suggestions import suggest_new_categories
from logic.category_store import load_categories, save_categories

# Zero-Shot-Klassifikator initialisieren
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

CONFIDENCE_THRESHOLD = 50.0


def classify(text: str) -> Tuple[str, float, list[str]]:
    print("=== Klassifikation gestartet ===")
    print("User Input:", text)

    categories = load_categories()
    print("Geladene Kategorien:", categories)

    dynamic: list[str] = suggest_new_categories(
        text,
        current_category="",
        categories=categories
    )
    print("GPT-Vorschläge (dynamic):", dynamic)

    updated = categories + [c for c in dynamic if c not in categories]
    updated = sorted(set(updated))
    print("Erweiterte Kategorienliste (updated):", updated)

    if len(updated) > len(categories):
        print("Neue Kategorien gespeichert.")
        save_categories(updated)

    result: Any = classifier(text, updated)
    print("Zero-Shot Ergebnis:", result)

    best_label: str = result["labels"][0]
    score: float = round(result["scores"][0] * 100, 2)
    print(f"Top Label: '{best_label}' mit Score: {score} %")

    # -------- Fallback-Logik mit Logging --------
    if score < CONFIDENCE_THRESHOLD:
        if dynamic:
            print(f"Low confidence ({score} %) – GPT-Vorschlag überschreibt Label: {dynamic[0]}")
            best_label = dynamic[0]
            if best_label not in updated:
                updated.append(best_label)
                save_categories(sorted(set(updated)))
        else:
            print(f"Low confidence ({score} %) – keine GPT-Vorschläge, setze Label auf 'unklar'")
            best_label = "unklar"
    else:
        print("Score ausreichend hoch, kein Fallback nötig.")
    # --------------------------------------------

    print("Finales Label:", best_label)
    print("=== Klassifikation abgeschlossen ===\n")
    return best_label, score, updated
