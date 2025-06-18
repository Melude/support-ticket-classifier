from transformers.pipelines import pipeline
from typing import Tuple, Any
from logic.suggestions import suggest_new_categories
from logic.category_store import load_categories, save_categories

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Ab welchem Score soll GPT-Vorschlag bevorzugt werden?
CONFIDENCE_THRESHOLD = 60.0

def classify(text: str) -> Tuple[str, float, list[str]]:
    categories = load_categories()

    # GPT-Vorschläge analysieren
    dynamic = suggest_new_categories(text, current_category="", categories=categories)
    updated = categories + [c for c in dynamic if c not in categories]

    # Kategorien speichern, wenn erweitert
    if len(updated) > len(categories):
        save_categories(updated)

    # Klassifikation mit Zero-Shot
    result: Any = classifier(text, updated)
    best_label = result["labels"][0]
    score = round(result["scores"][0] * 100, 2)

    # Wenn Confidence zu niedrig → GPT-Vorschlag übernehmen
    if dynamic and score < CONFIDENCE_THRESHOLD:
        print(f"Low confidence ({score}%) – we trust GPT instead: {dynamic[0]}")
        best_label = dynamic[0]
        updated.append(best_label)
        updated = sorted(set(updated))

    return best_label, score, updated
