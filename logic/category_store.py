import yaml
from pathlib import Path

CATEGORY_FILE = Path("data/categories.yaml")

def load_categories() -> list[str]:
    if not CATEGORY_FILE.exists():
        return []
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("categories", [])

def save_categories(categories: list[str]) -> None:
    unique = sorted(set(categories))  # keine Duplikate
    with open(CATEGORY_FILE, "w", encoding="utf-8") as f:
        yaml.dump({"categories": unique}, f, allow_unicode=True)
