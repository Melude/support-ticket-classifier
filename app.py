import gradio as gr
from transformers.pipelines import pipeline

# Modell laden
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Kategorien
CATEGORIES = [
    "Rechnungen",
    "Technisches Problem",
    "Zugangsdaten vergessen",
    "Kündigung",
    "Allgemeine Anfrage"
]

# Vorformulierte Antworten
RESPONSE_TEMPLATES = {
    "Rechnungen": "Du findest alle Rechnungen im Kundenbereich unter 'Meine Dokumente'.",
    "Technisches Problem": "Bitte beschreibe das Problem genauer oder starte dein Gerät neu.",
    "Zugangsdaten vergessen": "Nutze die 'Passwort vergessen'-Funktion auf der Login-Seite.",
    "Kündigung": "Kündigungen kannst du im Kundenportal unter 'Verträge' einreichen.",
    "Allgemeine Anfrage": "Wir melden uns schnellstmöglich bei dir zurück."
}

# Klassifikationslogik
def classify_ticket(text):
    result = classifier(text, CATEGORIES)
    best_label = result["labels"][0]
    score = round(result["scores"][0] * 100, 2)
    reply = RESPONSE_TEMPLATES.get(best_label, "Wir kümmern uns um dein Anliegen.")
    return f"{best_label} ({score}%)", reply

# UI bauen
with gr.Blocks(title="Support Ticket Classifier") as demo:
    gr.Markdown("### Support Ticket Classifier\nOrdnet Anfragen automatisch Kategorien zu und schlägt passende Antworten vor.")

    with gr.Row():
        input_box = gr.Textbox(label="Support-Anfrage", placeholder="Was ist dein Anliegen?")
    
    with gr.Row():
        category_output = gr.Textbox(label="Erkannte Kategorie")
        reply_output = gr.Textbox(label="Antwortvorschlag")

    submit_button = gr.Button("Klassifizieren")

    submit_button.click(fn=classify_ticket, inputs=input_box, outputs=[category_output, reply_output])

# Start
demo.launch()
