import gradio as gr
from logic.classifier import classify
from logic.responder import generate_response
from logic.category_store import load_categories

def classify_and_respond(user_input: str):
    if not user_input.strip():
        return "", "", "", ""

    category, confidence, used_categories = classify(user_input)
    response = generate_response(category, user_input)
    return category, f"{confidence} %", response, ", ".join(used_categories)

with gr.Blocks(title="Support-Ticket Classifier") as demo:
    gr.Markdown("""
    # Support-Ticket Classifier  
    Dieses Tool analysiert eingehende Kundenanfragen, ordnet sie automatisch einer passenden Kategorie zu  
    und liefert eine kurze, professionelle GPT-Antwort.  
    """)

    with gr.Row():
        user_input = gr.Textbox(
            label="Kundenanfrage",
            placeholder="Was ist dein Anliegen?",
            lines=3,
            autofocus=True
        )

    with gr.Row():
        category_output = gr.Textbox(label="Erkannte Kategorie", interactive=False)
        confidence_output = gr.Textbox(label="Konfidenz", interactive=False)

    reply_output = gr.Textbox(label="GPT-Antwort", lines=4, interactive=False)
    categories_output = gr.Textbox(label="Verwendete Kategorien", interactive=False)

    with gr.Row():
        submit_btn = gr.Button("Anfrage analysieren")

    submit_btn.click(
        fn=classify_and_respond,
        inputs=user_input,
        outputs=[category_output, confidence_output, reply_output, categories_output]
    )

demo.launch()
