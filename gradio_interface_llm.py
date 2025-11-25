import gradio as gr
import ollama

# Initialize Ollama client
client = ollama.Client()

# Choose model
MODEL = "gpt-oss:120b-cloud"

# Prompt function
def make_prompt(text):
    return f"""
    You are a transliteration system. Convert the following Hindi text written in Roman script **literally** into Devanagari script.
    Do not translate, correct, or change the spelling. Output only the Devanagari text.

    Roman text: "{text}"
    """

# Function to transliterate (used by Gradio)
def transliterate(text, mode):
    if not text.strip():
        return "Please enter text."
    
    if mode == "Word":
        prompt = make_prompt(text.strip())
    else:  # Sentence mode
        prompt = make_prompt(text.strip())

    try:
        response = client.generate(model=MODEL, prompt=prompt)
        return response.response.strip()
    except Exception as e:
        return f"Error: {e}"

# Build Gradio UI
with gr.Blocks(title="Roman → Devanagari Transliterator") as demo:
    gr.Markdown("## 🪶 Roman → Devanagari Transliteration (LLM via Ollama)")
    gr.Markdown("Type Hindi in Roman (like 'namaste', 'aap kaise ho') and get Devanagari output (like 'नमस्ते', 'आप कैसे हो').")

    with gr.Row():
        mode = gr.Radio(["Word", "Sentence"], value="Word", label="Mode")
    
    input_text = gr.Textbox(label="Enter Roman Hindi text", placeholder="e.g. namaste or aap kaise ho")
    output_text = gr.Textbox(label="Devanagari Output")

    run_btn = gr.Button("Transliterate")
    run_btn.click(fn=transliterate, inputs=[input_text, mode], outputs=output_text)

# Launch app
demo.launch()
