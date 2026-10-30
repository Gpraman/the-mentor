# ==========================================
# THE MENTOR - Render.com Version
# ==========================================

import os
import requests
from datetime import datetime
import gradio as gr
from PIL import Image
from google import genai
from google.genai import types

# ==========================================
# 1. API KEY & CONFIGURATION SETUP
# ==========================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not found. Please add it in Render settings.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-flash-lite-latest"

# ** ΒΑΛΕ ΕΔΩ ΤΟ WEB APP URL ΠΟΥ ΠΗΡΕΣ ΑΠΟ ΤΟ GOOGLE APPS SCRIPT **
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzsRip04pnl0_QI7-PZBCV1o_W9JbiTx3zt1wOLc1nXrhwGdWMsLFS8-dKirLupyTXiXA/exec"

def save_to_google_sheets(photo_title, evaluation_text, ans1, ans2, ans3, ans4):
    payload = {
        "photograph_title": photo_title,
        "ai_evaluation": evaluation_text,
        "q1": ans1,
        "q2": ans2,
        "q3": ans3,
        "q4": ans4
    }
    try:
        requests.post(GOOGLE_SCRIPT_URL, json=payload)
    except Exception as e:
        print(f"Error saving to Google Sheets: {e}")

# ==========================================
# 2. AI EVALUATION FUNCTION
# ==========================================

def evaluate_photograph(image, title):
    if image is None:
        return "Please upload a photograph first."
    
    try:
        prompt = (
            "You are 'The Mentor', an expert, constructive, and encouraging photography critic. "
            "Analyze the provided photograph. Comment on composition, lighting, subject matter, "
            "and emotional impact. Provide constructive feedback to help the photographer improve."
        )
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[image, prompt]
        )
        
        evaluation = response.text
        return evaluation
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        return "An error occurred during evaluation. Please check your connection and try again."

# ==========================================
# 3. GRADIO USER INTERFACE
# ==========================================

custom_css = """
body { background-color: #f9f9f9; font-family: sans-serif; }
"""

with gr.Blocks(title="THE MENTOR", css=custom_css) as app:
    gr.Markdown("# THE MENTOR\n### AI Photography Critic")
    
    with gr.Tabs():
        with gr.TabItem("Photographic Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload your photograph")
                    title_input = gr.Textbox(label="Photograph Title", placeholder="Enter title...")
                    eval_btn = gr.Button("Evaluate Photograph", variant="primary")
                    clear_btn = gr.Button("Clear Photograph")
                
                with gr.Column(scale=1):
                    eval_output = gr.Textbox(label="AI Evaluation", lines=15)
            
            eval_btn.click(
                fn=evaluate_photograph,
                inputs=[img_input, title_input],
                outputs=eval_output
            )
            
            clear_btn.click(
                fn=lambda: (None, "", ""),
                inputs=[],
                outputs=[img_input, title_input, eval_output]
            )

        with gr.TabItem("Research Questionnaire"):
            gr.Markdown("### Please help us with our research by answering a few short questions:")
            q1 = gr.Textbox(label="1. How would you rate the usefulness of the AI critique?")
            q2 = gr.Textbox(label="2. Did the feedback inspire you to look at your photo differently?")
            q3 = gr.Textbox(label="3. Would you use a tool like this regularly?")
            q4 = gr.Textbox(label="4. Any additional comments or suggestions?")
            
            submit_survey_btn = gr.Button("Submit Questionnaire & Save", variant="primary")
            survey_output = gr.Textbox(label="Status", interactive=False)
            
            def handle_submission(t, eval_txt, a1, a2, a3, a4):
                save_to_google_sheets(t, eval_txt, a1, a2, a3, a4)
                return "Thank you! Your responses have been successfully recorded."

            submit_survey_btn.click(
                fn=handle_submission,
                inputs=[title_input, eval_output, q1, q2, q3, q4],
                outputs=survey_output
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=10000, share=False)
