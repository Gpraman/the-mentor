# ============================================================
# THE MENTOR - Render.com Version
# ============================================================

import os
import csv
from datetime import datetime
import gradio as gr
from PIL import Image
from google import genai
from google.genai import types

# ============================================================
# 1. API KEY SETUP (Ρυθμισμένο για Server/Render)
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not found. Please add it in Render Settings.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-flash-lite-latest"

# ============================================================
# 2. DATA FILES
# ============================================================
QUESTIONNAIRE_FILE = "questionnaire_responses.csv"

# ============================================================
# 3. MENTOR PERSONA & CRITERIA
# ============================================================
SYSTEM_INSTRUCTION = """
You are THE MENTOR, a strict but fair AI photography critic
and educational interlocutor for beginner and amateur photographers.

Your role is to evaluate photographs critically and help the
photographer develop a more conscious photographic practice.

You must be honest, specific, analytical and constructive.
Avoid empty praise, generic comments and exaggerated criticism.

IMPORTANT PRINCIPLES:
- Do not pretend that artistic value can be measured objectively.
- Explain your judgments using visible evidence in the image.
- Distinguish technical problems from deliberate artistic choices.
- Do not assume the photographer's intention if it is not provided.
- A technically imperfect photograph can still have strong artistic impact.
- Do not automatically reward conventional or commercial aesthetics.
- Treat scores as reasoned critical estimates, not absolute truths.

PHASE 1 — PHOTOGRAPHIC EVALUATION
Evaluate the photograph using three criteria:
1. Technical execution
2. Composition and structure
3. Artistic intent and impact

For each criterion:
- Give a score from 1 to 10.
- Explain the score with specific observations.

Then provide:
- An overall score from 1 to 10.
- A concise critical assessment of approximately 80–100 words.
- One principal strength.
- One principal weakness or unresolved issue.

PHASE 2 — EDUCATIONAL DIALOGUE & PROMPT ENGINEERING
When the user asks questions:
- Answer in relation to the actual photograph and previous evaluation.
- Offer practical, specific suggestions.

If the user requests an image-generation prompt (AI Reimagining):
- Provide a highly detailed text-to-image prompt (in English, suitable for Midjourney or DALL-E).
- Format the final prompt clearly inside a markdown code block.
"""

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def call_gemini(contents):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.5,
            max_output_tokens=1800
        )
    )
    if not response.text:
        raise RuntimeError("The model did not return any text.")
    return response.text.strip()

def prepare_image(image):
    if image is None:
        return None
    image_copy = image.copy()
    image_copy.thumbnail((1920, 1920))
    if image_copy.mode not in ("RGB", "L"):
        image_copy = image_copy.convert("RGB")
    return image_copy

# ============================================================
# 5. PHOTOGRAPH EVALUATION
# ============================================================
def evaluate_artwork(image, title):
    if image is None:
        return "Please upload a photograph first.", []
    if not title or not title.strip():
        return "Please provide a title for your photograph before evaluation.", []

    try:
        prepared = prepare_image(image)
        prompt = f"""
Evaluate the submitted photograph.
Photograph title: {title.strip()}

First, briefly identify what is visibly present in the image.
Then evaluate it using the three criteria defined in your instructions.

Use this structure:
## 1. Technical execution
Score: X/10
Critical reasoning:

## 2. Composition and structure
Score: X/10
Critical reasoning:

## 3. Artistic intent and impact
Score: X/10
Critical reasoning:

## Overall evaluation
Overall score: X/10

Provide a concise critical assessment of approximately 80–100 words.
**Principal strength:** ...
**Principal weakness or unresolved issue:** ...
"""
        result = call_gemini([prepared, prompt])
        initial_history = [{"role": "assistant", "content": "The initial evaluation is complete. You can now discuss the feedback, request an AI-generated prompt to reimagine your work, or explore potential directions for improvement."}]
        return result, initial_history
    except Exception as e:
        print("Evaluation error:", repr(e))
        return "An error occurred during evaluation. Please check your connection and try again.", []

# ============================================================
# 6. MENTOR DIALOGUE
# ============================================================
def chat_with_mentor(user_message, chat_history, image, title):
    history = list(chat_history or [])
    if not user_message or not user_message.strip():
        return "", history
    if image is None:
        history.append({"role": "user", "content": user_message.strip()})
        history.append({"role": "assistant", "content": "Please upload a photograph first so I can comment on it accurately."})
        return "", history

    try:
        previous_messages = history[-12:]
        conversation_text = ""
        for msg in previous_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conversation_text += f"\n{role.upper()}: {content}\n"

        prompt = f"""
The photograph title is: {title or "Untitled"}.
Previous dialogue:
{conversation_text}

New user message:
{user_message.strip()}

Respond as THE MENTOR. Keep your response specific, useful, and connected to the photograph.
"""
        prepared = prepare_image(image)
        response_text = call_gemini([prepared, prompt])

        history.append({"role": "user", "content": user_message.strip()})
        history.append({"role": "assistant", "content": response_text})
        return "", history

    except Exception as e:
        print("Chat error:", repr(e))
        history.append({"role": "user", "content": user_message.strip()})
        history.append({"role": "assistant", "content": "I couldn't complete the response. Please try again in a moment."})
        return "", history

def request_ai_prompt(chat_history, image, title):
    request_msg = "Based on your critique, could you provide a detailed AI image-generation prompt (e.g., for Midjourney) to reimagine my photograph, addressing its weaknesses and enhancing its core concept?"
    return chat_with_mentor(request_msg, chat_history, image, title)

def clear_image():
    return None, "", "", []

# ============================================================
# 7. QUESTIONNAIRE
# ============================================================
QUESTION_FIELDS = [
    "q01_experience", "q02_level", "q03_frequency", "q04_feedback_frequency",
    "q05_ai_familiarity", "q06_understandable", "q07_important_elements",
    "q08_strengths_weaknesses", "q09_score_matches", "q10_reasoning",
    "q11_improvement", "q12_disagreement", "q13_fairness", "q14_consider_feedback",
    "q15_ai_or_human", "q16_artistic_intent", "q17_score_conflict",
    "q18_human_ai_difference", "q19_experimentation", "q20_personal_choices",
    "q21_self_evaluation", "q22_against_intent", "q23_useful_advice",
    "q24_advantage", "q25_limitation", "q26_reuse", "q27_suggested_change"
]

SCALE_OPTIONS = [
    "1 — Strongly Disagree", "2 — Disagree", "3 — Neither agree nor disagree",
    "4 — Agree", "5 — Strongly Agree", "Cannot judge"
]

def submit_questionnaire(consent, *answers):
    if not consent:
        return "You must confirm you have read the participation notice to submit."
    if len(answers) != len(QUESTION_FIELDS):
        return "There was a problem with the form data."

    required_indexes = [0, 1, 2, 4, 5, 8, 12, 16, 18, 25]
    for index in required_indexes:
        if answers[index] is None or str(answers[index]).strip() == "":
            return "Not all mandatory questions have been answered. Please check the required fields."

    row = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for field, answer in zip(QUESTION_FIELDS, answers):
        row[field] = answer if answer is not None else ""

    fieldnames = ["timestamp"] + QUESTION_FIELDS
    file_exists = os.path.exists(QUESTIONNAIRE_FILE) and os.path.getsize(QUESTIONNAIRE_FILE) > 0

    try:
        with open(QUESTIONNAIRE_FILE, "a", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        return "### Submission Complete\n\nThank you for participating in THE MENTOR research."
    except Exception as e:
        print("Questionnaire save error:", repr(e))
        return "Could not save the responses. Please try again."

# ============================================================
# 8. GRADIO UI ENVIRONMENT & CSS
# ============================================================
custom_css = """
#privacy-text, #privacy-text * {
    color: #e65c5c !important;
    font-size: 1.1em !important;
    text-align: center !important;
    margin-top: 5px !important;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
#intro-wrapper {
    min-height: 100vh !important;
    width: 100% !important;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: linear-gradient(-45deg, var(--background-fill-primary), var(--background-fill-secondary), var(--border-color-primary), var(--background-fill-primary));
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    border: none !important; 
    margin: 0 !important;
}
.intro-title {
    font-size: 5rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.3em !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    color: var(--body-text-color) !important;
}
#entrance-btn {
    font-size: 1.2rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.2em !important;
    padding: 15px 50px !important;
    background: transparent !important;
    border: 1px solid var(--primary-500) !important;
    color: var(--primary-500) !important;
    border-radius: 50px !important;
    transition: all 0.5s ease !important;
    cursor: pointer;
}
#entrance-btn:hover {
    background: var(--primary-500) !important;
    color: white !important;
    box-shadow: 0 0 25px var(--primary-500) !important;
    transform: scale(1.02);
}
"""

with gr.Blocks(title="THE MENTOR", css=custom_css) as app:
    with gr.Column(visible=True, elem_id="intro-wrapper") as intro_page:
        gr.HTML("<h1 class='intro-title'>THE MENTOR</h1>")
        entrance_btn = gr.Button("ENTRANCE", elem_id="entrance-btn")

    with gr.Column(visible=False) as main_page:
        gr.Markdown("# THE MENTOR\n### AI Photography Critic")
        with gr.Tab("Photographic Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Upload your photograph", type="pil")
                    title_input = gr.Textbox(label="Photograph Title", placeholder="Provide a title for your photograph", lines=1)
                    with gr.Row():
                        evaluate_button = gr.Button("Evaluate Photograph", variant="primary")
                        clear_button = gr.Button("Clear Photograph")
                    gr.HTML("<em>* Privacy Notice: Uploaded images are not saved or stored by this application. Your copyrights remain fully intact and unaffected.</em>", elem_id="privacy-text")
                with gr.Column(scale=2):
                    evaluation_output = gr.Markdown(label="THE MENTOR Evaluation")
            
            gr.Markdown("## Discuss the Evaluation")
            chatbot = gr.Chatbot(label="Dialogue with THE MENTOR", height=350)
            with gr.Row():
                message_input = gr.Textbox(label="Your message", placeholder="Ask about composition, lighting, the score, or potential improvements...", lines=2, scale=4)
                with gr.Column(scale=1):
                    send_button = gr.Button("Send", variant="primary")
                    prompt_btn = gr.Button("🎨 Suggest AI Reimagining", variant="secondary")

            evaluate_button.click(fn=evaluate_artwork, inputs=[image_input, title_input], outputs=[evaluation_output, chatbot])
            send_button.click(fn=chat_with_mentor, inputs=[message_input, chatbot, image_input, title_input], outputs=[message_input, chatbot])
            message_input.submit(fn=chat_with_mentor, inputs=[message_input, chatbot, image_input, title_input], outputs=[message_input, chatbot])
            prompt_btn.click(fn=request_ai_prompt, inputs=[chatbot, image_input, title_input], outputs=[message_input, chatbot])
            clear_button.click(fn=clear_image, inputs=[], outputs=[image_input, title_input, evaluation_output, chatbot])

        with gr.Tab("Research Questionnaire"):
            gr.Markdown("# Research Questionnaire\nAnswer based on your experience using THE MENTOR.")
            questionnaire_components = []
            
            gr.Markdown("## A. Your Photographic Experience")
            q01 = gr.Radio(["Less than 1 year", "1–3 years", "4–7 years", "More than 7 years"], label="1. How long have you been involved in photography?", info="Required")
            q02 = gr.Radio(["Beginner", "Amateur", "Advanced Amateur", "Other"], label="2. How would you describe your level?", info="Required")
            q03 = gr.Radio(["Daily", "Sometimes", "Rarely"], label="3. How often do you take photographs?", info="Required")
            q04 = gr.Radio(["Very often", "Sometimes", "Rarely", "Never"], label="4. How often do you ask others to critique your photos?")
            q05 = gr.Radio(["Not at all", "A little", "Moderately", "Quite a bit", "Very much"], label="5. How familiar are you with AI tools?", info="Required")
            questionnaire_components += [q01, q02, q03, q04, q05]

            gr.Markdown("## B. The Photograph Evaluation")
            q06 = gr.Radio(SCALE_OPTIONS, label="6. Were MENTOR's comments understandable?", info="Required")
            q07 = gr.Radio(SCALE_OPTIONS, label="7. Did it identify important elements of your photograph?")
            q08 = gr.Radio(SCALE_OPTIONS, label="8. Did it help you understand its strengths and weaknesses?")
            q09 = gr.Radio(SCALE_OPTIONS, label="9. Did its score match your own estimation?", info="Required")
            q10 = gr.Radio(SCALE_OPTIONS, label="10. Did you understand the reasons for this score?")
            q11 = gr.Radio(SCALE_OPTIONS, label="11. Did it give you a specific idea on how to improve?")
            q12 = gr.Textbox(label="12. Was there any comment or score you disagreed with? What was it?", lines=3)
            questionnaire_components += [q06, q07, q08, q09, q10, q11, q12]

            gr.Markdown("## C. Trust and Human Judgment")
            q13 = gr.Radio(SCALE_OPTIONS, label="13. Do you consider the evaluation fair?", info="Required")
            q14 = gr.Radio(SCALE_OPTIONS, label="14. Would you take MENTOR's remarks seriously?")
            q15 = gr.Radio(SCALE_OPTIONS, label="15. Would you want to know if the critique comes from AI or a human before reading it?")
            q16 = gr.Radio(SCALE_OPTIONS, label="16. Should an evaluation consider artistic intent, even when not fully visible in the photograph?")
            q17 = gr.Radio(["My own judgment", "I would re-examine the photo", "Human opinion", "MENTOR", "I don't know"], label="17. What would you do if MENTOR's score was very different from yours?", info="Required")
            q18 = gr.Textbox(label="18. What do you consider the most significant difference between human and AI photographic critique?", lines=3)
            questionnaire_components += [q13, q14, q15, q16, q17, q18]

            gr.Markdown("## D. Learning and Creative Freedom")
            q19 = gr.Radio(SCALE_OPTIONS, label="19. Did MENTOR's suggestions give you ideas for new photographic experiments?", info="Required")
            q20 = gr.Radio(SCALE_OPTIONS, label="20. Did its suggestions leave room for your own artistic choices?")
            q21 = gr.Radio(SCALE_OPTIONS, label="21. Did the experience help you evaluate your own photos more consciously?")
            q22 = gr.Radio(["Yes", "No", "I don't know"], label="22. Was there any suggestion that would change your photograph in a way contrary to your intent?")
            q23 = gr.Textbox(label="23. What was the most useful advice you received and why?", lines=3)
            questionnaire_components += [q19, q20, q21, q22, q23]

            gr.Markdown("## E. Overall Assessment")
            q24 = gr.Textbox(label="24. What do you consider the biggest advantage of AI photographic critique?", lines=3)
            q25 = gr.Textbox(label="25. What is the most significant limitation or risk?", lines=3)
            q26 = gr.Radio(["Yes", "No", "Maybe"], label="26. Would you use THE MENTOR again?", info="Required")
            q27 = gr.Textbox(label="27. If you could change one thing about THE MENTOR, what would it be?", lines=3)
            questionnaire_components += [q24, q25, q26, q27]

            consent = gr.Checkbox(label="I have read the participation notice and agree to the use of my responses for the described research purpose.")
            submit_button = gr.Button("Submit Questionnaire", variant="primary")
            questionnaire_output = gr.Markdown()

            submit_button.click(fn=submit_questionnaire, inputs=[consent] + questionnaire_components, outputs=questionnaire_output)

    def enter_app():
        return gr.Column(visible=False), gr.Column(visible=True)

    entrance_btn.click(fn=enter_app, inputs=[], outputs=[intro_page, main_page])

# Εκκίνηση με τις ρυθμίσεις του Render (server_name="0.0.0.0", server_port=10000)
app.launch(server_name="0.0.0.0", server_port=10000, css=custom_css)
