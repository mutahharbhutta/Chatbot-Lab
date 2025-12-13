import os
import requests
import gradio as gr

# Groq OpenAI-compatible endpoint
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Best general-purpose pick on Groq (strong + versatile)
DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are "Kasoti Book Oracle", a fun and sharp book-guessing mentor.

Core behavior:
- You can run in 3 modes:
  1) Clue Mode: give 3–8 progressive clues (easy -> hard) about a book without revealing the title first.
  2) Quiz Mode: ask short MCQs or quick questions about books/authors/characters/genres.
  3) Study Mode: give a compact summary + author + genre + 3 key identifiers.

Rules:
- If the user asks for a clue game, DO NOT reveal the book title until they say "reveal" or after 3 wrong guesses.
- Keep responses clear, energetic, and classroom-friendly.
- If user provides a book name, you can generate Kasoti-style clues for it.
"""

def query_groq(user_message, chat_history, model_name, temperature, max_tokens, mode):
    if not GROQ_API_KEY:
        return "Missing GROQ_API_KEY. Add it in Hugging Face Space → Settings → Secrets."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Add a small mode instruction on top of system prompt
    mode_hint = f"Current mode: {mode}. Follow the mode rules strongly."
    messages = [{"role": "system", "content": SYSTEM_PROMPT.strip() + "\n\n" + mode_hint}]

    for u, b in chat_history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens)
    }

    r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    return f"Error {r.status_code}: {r.text}"

def respond(message, chat_history, model_name, temperature, max_tokens, mode):
    reply = query_groq(message, chat_history, model_name, temperature, max_tokens, mode)
    chat_history.append((message, reply))
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Kasoti Book Oracle (Powered by Groq)")

    with gr.Row():
        mode = gr.Dropdown(
            ["Clue Mode", "Quiz Mode", "Study Mode"],
            value="Clue Mode",
            label="Bot Mode (UI improvement)"
        )
        model_name = gr.Dropdown(
            [
                "llama-3.3-70b-versatile",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "deepseek-r1-distill-llama-70b"
            ],
            value=DEFAULT_MODEL,
            label="Model"
        )

    with gr.Row():
        temperature = gr.Slider(0.0, 1.2, value=0.7, step=0.1, label="Creativity (temperature)")
        max_tokens = gr.Slider(128, 1024, value=512, step=64, label="Response length (max_tokens)")

    chatbot = gr.Chatbot(height=420)
    state = gr.State([])

    msg = gr.Textbox(label="Type here… (e.g., 'Give me clues for a Pakistani novel')")

    with gr.Row():
        clear = gr.Button("Clear Chat")
        example1 = gr.Button("Start a clue game")
        example2 = gr.Button("Quiz me")

    msg.submit(respond, [msg, state, model_name, temperature, max_tokens, mode], [msg, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, state])

    example1.click(lambda: "Clue Mode: Give me 6 progressive clues for a famous Pakistani book. Don’t reveal the title.",
                   None, msg)
    example2.click(lambda: "Quiz Mode: Give me 5 quick MCQs about famous world literature.",
                   None, msg)

demo.launch()
