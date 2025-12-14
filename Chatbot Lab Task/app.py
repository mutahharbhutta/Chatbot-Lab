import os
import requests
import gradio as gr

# Groq OpenAI-compatible endpoint
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

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
""".strip()


def detect_mode_and_clean_text(text: str, current_mode: str):
    """
    1) If user explicitly writes 'Quiz Mode:' / 'Clue Mode:' / 'Study Mode:', respect it.
    2) Otherwise, auto-detect mode from keywords (e.g., 'mcqs' -> Quiz Mode).
    """
    t = text.strip()
    low = t.lower()

    # ---- 1) Explicit prefixes (highest priority) ----
    if low.startswith("quiz mode:"):
        return "Quiz Mode", t.split(":", 1)[1].strip()
    if low.startswith("clue mode:"):
        return "Clue Mode", t.split(":", 1)[1].strip()
    if low.startswith("study mode:"):
        return "Study Mode", t.split(":", 1)[1].strip()

    # ---- 2) Keyword-based auto-detect (no prefix) ----
    quiz_keywords = [
        "mcq", "mcqs", "quiz", "multiple choice", "objective", "test me",
        "choose the correct", "choose the right", "options", "a)", "b)", "c)", "d)"
    ]
    study_keywords = [
        "summary", "summarize", "explain", "notes", "key points", "study", "revision",
        "overview", "theme", "themes", "plot", "characters", "author", "genre"
    ]
    clue_keywords = [
        "clue", "clues", "hint", "hints", "guess", "kasoti", "don't reveal", "dont reveal",
        "next clue", "harder clue", "reveal"
    ]

    quiz_score = sum(1 for k in quiz_keywords if k in low)
    study_score = sum(1 for k in study_keywords if k in low)
    clue_score = sum(1 for k in clue_keywords if k in low)

    best = max(quiz_score, study_score, clue_score)

    # If no strong keyword signal, keep current dropdown mode
    if best == 0:
        return current_mode, t

    # Resolve ties by priority: Quiz > Study > Clue
    if quiz_score == best:
        return "Quiz Mode", t
    if study_score == best:
        return "Study Mode", t
    return "Clue Mode", t


def query_groq(user_message, chat_history, model_name, temperature, max_tokens, mode):
    if not GROQ_API_KEY:
        return "❌ Missing GROQ_API_KEY. Add it in Hugging Face Space → Settings → Secrets."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    mode_hint = f"Current mode: {mode}. Follow the mode rules strongly."
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + mode_hint}]

    for m in chat_history:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return (
            f"❌ Groq error {r.status_code}. Try another model.\n\nDetails: {r.text}"
        )
    except Exception as e:
        return f"❌ Request failed: {repr(e)}"


def respond(message, chat_history, model_name, temperature, max_tokens, current_mode):
    # ✅ Auto-detect mode (with or without prefixes)
    new_mode, cleaned_message = detect_mode_and_clean_text(message, current_mode)

    reply = query_groq(
        cleaned_message, chat_history, model_name, temperature, max_tokens, new_mode
    )

    chat_history.append({"role": "user", "content": cleaned_message})
    chat_history.append({"role": "assistant", "content": reply})

    return "", chat_history, chat_history, gr.update(value=new_mode)


def clear_chat():
    return [], [], gr.update(value="Clue Mode")


with gr.Blocks() as demo:
    gr.Markdown("## 📚 Kasoti Book Oracle (Powered by Groq)")
    gr.Markdown(
        "### How to use\n"
        "- Select a **Mode** OR just type naturally (e.g., 'give MCQs' → Quiz Mode)\n"
        "- You can also type explicit prefixes: **Clue Mode:** / **Quiz Mode:** / **Study Mode:**\n"
        "- Press **Enter** to send\n"
        "- Use **Clear Chat** to restart\n"
    )

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

    chatbot = gr.Chatbot(
        height=420,
        type="messages",
        # autoscroll=True
    )

    state = gr.State([])

    msg = gr.Textbox(label="Type here… (e.g., 'Give me 5 MCQs about literature')")

    gr.Examples(
        examples=[
            "Give me 5 MCQs about famous world literature.",
            "Summarize Animal Farm with author, genre, and 3 key identifiers.",
            "Give me clues for a famous Pakistani novel. Don’t reveal the title."
        ],
        inputs=msg
    )

    with gr.Row():
        clear = gr.Button("Clear Chat")

    msg.submit(
        respond,
        [msg, state, model_name, temperature, max_tokens, mode],
        [msg, chatbot, state, mode]
    )

    clear.click(clear_chat, None, [chatbot, state, mode])


IS_HF = os.environ.get("SPACE_ID") is not None
demo.launch(share=not IS_HF)
