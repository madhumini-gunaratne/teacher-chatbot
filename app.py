from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import os, json
import numpy as np

# --- env + client ---
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- models ---
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = "text-embedding-3-small"  # unused while in dummy mode

# ---- Persona (system prompt) ----
def read_text(path, default=""):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return default

SYSTEM_PROMPT = read_text(
    "data/system_prompt.txt",
    "You are a calm, patient teacher. Explain step-by-step with small examples and check understanding."
)

# ---- Simple style booster (keeps answers short & step-by-step) ----
SIMPLE_STYLE = (
    "Always answer in this exact format:\n"
    "- One-sentence idea.\n"
    "- Steps: 3–5 numbered lines, one short line each.\n"
    "- Use tiny examples (numbers or A–Z) if helpful.\n"
    "- Avoid jargon; define any new term in plain words.\n"
    "- End with: 'TL;DR: ...' one line.\n"
    "Hard limit: 120 words total."
)

# ---- Load vector index (dummy-friendly) ----
INDEX_PATH = "data/index.json"
INDEX = {"model": EMBED_MODEL, "chunks": []}
EMB = None  # would hold embeddings if using real retrieval

def load_index():
    """Loads data/index.json if present. Works with dummy embeddings too."""
    global INDEX, EMB
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            INDEX = json.load(f)
        # Build a matrix only if real embeddings exist and are numeric
        try:
            EMB = np.array([c["embedding"] for c in INDEX["chunks"]], dtype=np.float32)
            norms = np.linalg.norm(EMB, axis=1, keepdims=True) + 1e-10
            EMB[:] = EMB / norms
        except Exception:
            EMB = None  # stay in dummy mode
        print(f"Loaded index with {len(INDEX['chunks'])} chunks.")
    else:
        print("No index found. (Dummy mode).")

load_index()

# ---- Dummy retrieval (returns all chunks) ----
def embed_query(q: str):
    # Skipped while in dummy mode
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def retrieve(query: str, k=4):
    # In dummy mode, just return all chunks (or first k if you prefer)
    return INDEX["chunks"][:max(1, k)] if INDEX["chunks"] else []

def build_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c.get('source','notes')}] {c.get('text','')}")
    return "\n\n".join(parts)

# ---- routes ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "No 'message' in JSON body."}), 400

    # 1) retrieve notes (dummy or real)
    top_chunks = retrieve(user_message, k=4)
    context_block = build_context(top_chunks) if top_chunks else ""

    # 2) construct messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": SIMPLE_STYLE},
    ]
    if context_block:
        messages.append({
            "role": "system",
            "content": (
                "Use the following course materials to answer. "
                "Cite the source filename inline like [Source: name]. "
                "If not covered, say you don't have that info.\n\n" + context_block
            )
        })
    messages.append({"role": "user", "content": user_message})

    # 3) call model (with short, deterministic settings)
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        # Fallback: synthesize from context so UI always shows something useful
        if context_block:
            fallback = (
                "One-line idea: Answer based on the course notes below.\n"
                "1) Read the key lines from notes.\n"
                "2) State the concept in plain words.\n"
                "3) Give a tiny example (A=0..Z=25 or small numbers).\n"
                "4) End with a one-line TL;DR.\n\n"
                f"{context_block}\n\n"
                "TL;DR: Used the notes above to summarize."
            )
        else:
            fallback = "I couldn't reach the model and I have no notes loaded yet."
        return jsonify({"reply": fallback, "error": f"{type(e).__name__}: {e}"}), 200

if __name__ == "__main__":
    # Change the port/host as you like (e.g., host='0.0.0.0' to test on phone)
    app.run(debug=True, port=8000)