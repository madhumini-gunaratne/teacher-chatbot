from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import os, json
import numpy as np
from datetime import timedelta

# Load environment variables and initialize Flask app
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
app.permanent_session_lifetime = timedelta(minutes=30)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# AI model configuration
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL = "text-embedding-3-small"

# System prompt and teaching style configuration
def read_text(path, default=""):
    """Read text file with fallback to default value."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return default

def get_teaching_style():
    """Get aggregated teaching style from transcripts."""
    transcripts_dir = Path("data/media/transcripts")
    if not transcripts_dir.exists():
        return {}

    style_scores = {}
    transcript_count = 0

    for transcript_file in transcripts_dir.glob("*.json"):
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
                if 'teaching_style' in transcript:
                    for style, score in transcript['teaching_style'].items():
                        style_scores[style] = style_scores.get(style, 0) + score
                    transcript_count += 1
        except Exception as e:
            print(f"Error reading transcript {transcript_file}: {e}")

    # Average the scores
    if transcript_count > 0:
        return {style: score / transcript_count for style, score in style_scores.items()}
    return {}

SYSTEM_PROMPT = read_text("data/system_prompt.txt", "")

# Style guide for engaging explanations
SIMPLE_STYLE = (
    "Make learning fun with this format! 🌟\n"
    "1. Hook: Start with an intriguing question or fun fact 🤔\n"
    "2. Adventure Time: Take students on a journey through the concept with 3-4 exciting stops 🚀\n"
    "3. Real-Life Magic: Share a fun, relatable example or story 🎯\n"
    "4. Mind-Blown Moment: Connect it to something surprising or interesting 💡\n"
    "5. Challenge: End with a fun 'what if' question or mini-puzzle 🎮\n"
    "6. Victory Lap: Sum it up with an encouraging TL;DR 🏆\n\n"
    "Remember: Keep it fun but informative, use emojis wisely, and celebrate learning moments!"
)

# Vector index configuration
INDEX_PATH = "data/index.json"
INDEX = {"model": EMBED_MODEL, "chunks": []}
EMB = None  # would hold embeddings if using real retrieval

def load_index():
    """Load vector index from disk."""
    global INDEX, EMB
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            INDEX = json.load(f)
        # Build embedding matrix if real embeddings exist
        try:
            EMB = np.array([c["embedding"] for c in INDEX["chunks"]], dtype=np.float32)
            norms = np.linalg.norm(EMB, axis=1, keepdims=True) + 1e-10
            EMB[:] = EMB / norms
        except Exception:
            EMB = None
        print(f"Loaded index with {len(INDEX['chunks'])} chunks.")
    else:
        print("No index found. (Dummy mode).")

load_index()

def embed_query(q: str):
    """Get embeddings for query using OpenAI API."""
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=q)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return np.array([0.0] * 1536, dtype=np.float32)  # Default embedding dimension

def retrieve(query: str, k=4):
    """Find most relevant chunks using embeddings similarity."""
    if not INDEX["chunks"] or EMB is None:
        # If no embeddings, just return the first k chunks
        return INDEX["chunks"][:max(1, k)] if INDEX["chunks"] else []

    try:
        q_emb = embed_query(query)
        q_emb_normalized = q_emb / (np.linalg.norm(q_emb) + 1e-10)

        if EMB.shape[1] != len(q_emb):
            print(f"Dimension mismatch: EMB has {EMB.shape[1]} dims, query has {len(q_emb)} dims. Returning all chunks.")
            return INDEX["chunks"][:max(1, k)] if INDEX["chunks"] else []

        scores = EMB @ q_emb_normalized
        top_k = np.argsort(scores)[-k:][::-1]
        return [INDEX["chunks"][i] for i in top_k]
    except Exception as e:
        print(f"Error in retrieve: {e}")
        return INDEX["chunks"][:max(1, k)] if INDEX["chunks"] else []

def build_context(chunks):
    """Build context string from retrieved chunks."""
    parts = []
    for c in chunks:
        source = c.get('source', 'notes')
        text = c.get('text', '')
        timestamp = ''

        if 'LECTURE:' in source and 'segments' in c:
            for seg in c['segments']:
                if seg['text'] in text:
                    timestamp = f" [Time: {seg['start']}]"
                    break

        parts.append(f"[Source: {source}{timestamp}] {text}")
    return "\n\n".join(parts)

def adapt_response_to_style(teaching_style):
    """Generate style instructions based on teacher's style."""
    style_prompts = []

    if not teaching_style:
        return ""

    if teaching_style.get('interactive', 0) > 0.2:
        style_prompts.append("Ask engaging questions throughout your explanation")
    if teaching_style.get('encouraging', 0) > 0.2:
        style_prompts.append("Use encouraging phrases and positive reinforcement")
    if teaching_style.get('methodical', 0) > 0.2:
        style_prompts.append("Break down concepts step by step")
    if teaching_style.get('analogies', 0) > 0.2:
        style_prompts.append("Use relevant analogies and metaphors")
    if teaching_style.get('humor', 0) > 0.2:
        style_prompts.append("Include appropriate light-hearted comments")
    if teaching_style.get('empathetic', 0) > 0.2:
        style_prompts.append("Show understanding of potential confusion points")

    return "\n".join(style_prompts)

def parse_model_selection(model_string):
    """Parse UI model selection and return provider and model name."""
    model_map = {
        "openai-gpt35": ("openai", "gpt-3.5-turbo"),
        "openai-gpt4": ("openai", "gpt-4"),
        "anthropic-opus": ("anthropic", "claude-3-opus-20240229"),
        "anthropic-sonnet": ("anthropic", "claude-3-sonnet-20240229"),
        "anthropic-haiku": ("anthropic", "claude-3-haiku-20240307"),
        "google-gemini": ("google", "gemini-pro"),
    }
    
    provider, model = model_map.get(model_string, ("openai", "gpt-3.5-turbo"))
    return provider, model

# Flask Routes
@app.route("/")
def home():
    """Render home page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests and return AI responses."""
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    print("=== Chat Request ===")
    print("Method:", request.method)
    print("Headers:", dict(request.headers))

    try:
        payload = request.get_json(silent=True)
        print("Received payload:", payload)

        if not payload:
            print("No payload received")
            return jsonify({"error": "No JSON payload received"}), 400

        user_message = payload.get("message", "").strip()
        print("User message:", user_message)

        if not user_message:
            return jsonify({"error": "No 'message' in JSON body."}), 400

        # Parse selected model from UI
        selected_model_str = payload.get("model", "openai-gpt35")
        provider, model = parse_model_selection(selected_model_str)
        print(f"Using provider: {provider}, model: {model}")

        # Initialize session if not exists
        if 'messages' not in session:
            session['messages'] = []

        # Retrieve relevant context
        try:
            top_chunks = retrieve(user_message, k=4)
            context_block = build_context(top_chunks) if top_chunks else ""
        except Exception as e:
            print(f"Error retrieving context: {e}")
            context_block = ""

        # Get teaching style
        try:
            teaching_style = get_teaching_style()
            style_instructions = adapt_response_to_style(teaching_style)
        except Exception as e:
            print(f"Error getting teaching style: {e}")
            style_instructions = ""

        # Build message list
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": SIMPLE_STYLE},
        ]

        # Add conversation history (last 5 messages)
        messages.extend(session['messages'][-5:])

        if style_instructions:
            messages.append({
                "role": "system",
                "content": f"Match the teacher's style:\n{style_instructions}"
            })
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

        # Call OpenAI API
        try:
            print(f"Calling {provider.upper()} API with model: {model}")
            reply = call_ai_model(messages, provider=provider, model=model)
            print(f"Got reply: {reply[:100]}...")

            # Update conversation history
            session['messages'].append({"role": "user", "content": user_message})
            session['messages'].append({"role": "assistant", "content": reply})
            session.modified = True

            return jsonify({"reply": reply, "provider": provider, "model": model})
        except Exception as e:
            print(f"Error calling {provider} API: {e}")
            # Fallback response
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

    except Exception as e:
        print(f"Unexpected error in /chat: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Server error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)