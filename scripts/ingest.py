import os, json, re
from pathlib import Path
from tqdm import tqdm  
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SRC_DIR = Path("data/source")
OUT_PATH = Path("data/index.json")
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_pdf(p: Path) -> str:
    reader = PdfReader(str(p))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append(f"[PAGE {i}]\n{txt}\n")
    return "\n".join(out)

def normalize_ws(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    i = 0
    while i < len(s):
        chunks.append(s[i:i+size])
        i += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def file_to_chunks(path: Path):
    if path.suffix.lower() in {".txt", ".md"}:
        text = read_text(path)
    elif path.suffix.lower() == ".pdf":
        text = read_pdf(path)
    else:
        return []
    text = normalize_ws(text)
    chunks = chunk_text(text)
    return [
        {"id": f"{path.name}::chunk{i}", "source": path.name, "text": ch}
        for i, ch in enumerate(chunks)
    ]

def embed_batch(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    assert SRC_DIR.exists(), f"{SRC_DIR} not found"
    all_chunks = []
    for p in sorted(SRC_DIR.iterdir()):
        if p.suffix.lower() in {".txt", ".md", ".pdf"}:
            all_chunks += file_to_chunks(p)
    if not all_chunks:
        print("No .txt/.md/.pdf files found in data/source/")
        return

    # embed in small batches
    BATCH = 64
    embeds = []
    for i in tqdm(range(0, len(all_chunks), BATCH), desc="Embedding"):
        batch = all_chunks[i:i+BATCH]
        embeds += embed_batch([c["text"] for c in batch])

    for c, v in zip(all_chunks, embeds):
        c["embedding"] = v

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": EMBED_MODEL, "chunks": all_chunks}, f)

    print(f"Saved {len(all_chunks)} chunks -> {OUT_PATH}")

if __name__ == "__main__":
    main()