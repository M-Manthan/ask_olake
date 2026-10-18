
import json
import os
from pathlib import Path

from http.server import BaseHTTPRequestHandler

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "olake_docs.json"

ALLOWED_MODELS = [
    "z-ai/glm-4.7-flash",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v4.1-flash",
]


def load_documents():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Documentation index is missing.")

    documents = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    documents = [
        doc for doc in documents
        if isinstance(doc, dict)
        and isinstance(doc.get("text"), str)
        and doc["text"].strip()
    ]

    if not documents:
        raise ValueError("Documentation index is empty.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
    )
    matrix = vectorizer.fit_transform(
        [doc["text"] for doc in documents]
    )

    return documents, vectorizer, matrix


def answer_question(question):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    documents, vectorizer, matrix = load_documents()

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ids = scores.argsort()[::-1][:4]

    results = []
    for i in ids:
        if scores[i] > 0:
            results.append(documents[int(i)])

    if not results:
        return {
            "answer": (
                "I couldn't find relevant information in the indexed "
                "OLake documentation. Try rephrasing your question."
            ),
            "sources": [],
        }

    context_parts = []
    sources = []

    for item in results:
        title = str(item.get("title", "OLake documentation"))
        url = str(item.get("url", ""))
        snippet = item["text"][:4000]

        context_parts.append(
            f"Title: {title}\nURL: {url}\nContent:\n{snippet}"
        )

        sources.append({
            "title": title,
            "url": url,
        })

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=ALLOWED_MODELS[0],
        extra_body={"models": ALLOWED_MODELS[1:]},
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Ask OLake, an assistant for OLake. "
                    "Answer using only the supplied documentation context. "
                    "If the context does not support an answer, say so. "
                    "Be clear, practical, and concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Documentation context:\n"
                    + "\n\n---\n\n".join(context_parts)
                ),
            },
        ],
    )

    answer = response.choices[0].message.content or (
        "I couldn't generate an answer. Please try again."
    )

    return {"answer": answer, "sources": sources}


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_GET(self):
        self._send_json(200, {
            "status": "ok",
            "service": "Ask OLake API",
        })

    def do_POST(self):
        if self.path.rstrip("/") != "/api":
            self._send_json(404, {"error": "Endpoint not found."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20000:
                self._send_json(400, {"error": "Invalid request size."})
                return

            payload = json.loads(self.rfile.read(length))
            question = payload.get("question", "")

            if not isinstance(question, str) or not question.strip():
                self._send_json(400, {
                    "error": "Please provide a question."
                })
                return

            if len(question) > 2000:
                self._send_json(400, {
                    "error": "Question must be 2000 characters or fewer."
                })
                return

            result = answer_question(question.strip())
            self._send_json(200, result)

        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Invalid JSON request."})
        except Exception:
            self._send_json(500, {
                "error": (
                    "Ask OLake couldn't process the request. "
                    "Check the deployment logs and configuration."
                )
            })
