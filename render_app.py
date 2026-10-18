
from flask import Flask, request, jsonify
from api.index import answer_question

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Ask OLake Render API"
    })


@app.route("/api", methods=["GET", "POST", "OPTIONS"])
def api():
    if request.method == "OPTIONS":
        return "", 204

    if request.method == "GET":
        return jsonify({
            "status": "ok",
            "service": "Ask OLake Render API"
        })

    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please provide a question."}), 400

    if len(question) > 2000:
        return jsonify({"error": "Question is too long."}), 400

    try:
        result = answer_question(question.strip())
        return jsonify(result)
    except Exception:
        app.logger.exception("Ask OLake request failed")
        return jsonify({
            "error": "Ask OLake couldn't process the request. Check Render logs and configuration."
        }), 500
