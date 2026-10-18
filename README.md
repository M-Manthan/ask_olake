# Ask OLake — Hackathon MVP

A Streamlit documentation chatbot for OLake Go. It indexes official OLake documentation, retrieves relevant pages using TF-IDF, and uses OpenRouter to write a grounded answer with source URLs.

## Features
- Chat interface with conversation history
- Searches locally indexed official OLake docs
- Answers with source links
- Suggested starter questions
- Simple Python stack suitable for a hackathon prototype

## Requirements
- Python 3.10+
- Internet connection for documentation indexing and OpenRouter API calls
- OpenRouter API key

## Windows setup (PowerShell)

```powershell
cd ask_olake
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your OpenRouter API key as `OPENROUTER_API_KEY`. The app also reads this key from Streamlit Secrets when deployed.

Index the official documentation:

```powershell
python ingest_docs.py
```

The generated `data/olake_docs.json` is part of the project so the documentation index is available in deployments. The script writes relative to its own location, not the current working directory.

Run the chatbot:

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Demo questions
- What is OLake?
- How do I create my first ingestion job?
- Which source databases does OLake support?
- Explain Change Data Capture (CDC).
- What should I check if my source connection fails?

## Important notes
- The first indexing run requires internet access. If OLake's website structure changes, update `SEED_URLS` in `ingest_docs.py`.
- The retriever is a lightweight TF-IDF baseline, not a production vector database.
- Keep `.env` private. Never commit API keys.
- For Streamlit Community Cloud, add `OPENROUTER_API_KEY` in the app's Secrets settings.
- The assistant is instructed not to invent undocumented details, but always verify commands and configuration against the linked official docs.
- This is a hackathon MVP; production deployment should add rate limiting, stronger retrieval, tests, logging, and privacy controls.

## Official sources
- https://olake.io/docs/
- https://olake.io/docs/getting-started/creating-first-pipeline/
- https://olake.io/docs/getting-started/playground/
- https://olake.io/docs/community/setting-up-a-dev-env/
