# AryaGPT v2 🤖

An agentic, self-updating personal AI assistant that answers questions about **Arya Shah** — built with LangGraph, ChromaDB, and Streamlit.

---

## Features

- **Agentic RAG** — LangGraph agent with 8 tools: knowledge-base retrieval, web search (DuckDuckGo, Arya-scoped), GitHub activity, date lookup, document summarizer, conversation history search, elevator pitch generator, and follow-up question suggester
- **Local embeddings** — `all-MiniLM-L6-v2` + cross-encoder reranking (no API key needed for embeddings)
- **Multi-provider LLM** — Groq, Together AI, or OpenAI, switchable live via admin panel
- **Admin panel** — password-gated, cookie-authenticated; change provider/model and it instantly applies to all live visitors
- **Self-updating KB** — push any file to `data/` → GitHub Action rebuilds ChromaDB → Streamlit redeploys automatically
- **Eval dashboard** — P50/P95 latency, per-provider breakdown, per-query detail table, latency over time chart
- **Streaming UI** — dark-themed Streamlit app with source citations

---

## Project Structure

```
aryagpt/
├── app.py                      # Main chat UI
├── pages/
│   └── admin.py                # Admin panel (password-gated)
├── agent/
│   ├── graph.py                # LangGraph agent
│   ├── tools.py                # 8 tool implementations
│   └── prompts.py              # System prompts
├── rag/
│   ├── ingest.py               # Ingestion pipeline
│   ├── retriever.py            # Retrieval + reranking
│   └── embeddings.py           # Local embedding model
├── admin/
│   └── config.py               # Provider/model config
├── db/
│   └── logger.py               # SQLite logger + eval metrics
├── data/                       # Drop your files here
├── chroma_db/                  # Auto-managed vector store
├── scripts/
│   └── ingest.py               # CLI ingest tool
├── .github/workflows/
│   └── update_kb.yml           # Auto-rebuild KB on data push
├── .streamlit/
│   ├── config.toml             # Theme config
│   └── secrets.toml            # API keys (DO NOT COMMIT)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

Edit `.streamlit/secrets.toml`:

```toml
ADMIN_PASSWORD = "your_secure_password"
ADMIN_USERNAME = "arya"
ADMIN_NAME     = "Arya Shah"

GROQ_API_KEY     = "gsk_..."
TOGETHER_API_KEY = "..."
OPENAI_API_KEY   = "sk-..."   # optional
```

### 3. Add your data

Drop any files into `data/`:
- `.pdf` — resume, documents
- `.csv` — structured Q&A
- `.md` / `.txt` — bios, project descriptions
- `.json` — structured data

### 4. Build the knowledge base

```bash
python scripts/ingest.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Adding/Updating Data

**Method 1 — GitHub push (recommended):**
```bash
# Drop your file into data/
git add data/your_new_file.pdf
git commit -m "add updated resume"
git push
```
The GitHub Action runs automatically, rebuilds the ChromaDB index, commits it back, and Streamlit Cloud redeploys. Live in ~2 minutes.

**Method 2 — Admin panel:**
Navigate to `/admin` → Knowledge Base tab → click **Run Re-Ingest Now**.

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo, `app.py`
3. In **Advanced settings → Secrets**, paste your `secrets.toml` contents
4. Deploy

The GitHub Action needs a `GITHUB_TOKEN` secret with write access — this is provided automatically by GitHub Actions.

---

## Admin Panel

Navigate to `/admin` (linked in the Streamlit sidebar). Login with your `ADMIN_USERNAME` / `ADMIN_PASSWORD`.

**Tabs:**
| Tab | What you can do |
|---|---|
| ⚙️ Provider & Model | Switch LLM provider/model live — applies instantly to all visitors |
| 📚 Knowledge Base | View KB stats, list source files, trigger manual re-ingest |
| 💬 Conversation Logs | Browse and search all conversations (SQLite) |
| 📊 Eval Dashboard | P50/P95 latency, provider breakdown, per-query table, latency chart |

---

## Environment Variables

All secrets should be in `.streamlit/secrets.toml` (local) or Streamlit Cloud secrets (production).

| Variable | Required | Description |
|---|---|---|
| `ADMIN_PASSWORD` | ✅ | Admin panel login password |
| `ADMIN_USERNAME` | optional | Admin panel login username |
| `GROQ_API_KEY` | ✅ (if using Groq) | Groq API key |
| `TOGETHER_API_KEY` | ✅ (if using Together) | Together AI API key |
| `OPENAI_API_KEY` | optional | OpenAI API key |
