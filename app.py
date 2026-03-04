"""AryaGPT v2 — Main Streamlit chat UI."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_server_state import server_state, server_state_lock

from admin.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
)
from agent.graph import run_agent
from db.logger import get_recent_conversations, init_db, log_conversation, log_eval

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AryaGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AryaGPT v2 — An agentic AI assistant for Arya Shah. Built with LangGraph + Streamlit.",
    },
)

# ---------------------------------------------------------------------------
# Propagate secrets → environment variables (needed by agent LLM factory)
# ---------------------------------------------------------------------------

_SECRET_KEYS = ["GROQ_API_KEY", "TOGETHER_API_KEY", "OPENAI_API_KEY"]
for _k in _SECRET_KEYS:
    if _k not in os.environ:
        try:
            os.environ[_k] = st.secrets[_k]
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Initialise DB
# ---------------------------------------------------------------------------

init_db()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* Chat container */
.stChatMessage { border-radius: 12px; padding: 4px 8px; }

/* User bubble */
[data-testid="stChatMessageContent"][class*="user"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 18px 18px 4px 18px;
}

/* Assistant bubble */
[data-testid="stChatMessageContent"][class*="assistant"] {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 18px 18px 18px 4px;
}

/* Sidebar header */
.sidebar-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #cba6f7;
    margin-bottom: 0.5rem;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: #313244;
    color: #cdd6f4;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin: 2px;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%);
    border: 1px solid #45475a;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

/* Metric pill */
.metric-pill {
    background: #313244;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #a6e3a1;
    display: inline-block;
    margin-right: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None


# ---------------------------------------------------------------------------
# Track active sessions via server_state
# ---------------------------------------------------------------------------

with server_state_lock["active_sessions"]:
    if "active_sessions" not in server_state:
        server_state.active_sessions = set()
    server_state.active_sessions = server_state.active_sessions | {st.session_state.session_id}


def _get_active_provider() -> str:
    try:
        return server_state.get("active_provider", DEFAULT_PROVIDER)
    except Exception:
        return DEFAULT_PROVIDER


def _get_active_model() -> str:
    try:
        return server_state.get("active_model", DEFAULT_MODEL)
    except Exception:
        return DEFAULT_MODEL


def _get_active_temperature() -> float:
    try:
        return float(server_state.get("active_temperature", DEFAULT_TEMPERATURE))
    except Exception:
        return DEFAULT_TEMPERATURE


def _get_active_max_tokens() -> int:
    try:
        return int(server_state.get("active_max_tokens", DEFAULT_MAX_TOKENS))
    except Exception:
        return DEFAULT_MAX_TOKENS


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🤖 AryaGPT")
    st.caption("An agentic AI assistant for Arya Shah")
    st.divider()

    # KB status
    st.markdown('<div class="sidebar-header">📚 Knowledge Base</div>', unsafe_allow_html=True)
    meta_path = Path("chroma_db") / "ingest_meta.json"
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            kb_meta = json.load(f)
        st.success(f"✅ {kb_meta.get('total_chunks', '?')} chunks indexed")
        st.caption(f"Last updated: {kb_meta.get('last_ingest_human', 'unknown')}")
    else:
        st.warning("⚠️ Knowledge base not yet built. Run ingest first.")

    st.divider()

    # Active provider info
    provider = _get_active_provider()
    model = _get_active_model()
    provider_display = {"groq": "Groq ⚡", "together": "Together AI 🤝", "openai": "OpenAI 🌐"}.get(provider, provider)
    st.markdown('<div class="sidebar-header">⚙️ Active Model</div>', unsafe_allow_html=True)
    st.info(f"**{provider_display}**\n\n`{model}`")

    st.divider()

    # UI controls
    st.markdown('<div class="sidebar-header">🎛️ Display Options</div>', unsafe_allow_html=True)
    st.session_state.show_sources = st.toggle("Show source citations", value=st.session_state.show_sources)

    if st.session_state.last_latency:
        st.divider()
        st.markdown('<div class="sidebar-header">📊 Last Response</div>', unsafe_allow_html=True)
        lat = st.session_state.last_latency
        st.markdown(
            f'<span class="metric-pill">⏱ Total: {lat["total_ms"]:.0f}ms</span>'
            f'<span class="metric-pill">🔍 Retrieval: {lat["retrieval_ms"]:.0f}ms</span>'
            f'<span class="metric-pill">🤖 LLM: {lat["llm_ms"]:.0f}ms</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.session_state.last_latency = None
        st.rerun()

    st.divider()
    st.markdown(
        "**Arya Shah** — Software Engineer & AI/ML enthusiast\n\n"
        "📧 [aryaforhire@gmail.com](mailto:aryaforhire@gmail.com)\n\n"
        "💼 [LinkedIn](https://www.linkedin.com/in/arya--shah/)\n\n"
        "💻 [GitHub](https://github.com/aryashah2k)",
        unsafe_allow_html=False,
    )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("🤖 AryaGPT")

# Welcome message
if not st.session_state.messages:
    st.markdown(
        """
<div class="welcome-card">
<h3>👋 Welcome! I'm AryaGPT</h3>
<p>I'm an agentic AI assistant specialized in answering questions about <strong>Arya Shah</strong> — 
his background, education, work experience, projects, research, and more.</p>

<p><strong>Try asking me:</strong></p>
<ul>
<li>🎓 What is Arya's educational background?</li>
<li>💼 Tell me about his work experience at ZS Associates</li>
<li>🔬 What research papers has he published?</li>
<li>🏆 Can you create an elevator pitch about Arya for a fintech startup?</li>
<li>💻 What are his latest GitHub projects?</li>
<li>📜 List Arya's patents</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and st.session_state.show_sources:
            badges = "".join(
                f'<span class="source-badge">📄 {src}</span>' for src in msg["sources"]
            )
            st.markdown(f"<div style='margin-top:8px'>{badges}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if user_input := st.chat_input("Ask me about Arya Shah…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = run_agent(
                    user_message=user_input,
                    chat_history=st.session_state.lc_history,
                    provider=_get_active_provider(),
                    model=_get_active_model(),
                    temperature=_get_active_temperature(),
                    max_tokens=_get_active_max_tokens(),
                    conversation_id=st.session_state.session_id,
                )
                answer = result["answer"]
                sources = result.get("sources", [])
                total_ms = result.get("total_latency_ms", 0)
                retrieval_ms = result.get("retrieval_latency_ms", 0)
                llm_ms = result.get("llm_latency_ms", 0)

            except Exception as e:
                answer = (
                    f"I encountered an error while processing your request: `{e}`\n\n"
                    "Please check that the API keys are configured correctly in the admin panel."
                )
                sources = []
                total_ms = retrieval_ms = llm_ms = 0

        st.markdown(answer)

        if sources and st.session_state.show_sources:
            badges = "".join(
                f'<span class="source-badge">📄 {src}</span>' for src in sources
            )
            st.markdown(f"<div style='margin-top:8px'>{badges}</div>", unsafe_allow_html=True)

    # Update state
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.lc_history.append(HumanMessage(content=user_input))
    st.session_state.lc_history.append(AIMessage(content=answer))

    st.session_state.last_latency = {
        "total_ms": total_ms,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
    }

    # Log to SQLite
    try:
        log_conversation(
            session_id=st.session_state.session_id,
            user_msg=user_input,
            bot_msg=answer,
            provider=_get_active_provider(),
            model=_get_active_model(),
            sources=sources,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )
        log_eval(
            session_id=st.session_state.session_id,
            query=user_input,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            provider=_get_active_provider(),
            model=_get_active_model(),
            chunks_returned=len(sources),
            answer_length=len(answer),
        )
    except Exception:
        pass

    st.rerun()
