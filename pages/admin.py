"""AryaGPT Admin Panel — password-gated, provider control, KB management, eval dashboard."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
from streamlit_server_state import server_state, server_state_lock

from admin.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_MODELS,
)
from db.logger import get_eval_metrics, get_recent_conversations, init_db

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AryaGPT Admin",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Propagate secrets → environment variables
_SECRET_KEYS = ["GROQ_API_KEY", "TOGETHER_API_KEY", "OPENAI_API_KEY"]
for _k in _SECRET_KEYS:
    if _k not in os.environ:
        try:
            os.environ[_k] = st.secrets[_k]
        except Exception:
            pass

init_db()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.admin-header {
    background: linear-gradient(135deg, #1e1e2e 0%, #181825 100%);
    border: 1px solid #45475a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #cba6f7;
}
.metric-label {
    font-size: 0.8rem;
    color: #6c7086;
    margin-top: 4px;
}
.provider-active {
    border: 2px solid #a6e3a1 !important;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #cba6f7;
    border-bottom: 1px solid #313244;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Authentication setup
# ---------------------------------------------------------------------------

def _get_admin_credentials() -> tuple:
    """Return (username, plain_password, display_name) from secrets or env."""
    try:
        return (
            st.secrets.get("ADMIN_USERNAME", "arya"),
            st.secrets["ADMIN_PASSWORD"],
            st.secrets.get("ADMIN_NAME", "Arya Shah"),
        )
    except Exception:
        return (
            os.environ.get("ADMIN_USERNAME", "arya"),
            os.environ.get("ADMIN_PASSWORD", "changeme"),
            os.environ.get("ADMIN_NAME", "Arya Shah"),
        )


admin_username, admin_password, admin_name = _get_admin_credentials()

# streamlit-authenticator requires a bcrypt hash in credentials, not plain text
import bcrypt as _bcrypt
_hashed_password = _bcrypt.hashpw(admin_password.encode(), _bcrypt.gensalt()).decode()

credentials = {
    "usernames": {
        admin_username: {
            "name": admin_name,
            "password": _hashed_password,
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "aryagpt_admin_auth",
    "aryagpt_admin_cookie_key_v2",
    cookie_expiry_days=7,
)

authenticator.login(key="aryagpt_admin_login", location="main")

# ---------------------------------------------------------------------------
# Not authenticated
# ---------------------------------------------------------------------------

authentication_status = st.session_state.get("authentication_status", None)
name = st.session_state.get("name", "")

if authentication_status is False:
    st.error("❌ Incorrect username or password.")
    st.stop()

if authentication_status is None:
    st.info("Please enter your admin credentials above.")
    st.stop()

# ---------------------------------------------------------------------------
# AUTHENTICATED — render admin panel
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="admin-header"><h2>🛡️ AryaGPT Admin Panel</h2>'
    f'<p>Welcome, <strong>{name}</strong> — you have full admin access.</p></div>',
    unsafe_allow_html=True,
)

authenticator.logout(location="sidebar")

# ---------------------------------------------------------------------------
# Helper: get/set server_state safely
# ---------------------------------------------------------------------------

def _ss_get(key: str, default):
    try:
        return server_state.get(key, default)
    except Exception:
        return default


def _ss_set(key: str, value) -> None:
    try:
        with server_state_lock[key]:
            setattr(server_state, key, value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_provider, tab_kb, tab_logs, tab_evals = st.tabs([
    "⚙️ Provider & Model",
    "📚 Knowledge Base",
    "💬 Conversation Logs",
    "📊 Eval Dashboard",
])

# ===========================================================================
# TAB 1: Provider & Model
# ===========================================================================

with tab_provider:
    st.markdown('<div class="section-title">Active LLM Provider</div>', unsafe_allow_html=True)

    current_provider = _ss_get("active_provider", DEFAULT_PROVIDER)
    current_model = _ss_get("active_model", DEFAULT_MODEL)
    current_temp = float(_ss_get("active_temperature", DEFAULT_TEMPERATURE))
    current_max_tokens = int(_ss_get("active_max_tokens", DEFAULT_MAX_TOKENS))

    st.info(
        f"🟢 **Currently active:** `{PROVIDER_DISPLAY_NAMES.get(current_provider, current_provider)}` "
        f"/ `{current_model}` — "
        f"temp={current_temp}, max_tokens={current_max_tokens}\n\n"
        f"Changes apply **immediately** to all active visitor sessions."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Select Provider**")
        provider_options = list(PROVIDER_MODELS.keys())
        provider_labels = [PROVIDER_DISPLAY_NAMES[p] for p in provider_options]
        selected_provider_label = st.radio(
            "Provider",
            provider_labels,
            index=provider_options.index(current_provider) if current_provider in provider_options else 0,
            label_visibility="collapsed",
        )
        selected_provider = provider_options[provider_labels.index(selected_provider_label)]

    with col2:
        st.markdown("**Select Model**")
        model_options = PROVIDER_MODELS[selected_provider]
        default_model_idx = (
            model_options.index(current_model)
            if current_model in model_options
            else 0
        )
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=default_model_idx,
            label_visibility="collapsed",
        )

        st.markdown("**Temperature**")
        selected_temp = st.slider("Temperature", 0.0, 1.0, current_temp, 0.05, label_visibility="collapsed")

        st.markdown("**Max Output Tokens**")
        selected_max_tokens = st.slider("Max tokens", 256, 4096, current_max_tokens, 128, label_visibility="collapsed")

    st.divider()

    # API key status check
    st.markdown('<div class="section-title">API Key Status</div>', unsafe_allow_html=True)
    key_cols = st.columns(3)
    for i, (prov, env_key) in enumerate([
        ("groq", "GROQ_API_KEY"),
        ("together", "TOGETHER_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ]):
        key_val = ""
        try:
            key_val = st.secrets.get(env_key, "") or os.environ.get(env_key, "")
        except Exception:
            key_val = os.environ.get(env_key, "")

        with key_cols[i]:
            if key_val:
                st.success(f"✅ {PROVIDER_DISPLAY_NAMES[prov]}\n\n`{env_key[:8]}...`")
            else:
                st.error(f"❌ {PROVIDER_DISPLAY_NAMES[prov]}\n\n`{env_key}` not set")

    st.divider()

    if st.button("💾 Apply Settings", type="primary", use_container_width=True):
        _ss_set("active_provider", selected_provider)
        _ss_set("active_model", selected_model)
        _ss_set("active_temperature", selected_temp)
        _ss_set("active_max_tokens", selected_max_tokens)
        # Also set environment variable for the current process
        env_key = {"groq": "GROQ_API_KEY", "together": "TOGETHER_API_KEY", "openai": "OPENAI_API_KEY"}.get(selected_provider, "")
        st.success(
            f"✅ Settings applied! All visitors will now use **{PROVIDER_DISPLAY_NAMES[selected_provider]}** / `{selected_model}`."
        )
        st.rerun()

# ===========================================================================
# TAB 2: Knowledge Base
# ===========================================================================

with tab_kb:
    st.markdown('<div class="section-title">Knowledge Base Status</div>', unsafe_allow_html=True)

    meta_path = Path("chroma_db") / "ingest_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            kb_meta = json.load(f)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Chunks", kb_meta.get("total_chunks", "?"))
        with m2:
            st.metric("Last Added", kb_meta.get("last_added", "?"))
        with m3:
            st.metric("Last Skipped", kb_meta.get("last_skipped", "?"))

        st.caption(f"⏰ Last ingest: {kb_meta.get('last_ingest_human', 'unknown')}")
    else:
        st.warning("⚠️ No ingest metadata found. Run the ingest pipeline first.")

    st.divider()

    # Data files list
    st.markdown('<div class="section-title">Source Files in data/</div>', unsafe_allow_html=True)
    data_path = Path("data")
    if data_path.exists():
        files = sorted(data_path.rglob("*"))
        file_data = []
        for f in files:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                file_data.append({"File": f.name, "Type": f.suffix, "Size (KB)": f"{size_kb:.1f}"})
        if file_data:
            import pandas as pd
            st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)
        else:
            st.info("No data files found.")
    else:
        st.warning("data/ directory not found.")

    st.divider()

    # Manual re-ingest
    st.markdown('<div class="section-title">Manual Re-Ingest</div>', unsafe_allow_html=True)
    st.caption(
        "Trigger a full re-ingest of all files in `data/`. "
        "Only new/changed files will be added (deduplication via content hash)."
    )

    if st.button("🔄 Run Re-Ingest Now", type="primary", use_container_width=True):
        with st.spinner("Running ingest pipeline…"):
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/ingest.py", "--data-dir", "data/", "--chroma-dir", "chroma_db/"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    st.success("✅ Re-ingest completed successfully!")
                    st.code(result.stdout, language="text")
                else:
                    st.error("❌ Ingest failed.")
                    st.code(result.stderr, language="text")
            except subprocess.TimeoutExpired:
                st.error("❌ Ingest timed out after 5 minutes.")
            except Exception as e:
                st.error(f"❌ Error running ingest: {e}")
        st.rerun()

# ===========================================================================
# TAB 3: Conversation Logs
# ===========================================================================

with tab_logs:
    st.markdown('<div class="section-title">Recent Conversations</div>', unsafe_allow_html=True)

    n_logs = st.slider("Show last N conversations", 10, 200, 50, 10)

    try:
        conversations = get_recent_conversations(limit=n_logs)
    except Exception as e:
        conversations = []
        st.error(f"Could not load conversations: {e}")

    if not conversations:
        st.info("No conversations logged yet.")
    else:
        import pandas as pd

        df = pd.DataFrame(conversations)
        display_cols = ["ts_human", "session_id", "provider", "model", "user_msg", "bot_msg", "total_ms"]
        available_cols = [c for c in display_cols if c in df.columns]
        df_display = df[available_cols].copy()

        if "session_id" in df_display.columns:
            df_display["session_id"] = df_display["session_id"].str[:8] + "…"
        if "bot_msg" in df_display.columns:
            df_display["bot_msg"] = df_display["bot_msg"].str[:120] + "…"
        if "total_ms" in df_display.columns:
            df_display["total_ms"] = df_display["total_ms"].apply(lambda x: f"{x:.0f}ms")

        df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]

        search_term = st.text_input("🔍 Filter conversations", placeholder="Search user messages…")
        if search_term:
            mask = df_display.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
            df_display = df_display[mask]

        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df_display)} of {len(conversations)} records")

    st.divider()

    # Active sessions
    st.markdown('<div class="section-title">Active Sessions</div>', unsafe_allow_html=True)
    try:
        active = _ss_get("active_sessions", set())
        st.metric("Live Visitor Sessions", len(active))
        if active:
            for sid in list(active)[:10]:
                st.caption(f"🟢 `{str(sid)[:16]}…`")
    except Exception:
        st.info("Could not retrieve active session count.")

# ===========================================================================
# TAB 4: Eval Dashboard
# ===========================================================================

with tab_evals:
    st.markdown('<div class="section-title">RAG Evaluation Dashboard</div>', unsafe_allow_html=True)

    try:
        metrics = get_eval_metrics()
    except Exception as e:
        metrics = {}
        st.error(f"Could not load eval metrics: {e}")

    if not metrics or metrics.get("total_queries", 0) == 0:
        st.info("No eval data yet. Metrics will appear after the first conversations.")
    else:
        # Top-level metrics
        st.markdown("#### Latency Percentiles")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.metric("P50 Total", f"{metrics['p50_total_ms']:.0f} ms")
        with c2:
            st.metric("P95 Total", f"{metrics['p95_total_ms']:.0f} ms")
        with c3:
            st.metric("P50 Retrieval", f"{metrics['p50_retrieval_ms']:.0f} ms")
        with c4:
            st.metric("P95 Retrieval", f"{metrics['p95_retrieval_ms']:.0f} ms")
        with c5:
            st.metric("P50 LLM", f"{metrics['p50_llm_ms']:.0f} ms")
        with c6:
            st.metric("P95 LLM", f"{metrics['p95_llm_ms']:.0f} ms")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Volume & Averages")
            st.metric("Total Queries", metrics["total_queries"])
            st.metric("Avg Total Latency", f"{metrics['avg_total_ms']:.0f} ms")

        with col_b:
            st.markdown("#### Provider Breakdown")
            import pandas as pd
            breakdown = metrics.get("provider_breakdown", {})
            if breakdown:
                bd_rows = []
                for prov, stats in breakdown.items():
                    bd_rows.append({
                        "Provider": PROVIDER_DISPLAY_NAMES.get(prov, prov),
                        "Queries": stats["count"],
                        "Avg Latency (ms)": f"{stats['avg_total_ms']:.0f}",
                    })
                st.dataframe(pd.DataFrame(bd_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No provider data yet.")

        st.divider()

        # Latency over time chart
        st.markdown("#### Latency Over Time")
        try:
            from db.logger import _get_conn, DB_PATH
            import pandas as pd

            conn = _get_conn()
            df_evals = pd.read_sql_query(
                "SELECT ts, total_ms, retrieval_ms, llm_ms, provider FROM evals ORDER BY ts",
                conn,
            )
            conn.close()

            if not df_evals.empty:
                import datetime as dt
                df_evals["datetime"] = pd.to_datetime(df_evals["ts"], unit="s")
                df_evals = df_evals.set_index("datetime")

                chart_cols = st.multiselect(
                    "Metrics to plot",
                    ["total_ms", "retrieval_ms", "llm_ms"],
                    default=["total_ms"],
                )
                if chart_cols:
                    st.line_chart(df_evals[chart_cols], use_container_width=True)
            else:
                st.info("Not enough data to plot yet.")
        except Exception as e:
            st.warning(f"Could not render latency chart: {e}")

        st.divider()

        # Per-query detail table
        st.markdown("#### Per-Query Detail")
        try:
            from db.logger import _get_conn
            import pandas as pd

            n_evals = st.slider("Show last N eval records", 10, 500, 100, 10, key="eval_n")
            conn = _get_conn()
            df_q = pd.read_sql_query(
                f"SELECT ts, query, provider, model, total_ms, retrieval_ms, llm_ms, chunks_returned, answer_length "
                f"FROM evals ORDER BY ts DESC LIMIT {n_evals}",
                conn,
            )
            conn.close()

            if not df_q.empty:
                import datetime as dt
                df_q["ts"] = pd.to_datetime(df_q["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
                df_q["total_ms"] = df_q["total_ms"].apply(lambda x: f"{x:.0f}")
                df_q["retrieval_ms"] = df_q["retrieval_ms"].apply(lambda x: f"{x:.0f}")
                df_q["llm_ms"] = df_q["llm_ms"].apply(lambda x: f"{x:.0f}")
                df_q["query"] = df_q["query"].str[:80] + "…"
                st.dataframe(df_q, use_container_width=True, hide_index=True)
            else:
                st.info("No eval records yet.")
        except Exception as e:
            st.warning(f"Could not load eval records: {e}")
