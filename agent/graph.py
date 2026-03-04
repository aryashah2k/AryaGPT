"""LangGraph agentic RAG graph for AryaGPT."""

from __future__ import annotations

import json
import os
import time
from typing import Annotated, Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.prompts import SYSTEM_PROMPT, REFLECTION_PROMPT, ELEVATOR_PITCH_PROMPT
from agent.tools import (
    retrieve_context,
    web_search,
    get_github_activity,
    get_current_date,
)
from rag.retriever import retrieve, format_context

# Only bind 4 tools — Groq tool-calling breaks with too many simultaneous tools
CORE_TOOLS = [retrieve_context, web_search, get_github_activity, get_current_date]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    provider: str
    model: str
    temperature: float
    max_tokens: int
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    sources_used: list[str]
    retry_count: int
    conversation_id: str


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm(provider: str, model: str, temperature: float, max_tokens: int) -> BaseChatModel:
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    elif provider == "together":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def _call_agent(state: AgentState) -> dict:
    """Main agent node: calls the LLM with core tools bound."""
    llm = _build_llm(
        state["provider"],
        state["model"],
        state["temperature"],
        state["max_tokens"],
    )
    llm_with_tools = llm.bind_tools(CORE_TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])

    t0 = time.perf_counter()
    response = llm_with_tools.invoke(messages)
    llm_latency = (time.perf_counter() - t0) * 1000

    return {
        "messages": [response],
        "llm_latency_ms": state.get("llm_latency_ms", 0) + llm_latency,
        "total_latency_ms": state.get("total_latency_ms", 0) + llm_latency,
    }


def _handle_tool_results(state: AgentState) -> dict:
    """Post-process tool results: extract latency from retriever, collect sources."""
    messages = list(state["messages"])
    extra_latency = 0.0
    sources: list[str] = list(state.get("sources_used", []))

    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        content = msg.content or ""
        # Extract sources from retrieve_context tool output
        if "[Retrieved from:" in content:
            src_part = content.split("[Retrieved from:")[-1].strip().rstrip("]")
            for s in src_part.split(","):
                s = s.strip()
                if s and s not in sources:
                    sources.append(s)

    return {
        "sources_used": sources,
        "retrieval_latency_ms": state.get("retrieval_latency_ms", 0) + extra_latency,
        "total_latency_ms": state.get("total_latency_ms", 0) + extra_latency,
    }


def _reflection_node(state: AgentState) -> dict:
    """Reflect on the last AI message for hallucination / grounding check."""
    messages = list(state["messages"])

    last_ai = None
    context_pieces = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai = msg
        if isinstance(msg, ToolMessage):
            context_pieces.append(msg.content or "")

    if last_ai is None or state.get("retry_count", 0) >= 1:
        return {}

    if not context_pieces:
        return {}

    context = "\n\n---\n\n".join(context_pieces[:3])
    prompt = REFLECTION_PROMPT.format(context=context, answer=last_ai.content)

    llm = _build_llm(
        state["provider"],
        state["model"],
        0.0,
        256,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(response.content.strip().strip("```json").strip("```").strip())
        if data.get("needs_retry"):
            return {
                "retry_count": state.get("retry_count", 0) + 1,
                "messages": [
                    HumanMessage(
                        content=f"[RETRY] Your previous answer may have contained unsupported details. "
                                f"Reason: {data.get('reason', '')}. "
                                f"Please re-answer based strictly on the retrieved context."
                    )
                ],
            }
    except Exception:
        pass

    return {}



# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _should_continue(state: AgentState) -> str:
    """Route after agent call: if last message has tool calls → tools, else → reflect."""
    last = list(state["messages"])[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "reflect"


def _after_reflect(state: AgentState) -> str:
    """After reflection: if retry needed → agent, else → end."""
    if state.get("retry_count", 0) > 0:
        msgs = list(state["messages"])
        if isinstance(msgs[-1], HumanMessage) and msgs[-1].content.startswith("[RETRY]"):
            return "agent"
    return END


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    tool_node = ToolNode(CORE_TOOLS)

    builder = StateGraph(AgentState)

    builder.add_node("agent", _call_agent)
    builder.add_node("tools", tool_node)
    builder.add_node("post_tool", _handle_tool_results)
    builder.add_node("reflect", _reflection_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "reflect": "reflect"},
    )
    builder.add_edge("tools", "post_tool")
    builder.add_edge("post_tool", "agent")
    builder.add_conditional_edges("reflect", _after_reflect, {"agent": "agent", END: END})

    return builder.compile()


# Singleton graph instance
_graph = None
_graph_lock = __import__("threading").Lock()


def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Public invoke function
# ---------------------------------------------------------------------------

def run_agent(
    user_message: str,
    chat_history: list,
    provider: str = "groq",
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    conversation_id: str = "",
) -> dict:
    """
    Run the agent for a single user turn.

    Groq and Together AI use direct RAG (reliable, no tool-calling quirks).
    OpenAI uses the full agentic LangGraph graph with tool-calling.
    """
    t_total = time.perf_counter()

    # Direct RAG path for Groq/Together (avoids tool_use_failed errors)
    if provider in ("groq", "together"):
        return _fallback_direct_rag(
            user_message=user_message,
            chat_history=chat_history,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            t_total=t_total,
        )

    # Full agentic graph for OpenAI
    graph = get_graph()

    initial_state: AgentState = {
        "messages": list(chat_history) + [HumanMessage(content=user_message)],
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retrieval_latency_ms": 0.0,
        "llm_latency_ms": 0.0,
        "total_latency_ms": 0.0,
        "sources_used": [],
        "retry_count": 0,
        "conversation_id": conversation_id,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        err_str = str(e)
        if "tool_use_failed" in err_str or "Failed to call a function" in err_str:
            return _fallback_direct_rag(
                user_message=user_message,
                chat_history=chat_history,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                t_total=t_total,
            )
        raise

    total_ms = (time.perf_counter() - t_total) * 1000

    answer = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            answer = msg.content
            break

    return {
        "answer": answer,
        "sources": final_state.get("sources_used", []),
        "retrieval_latency_ms": final_state.get("retrieval_latency_ms", 0),
        "llm_latency_ms": final_state.get("llm_latency_ms", 0),
        "total_latency_ms": total_ms,
    }


def _detect_intent(msg: str) -> dict:
    """Return flags for which live tools to call based on message keywords."""
    m = msg.lower()
    return {
        "github": any(w in m for w in ("github", "repo", "repositories", "open source", "package", "catvision", "metaheurustics", "sculptor", "personaverse")),
        "web": any(w in m for w in ("news", "recent", "latest news", "search", "article")),
        "date": any(w in m for w in ("today", "current date", "what year", "what day", "right now", "currently")),
    }


def _fallback_direct_rag(
    user_message: str,
    chat_history: list,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    t_total: float,
) -> dict:
    """
    Primary RAG path for Groq/Together: retrieves KB context + injects live tool
    data (GitHub, web, date) based on query intent, then calls LLM with no tool binding.
    """
    from rag.retriever import retrieve, format_context
    from agent.tools import get_github_activity, web_search, get_current_date

    intent = _detect_intent(user_message)
    extra_context_parts = []

    # Always retrieve from KB
    t_retr = time.perf_counter()
    result = retrieve(user_message)
    retrieval_ms = (time.perf_counter() - t_retr) * 1000

    chunks = result.get("chunks", [])
    kb_context = format_context(chunks) if chunks else ""
    sources = list({c["source"] for c in chunks})

    if kb_context:
        extra_context_parts.append(f"## Knowledge Base\n{kb_context}")

    # Inject live GitHub data if relevant
    if intent["github"]:
        try:
            github_data = get_github_activity.invoke({"detail": "repos"})
            extra_context_parts.append(f"## Live GitHub Data (fetched now)\n{github_data}")
            sources.append("github.com/aryashah2k")
        except Exception:
            pass

    # Inject web search if relevant
    if intent["web"]:
        try:
            web_data = web_search.invoke({"query": user_message})
            extra_context_parts.append(f"## Live Web Search Results\n{web_data}")
        except Exception:
            pass

    # Inject current date if relevant
    if intent["date"]:
        try:
            date_data = get_current_date.invoke({"dummy": ""})
            extra_context_parts.append(f"## Current Date\n{date_data}")
        except Exception:
            pass

    context_block = "\n\n---\n\n".join(extra_context_parts) if extra_context_parts else "No relevant context found."

    augmented_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{context_block}\n\n"
        f"## User Question\n{user_message}\n\n"
        f"Answer using the context above. End with 2-3 follow-up questions under **You might also ask:**"
    )

    llm = _build_llm(provider, model, temperature, max_tokens)

    t_llm = time.perf_counter()
    response = llm.invoke(
        list(chat_history) + [HumanMessage(content=augmented_prompt)]
    )
    llm_ms = (time.perf_counter() - t_llm) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000

    return {
        "answer": response.content,
        "sources": sources,
        "retrieval_latency_ms": retrieval_ms,
        "llm_latency_ms": llm_ms,
        "total_latency_ms": total_ms,
    }
