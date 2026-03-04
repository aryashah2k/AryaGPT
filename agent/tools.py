"""LangGraph tool implementations for AryaGPT."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import requests
from ddgs import DDGS
from langchain_core.tools import tool

from rag.retriever import retrieve, format_context

GITHUB_USERNAME = "aryashah2k"
WEB_SEARCH_SUFFIX = "Arya Shah"


# ---------------------------------------------------------------------------
# Tool 1: Retrieve from knowledge base
# ---------------------------------------------------------------------------

@tool
def retrieve_context(query: str) -> str:
    """Search Arya Shah's knowledge base for relevant information.
    Use this for any question about Arya's background, skills, experience, education, projects, or personal details.
    """
    result = retrieve(query)
    chunks = result.get("chunks", [])
    if not chunks:
        return "No relevant information found in the knowledge base."
    context = format_context(chunks)
    sources = list({c["source"] for c in chunks})
    return f"{context}\n\n[Retrieved from: {', '.join(sources)}]"


# ---------------------------------------------------------------------------
# Tool 2: Web search (Arya-scoped)
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for live or recent information about Arya Shah.
    The search is automatically scoped to Arya Shah. Use this when the knowledge base
    doesn't have up-to-date information (e.g., recent news, latest projects).
    """
    scoped_query = f"{query} {WEB_SEARCH_SUFFIX}"
    try:
        ddgs = DDGS()
        results = list(ddgs.text(scoped_query, max_results=5))
        if not results:
            return "No web results found."
        formatted = []
        for r in results:
            formatted.append(f"**{r.get('title', '')}**\n{r.get('body', '')}\nURL: {r.get('href', '')}")
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {e}"


# ---------------------------------------------------------------------------
# Tool 3: GitHub activity
# ---------------------------------------------------------------------------

@tool
def get_github_activity(detail: str = "repos") -> str:
    """Fetch Arya Shah's public GitHub activity: repos, pinned projects, contribution stats.
    Use this when asked about Arya's GitHub, open source contributions, or latest projects.
    detail: 'repos' for repository list, 'profile' for profile overview.
    """
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "AryaGPT"}
    try:
        profile_resp = requests.get(
            f"https://api.github.com/users/{GITHUB_USERNAME}",
            headers=headers,
            timeout=8,
        )
        profile = profile_resp.json() if profile_resp.status_code == 200 else {}

        # Fetch up to 100 repos sorted by most recently pushed, exclude forks
        repos_resp = requests.get(
            f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
            f"?sort=pushed&direction=desc&per_page=100&type=owner",
            headers=headers,
            timeout=10,
        )
        all_repos = repos_resp.json() if repos_resp.status_code == 200 else []

        if isinstance(all_repos, list):
            # Filter out forks, sort by stars then pushed
            own_repos = [r for r in all_repos if not r.get("fork", False)]
            # Show top 15 by stars, then fill with most recently pushed
            by_stars = sorted(own_repos, key=lambda r: r.get("stargazers_count", 0), reverse=True)[:5]
            by_pushed = [r for r in own_repos if r not in by_stars][:10]
            display_repos = by_stars + by_pushed

            repo_lines = []
            for r in display_repos:
                stars = r.get("stargazers_count", 0)
                lang = r.get("language") or "N/A"
                desc = (r.get("description") or "").strip()
                pushed = (r.get("pushed_at") or "")[:10]
                repo_lines.append(
                    f"- **{r['name']}** ({lang}, ⭐{stars}, last pushed {pushed}): {desc}\n"
                    f"  URL: https://github.com/{GITHUB_USERNAME}/{r['name']}"
                )
            repos_text = "\n".join(repo_lines)
            total_shown = len(display_repos)
        else:
            repos_text = "Could not fetch repositories."
            total_shown = 0

        # Also check PyPI for published packages
        pypi_text = ""
        try:
            pypi_resp = requests.get(
                f"https://pypi.org/pypi/{GITHUB_USERNAME}/json",
                timeout=5,
            )
            if pypi_resp.status_code != 200:
                # Try a search-style lookup for known packages
                pypi_text = "(Check https://pypi.org/user/aryashah2k for published Python packages)"
        except Exception:
            pass

        return (
            f"**GitHub Profile: {GITHUB_USERNAME}**\n"
            f"Public repos: {profile.get('public_repos', '?')} | "
            f"Followers: {profile.get('followers', '?')} | "
            f"Following: {profile.get('following', '?')}\n\n"
            f"**Repositories (top by stars + {total_shown} most recent, own only — forks excluded):**\n"
            f"{repos_text}\n\n"
            f"{pypi_text}\n"
            f"Full profile: https://github.com/{GITHUB_USERNAME}"
        )
    except Exception as e:
        return f"GitHub API request failed: {e}"


# ---------------------------------------------------------------------------
# Tool 4: Current date
# ---------------------------------------------------------------------------

@tool
def get_current_date(dummy: str = "") -> str:
    """Get today's date and time in UTC. Use this to answer temporal questions like
    'what is Arya doing now' or 'how long has he been working at X'.
    """
    now = datetime.now(timezone.utc)
    return f"Current date and time (UTC): {now.strftime('%A, %B %d, %Y at %H:%M UTC')}"


# ---------------------------------------------------------------------------
# Tool 5: Summarize a document or URL
# ---------------------------------------------------------------------------

@tool
def summarize_document(url_or_text: str) -> str:
    """Fetch and summarize a URL or a block of text provided by the user.
    Use this when the user pastes a link or document they want summarized in the context of Arya.
    """
    text = url_or_text.strip()
    if text.startswith("http://") or text.startswith("https://"):
        try:
            resp = requests.get(text, timeout=10, headers={"User-Agent": "AryaGPT"})
            resp.raise_for_status()
            from html.parser import HTMLParser

            class _TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.parts = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "footer"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "footer"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        stripped = data.strip()
                        if stripped:
                            self.parts.append(stripped)

            parser = _TextExtractor()
            parser.feed(resp.text)
            text = " ".join(parser.parts)[:4000]
        except Exception as e:
            return f"Could not fetch URL: {e}"
    else:
        text = text[:4000]

    return f"[Document content for summarization — {len(text)} chars]\n\n{text}"


# ---------------------------------------------------------------------------
# Tool 6: Query conversation history
# ---------------------------------------------------------------------------

@tool
def query_conversation_history(query: str) -> str:
    """Search the current conversation history for previously discussed topics.
    Use this when the user asks about something mentioned earlier in the chat,
    like 'what did you say about his patents earlier?'.
    Note: Returns a prompt to use the in-memory history — the graph handles actual lookup.
    """
    return f"[HISTORY_QUERY:{query}]"


# ---------------------------------------------------------------------------
# Tool 7: Generate elevator pitch
# ---------------------------------------------------------------------------

@tool
def generate_elevator_pitch(target: str) -> str:
    """Generate a tailored elevator pitch about Arya Shah for a specific company or role.
    target should describe the company, role, or context, e.g. 'a fintech startup as a senior ML engineer'.
    This tool retrieves relevant context and returns it for pitch generation.
    """
    result = retrieve(f"Arya Shah skills experience projects {target}")
    chunks = result.get("chunks", [])
    context = format_context(chunks) if chunks else "No specific context found."
    return json.dumps({"target": target, "context": context, "action": "GENERATE_PITCH"})


# ---------------------------------------------------------------------------
# Tool 8: Suggest follow-up questions
# ---------------------------------------------------------------------------

@tool
def suggest_questions(topic: str) -> str:
    """Generate 3 relevant follow-up questions about Arya Shah based on the current topic.
    Use this at the end of a response to proactively engage the visitor.
    topic should be a brief description of what was just discussed.
    """
    result = retrieve(topic)
    chunks = result.get("chunks", [])
    context_preview = format_context(chunks[:2]) if chunks else ""
    return json.dumps({"topic": topic, "context_preview": context_preview, "action": "SUGGEST_QUESTIONS"})


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    retrieve_context,
    web_search,
    get_github_activity,
    get_current_date,
    summarize_document,
    query_conversation_history,
    generate_elevator_pitch,
    suggest_questions,
]
