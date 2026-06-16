"""
web_search.py
=============
A lightweight, free web-lookup tool (DuckDuckGo, no API key) used by the chatbot
ONLY for live facts the language model can't know - things newer than its
training cutoff, like 2026 squads, line-ups, injuries, or current news.

Predictions and research findings still come from OUR model/data; this only fills
the "current events" gap. Results are returned as plain text and clearly labeled
as web-sourced so the assistant can attribute them honestly.
"""

# Keywords that suggest a question needs live/current info from the web rather
# than our model. (Players, line-ups, injuries, news, coaches, etc.)
WEB_TRIGGERS = [
    "player", "players", "playing", "plays", "play for", "lineup", "line-up",
    "line up", "squad", "roster", "named", "selected", "called up", "call-up",
    "injury", "injured", "injuries", "suspended", "coach", "manager", "captain",
    "transfer", "news", "scorer", "goalkeeper", "striker", "midfielder",
    "defender", "formation", "starting", "starting xi", "bench", "who is in",
    "referee", "who scored", "top scorer", "ticket", "capacity", "latest",
]


def needs_web(question: str) -> bool:
    """True if the question looks like it needs current/live info."""
    q = question.lower()
    return any(trigger in q for trigger in WEB_TRIGGERS)


def web_search(query: str, max_results: int = 5) -> str:
    """Return top web results as plain text, or '' if search is unavailable."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
    except Exception:
        return ""

    if not results:
        return ""

    lines = []
    for r in results:
        title = r.get("title") or ""
        body = r.get("body") or r.get("content") or r.get("snippet") or ""
        url = r.get("href") or r.get("url") or r.get("link") or ""
        lines.append(f"- {title}: {body} ({url})")
    return "\n".join(lines)


if __name__ == "__main__":
    print("needs_web('who are the players?') ->", needs_web("who are the players?"))
    print("needs_web('who will win?')       ->", needs_web("who will win?"))
    print("\nSample search:")
    print(web_search("Belgium World Cup 2026 squad players", max_results=3))
