"""
analyst.py
==========
The AI chatbot layer. It uses Groq's free LLM API (Llama 4 Scout) to answer
questions in plain English - but grounded in OUR real data, not generic football
knowledge. Before each answer we inject a "context" block containing:

  - tournament facts,
  - the current top teams by our Elo rating,
  - our model's honest performance + what drives its predictions,
  - the research hypotheses, and
  - a couple of real numbers computed from our feature table.

This is a lightweight form of RAG (Retrieval-Augmented Generation): we retrieve
our own findings and feed them to the model so its answers are accurate to this
project.

Run directly to test (needs GROQ_API_KEY in .env):
    python -m src.analyst
"""

import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from src.team_form import load_latest_strength
from src.web_search import needs_web, web_search

load_dotenv()

# Llama 4 Scout on Groq - fast and free. Change here if you prefer another model.
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

FEATURES_PATH = "data/processed/features.csv"
SCHEDULE_PATH = "data/processed/schedule_2026.csv"
TODAY = date.today().isoformat()


def _client():
    """Create the Groq client, or None if no key is configured."""
    key = os.environ.get("GROQ_API_KEY")
    if not key or key == "your_groq_key_here":
        return None
    return Groq(api_key=key)


def build_research_context() -> str:
    """Assemble a context block from our real artifacts for the model to use."""
    lines = [
        "FIFA 2026 WORLD CUP - PROJECT FACTS:",
        "- Hosts: USA, Canada, Mexico. 48 teams, 104 matches. Final: MetLife Stadium, July 19 2026.",
        "- Highest-altitude venue: Estadio Azteca, Mexico City (~2,240 m).",
        "",
        "OUR MODEL:",
        "- XGBoost predicts match outcome (home win / draw / away win) from team strength",
        "  (Elo), recent form, travel distance, time-zone shift, altitude, and weather.",
        "- ~51-53% cross-validated accuracy vs ~42% baseline; the strongest single",
        "  predictor is the Elo difference between teams. Draws are hardest to predict.",
        "- Travel distance is a home->venue displacement proxy, not in-tournament travel.",
        "",
        "RESEARCH HYPOTHESES:",
        "- H1: travel distance >5,000 km reduces win probability",
        "- H2: fewer rest days reduce scoring (little effect found at finals level)",
        "- H3: humidity >80% reduces total goals",
        "- H4: altitude >1,500 m disadvantages non-adapted teams",
        "- H5: time-zone shift >5 h reduces defensive performance",
    ]

    # Current top teams by our Elo.
    strength = load_latest_strength()
    if strength:
        top = sorted(strength.items(), key=lambda kv: kv[1]["elo"], reverse=True)[:10]
        lines.append("")
        lines.append("CURRENT TOP 10 TEAMS BY OUR ELO RATING:")
        for i, (team, s) in enumerate(top, 1):
            lines.append(f"  {i}. {team} - Elo {s['elo']:.0f}, recent form {s['form']:.2f}")

    # The 2026 schedule: today's date, the next upcoming fixtures (with our
    # predictions), and the most recent results - so the bot can answer
    # "what's the next match?" and schedule questions.
    if os.path.exists(SCHEDULE_PATH):
        sched = pd.read_csv(SCHEDULE_PATH)
        upcoming = sched[sched["status"] == "Upcoming"].head(8)
        played = sched[sched["status"] == "Played"].tail(6)

        lines.append("")
        lines.append(f"TODAY'S DATE: {TODAY}.")
        lines.append("(All World Cup matches are at NEUTRAL venues. 'home'/'away' is only the "
                     "fixture's listed order, NOT a real home team. The stadium's country is "
                     "irrelevant. Only host nations USA, Canada, Mexico ever play a true home game. "
                     "Always refer to the two teams by name, never as 'the home team'.)")
        if not upcoming.empty:
            lines.append("NEXT UPCOMING MATCHES (with our model's predicted win %):")
            for _, r in upcoming.iterrows():
                lines.append(
                    f"  - {r['date']}: {r['home_team']} vs {r['away_team']} at {r['venue']} -> "
                    f"{r['home_team']} {r['pred_home_%']}%, draw {r['pred_draw_%']}%, "
                    f"{r['away_team']} {r['pred_away_%']}%"
                )
        if not played.empty:
            lines.append("MOST RECENT RESULTS:")
            for _, r in played.iterrows():
                lines.append(f"  - {r['date']}: {r['home_team']} {r['score']} {r['away_team']} ({r['actual']})")

    # A couple of real numbers from the feature table.
    if os.path.exists(FEATURES_PATH):
        df = pd.read_csv(FEATURES_PATH)
        try:
            hi = df[df["humidity"] > 80]["total_goals"].mean()
            lo = df[df["humidity"] <= 80]["total_goals"].mean()
            lines.append("")
            lines.append("REAL FINDINGS FROM OUR DATA:")
            lines.append(f"  - Avg goals in high humidity (>80%): {hi:.2f}")
            lines.append(f"  - Avg goals in normal humidity (<=80%): {lo:.2f}")
            lines.append(f"  - Matches analysed: {len(df)}")
        except Exception:
            pass

    return "\n".join(lines)


def ask_analyst(question: str, chat_history: list = None) -> str:
    """Answer a question using Groq, grounded in our project context."""
    chat_history = chat_history or []
    client = _client()
    if client is None:
        return ("No Groq API key found. Add GROQ_API_KEY to your .env file "
                "(get a free key at https://console.groq.com).")

    system_prompt = (
        "You are a friendly FIFA 2026 assistant. Use the project data below to answer in simple, "
        "plain English - like texting a knowledgeable friend. "
        "Keep answers SHORT: usually 1-3 sentences. Give the direct answer first. "
        "Avoid jargon and do NOT bring up research hypotheses, model accuracy, or technical "
        "details unless the user specifically asks. "
        "World Cup matches are at neutral venues, so never say 'the home team' or guess a home "
        "team from the stadium - just name the two teams (e.g. 'Belgium are slight favourites, "
        "about 57%, over Egypt'). "
        "If LIVE WEB RESULTS are provided, use them for current facts (players, line-ups, "
        "injuries, news) and say the info is 'from the web'. Keep predictions/findings based on "
        "our own data. "
        "If the info isn't in your data or the web results, say so briefly instead of guessing.\n\n"
        + build_research_context()
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)

    # For live-info questions (players, line-ups, injuries, news), fetch the web -
    # the LLM's training is older than 2026 so it can't know these on its own.
    if needs_web(question):
        results = web_search(f"{question} FIFA World Cup 2026")
        if results:
            messages.append({
                "role": "system",
                "content": "LIVE WEB RESULTS (current, from the web - use for player/line-up/"
                           "injury/news questions; attribute as 'from the web'):\n" + results,
            })

    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.5,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Sorry, the AI request failed: {e}"


if __name__ == "__main__":
    print("STEP: Testing the AI analyst (needs GROQ_API_KEY)...\n")
    for q in [
        "Who are the favourites to win and why?",
        "Does weather really affect goals in our data?",
    ]:
        print(f"Q: {q}")
        print(f"A: {ask_analyst(q)}\n{'-'*70}")
