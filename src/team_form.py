"""
team_form.py
============
Adds "performance history" features to every match - i.e. HOW GOOD each team was
at the moment they played, based purely on their past results. This is the
data-driven stand-in for "player quality": a team is exactly as strong as the
results its players produce.

Two kinds of strength signal, both computed LEAK-FREE (using only matches that
happened BEFORE each game, never the game itself or the future):

  1. Elo rating  - the chess-style rating used (in spirit) by FIFA's own world
                   ranking. Every team starts at 1500; you gain points for
                   beating strong teams, lose points for losing to weak ones.

  2. Recent form - over each team's last N matches: win rate, average goals
                   scored, average goals conceded.

We replay the ENTIRE history (every international since 1872) in date order. For
each match we first RECORD the pre-match ratings, THEN update them with the
result. Because the record happens before the update, no future information can
leak into the features.

Run directly to preview the current top teams:
    python -m src.team_form
"""

import json
import os
from collections import defaultdict, deque

import pandas as pd

from src.data_collector import load_full_history

# Elo tuning constants (standard, well-known values for international football).
START_ELO = 1500.0
K_FACTOR = 30.0          # how fast ratings move after each result
HOME_ADVANTAGE = 65.0    # Elo points added to the home side's strength
FORM_WINDOW = 10         # how many recent matches count as "current form"

STRENGTH_PATH = "data/processed/latest_strength.json"

# New columns this module adds to the match table.
HISTORY_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form", "away_form",
    "home_gf", "home_ga", "away_gf", "away_ga",
    "home_rest_days", "away_rest_days", "rest_diff_days",
]

# Rest days are capped: beyond this a team is simply "fully rested", and very
# long international gaps (months) would otherwise dwarf the fatigue signal.
REST_CAP_DAYS = 30


def _expected(rating_a: float, rating_b: float) -> float:
    """Elo expected score for A vs B (0..1)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def add_history_features(full_df: pd.DataFrame):
    """
    Return (df_with_history_columns, latest_strength_dict).

    df_with_history_columns is the input rows plus the HISTORY_COLS, each holding
    the teams' strength/form AS OF just before that match (leak-free).
    latest_strength_dict holds every team's most recent rating + form, for use
    when predicting brand-new (2026) matches.
    """
    elo = defaultdict(lambda: START_ELO)
    recent = defaultdict(lambda: deque(maxlen=FORM_WINDOW))   # each item: (points, gf, ga)
    last_played = {}                                          # team -> date of previous match

    records = []
    for row in full_df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        hs, as_ = row.home_score, row.away_score
        date = row.date

        # --- 1. RECORD pre-match strength/form/rest (no future info) ---
        home_elo, away_elo = elo[h], elo[a]
        home_rest = _rest_days(last_played.get(h), date)
        away_rest = _rest_days(last_played.get(a), date)
        records.append({
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "elo_diff": round(home_elo - away_elo, 1),
            "home_form": _winrate(recent[h]),
            "away_form": _winrate(recent[a]),
            "home_gf": _avg(recent[h], idx=1),
            "home_ga": _avg(recent[h], idx=2),
            "away_gf": _avg(recent[a], idx=1),
            "away_ga": _avg(recent[a], idx=2),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_diff_days": home_rest - away_rest,
        })
        last_played[h] = date
        last_played[a] = date

        # --- 2. UPDATE ratings using the actual result ---
        if hs > as_:
            home_points, away_points = 1.0, 0.0
        elif hs < as_:
            home_points, away_points = 0.0, 1.0
        else:
            home_points, away_points = 0.5, 0.5

        exp_home = _expected(home_elo + HOME_ADVANTAGE, away_elo)
        change = K_FACTOR * (home_points - exp_home)
        elo[h] = home_elo + change
        elo[a] = away_elo - change

        recent[h].append((home_points, hs, as_))
        recent[a].append((away_points, as_, hs))

    history_df = pd.DataFrame(records, index=full_df.index)
    out = pd.concat([full_df, history_df], axis=1)

    # Snapshot every team's latest strength for future-match prediction.
    latest = {
        team: {
            "elo": round(elo[team], 1),
            "form": _winrate(recent[team]),
            "gf": _avg(recent[team], idx=1),
            "ga": _avg(recent[team], idx=2),
        }
        for team in elo
    }
    return out, latest


def _rest_days(prev_date, this_date) -> int:
    """Days since a team's previous match, capped (no prior match -> fully rested)."""
    if prev_date is None:
        return REST_CAP_DAYS
    return int(min((this_date - prev_date).days, REST_CAP_DAYS))


def _winrate(dq: deque) -> float:
    """Average match points (1=win, .5=draw, 0=loss) over the recent window."""
    if not dq:
        return 0.5
    return round(sum(item[0] for item in dq) / len(dq), 3)


def _avg(dq: deque, idx: int) -> float:
    """Average of goals-for (idx=1) or goals-against (idx=2) over the window."""
    if not dq:
        return 1.2          # neutral prior: ~average international scoreline
    return round(sum(item[idx] for item in dq) / len(dq), 2)


def save_latest_strength(latest: dict, dest: str = STRENGTH_PATH) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(latest, f, indent=2)


def load_latest_strength(path: str = STRENGTH_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    print("STEP: Building team strength (Elo) + form from full match history...")
    full = load_full_history()
    print(f"   Replaying {len(full):,} international matches in date order...")
    _, latest = add_history_features(full)
    save_latest_strength(latest)

    ranking = sorted(latest.items(), key=lambda kv: kv[1]["elo"], reverse=True)
    print("\n   Current top 15 teams by Elo (data-driven strength):")
    for i, (team, s) in enumerate(ranking[:15], 1):
        print(f"   {i:>2}. {team:<16} Elo {s['elo']:>7.1f}   form {s['form']:.2f}")
