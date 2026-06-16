"""
simulator.py
============
A Monte-Carlo simulation of the whole FIFA 2026 tournament: it plays the entire
event thousands of times and counts how often each team wins their group,
qualifies, reaches the final, and lifts the trophy.

How it works (and the honest simplifications):
  - GROUPS are reconstructed from the real 72 group-stage fixtures (teams that
    play each other are in the same group). Matches already PLAYED use their real
    result; unplayed matches are simulated.
  - MATCH probabilities come from Elo (the dominant signal in our model):
        P(A beats B) = 1 / (1 + 10**((EloB - EloA) / 400))
    For group games we add a draw probability that shrinks as the Elo gap grows.
  - QUALIFICATION follows the 2026 format: top 2 of each group + the 8 best
    third-placed teams advance (32 teams).
  - KNOCKOUT is a single-elimination bracket with a RANDOM draw each simulation.
    This is a simplification of FIFA's exact positional bracket (which depends on
    which specific third-placed teams advance), but over thousands of runs it
    gives strength-based odds for each stage.

Run directly:
    python -m src.simulator
"""

import math
import random
from collections import defaultdict

import pandas as pd

from src.schedule import load_2026_schedule
from src.team_form import load_latest_strength, START_ELO

random.seed(42)                      # reproducible odds across runs

DRAW_BASE = 0.28                     # typical international draw rate (neutral)


def reconstruct_groups(sched: pd.DataFrame):
    """Infer the 12 groups of 4 from who plays whom in the group fixtures."""
    opponents = defaultdict(set)
    for _, m in sched.iterrows():
        opponents[m["home_team"]].add(m["away_team"])
        opponents[m["away_team"]].add(m["home_team"])

    groups = []
    seen = set()
    for team, opps in opponents.items():
        members = frozenset({team} | opps)
        if members not in seen and len(members) == 4:
            seen.add(members)
            groups.append(sorted(members))
    return groups


def compute_group_standings():
    """Build live group tables + fixtures from real results (updates as games are played)."""
    sched = load_2026_schedule()
    groups = sorted(reconstruct_groups(sched), key=lambda g: sorted(g))

    out = []
    for idx, members in enumerate(groups, 1):
        mset = set(members)
        stats = {t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0} for t in members}
        fixtures = []
        for _, m in sched.iterrows():
            if m["home_team"] in mset and m["away_team"] in mset:
                h, a = m["home_team"], m["away_team"]
                if m["status"] == "Played":
                    hs, as_ = int(m["home_score"]), int(m["away_score"])
                    for tm, gf, ga in [(h, hs, as_), (a, as_, hs)]:
                        stats[tm]["P"] += 1
                        stats[tm]["GF"] += gf
                        stats[tm]["GA"] += ga
                    if hs > as_:
                        stats[h]["W"] += 1; stats[h]["Pts"] += 3; stats[a]["L"] += 1
                    elif as_ > hs:
                        stats[a]["W"] += 1; stats[a]["Pts"] += 3; stats[h]["L"] += 1
                    else:
                        stats[h]["D"] += 1; stats[a]["D"] += 1; stats[h]["Pts"] += 1; stats[a]["Pts"] += 1
                    fixtures.append({"date": m["date"].strftime("%Y-%m-%d"),
                                     "match": f"{h} {hs}-{as_} {a}", "status": "Played"})
                else:
                    fixtures.append({"date": m["date"].strftime("%Y-%m-%d"),
                                     "match": f"{h} vs {a}", "status": "Upcoming"})

        rows = []
        for t in members:
            s = stats[t]
            rows.append({"Team": t, "P": s["P"], "W": s["W"], "D": s["D"], "L": s["L"],
                         "GF": s["GF"], "GA": s["GA"], "GD": s["GF"] - s["GA"], "Pts": s["Pts"]})
        table = (pd.DataFrame(rows)
                 .sort_values(["Pts", "GD", "GF"], ascending=False)
                 .reset_index(drop=True))
        table.index = table.index + 1                     # 1-based position
        fixtures_df = pd.DataFrame(fixtures).sort_values("date").reset_index(drop=True)
        out.append((f"Group {idx}", table, fixtures_df))
    return out


def _match_probs(elo_a: float, elo_b: float):
    """Return (P(A win), P(draw), P(B win)) for a neutral-venue group match."""
    e = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    p_draw = DRAW_BASE * math.exp(-abs(elo_a - elo_b) / 200.0)
    return (1 - p_draw) * e, p_draw, (1 - p_draw) * (1 - e)


def _knockout_winner(a, b, elo):
    """Pick a single-match winner (no draws) from Elo."""
    p_a = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
    return a if random.random() < p_a else b


def _simulate_group(members, fixtures, elo):
    """Play a group's 6 matches; return teams ranked 1st..4th with points."""
    points = {t: 0 for t in members}
    for f in fixtures:
        a, b = f["home_team"], f["away_team"]
        if f["played"]:                       # use the real result
            hs, as_ = f["hs"], f["as_"]
            res = "A" if hs > as_ else "B" if as_ > hs else "D"
        else:                                  # simulate it
            pa, pd_, pb = _match_probs(elo[a], elo[b])
            r = random.random()
            res = "A" if r < pa else "D" if r < pa + pd_ else "B"
        if res == "A":
            points[a] += 3
        elif res == "B":
            points[b] += 3
        else:
            points[a] += 1
            points[b] += 1

    # Rank by points, then Elo, then a tiny random jitter to break ties.
    ranked = sorted(members, key=lambda t: (points[t], elo[t], random.random()), reverse=True)
    return ranked, points


def _knockout(qualifiers, elo, stage_counts):
    """Run a single-elimination bracket; record how far each team gets."""
    bracket = qualifiers[:]
    random.shuffle(bracket)                    # random draw each simulation
    round_names = {32: "qualify", 16: "r16", 8: "quarter", 4: "semi", 2: "final"}

    while len(bracket) > 1:
        if len(bracket) in round_names:
            for t in bracket:
                stage_counts[round_names[len(bracket)]][t] += 1
        winners = []
        for i in range(0, len(bracket), 2):
            winners.append(_knockout_winner(bracket[i], bracket[i + 1], elo))
        bracket = winners
    stage_counts["champion"][bracket[0]] += 1
    return bracket[0]


def simulate_tournament(n_sims: int = 2000) -> pd.DataFrame:
    """Run the whole tournament n_sims times; return per-team stage probabilities."""
    sched = load_2026_schedule()
    strength = load_latest_strength()
    elo = defaultdict(lambda: START_ELO, {t: s["elo"] for t, s in strength.items()})

    groups = reconstruct_groups(sched)

    # Pre-build each group's fixture list once.
    group_fixtures = []
    for members in groups:
        mset = set(members)
        fx = []
        for _, m in sched.iterrows():
            if m["home_team"] in mset and m["away_team"] in mset:
                fx.append({
                    "home_team": m["home_team"], "away_team": m["away_team"],
                    "played": m["status"] == "Played",
                    "hs": m["home_score"], "as_": m["away_score"],
                })
        group_fixtures.append((members, fx))

    stage_counts = {k: defaultdict(int) for k in
                    ["win_group", "qualify", "r16", "quarter", "semi", "final", "champion"]}

    for _ in range(n_sims):
        qualifiers, thirds = [], []
        for members, fx in group_fixtures:
            ranked, points = _simulate_group(members, fx, elo)
            stage_counts["win_group"][ranked[0]] += 1
            qualifiers.extend(ranked[:2])          # top 2 auto-qualify
            thirds.append((ranked[2], points[ranked[2]]))

        # 8 best third-placed teams complete the 32.
        thirds.sort(key=lambda x: (x[1], elo[x[0]], random.random()), reverse=True)
        qualifiers.extend(t for t, _ in thirds[:8])

        _knockout(qualifiers, elo, stage_counts)

    all_teams = sorted({t for members in groups for t in members})
    rows = []
    for t in all_teams:
        rows.append({
            "team": t,
            "elo": round(elo[t]),
            "win_group_%": round(100 * stage_counts["win_group"][t] / n_sims, 1),
            "qualify_%":   round(100 * stage_counts["qualify"][t] / n_sims, 1),
            "semi_%":      round(100 * stage_counts["semi"][t] / n_sims, 1),
            "final_%":     round(100 * stage_counts["final"][t] / n_sims, 1),
            "champion_%":  round(100 * stage_counts["champion"][t] / n_sims, 1),
        })
    return pd.DataFrame(rows).sort_values("champion_%", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("STEP: Simulating the FIFA 2026 tournament (this runs thousands of times)...")
    table = simulate_tournament(n_sims=2000)
    print(f"\n   Title odds — top 16 of {len(table)} teams:\n")
    print(table.head(16).to_string(index=False))
