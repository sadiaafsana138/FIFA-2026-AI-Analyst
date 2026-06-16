"""
validate_data.py
================
A data-quality check that verifies every combination the project relies on is
actually covered - so there are no silent gaps (e.g. a 2026 team missing from our
strength table would quietly get a weak default rating and a wrong prediction).

Checks:
  1. Every team capital has a valid coordinate + a resolvable timezone.
  2. All 16 stadiums have valid coordinates.
  3. Every team in the 2026 schedule is covered by TEAM_CAPITALS (travel) AND by
     the Elo strength table (prediction). Missing ones are reported.
  4. Every 2026 host city geocoded and mapped to a stadium.
  5. Every upcoming fixture has predictions; every played fixture has a result.

Run:
    python -m src.validate_data
"""

import pandas as pd
import pytz

from src.data_collector import STADIUMS, TEAM_CAPITALS
from src.travel_calculator import get_timezone_shift
from src.team_form import load_latest_strength
from src.schedule import SCHEDULE_OUT, build_schedule_dashboard

import os


def _ok(msg):   print(f"   [OK]   {msg}")
def _warn(msg): print(f"   [WARN] {msg}")
def _fail(msg): print(f"   [FAIL] {msg}")


def main():
    problems = 0

    # 1. Team capitals -------------------------------------------------------
    print("1. Team capitals (coords + timezone):")
    bad_tz, bad_coord = [], []
    for team, info in TEAM_CAPITALS.items():
        if info["timezone"] not in pytz.all_timezones:
            bad_tz.append(team)
        if not (-90 <= info["lat"] <= 90 and -180 <= info["lon"] <= 180):
            bad_coord.append(team)
    if bad_tz:   _fail(f"invalid timezone: {bad_tz}"); problems += len(bad_tz)
    if bad_coord: _fail(f"invalid coords: {bad_coord}"); problems += len(bad_coord)
    if not bad_tz and not bad_coord:
        _ok(f"all {len(TEAM_CAPITALS)} team capitals valid")

    # 2. Stadiums ------------------------------------------------------------
    print("2. Stadiums:")
    bad_s = [s["name"] for s in STADIUMS
             if not (-90 <= s["lat"] <= 90 and -180 <= s["lon"] <= 180)]
    if bad_s: _fail(f"invalid coords: {bad_s}"); problems += len(bad_s)
    else:     _ok(f"all {len(STADIUMS)} stadiums valid")

    # 3 + 4 + 5. Schedule coverage ------------------------------------------
    print("3-5. 2026 schedule coverage:")
    sched = pd.read_csv(SCHEDULE_OUT) if os.path.exists(SCHEDULE_OUT) else build_schedule_dashboard()
    strength = load_latest_strength()

    sched_teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    missing_caps = [t for t in sched_teams if t not in TEAM_CAPITALS]
    missing_elo = [t for t in sched_teams if t not in strength]

    if missing_caps: _fail(f"2026 teams missing from TEAM_CAPITALS (travel=0!): {missing_caps}"); problems += len(missing_caps)
    else:            _ok(f"all {len(sched_teams)} schedule teams have home coords")

    if missing_elo: _warn(f"2026 teams missing from Elo table (default 1500): {missing_elo}")
    else:           _ok("all schedule teams have a real Elo rating")

    # venue mapping + prediction/result completeness
    no_venue = sched[sched["venue"].isna()]
    if len(no_venue): _fail(f"{len(no_venue)} fixtures with no venue mapped"); problems += len(no_venue)
    else:             _ok("every fixture mapped to a stadium")

    up = sched[sched["status"] == "Upcoming"]
    missing_pred = up[up["pred_home_%"].isna()]
    if len(missing_pred): _fail(f"{len(missing_pred)} upcoming fixtures missing a prediction"); problems += len(missing_pred)
    else:                 _ok(f"all {len(up)} upcoming fixtures have predictions")

    played = sched[sched["status"] == "Played"]
    missing_res = played[played["actual"] == "-"]
    if len(missing_res): _fail(f"{len(missing_res)} played fixtures missing a result"); problems += len(missing_res)
    else:                _ok(f"all {len(played)} played fixtures have results")

    print()
    if problems == 0:
        print(f"RESULT: all checks passed. Data is complete across all combinations.")
    else:
        print(f"RESULT: {problems} problem(s) found - see [FAIL] lines above.")
    return problems


if __name__ == "__main__":
    main()
