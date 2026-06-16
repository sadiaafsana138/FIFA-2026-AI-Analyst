"""
schedule.py
===========
Builds the FIFA 2026 match-schedule dashboard data: the REAL 2026 World Cup
fixtures from our dataset, enriched so that

  - upcoming matches get our model's win / draw / loss prediction, and
  - already-played matches show the actual result (so you can see if the model
    would have been right).

Each fixture's host city is mapped to the nearest of our 16 venues (so we can
reuse the venue's altitude/coordinates for the prediction).

NOTE on time zones: the source data has match DATES, not kick-off times, so we
show dates (Bangladesh, UTC+6, shares the same calendar date for these fixtures).
Exact BD kick-off clock-times can be layered in later from the official schedule.

Run directly to preview:
    python -m src.schedule
"""

import os

import pandas as pd
from geopy.distance import geodesic

from src.data_collector import STADIUMS, RAW_MATCHES_PATH, download_match_history
from src.feature_engineer import geocode_city, _load_geocode_cache
from src.model import predict_match, MODEL_PATH

import pickle

SCHEDULE_OUT = "data/processed/schedule_2026.csv"


def load_2026_schedule() -> pd.DataFrame:
    """Load the real 2026 World Cup fixtures (played + upcoming) from the dataset."""
    download_match_history()
    df = pd.read_csv(RAW_MATCHES_PATH, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")].copy()
    wc = wc.sort_values("date").reset_index(drop=True)
    wc["status"] = wc["home_score"].notna().map({True: "Played", False: "Upcoming"})
    return wc


def _nearest_stadium(lat: float, lon: float) -> dict:
    """Find the FIFA 2026 venue closest to a match's host-city coordinates."""
    return min(STADIUMS, key=lambda s: geodesic((lat, lon), (s["lat"], s["lon"])).km)


def build_schedule_dashboard() -> pd.DataFrame:
    """Return the enriched schedule: fixtures + venue + prediction or actual result."""
    wc = load_2026_schedule()
    geo_cache = _load_geocode_cache()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    rows = []
    for _, m in wc.iterrows():
        coords = geocode_city(str(m["city"]), str(m["country"]), geo_cache)
        venue = _nearest_stadium(*coords)["name"] if coords else STADIUMS[0]["name"]
        date_str = m["date"].strftime("%Y-%m-%d")

        row = {
            "date": date_str,
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "venue": venue,
            "city": m["city"],
            "status": m["status"],
        }

        if m["status"] == "Played":
            hs, as_ = int(m["home_score"]), int(m["away_score"])
            row["score"] = f"{hs} - {as_}"
            row["actual"] = (
                f"{m['home_team']} win" if hs > as_
                else f"{m['away_team']} win" if as_ > hs
                else "Draw"
            )
            row["pred_home_%"] = row["pred_draw_%"] = row["pred_away_%"] = None
        else:
            pred = predict_match(m["home_team"], m["away_team"], venue, date_str, model)
            row["score"] = "-"
            row["actual"] = "-"
            row["pred_home_%"] = pred["home_win"]
            row["pred_draw_%"] = pred["draw"]
            row["pred_away_%"] = pred["away_win"]

        rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SCHEDULE_OUT), exist_ok=True)
    out.to_csv(SCHEDULE_OUT, index=False)        # cache for the chatbot to read
    return out


if __name__ == "__main__":
    print("STEP: Building the 2026 match-schedule dashboard...")
    sched = build_schedule_dashboard()
    print(f"   {len(sched)} fixtures ({(sched['status'] == 'Played').sum()} played, "
          f"{(sched['status'] == 'Upcoming').sum()} upcoming)\n")

    print("   Recently played (with actual results):")
    played = sched[sched["status"] == "Played"][["date", "home_team", "away_team", "venue", "score", "actual"]]
    print(played.head(6).to_string(index=False))

    print("\n   Upcoming (with model predictions, % home/draw/away):")
    up = sched[sched["status"] == "Upcoming"][["date", "home_team", "away_team", "venue", "pred_home_%", "pred_draw_%", "pred_away_%"]]
    print(up.head(8).to_string(index=False))
