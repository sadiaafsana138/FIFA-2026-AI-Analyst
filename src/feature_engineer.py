"""
feature_engineer.py
===================
This is where everything comes together. We walk through every World Cup finals
match (2010-2022) and, for each one, attach the real-world conditions it was
played under:

    - WHERE it was played   -> geocode the host city to lat/lon (cached)
    - the WEATHER that day   -> from weather_pipeline (temp, humidity, rain, wind)
    - the ALTITUDE           -> real ground elevation at that location
    - the TRAVEL burden      -> each team's distance + time-zone shift to the venue
    - the RESULT             -> home win / draw / away win, and total goals

The output is data/processed/features.csv: one clean row per match, ready for
the machine-learning model to train on.

IMPORTANT (methodology): unlike a naive approach that pins every match to a 2026
stadium, we use each match's REAL host city. So the weather and travel features
describe what actually happened, which makes the model honest.

Run it directly to build the feature table:
    python -m src.feature_engineer
"""

import json
import os

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

from src.data_collector import load_full_history, TEAM_CAPITALS
from src.travel_calculator import get_travel_distance, get_timezone_shift
from src.weather_pipeline import get_weather
from src.team_form import add_history_features, save_latest_strength, HISTORY_COLS

GEOCODE_CACHE = "data/raw/geocode_cache.json"
FEATURES_PATH = "data/processed/features.csv"

# One geocoder for the whole run. Nominatim is the free OpenStreetMap service;
# it asks us to identify ourselves and to send at most ~1 request per second.
_geolocator = Nominatim(user_agent="fifa2026-analytics")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1.1)


def _load_geocode_cache() -> dict:
    if os.path.exists(GEOCODE_CACHE):
        with open(GEOCODE_CACHE) as f:
            return json.load(f)
    return {}


def _save_geocode_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(GEOCODE_CACHE), exist_ok=True)
    with open(GEOCODE_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def geocode_city(city: str, country: str, cache: dict):
    """Return (lat, lon) for a host city, using a disk cache to avoid re-querying."""
    key = f"{city}, {country}"
    if key in cache:
        return cache[key]

    try:
        location = _geocode(key)
        if location:
            coords = [round(location.latitude, 4), round(location.longitude, 4)]
            cache[key] = coords
            _save_geocode_cache(cache)
            return coords
    except Exception as e:
        print(f"   Geocoding failed for {key}: {e}")

    cache[key] = None          # remember the failure so we don't retry it
    _save_geocode_cache(cache)
    return None


def build_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich each match with location, weather, altitude, and travel features."""
    geo_cache = _load_geocode_cache()
    rows = []

    for _, m in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Enriching matches"):
        # 1) Where was it played? (real host city -> coordinates)
        coords = geocode_city(str(m["city"]), str(m["country"]), geo_cache)
        if not coords:
            continue                       # skip matches we cannot locate
        lat, lon = coords
        date_str = str(m["date"])[:10]

        # 2) Weather + altitude at that place/date.
        weather = get_weather(lat, lon, date_str)

        # 3) Travel burden for each team to this venue.
        home_dist = get_travel_distance(m["home_team"], lat, lon)
        away_dist = get_travel_distance(m["away_team"], lat, lon)
        home_tz = get_timezone_shift(m["home_team"], lat, lon)
        away_tz = get_timezone_shift(m["away_team"], lat, lon)

        # 4) The result, encoded for the model.
        hs, as_ = m["home_score"], m["away_score"]
        if hs > as_:
            result = 1          # home win
        elif hs == as_:
            result = 0          # draw
        else:
            result = -1         # away win

        feature_row = {
            "date":            date_str,
            "home_team":       m["home_team"],
            "away_team":       m["away_team"],
            "city":            m["city"],
            "country":         m["country"],
            "home_score":      hs,
            "away_score":      as_,
            "result":          result,
            "total_goals":     hs + as_,
            "altitude_m":      weather.get("elevation_m"),
            "home_travel_km":  home_dist,
            "away_travel_km":  away_dist,
            "travel_diff_km":  (away_dist - home_dist) if (home_dist is not None and away_dist is not None) else None,
            "home_tz_shift":   home_tz,
            "away_tz_shift":   away_tz,
            "temp_avg":        weather.get("temp_avg"),
            "humidity":        weather.get("humidity"),
            "precip_mm":       weather.get("precip_mm"),
            "wind_kmh":        weather.get("wind_kmh"),
        }
        # Carry the leak-free strength/form features computed in team_form.
        for col in HISTORY_COLS:
            feature_row[col] = m[col]
        rows.append(feature_row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"\n   Feature matrix built: {len(df)} rows x {len(df.columns)} columns")
    print(f"   Saved to {FEATURES_PATH}")
    return df


def load_world_cup_finals() -> pd.DataFrame:
    """
    Get World Cup FINALS matches (2010+) WITH leak-free strength/form features.

    We compute Elo + form over the FULL history first (so strength reflects every
    match a team ever played), then keep only the finals we want to analyse.
    """
    full = load_full_history()                          # every match, 1872+
    full_with_history, latest = add_history_features(full)
    save_latest_strength(latest)                        # for predicting 2026 games

    finals = full_with_history[
        (full_with_history["tournament"] == "FIFA World Cup")
        & (full_with_history["date"] >= "2010-01-01")
    ].copy()
    print(f"   Using {len(finals)} World Cup finals matches (2010-present), with Elo + form")
    return finals


if __name__ == "__main__":
    print("STEP: Building the ML feature matrix...")
    print("(First run geocodes host cities + fetches weather - a few minutes.")
    print(" Every later run is fast because both are cached.)\n")
    finals = load_world_cup_finals()
    build_features(finals)
