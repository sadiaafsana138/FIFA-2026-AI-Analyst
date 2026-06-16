"""
travel_calculator.py
====================
Turns raw coordinates into two human-meaningful "burden" numbers for a team
playing at a given stadium:

  1. get_travel_distance() -> kilometres from the team's home city to the venue
     (great-circle distance, i.e. the real distance over the curved Earth).

  2. get_timezone_shift()  -> how many hours the team's body clock must adjust
     (jet lag). Anchored to the 2026 tournament dates so daylight-saving is
     handled correctly.

These two functions are later fed into the ML model as features, on the theory
that long flights + big time-zone jumps hurt performance.

Run it directly to test:
    python -m src.travel_calculator
"""

from datetime import datetime

import pytz
from geopy.distance import geodesic
from timezonefinder import TimezoneFinder

from src.data_collector import TEAM_CAPITALS

# TimezoneFinder loads a big lookup table, so build it once and reuse it.
_tf = TimezoneFinder()

# Anchor time-zone math to the tournament window so DST offsets are correct
# for when the matches are actually played (June-July 2026).
_REFERENCE_DATE = datetime(2026, 6, 15, 18, 0, 0)


def get_travel_distance(team: str, stadium_lat: float, stadium_lon: float):
    """Great-circle distance (km) from a team's home city to a stadium."""
    if team not in TEAM_CAPITALS:
        return None
    home = TEAM_CAPITALS[team]
    home_coords = (home["lat"], home["lon"])
    stadium_coords = (stadium_lat, stadium_lon)
    return round(geodesic(home_coords, stadium_coords).kilometers, 1)


def get_timezone_shift(team: str, stadium_lat: float, stadium_lon: float):
    """Absolute hour difference between a team's home timezone and the venue."""
    if team not in TEAM_CAPITALS:
        return None

    home_tz_str = TEAM_CAPITALS[team]["timezone"]
    stadium_tz_str = _tf.timezone_at(lat=stadium_lat, lng=stadium_lon)
    if not stadium_tz_str:
        return None

    home_tz = pytz.timezone(home_tz_str)
    stadium_tz = pytz.timezone(stadium_tz_str)

    # utcoffset at the reference date gives each zone's offset from UTC (in hours).
    home_offset = home_tz.utcoffset(_REFERENCE_DATE).total_seconds() / 3600
    stadium_offset = stadium_tz.utcoffset(_REFERENCE_DATE).total_seconds() / 3600

    return round(abs(home_offset - stadium_offset), 1)


if __name__ == "__main__":
    print("STEP: Testing travel + timezone calculations...")
    tests = [
        ("Japan",     40.8135, -74.0745, "MetLife Stadium (New York)"),
        ("France",    19.3029, -99.1505, "Estadio Azteca (Mexico City)"),
        ("Mexico",    19.3029, -99.1505, "Estadio Azteca (home!)"),
    ]
    for team, lat, lon, label in tests:
        dist = get_travel_distance(team, lat, lon)
        tz = get_timezone_shift(team, lat, lon)
        print(f"   {team:>9} -> {label:<32}: {dist:>8,} km, {tz}h time-zone shift")
