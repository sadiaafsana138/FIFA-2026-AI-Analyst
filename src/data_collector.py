"""
data_collector.py
==================
This is the "ground truth" file for our project. It defines three things:

  1. STADIUMS      - the 16 real FIFA 2026 venues, with GPS coordinates + altitude.
  2. TEAM_CAPITALS - where each national team "comes from" (home city + timezone).
  3. match history - ~150 years of real international results, AUTO-DOWNLOADED
                     from a public GitHub mirror (no Kaggle account required).

Everything else in the project (weather, travel distance, the ML model, the maps,
the chatbot) is built on top of the data this file produces.

Run it directly to download + save the data:
    python -m src.data_collector
"""

import os
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. FIFA 2026 STADIUMS (the real 16 host venues)
# ---------------------------------------------------------------------------
# Each stadium has latitude/longitude (for distance + weather lookups) and
# altitude in metres (high altitude like Mexico City affects player stamina).
STADIUMS = [
    # ---- United States (11 venues) ----
    {"name": "MetLife Stadium",        "city": "New York/NJ",     "country": "USA",    "lat": 40.8135, "lon": -74.0745,  "altitude_m": 7},
    {"name": "Mercedes-Benz Stadium",  "city": "Atlanta",         "country": "USA",    "lat": 33.7554, "lon": -84.4008,  "altitude_m": 320},
    {"name": "Gillette Stadium",       "city": "Boston",          "country": "USA",    "lat": 42.0909, "lon": -71.2643,  "altitude_m": 32},
    {"name": "AT&T Stadium",           "city": "Dallas",          "country": "USA",    "lat": 32.7473, "lon": -97.0945,  "altitude_m": 150},
    {"name": "NRG Stadium",            "city": "Houston",         "country": "USA",    "lat": 29.6847, "lon": -95.4107,  "altitude_m": 15},
    {"name": "Arrowhead Stadium",      "city": "Kansas City",     "country": "USA",    "lat": 39.0489, "lon": -94.4839,  "altitude_m": 270},
    {"name": "SoFi Stadium",           "city": "Los Angeles",     "country": "USA",    "lat": 33.9535, "lon": -118.3392, "altitude_m": 30},
    {"name": "Hard Rock Stadium",      "city": "Miami",           "country": "USA",    "lat": 25.9580, "lon": -80.2389,  "altitude_m": 2},
    {"name": "Lincoln Financial Field","city": "Philadelphia",    "country": "USA",    "lat": 39.9008, "lon": -75.1675,  "altitude_m": 12},
    {"name": "Levi's Stadium",         "city": "San Francisco",   "country": "USA",    "lat": 37.4030, "lon": -121.9700, "altitude_m": 9},
    {"name": "Lumen Field",            "city": "Seattle",         "country": "USA",    "lat": 47.5952, "lon": -122.3316, "altitude_m": 5},
    # ---- Mexico (3 venues) ----
    {"name": "Estadio Azteca",         "city": "Mexico City",     "country": "Mexico", "lat": 19.3029, "lon": -99.1505,  "altitude_m": 2240},
    {"name": "Estadio Akron",          "city": "Guadalajara",     "country": "Mexico", "lat": 20.6817, "lon": -103.4626, "altitude_m": 1566},
    {"name": "Estadio BBVA",           "city": "Monterrey",       "country": "Mexico", "lat": 25.6692, "lon": -100.2444, "altitude_m": 530},
    # ---- Canada (2 venues) ----
    {"name": "BMO Field",              "city": "Toronto",         "country": "Canada", "lat": 43.6332, "lon": -79.4186,  "altitude_m": 76},
    {"name": "BC Place",               "city": "Vancouver",       "country": "Canada", "lat": 49.2767, "lon": -123.1117, "altitude_m": 3},
]

# ---------------------------------------------------------------------------
# 2. TEAM HOME LOCATIONS (capital/main city coords + timezone)
# ---------------------------------------------------------------------------
# Used to measure how far each team must travel and how big their time-zone
# jump is. Easily extend this dict with more teams later.
TEAM_CAPITALS = {
    "Argentina":          {"lat": -34.6037, "lon": -58.3816, "timezone": "America/Argentina/Buenos_Aires"},
    "Brazil":             {"lat": -15.7801, "lon": -47.9292, "timezone": "America/Sao_Paulo"},
    "France":             {"lat":  48.8566, "lon":   2.3522, "timezone": "Europe/Paris"},
    "England":            {"lat":  51.5074, "lon":  -0.1278, "timezone": "Europe/London"},
    "Spain":              {"lat":  40.4168, "lon":  -3.7038, "timezone": "Europe/Madrid"},
    "Germany":            {"lat":  52.5200, "lon":  13.4050, "timezone": "Europe/Berlin"},
    "Portugal":           {"lat":  38.7169, "lon":  -9.1399, "timezone": "Europe/Lisbon"},
    "Netherlands":        {"lat":  52.3676, "lon":   4.9041, "timezone": "Europe/Amsterdam"},
    "Belgium":            {"lat":  50.8503, "lon":   4.3517, "timezone": "Europe/Brussels"},
    "Croatia":            {"lat":  45.8150, "lon":  15.9819, "timezone": "Europe/Zagreb"},
    "Italy":              {"lat":  41.9028, "lon":  12.4964, "timezone": "Europe/Rome"},
    "Morocco":            {"lat":  33.9716, "lon":  -6.8498, "timezone": "Africa/Casablanca"},
    "Senegal":            {"lat":  14.6928, "lon": -17.4467, "timezone": "Africa/Dakar"},
    "Nigeria":            {"lat":   9.0765, "lon":   7.3986, "timezone": "Africa/Lagos"},
    "Japan":              {"lat":  35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
    "South Korea":        {"lat":  37.5665, "lon": 126.9780, "timezone": "Asia/Seoul"},
    "Australia":          {"lat": -35.2809, "lon": 149.1300, "timezone": "Australia/Sydney"},
    "Saudi Arabia":       {"lat":  24.7136, "lon":  46.6753, "timezone": "Asia/Riyadh"},
    "Mexico":             {"lat":  19.4326, "lon": -99.1332, "timezone": "America/Mexico_City"},
    "United States":      {"lat":  38.9072, "lon": -77.0369, "timezone": "America/New_York"},
    "Canada":             {"lat":  45.4215, "lon": -75.6972, "timezone": "America/Toronto"},
    "Uruguay":            {"lat": -34.9011, "lon": -56.1645, "timezone": "America/Montevideo"},
    "Colombia":           {"lat":   4.7110, "lon": -74.0721, "timezone": "America/Bogota"},
    "Ecuador":            {"lat":  -0.1807, "lon": -78.4678, "timezone": "America/Guayaquil"},
    "Norway":             {"lat":  59.9139, "lon":  10.7522, "timezone": "Europe/Oslo"},
    "Switzerland":        {"lat":  46.9480, "lon":   7.4474, "timezone": "Europe/Zurich"},
    # --- Expanded to cover every World Cup nation in the dataset ---
    "Algeria":            {"lat":  36.7538, "lon":   3.0588, "timezone": "Africa/Algiers"},
    "Austria":            {"lat":  48.2082, "lon":  16.3738, "timezone": "Europe/Vienna"},
    "Bosnia and Herzegovina": {"lat": 43.8563, "lon": 18.4131, "timezone": "Europe/Sarajevo"},
    "Cameroon":           {"lat":   3.8480, "lon":  11.5021, "timezone": "Africa/Douala"},
    "Cape Verde":         {"lat":  14.9330, "lon": -23.5133, "timezone": "Atlantic/Cape_Verde"},
    "Chile":              {"lat": -33.4489, "lon": -70.6693, "timezone": "America/Santiago"},
    "Costa Rica":         {"lat":   9.9281, "lon": -84.0907, "timezone": "America/Costa_Rica"},
    "Curaçao":            {"lat":  12.1091, "lon": -68.9316, "timezone": "America/Curacao"},
    "Czech Republic":     {"lat":  50.0755, "lon":  14.4378, "timezone": "Europe/Prague"},
    "DR Congo":           {"lat":  -4.4419, "lon":  15.2663, "timezone": "Africa/Kinshasa"},
    "Denmark":            {"lat":  55.6761, "lon":  12.5683, "timezone": "Europe/Copenhagen"},
    "Egypt":              {"lat":  30.0444, "lon":  31.2357, "timezone": "Africa/Cairo"},
    "Ghana":              {"lat":   5.6037, "lon":  -0.1870, "timezone": "Africa/Accra"},
    "Greece":             {"lat":  37.9838, "lon":  23.7275, "timezone": "Europe/Athens"},
    "Haiti":              {"lat":  18.5944, "lon": -72.3074, "timezone": "America/Port-au-Prince"},
    "Honduras":           {"lat":  14.0723, "lon": -87.1921, "timezone": "America/Tegucigalpa"},
    "Iceland":            {"lat":  64.1466, "lon": -21.9426, "timezone": "Atlantic/Reykjavik"},
    "Iran":               {"lat":  35.6892, "lon":  51.3890, "timezone": "Asia/Tehran"},
    "Iraq":               {"lat":  33.3152, "lon":  44.3661, "timezone": "Asia/Baghdad"},
    "Ivory Coast":        {"lat":   5.3600, "lon":  -4.0083, "timezone": "Africa/Abidjan"},
    "Jordan":             {"lat":  31.9454, "lon":  35.9284, "timezone": "Asia/Amman"},
    "New Zealand":        {"lat": -41.2865, "lon": 174.7762, "timezone": "Pacific/Auckland"},
    "North Korea":        {"lat":  39.0392, "lon": 125.7625, "timezone": "Asia/Pyongyang"},
    "Panama":             {"lat":   8.9824, "lon": -79.5199, "timezone": "America/Panama"},
    "Paraguay":           {"lat": -25.2637, "lon": -57.5759, "timezone": "America/Asuncion"},
    "Peru":               {"lat": -12.0464, "lon": -77.0428, "timezone": "America/Lima"},
    "Poland":             {"lat":  52.2297, "lon":  21.0122, "timezone": "Europe/Warsaw"},
    "Qatar":              {"lat":  25.2854, "lon":  51.5310, "timezone": "Asia/Qatar"},
    "Russia":             {"lat":  55.7558, "lon":  37.6173, "timezone": "Europe/Moscow"},
    "Scotland":           {"lat":  55.9533, "lon":  -3.1883, "timezone": "Europe/London"},
    "Serbia":             {"lat":  44.7866, "lon":  20.4489, "timezone": "Europe/Belgrade"},
    "Slovakia":           {"lat":  48.1486, "lon":  17.1077, "timezone": "Europe/Bratislava"},
    "Slovenia":           {"lat":  46.0569, "lon":  14.5058, "timezone": "Europe/Ljubljana"},
    "South Africa":       {"lat": -25.7479, "lon":  28.2293, "timezone": "Africa/Johannesburg"},
    "Sweden":             {"lat":  59.3293, "lon":  18.0686, "timezone": "Europe/Stockholm"},
    "Tunisia":            {"lat":  36.8065, "lon":  10.1815, "timezone": "Africa/Tunis"},
    "Turkey":             {"lat":  39.9334, "lon":  32.8597, "timezone": "Europe/Istanbul"},
    "Uzbekistan":         {"lat":  41.2995, "lon":  69.2401, "timezone": "Asia/Tashkent"},
    "Wales":              {"lat":  51.4837, "lon":  -3.1681, "timezone": "Europe/London"},
}

# Public GitHub mirror of the well-known "International football results
# 1872-present" dataset (same data people normally download from Kaggle).
MATCH_DATA_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
RAW_MATCHES_PATH = "data/raw/matches.csv"


def download_match_history(url: str = MATCH_DATA_URL, dest: str = RAW_MATCHES_PATH) -> str:
    """Download the raw match-results CSV if we don't already have it locally."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        print(f"   Match data already present at {dest} (skipping download)")
        return dest

    print(f"   Downloading match history from GitHub mirror...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()           # crash loudly if the download failed
    with open(dest, "wb") as f:
        f.write(response.content)
    print(f"   Saved raw match data to {dest}")
    return dest


def load_match_history(filepath: str = RAW_MATCHES_PATH) -> pd.DataFrame:
    """Load the CSV, then keep only World Cup matches from 2010 onward."""
    download_match_history(dest=filepath)         # make sure the file exists first
    df = pd.read_csv(filepath)

    # Keep only World Cup finals + qualifiers (the matches most like 2026).
    wc = df[df["tournament"].str.contains(
        "FIFA World Cup", na=False, case=False
    )].copy()

    # Keep recent history so the patterns are relevant to today's game.
    wc["date"] = pd.to_datetime(wc["date"], errors="coerce")
    recent = wc[wc["date"] >= "2010-01-01"].copy()

    print(f"   Loaded {len(recent):,} World Cup matches from 2010 onward")
    return recent


def load_full_history(filepath: str = RAW_MATCHES_PATH) -> pd.DataFrame:
    """Load the ENTIRE results history (every tournament, 1872-present).

    We need the full history - not just World Cup matches - so that team
    strength (Elo) and recent form are based on everything a team has played.
    """
    download_match_history(dest=filepath)
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    return df.sort_values("date").reset_index(drop=True)


def save_stadiums(dest: str = "data/raw/stadiums.csv") -> pd.DataFrame:
    """Write the stadium list to a CSV so other scripts / the app can read it."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df = pd.DataFrame(STADIUMS)
    df.to_csv(dest, index=False)
    print(f"   Saved {len(STADIUMS)} stadiums to {dest}")
    return df


if __name__ == "__main__":
    print("STEP: Collecting base data...")
    save_stadiums()
    matches = load_match_history()
    print("\nSample of the match data we will analyse:")
    print(matches[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]].head())
    print(f"\nDone. {len(TEAM_CAPITALS)} teams and {len(STADIUMS)} stadiums are configured.")
