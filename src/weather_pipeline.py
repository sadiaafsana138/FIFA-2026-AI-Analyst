"""
weather_pipeline.py
===================
Fetches REAL historical weather for any location + date from the OpenMeteo
archive API (100% free, no API key required).

For each (latitude, longitude, date) we return:
    temp_max, temp_min, temp_avg, humidity, precip_mm, wind_kmh

To avoid hammering the API (and to make the pipeline fast on repeat runs), every
successful result is cached to disk in data/weather/weather_cache.json. The cache
key is "lat_lon_date", so asking for the same match twice costs zero network time.

Run it directly to test one lookup:
    python -m src.weather_pipeline
"""

import os
import json
import time
import requests

CACHE_FILE = "data/weather/weather_cache.json"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_cache() -> dict:
    """Read the saved weather cache from disk (or return an empty one)."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    """Write the weather cache back to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_weather(lat: float, lon: float, date_str: str) -> dict:
    """
    Return historical weather for one place on one day (YYYY-MM-DD).
    Uses the on-disk cache first; only calls the API on a cache miss.
    Returns {} if the lookup fails so callers can degrade gracefully.
    """
    cache = load_cache()
    key = f"{lat}_{lon}_{date_str}"

    if key in cache:                 # cache hit -> instant, no network
        return cache[key]

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date_str,
        "end_date":   date_str,
        "daily":      "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "hourly":     "relativehumidity_2m",
        "timezone":   "auto",
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        hourly = data.get("hourly", {})
        elevation = data.get("elevation")     # real ground elevation (metres)

        temp_max = _first(daily.get("temperature_2m_max"))
        temp_min = _first(daily.get("temperature_2m_min"))
        precip   = _first(daily.get("precipitation_sum"))
        wind     = _first(daily.get("windspeed_10m_max"))

        # Average humidity over typical match hours (14:00-22:00 local).
        humidity_list = hourly.get("relativehumidity_2m", []) or []
        match_hours = humidity_list[14:22] if len(humidity_list) >= 22 else humidity_list
        match_hours = [h for h in match_hours if h is not None]
        avg_humidity = sum(match_hours) / len(match_hours) if match_hours else None

        # NOTE: use "is not None" (not "if temp_max") so a real 0.0 isn't dropped.
        temp_avg = (
            round((temp_max + temp_min) / 2, 1)
            if temp_max is not None and temp_min is not None
            else None
        )

        result = {
            "temp_max":     temp_max,
            "temp_min":     temp_min,
            "temp_avg":     temp_avg,
            "humidity":     round(avg_humidity, 1) if avg_humidity is not None else None,
            "precip_mm":    precip,
            "wind_kmh":     wind,
            "elevation_m":  elevation,
        }

        cache[key] = result
        save_cache(cache)
        time.sleep(0.3)              # be polite to the free API
        return result

    except Exception as e:
        print(f"   Weather fetch failed for {lat},{lon} on {date_str}: {e}")
        return {}


def _first(values):
    """OpenMeteo returns daily fields as 1-element lists; grab the value safely."""
    if isinstance(values, list) and values:
        return values[0]
    return None


if __name__ == "__main__":
    print("STEP: Testing the weather pipeline...")
    # Estadio Azteca, Mexico City, on a sample date.
    sample = get_weather(19.3029, -99.1505, "2023-06-15")
    print("Sample weather (Mexico City, 2023-06-15):")
    for k, v in sample.items():
        print(f"   {k:>10}: {v}")
