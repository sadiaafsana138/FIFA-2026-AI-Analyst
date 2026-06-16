---
title: FIFA 2026 Travel Climate Performance Analytics
emoji: ⚽
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# ⚽ FIFA 2026 — Travel, Climate & Performance Analytics
LIVE -- https://huggingface.co/spaces/sadiaafsana138/Fifa-2026-AI-Analyst

An end-to-end data-science platform that studies **how team strength, travel, climate, and
altitude affect FIFA World Cup match outcomes**, predicts the live 2026 tournament, simulates who
will win the cup, and answers questions through an AI assistant.

It combines a real data pipeline, an Elo rating engine, a machine-learning model, a Monte-Carlo
tournament simulator, an interactive dashboard, and a web-augmented chatbot.

> **Live demo:** _add your Hugging Face Space URL here_

---

## What it does (6 dashboard pages)

- 📅 **Schedule & Predictions** — every real 2026 World Cup fixture, with the **actual result** for
  games already played and the **model's win/draw/loss prediction** for upcoming ones.
- 📋 **Group Standings** — live group tables built from real results (points, goal difference,
  who's qualifying). Updates as new scores come in.
- 🔮 **Match Predictor** — pick any two teams + venue to get outcome probabilities.
- 🏆 **Tournament Simulator** — Monte-Carlo: plays the whole tournament thousands of times to
  estimate each team's odds of winning their group, qualifying, reaching the final, and lifting the cup.
- 📊 **Research Findings** — charts testing the hypotheses (humidity, altitude, travel) plus a
  cross-validated comparison of nine ML models.
- 🤖 **AI Analyst** — a Groq (Llama 4 Scout) chatbot grounded in the project's own data, with a
  **web-search fallback** for live info (squads, line-ups, news) the language model can't know.

A **🔄 Refresh latest results** button re-downloads the newest scores so every page updates.

---

## Research question

> *How do team strength, travel distance, rest, time-zone shifts, altitude, and weather affect
> the outcome of international football matches?*

### Hypotheses

| # | Hypothesis |
|---|---|
| H1 | Greater travel distance reduces win probability |
| H2 | Fewer rest days reduce scoring |
| H3 | High humidity (>80%) reduces total goals |
| H4 | Altitude (>1,500 m) disadvantages non-adapted teams |
| H5 | Large time-zone shifts (>5 h) reduce performance |

### Headline findings (honest)

- **Team strength (Elo) is by far the strongest predictor** of match outcome.
- The model reaches **~51–53% cross-validated accuracy** on a 3-way outcome vs a **~42% baseline** —
  all environmental/travel features together add a real but secondary edge.
- All nine benchmarked models beat the baseline by ~10 points and cluster near 51%, indicating the
  realistic ceiling for World Cup outcome prediction on this data.
- **H3 was not supported** here: average goals were slightly *higher* in high humidity (~2.87 vs ~2.52).
- **H2 showed little effect** at finals level (uniform schedules), so rest-days were kept for
  analysis but excluded from the predictor — a deliberate feature-selection decision.

---

## How it works (pipeline)

```
data_collector   →  16 real 2026 stadiums + team home bases + match history
        │                                       (auto-downloaded from a public mirror)
        ▼
team_form         →  Elo rating + recent form + rest days over 49,000 matches (leak-free)
weather_pipeline  →  real historical weather + elevation per location/date (OpenMeteo, cached)
travel_calculator →  great-circle distance + time-zone shift
        ▼
feature_engineer  →  geocode each real host city, merge everything → features.csv
        ▼
model             →  XGBoost (regularized), cross-validated against 8 other models
        ▼
schedule          →  real 2026 fixtures + predictions (upcoming) / results (played)
simulator         →  Monte-Carlo tournament: group standings + title odds
        ▼
app.py            →  Streamlit dashboard (6 pages)
analyst.py        →  Groq RAG chatbot + web_search.py (live web fallback)
validate_data.py  →  data-quality checks across all teams/venues/fixtures
```

---

## Tech stack

- **Data:** International results (public mirror), OpenMeteo (weather + elevation), OpenStreetMap (geocoding)
- **ML:** XGBoost, scikit-learn (cross-validation, 9-model comparison)
- **Geospatial:** geopy (great-circle distance), timezonefinder
- **Simulation:** Monte-Carlo (Elo-based match model)
- **LLM:** Groq API — Llama 4 Scout (retrieval-augmented), DuckDuckGo (`ddgs`) web search
- **App:** Streamlit, Plotly
- **Deploy:** Hugging Face Spaces

---

## Run locally

```powershell
# 1. Create + activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key (free: https://console.groq.com)
#    Copy .env.example to .env and paste your key:
#    GROQ_API_KEY=your_key_here

# 4. Build the data + model (first run fetches weather; later runs are cached)
python -m src.data_collector
python -m src.feature_engineer
python -m src.model
python -m src.schedule

# 5. (Optional) verify everything is consistent across all combinations
python -m src.validate_data

# 6. Launch the app
streamlit run app.py
```

---

## Project structure

```
.
├── app.py                  # Streamlit dashboard (6 pages)
├── requirements.txt
├── .env.example            # copy to .env and add your Groq key
├── src/
│   ├── data_collector.py   # stadiums, team home bases, match history
│   ├── weather_pipeline.py # OpenMeteo weather + elevation (cached)
│   ├── travel_calculator.py# distance + time-zone shift
│   ├── team_form.py        # Elo rating + recent form + rest days (leak-free)
│   ├── feature_engineer.py # builds the ML feature table
│   ├── model.py            # XGBoost train / predict / 9-model comparison
│   ├── schedule.py         # 2026 fixtures + predictions/results
│   ├── simulator.py        # Monte-Carlo tournament + group standings
│   ├── analyst.py          # Groq RAG chatbot
│   ├── web_search.py       # DuckDuckGo live web fallback
│   └── validate_data.py    # data-quality checks
└── data/                   # generated data, model, caches
```

---

## Deploy to Hugging Face Spaces (free)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. **New Space** → SDK: **Streamlit**.
3. Push this repo to the Space (or connect a GitHub repo).
4. In the Space's **Settings → Secrets**, add `GROQ_API_KEY` = your key.
   (Secrets are never committed — `.env` is git-ignored.)
5. The Space builds from `requirements.txt` and runs `app.py` automatically.

---

## Honest limitations & future work

- **Travel** is measured home-base → venue (a displacement/acclimatization proxy), not actual
  in-tournament travel between match cities.
- **Player-level data** (line-ups, injuries, individual ratings) is not freely available across all
  tournaments; team Elo is used as the honest aggregate of player performance, and the chatbot pulls
  live squad/line-up info from the web when asked.
- **Knockout bracket** is not yet drawn (the group stage is ongoing), so the simulator uses a
  randomised knockout draw each run; the real bracket appears in the schedule once published.
- **Group tiebreakers** use points → goal difference → goals scored (the main FIFA criteria);
  head-to-head and fair-play tiebreaks are future work.
- **Updates** are near-real-time (on refresh), gated by the free data source's update frequency; a
  live score feed would need a paid sports API.
- **Sample size** for finals matches is modest (~264), so findings are indicative, not definitive.
- **Future work:** real inter-match travel from base camps + fixtures; squad-strength features; a
  visual live knockout bracket; statistical significance tests for each hypothesis.
