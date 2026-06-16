"""
app.py  -  FIFA 2026 Analytics dashboard (Streamlit)
====================================================
Ties the whole project into one web app with four pages:

  1. Schedule & Predictions - real 2026 fixtures, results, and model predictions
  2. Match Predictor        - pick any two teams + venue, get win/draw/loss odds
  3. Research Findings       - charts that test the hypotheses + model comparison
  4. AI Analyst             - chat with the Groq-powered assistant

Run locally:
    streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_collector import STADIUMS, TEAM_CAPITALS
from src.model import predict_match, compare_models, FEATURES_PATH
from src.team_form import load_latest_strength
from src.schedule import build_schedule_dashboard, SCHEDULE_OUT
from src.simulator import simulate_tournament, compute_group_standings
from src.data_collector import RAW_MATCHES_PATH
from src.analyst import ask_analyst

st.set_page_config(page_title="FIFA 2026 Analytics", page_icon="⚽", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders (so heavy work runs once, not on every click)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading 2026 schedule + predictions...")
def get_schedule():
    return build_schedule_dashboard()


@st.cache_data(show_spinner=False)
def get_features():
    return pd.read_csv(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else pd.DataFrame()


@st.cache_data(show_spinner="Benchmarking models...")
def get_model_comparison():
    return compare_models()


@st.cache_data(show_spinner="Simulating the tournament thousands of times...")
def get_simulation(n_sims):
    return simulate_tournament(n_sims=n_sims)


@st.cache_data(show_spinner="Loading group standings...")
def get_standings():
    return compute_group_standings()


@st.cache_data(show_spinner=False)
def get_strength():
    return load_latest_strength()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚽ FIFA 2026 Analytics")
page = st.sidebar.radio(
    "Go to",
    ["📅 Schedule & Predictions", "📋 Group Standings", "🔮 Match Predictor",
     "🏆 Tournament Simulator", "📊 Research Findings", "🤖 AI Analyst"],
)
st.sidebar.divider()
if st.sidebar.button("🔄 Refresh latest results"):
    # Pull the newest scores: drop cached data + the downloaded match file, then rerun.
    for path in (RAW_MATCHES_PATH, SCHEDULE_OUT):
        if os.path.exists(path):
            os.remove(path)
    st.cache_data.clear()
    st.rerun()
st.sidebar.divider()
st.sidebar.metric("Teams", "48")
st.sidebar.metric("Matches", "104")
st.sidebar.metric("Model accuracy (CV)", "~51-53%")
st.sidebar.caption("Predicts outcomes from team strength, travel, climate & altitude.")


# --------------------------------------------------------------------------- #
# Page 1 - Schedule & Predictions
# --------------------------------------------------------------------------- #
if page == "📅 Schedule & Predictions":
    st.title("📅 FIFA 2026 — Schedule & Predictions")
    st.caption("Real fixtures. Actual results for played games; model predictions for upcoming ones. "
               "Dates shown are the match calendar date (same date in Bangladesh, UTC+6).")

    sched = get_schedule()
    played = sched[sched["status"] == "Played"]
    upcoming = sched[sched["status"] == "Upcoming"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total fixtures", len(sched))
    c2.metric("Played", len(played))
    c3.metric("Upcoming", len(upcoming))

    view = st.radio("Show", ["Upcoming (predictions)", "Played (results)", "All"], horizontal=True)
    teams = sorted(set(sched["home_team"]) | set(sched["away_team"]))
    team_filter = st.selectbox("Filter by team (optional)", ["All teams"] + teams)

    if view == "Upcoming (predictions)":
        data = upcoming
    elif view == "Played (results)":
        data = played
    else:
        data = sched

    if team_filter != "All teams":
        data = data[(data["home_team"] == team_filter) | (data["away_team"] == team_filter)]

    if view == "Played (results)":
        st.dataframe(
            data[["date", "home_team", "away_team", "venue", "score", "actual"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.dataframe(
            data[["date", "home_team", "away_team", "venue", "status",
                  "pred_home_%", "pred_draw_%", "pred_away_%", "score", "actual"]],
            use_container_width=True, hide_index=True,
        )
    st.caption("pred_home/draw/away_% = model's predicted probabilities for the match outcome.")


# --------------------------------------------------------------------------- #
# Page 2 - Group Standings
# --------------------------------------------------------------------------- #
elif page == "📋 Group Standings":
    st.title("📋 Group Standings")
    st.caption("Live group tables built from real results. Top 2 of each group (green) qualify; "
               "the 8 best 3rd-placed teams also advance. Use 'Refresh latest results' (sidebar) for newest scores.")

    standings = get_standings()
    st.info("Knockout bracket (Round of 32 → Final) appears once the group stage finishes — "
            "those matchups are decided by these results. Meanwhile, see the Tournament Simulator for title odds.")

    # Show groups two per row.
    for i in range(0, len(standings), 2):
        cols = st.columns(2)
        for col, (label, table, fixtures) in zip(cols, standings[i:i + 2]):
            with col:
                st.subheader(label)
                styled = table.style.apply(
                    lambda row: ["background-color: #1b5e20; color: white" if row.name <= 2 else "" for _ in row],
                    axis=1,
                )
                st.dataframe(styled, use_container_width=True)
                with st.expander("Matches"):
                    st.dataframe(fixtures, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page 3 - Match Predictor
# --------------------------------------------------------------------------- #
elif page == "🔮 Match Predictor":
    st.title("🔮 Match Predictor")
    st.caption("Pick any two teams and a venue to see the model's win / draw / loss probabilities.")

    teams = sorted(TEAM_CAPITALS.keys())
    venues = [s["name"] for s in STADIUMS]

    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Home team", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
        venue = st.selectbox("Venue", venues)
    with col2:
        away = st.selectbox("Away team", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)
        date_str = st.date_input("Match date").strftime("%Y-%m-%d")

    if st.button("Predict", type="primary"):
        if home == away:
            st.warning("Pick two different teams.")
        else:
            res = predict_match(home, away, venue, date_str)
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{home} win", f"{res['home_win']}%")
            m2.metric("Draw", f"{res['draw']}%")
            m3.metric(f"{away} win", f"{res['away_win']}%")

            chart_df = pd.DataFrame({
                "Outcome": [f"{home} win", "Draw", f"{away} win"],
                "Probability": [res["home_win"], res["draw"], res["away_win"]],
            })
            fig = px.bar(chart_df, x="Outcome", y="Probability", color="Outcome",
                         title="Predicted outcome probabilities (%)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Factors used in this prediction"):
                st.json(res["features"])


# --------------------------------------------------------------------------- #
# Page 3 - Tournament Simulator
# --------------------------------------------------------------------------- #
elif page == "🏆 Tournament Simulator":
    st.title("🏆 Tournament Simulator — Who Wins the Cup?")
    st.caption("Monte-Carlo: plays the whole tournament thousands of times using team strength (Elo), "
               "respecting games already played. Groups are real; the knockout draw is randomised per run.")

    n_sims = st.select_slider("Number of simulations", options=[500, 1000, 2000, 3000, 5000], value=2000)
    sim = get_simulation(n_sims)

    top = sim.head(12)
    fig = px.bar(top.sort_values("champion_%"), x="champion_%", y="team", orientation="h",
                 title=f"Title odds (top 12 of {len(sim)} teams, {n_sims:,} simulations)",
                 color="champion_%", color_continuous_scale="Greens",
                 labels={"champion_%": "Chance of winning the cup (%)", "team": ""})
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=460)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full odds by stage")
    st.caption("win_group_% · qualify_% · semi_% (reach semis) · final_% (reach final) · champion_%")
    st.dataframe(sim, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page 4 - Research Findings
# --------------------------------------------------------------------------- #
elif page == "📊 Research Findings":
    st.title("📊 Research Findings")
    st.caption("Testing whether environmental factors affect match outcomes (2010–2026 World Cup finals).")

    df = get_features()
    if df.empty:
        st.info("Run the pipeline first (python -m src.feature_engineer).")
    else:
        st.subheader("Team strength — top 12 by our Elo rating")
        strength = get_strength()
        if strength:
            top = sorted(strength.items(), key=lambda kv: kv[1]["elo"], reverse=True)[:12]
            elo_df = pd.DataFrame([{"Team": t, "Elo": s["elo"]} for t, s in top])
            fig0 = px.bar(elo_df.sort_values("Elo"), x="Elo", y="Team", orientation="h",
                          color="Elo", color_continuous_scale="Blues")
            fig0.update_layout(showlegend=False, coloraxis_showscale=False, height=420)
            st.plotly_chart(fig0, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("H3 — Humidity vs total goals")
            d3 = df.dropna(subset=["humidity", "total_goals"])
            fig3 = px.scatter(d3, x="humidity", y="total_goals", trendline="ols",
                              labels={"humidity": "Humidity (%)", "total_goals": "Goals in match"})
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            st.subheader("H4 — Altitude vs total goals")
            d4 = df.dropna(subset=["altitude_m", "total_goals"]).copy()
            d4["altitude_band"] = pd.cut(d4["altitude_m"], bins=[-1, 500, 1500, 4000],
                                         labels=["Low (<500m)", "Mid (500-1500m)", "High (>1500m)"])
            fig4 = px.box(d4, x="altitude_band", y="total_goals",
                          labels={"altitude_band": "Altitude band", "total_goals": "Goals in match"})
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("H1 — Travel difference vs result")
        d1 = df.dropna(subset=["travel_diff_km", "result"])
        fig1 = px.scatter(d1, x="travel_diff_km", y="result", trendline="ols",
                          labels={"travel_diff_km": "Away travel − Home travel (km)",
                                  "result": "Result (−1 away win, 0 draw, 1 home win)"})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Model comparison (5-fold cross-validation)")
        st.caption("Why XGBoost? All learned models beat the baseline; the top models are statistically tied.")
        st.dataframe(get_model_comparison(), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Page 4 - AI Analyst
# --------------------------------------------------------------------------- #
elif page == "🤖 AI Analyst":
    st.title("🤖 AI Analyst")
    st.caption("Ask about the tournament, predictions, or findings. Powered by Groq · Llama 4 Scout, "
               "grounded in this project's real data.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. Who are the favourites, and does altitude matter?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = ask_analyst(prompt, st.session_state.messages[:-1])
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
