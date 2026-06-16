"""
model.py
========
Trains an XGBoost classifier to predict a match result (home win / draw / away
win) from the environmental + travel features we built in feature_engineer.py.

It does three jobs:
  1. train_model()    -> trains, evaluates (accuracy + report), prints which
                         features matter most, and saves the model to disk.
  2. predict_match()  -> loads the saved model and predicts probabilities for a
                         hypothetical 2026 match at a chosen stadium.

Run it directly to train:
    python -m src.model
"""

import os
import pickle

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# The input columns the model learns from. (We deliberately exclude the score
# and the teams themselves - we want to learn from CONDITIONS, not team names.)
FEATURE_COLS = [
    # --- Team strength / performance history (the biggest signal) ---
    "home_elo",       "away_elo",       "elo_diff",
    "home_form",      "away_form",
    "home_gf",        "home_ga",        "away_gf",        "away_ga",
    # --- Travel + jet lag ---
    "home_travel_km", "away_travel_km", "travel_diff_km",
    "home_tz_shift",  "away_tz_shift",
    # NOTE: rest-day features (home_rest_days/away_rest_days/rest_diff_days) are
    # computed and saved in features.csv for hypothesis H2 analysis, but excluded
    # from the predictor: at World Cup finals every team rests ~equally, so they
    # added noise and did not generalize. (A deliberate feature-selection choice.)
    # --- Venue + climate ---
    "altitude_m",
    "temp_avg",       "humidity",
    "precip_mm",      "wind_kmh",
]

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "data/processed/model.pkl"

# Human-readable names for the 3 classes. We store the result as -1/0/1 but
# XGBoost needs 0/1/2, so we shift by +1 (see below).
CLASS_NAMES = ["Away Win", "Draw", "Home Win"]


def train_model(features_path: str = FEATURES_PATH):
    """Train + evaluate the XGBoost model, then save it. Returns (model, importances)."""
    df = pd.read_csv(features_path).dropna(subset=FEATURE_COLS + ["result"])
    print(f"   Training on {len(df)} fully-featured matches")

    X = df[FEATURE_COLS]
    y = df["result"] + 1          # shift -1/0/1  ->  0/1/2 for XGBoost

    # Hold out 20% of matches the model never sees, to test honestly.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Regularized settings: with only a few hundred matches we keep the trees
    # shallow and subsample rows/columns so the model generalizes instead of
    # memorizing - which also keeps the predicted probabilities realistic.
    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Honest evaluation on the held-out matches ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   Accuracy on unseen matches: {acc:.3f}")
    print("\n   Detailed report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # --- Which factors mattered most? ---
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("   Feature importances (higher = more influence on predictions):")
    print(importances.to_string())

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n   Model saved to {MODEL_PATH}")
    return model, importances


def compare_models(features_path: str = FEATURES_PATH) -> pd.DataFrame:
    """
    Benchmark several models with 5-fold cross-validation so the accuracy estimate
    is robust (averaged over folds) and our choice of XGBoost is justified.
    Returns a tidy comparison table.
    """
    df = pd.read_csv(features_path).dropna(subset=FEATURE_COLS + ["result"])
    X = df[FEATURE_COLS]
    y = df["result"] + 1

    # Our tuned XGBoost, reused inside the ensemble too.
    xgb = XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        min_child_weight=3, eval_metric="mlogloss", random_state=42,
    )
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

    # A full field of model families. "Baseline" is the bar every real model must clear.
    candidates = {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes":              GaussianNB(),
        "K-Nearest Neighbors":      make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "SVM (RBF)":                make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf")),
        "Logistic Regression":      logreg,
        "Random Forest":            rf,
        "Gradient Boosting":        GradientBoostingClassifier(random_state=42),
        "XGBoost (ours)":           xgb,
        "Ensemble (LR+RF+XGB)":     VotingClassifier(
            estimators=[("lr", logreg), ("rf", rf), ("xgb", xgb)], voting="soft",
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        rows.append({
            "model": name,
            "cv_accuracy_mean": round(scores.mean(), 3),
            "cv_accuracy_std": round(scores.std(), 3),
        })

    table = pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False).reset_index(drop=True)
    print("\n   5-fold cross-validated accuracy (mean +/- std):")
    for _, r in table.iterrows():
        print(f"      {r['model']:<26} {r['cv_accuracy_mean']:.3f} +/- {r['cv_accuracy_std']:.3f}")
    return table


def predict_match(home_team: str, away_team: str, venue: str, date_str: str, model=None) -> dict:
    """
    Predict win/draw/loss probabilities for a hypothetical match at a 2026 venue.
    Uses real altitude + travel for the venue; weather falls back to mild defaults
    because the weather archive does not cover future dates.
    """
    # Imported here (not at top) to avoid a circular import with feature modules.
    from src.data_collector import STADIUMS
    from src.travel_calculator import get_travel_distance, get_timezone_shift
    from src.weather_pipeline import get_weather
    from src.team_form import load_latest_strength, START_ELO

    if model is None:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    stadium = next((s for s in STADIUMS if s["name"] == venue), STADIUMS[0])
    lat, lon, alt = stadium["lat"], stadium["lon"], stadium["altitude_m"]

    weather = get_weather(lat, lon, date_str)      # {} for future dates -> defaults below

    home_dist = get_travel_distance(home_team, lat, lon) or 0
    away_dist = get_travel_distance(away_team, lat, lon) or 0

    # Each team's CURRENT strength + form (from the full match history).
    strength = load_latest_strength()
    h = strength.get(home_team, {"elo": START_ELO, "form": 0.5, "gf": 1.2, "ga": 1.2})
    a = strength.get(away_team, {"elo": START_ELO, "form": 0.5, "gf": 1.2, "ga": 1.2})

    features = {
        "home_elo":        h["elo"],
        "away_elo":        a["elo"],
        "elo_diff":        round(h["elo"] - a["elo"], 1),
        "home_form":       h["form"],
        "away_form":       a["form"],
        "home_gf":         h["gf"],
        "home_ga":         h["ga"],
        "away_gf":         a["gf"],
        "away_ga":         a["ga"],
        # Tournament matches are typically ~3-4 days apart; assume equal rest.
        "home_rest_days":  4,
        "away_rest_days":  4,
        "rest_diff_days":  0,
        "home_travel_km":  home_dist,
        "away_travel_km":  away_dist,
        "travel_diff_km":  away_dist - home_dist,
        "home_tz_shift":   get_timezone_shift(home_team, lat, lon) or 0,
        "away_tz_shift":   get_timezone_shift(away_team, lat, lon) or 0,
        "altitude_m":      alt,
        "temp_avg":        weather.get("temp_avg", 22),
        "humidity":        weather.get("humidity", 60),
        "precip_mm":       weather.get("precip_mm", 0),
        "wind_kmh":        weather.get("wind_kmh", 12),
    }

    X = pd.DataFrame([features])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]              # [P(away), P(draw), P(home)]

    return {
        "home_win": round(float(proba[2]) * 100, 1),
        "draw":     round(float(proba[1]) * 100, 1),
        "away_win": round(float(proba[0]) * 100, 1),
        "features": features,
    }


if __name__ == "__main__":
    print("STEP: Training the XGBoost match-outcome model...")
    model, importances = train_model()

    print("\n   Model comparison (why XGBoost?):")
    compare_models()

    print("\n   Example prediction - France (home) vs Brazil (away) at MetLife Stadium:")
    result = predict_match("France", "Brazil", "MetLife Stadium", "2026-07-19", model)
    print(f"      France win: {result['home_win']}%  |  Draw: {result['draw']}%  |  Brazil win: {result['away_win']}%")
