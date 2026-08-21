#!/usr/bin/env python3
"""
app.py -- Streamlit front-end for football_predictor.py

Run locally with:
    streamlit run app.py

WEEKLY WORKFLOW THIS IS BUILT AROUND
-------------------------------------
1. The model (10 seasons x however many divisions, pooled Dixon-Coles fit) is
   the expensive part -- downloading + fitting can take a few minutes. It is
   NOT refit on every page load. It's cached in `st.session_state` for the
   running session, and also pickled to disk (`model_cache.pkl`) so restarting
   the app doesn't lose it. You only need to click "Fit / refresh model" in
   the sidebar when you actually want fresh data folded in (e.g. once a
   week, after the latest results are in).
2. The cheap, weekly part is the "This week's fixtures" tab: upload a small
   CSV (or type games directly into the table) and get predictions in
   seconds, using whatever model is currently cached -- no retraining needed.

football_predictor.py must be in the same folder as this file.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from football_predictor import (
    DixonColes, load_multi_division_historical, ALL_DIVISIONS, DIVISION_NAMES,
    predict_fixtures, resolve_team_name,
    simulate_season, position_probability_table, style_position_table,
    simulate_cup,
)

MODEL_CACHE_PATH = "model_cache.pkl"


def _rerun():
    """Compatibility shim: st.rerun() is the modern name, older Streamlit used st.experimental_rerun()."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def save_model_cache(model, meta):
    try:
        with open(MODEL_CACHE_PATH, "wb") as f:
            pickle.dump({"model": model, "meta": meta}, f)
    except Exception as e:
        st.sidebar.warning(f"Could not save model cache to disk: {e}")


def load_model_cache():
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            with open(MODEL_CACHE_PATH, "rb") as f:
                d = pickle.load(f)
            return d.get("model"), d.get("meta")
        except Exception:
            return None, None
    return None, None


st.set_page_config(page_title="Football Predictor", page_icon="\u26bd", layout="wide")
st.title("\u26bd Football Predictor")
st.caption("Pooled Dixon-Coles model across whichever English divisions you train it on.")

# ---------------------------------------------------------------------------
# Sidebar: model setup / weekly refresh
# ---------------------------------------------------------------------------
st.sidebar.header("Model")

divisions = st.sidebar.multiselect(
    "Divisions to include",
    ALL_DIVISIONS,
    default=ALL_DIVISIONS,
    format_func=lambda d: f"{d} -- {DIVISION_NAMES.get(d, d)}",
)
n_seasons = st.sidebar.number_input("Seasons of history", min_value=1, max_value=15, value=10, step=1)

if "model" not in st.session_state:
    cached_model, cached_meta = load_model_cache()
    st.session_state["model"] = cached_model
    st.session_state["meta"] = cached_meta

refresh_clicked = st.sidebar.button(
    "\U0001f504 Fit / refresh model", type="primary",
    help="Downloads results and refits the model. Do this weekly (or whenever you want the "
         "latest results folded in) -- not on every page load.",
)
if refresh_clicked:
    if not divisions:
        st.sidebar.error("Select at least one division.")
    else:
        with st.spinner(f"Downloading {int(n_seasons)} season(s) x {len(divisions)} division(s) "
                         "and fitting the model -- this can take a few minutes..."):
            data = load_multi_division_historical(list(divisions), int(n_seasons))
            model = DixonColes().fit(data)
            meta = {
                "fitted_at": datetime.now(),
                "divisions": list(divisions),
                "n_seasons": int(n_seasons),
                "n_matches": len(data),
                "n_teams": len(model.teams),
            }
        st.session_state["model"] = model
        st.session_state["meta"] = meta
        save_model_cache(model, meta)
        st.sidebar.success("Model refreshed.")

model = st.session_state.get("model")
meta = st.session_state.get("meta")

if model is None:
    st.info(
        "\U0001f448 Click **Fit / refresh model** in the sidebar to get started. The first run "
        "downloads results and fits the model, which can take a few minutes; after that, "
        "predictions and simulations reuse the cached model instantly."
    )
    st.stop()

st.sidebar.caption(
    f"Model fitted: {meta['fitted_at']:%Y-%m-%d %H:%M}  \n"
    f"{meta['n_matches']:,} matches, {meta['n_teams']} teams  \n"
    f"Divisions: {', '.join(meta['divisions'])}"
)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_fixtures, tab_season, tab_cup, tab_teams, tab_lsc = st.tabs(
    ["\U0001f4c5 This week's fixtures", "\U0001f3c6 Season simulation",
     "\U0001f3af Cup simulator", "\U0001f465 Teams", "\U0001f4ca League strength"]
)

# --- This week's fixtures (the weekly-update tab) ---------------------------
with tab_fixtures:
    st.subheader("Predict this week's fixtures")
    st.caption(
        "This is the part you update manually each week: upload a CSV with `HomeTeam,AwayTeam` "
        "columns for the upcoming matches, or edit the table below directly. Team names must "
        "match football-data.co.uk's spelling -- check the Teams tab if unsure."
    )

    uploaded = st.file_uploader("Upload this week's fixtures CSV", type="csv", key="fixtures_csv")
    if uploaded is not None:
        fixtures_input = pd.read_csv(uploaded)
    elif "fixtures_editor_data" in st.session_state:
        fixtures_input = st.session_state["fixtures_editor_data"]
    else:
        fixtures_input = pd.DataFrame({"HomeTeam": [""] * 5, "AwayTeam": [""] * 5})

    fixtures_input = st.data_editor(fixtures_input, num_rows="dynamic", key="fixtures_editor",
                                     use_container_width=True)
    st.session_state["fixtures_editor_data"] = fixtures_input

    if st.button("Predict fixtures", type="primary"):
        rows = fixtures_input.dropna(subset=["HomeTeam", "AwayTeam"])
        rows = rows[(rows["HomeTeam"].astype(str).str.strip() != "") &
                    (rows["AwayTeam"].astype(str).str.strip() != "")]
        if len(rows) == 0:
            st.warning("Add at least one fixture (HomeTeam, AwayTeam) first.")
        else:
            with st.spinner("Predicting..."):
                summary = predict_fixtures(model, rows, verbose=False)
            st.session_state["last_prediction_summary"] = summary

    summary = st.session_state.get("last_prediction_summary")
    if summary is not None:
        if "Error" in summary.columns:
            errors = summary[summary["Error"].notna()]
            ok_rows = summary[summary["Error"].isna()]
        else:
            errors = pd.DataFrame()
            ok_rows = summary

        if len(errors):
            st.warning(f"{len(errors)} fixture(s) skipped -- unrecognized team name(s):")
            st.dataframe(errors[["HomeTeam", "AwayTeam", "Error"]], use_container_width=True)

        if len(ok_rows):
            st.dataframe(ok_rows, use_container_width=True)
            st.download_button("Download predictions CSV", ok_rows.to_csv(index=False),
                                "predictions.csv", "text/csv")

            st.markdown("#### Match detail")
            for _, r in ok_rows.iterrows():
                with st.expander(f"{r['HomeTeam']} vs {r['AwayTeam']} -- predicted "
                                  f"{r['PredictedResult']} ({r['Confidence%']:.0f}% confidence)"):
                    pred = model.predict_match(r["HomeTeam"], r["AwayTeam"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Home win", f"{pred['P(H)']*100:.1f}%")
                    c2.metric("Draw", f"{pred['P(D)']*100:.1f}%")
                    c3.metric("Away win", f"{pred['P(A)']*100:.1f}%")
                    st.write(f"Expected goals: **{pred['home_team']}** {pred['lambda_home']:.2f} - "
                             f"{pred['lambda_away']:.2f} **{pred['away_team']}**")
                    if pred["cross_division"]:
                        st.caption("Cross-division tie.")
                    st.write("Top 5 most probable scorelines:")
                    score_df = pd.DataFrame(pred["top_scorelines"])
                    score_df["prob"] = (score_df["prob"] * 100).round(1)
                    score_df = score_df.rename(columns={"score": "Score", "prob": "Probability %"})
                    st.table(score_df)

# --- Season simulation --------------------------------------------------
with tab_season:
    st.subheader("Full season Monte Carlo simulation")
    col1, col2 = st.columns(2)
    sim_division = col1.selectbox("Division to simulate", model.divisions,
                                   format_func=lambda d: f"{d} -- {DIVISION_NAMES.get(d, d)}")
    n_sims = col2.number_input("Number of simulations", min_value=100, max_value=100000,
                                value=10000, step=1000)

    st.caption(
        "football-data.co.uk doesn't publish a full future schedule, so upload the FULL remaining "
        "fixture list yourself (two columns: HomeTeam, AwayTeam). Optionally also upload results "
        "already played this season (HomeTeam, AwayTeam, FTHG, FTAG) -- if omitted, the table "
        "starts from zero."
    )
    fixtures_file = st.file_uploader("Remaining fixtures CSV", type="csv", key="season_fixtures_csv")
    played_file = st.file_uploader("Already-played results CSV (optional)", type="csv",
                                    key="season_played_csv")

    if st.button("Run season simulation", type="primary"):
        if fixtures_file is None:
            st.warning("Upload a remaining-fixtures CSV first.")
        else:
            fixtures_df = pd.read_csv(fixtures_file)
            played_df = pd.read_csv(played_file) if played_file is not None else \
                pd.DataFrame(columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            with st.spinner(f"Running {int(n_sims):,} simulations..."):
                table = simulate_season(model, played_df, fixtures_df, division=sim_division,
                                         n_sims=int(n_sims))
            st.session_state["last_season_table"] = table
            st.session_state["last_season_division"] = sim_division

    table = st.session_state.get("last_season_table")
    if table is not None:
        st.markdown(f"#### Projected {DIVISION_NAMES.get(st.session_state['last_season_division'], '')} table")
        st.dataframe(table, use_container_width=True)
        st.download_button("Download table CSV", table.to_csv(), "season_table.csv", "text/csv")

        st.markdown("#### Finishing-position probability grid")
        euro_spots = st.slider("European/qualification-zone columns to highlight", 0, 10, 5)
        releg_spots = st.slider("Relegation-zone columns to highlight", 0, 10, 3)
        styler = style_position_table(table, european_spots=euro_spots, relegation_spots=releg_spots)
        st.markdown(styler.to_html(), unsafe_allow_html=True)

        pos_grid = position_probability_table(table)
        st.download_button("Download position grid CSV", pos_grid.to_csv(index=False),
                            "position_grid.csv", "text/csv")

# --- Cup simulator --------------------------------------------------------
with tab_cup:
    st.subheader("Knockout cup simulator")
    st.caption(
        "Upload round-1 ties (HomeTeam,AwayTeam columns, power-of-2 rows -- e.g. 8, 16, 32, 64). "
        "Works across divisions since all team ratings sit on one shared scale."
    )
    cup_n_sims = st.number_input("Number of simulations", min_value=100, max_value=100000,
                                  value=10000, step=1000, key="cup_n_sims")
    cup_file = st.file_uploader("Round-1 ties CSV", type="csv", key="cup_fixtures_csv")

    if st.button("Run cup simulation", type="primary"):
        if cup_file is None:
            st.warning("Upload a round-1 ties CSV first.")
        else:
            ties_df = pd.read_csv(cup_file)
            round1_ties = list(ties_df[["HomeTeam", "AwayTeam"]].itertuples(index=False, name=None))
            try:
                with st.spinner(f"Simulating {len(round1_ties)}-tie bracket, {int(cup_n_sims):,} runs..."):
                    cup_result = simulate_cup(model, round1_ties, n_sims=int(cup_n_sims))
                st.session_state["last_cup_result"] = cup_result
            except ValueError as e:
                st.error(str(e))

    cup_result = st.session_state.get("last_cup_result")
    if cup_result is not None:
        st.dataframe(cup_result, use_container_width=True)
        st.download_button("Download cup results CSV", cup_result.to_csv(), "cup_simulation.csv", "text/csv")

# --- Teams / diagnostics --------------------------------------------------
with tab_teams:
    st.subheader("All teams in the fitted model")
    division_filter = st.selectbox(
        "Filter by division", ["All"] + list(model.divisions),
        format_func=lambda d: d if d == "All" else f"{d} -- {DIVISION_NAMES.get(d, d)}",
    )
    teams_df = model.list_teams(division=None if division_filter == "All" else division_filter)
    st.dataframe(teams_df, use_container_width=True)
    st.download_button("Download team list CSV", teams_df.to_csv(index=False), "teams.csv", "text/csv")

    st.markdown("#### Possible spelling duplicates")
    st.caption(
        "If a real club shows up here as two 'different' teams, standardise the spelling in your "
        "source data and refit, rather than overriding."
    )
    sim_df = model.find_similar_team_names()
    if len(sim_df) == 0:
        st.success("No suspicious near-duplicate names found.")
    else:
        st.dataframe(sim_df, use_container_width=True)

    st.markdown("#### Manually fix a team's division")
    st.caption(
        "Use this right after a real-world promotion/relegation is confirmed but before results "
        "in the new division have been recorded yet."
    )
    c1, c2, c3 = st.columns([2, 2, 1])
    team_to_fix = c1.selectbox("Team", model.teams, key="override_team")
    new_div = c2.selectbox("Correct division", model.divisions,
                            format_func=lambda d: f"{d} -- {DIVISION_NAMES.get(d, d)}", key="override_div")
    if c3.button("Apply"):
        model.override_team_division({team_to_fix: new_div})
        save_model_cache(model, meta)
        st.success(f"{team_to_fix} -> {DIVISION_NAMES.get(new_div, new_div)}")
        _rerun()

# --- League strength coefficients ----------------------------------------
with tab_lsc:
    st.subheader("League Strength Coefficients")
    st.caption(
        "How strong is a typical team in each division, on the pooled common scale -- derived "
        "from teams that were promoted/relegated across divisions, which link the rating scales "
        "together."
    )
    lsc = model.league_strength_coefficients()
    st.dataframe(lsc, use_container_width=True)

    for dv in model.divisions:
        st.write(
            f"**{DIVISION_NAMES.get(dv, dv)}**: home advantage {model.params['home_adv'][dv]:+.3f}, "
            f"tempo {model.params['tempo'][dv]:+.3f}"
        )
