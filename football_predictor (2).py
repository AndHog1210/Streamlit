#!/usr/bin/env python3
"""
football_predictor.py

Predicts football match outcomes (H/D/A) and most likely scorelines, simulates
a full league season, and simulates knockout cup competitions -- all from one
Dixon-Coles model fitted JOINTLY across multiple divisions.

WHY A JOINT (POOLED) MODEL?
----------------------------
If you fit a separate model per division, a team's attack/defense rating is
only ever compared against teams in its own division -- there is no way to
know whether "a good League One team" is stronger or weaker than "a bad
Championship team". But teams get promoted and relegated every season, so if
you pool ALL divisions' results together and give every team ONE rating that
is shared across whichever division(s) it played in, those promoted/relegated
teams act as "bridges" that link the rating scales of adjacent divisions
together on a single common axis. A team that went Championship -> Premier
League -> Championship contributes match data to both scales, which lets the
model triangulate how much weaker/stronger the divisions are relative to each
other. This is the same trick used to link chess rating pools or cross-region
Elo systems.

On top of the pooled team ratings, the model also fits a per-division
"tempo" term (average goals environment of that division) and a per-division
home-advantage term, and afterwards derives an explicit, human-readable
LEAGUE STRENGTH COEFFICIENT (LSC) per division -- basically "how strong is an
average team in this division, on the common pooled scale". This is reported
directly, and is also what makes it possible to sensibly predict/simulate a
cross-division match, e.g. a cup tie between a Premier League club and a
League Two club.

Data source: https://www.football-data.co.uk/englandm.php
  - Historical season files:  https://www.football-data.co.uk/mmz4281/<season>/<DIV>.csv
    e.g. season 2024/25 -> code "2425", so E0 2024/25 is .../mmz4281/2425/E0.csv
  - Near-term upcoming fixtures (all leagues, only a few weeks ahead):
    https://www.football-data.co.uk/fixtures.csv

Divisions (England): E0=Premier League, E1=Championship, E2=League One, E3=League Two

USAGE EXAMPLES
--------------
# Pull 10 seasons of all four divisions and fit the pooled model
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --show-lsc

# List every team available in the fitted model (exact spellings for --predict etc.)
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --list-teams

# Predict one match (works even cross-division, e.g. a cup tie)
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --predict "Arsenal,Wrexham"

# Predict a whole file of fixtures at once (e.g. this week's games)
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --predict-fixtures this_weeks_games.csv

# Simulate the rest of the Premier League season 10,000 times
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --simulate --division E0 --fixtures pl_fixtures.csv --sims 10000
# (this also prints/saves the full exact-finishing-position probability grid,
# not just the summary table -- see position_probability_table() / style_position_table())

# Simulate a knockout cup bracket (e.g. FA Cup 3rd round) 10,000 times
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --cup --cup-fixtures fa_cup_r3.csv --sims 10000

Requires: pandas, numpy, scipy, requests, matplotlib (matplotlib only needed for
the optional style_position_table() heatmap display)
"""

import argparse
import difflib
import io
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

warnings.filterwarnings("ignore")

BASE_URL = "https://www.football-data.co.uk"
FIXTURES_URL = f"{BASE_URL}/fixtures.csv"
ALL_DIVISIONS = ["E0", "E1", "E2", "E3"]
DIVISION_NAMES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def season_code(start_year: int) -> str:
    """2024 -> '2425' (2024/25 season file naming used by the site)."""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def fetch_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def infer_current_start_year() -> int:
    today = datetime.today()
    return today.year if today.month >= 7 else today.year - 1


def load_division_seasons(division: str, n_seasons: int, current_start_year: int = None) -> pd.DataFrame:
    """Download and concatenate the last n_seasons of results for one division."""
    if current_start_year is None:
        current_start_year = infer_current_start_year()

    frames = []
    for i in range(n_seasons):
        start_year = current_start_year - i
        code = season_code(start_year)
        url = f"{BASE_URL}/mmz4281/{code}/{division}.csv"
        try:
            df = fetch_csv(url)
        except Exception as e:
            print(f"  (skipping {url}: {e})", file=sys.stderr)
            continue
        df["SeasonStartYear"] = start_year
        df["Division"] = division
        frames.append(df)
        print(f"  loaded {len(df)} matches from {code}/{division}.csv")

    if not frames:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                                      "FTR", "SeasonStartYear", "Division"])
    return pd.concat(frames, ignore_index=True, sort=False)


def load_multi_division_historical(divisions, n_seasons: int, current_start_year: int = None) -> pd.DataFrame:
    """Download and pool results across several divisions/seasons into one dataframe."""
    frames = [load_division_seasons(d, n_seasons, current_start_year) for d in divisions]
    data = pd.concat(frames, ignore_index=True, sort=False)

    keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "SeasonStartYear", "Division"]
    data = data[[c for c in keep if c in data.columns]].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    data["FTHG"] = data["FTHG"].astype(int)
    data["FTAG"] = data["FTAG"].astype(int)
    return data


def load_fixtures_from_site(division: str) -> pd.DataFrame:
    """
    football-data.co.uk's fixtures.csv covers ALL leagues but typically only the
    next couple of weeks of matches, not a whole remaining season. Useful for
    near-term predictions; for a true full-season simulation, supply your own
    fixture list via --fixtures.
    """
    df = fetch_csv(FIXTURES_URL)
    df = df[df["Div"] == division][["Date", "HomeTeam", "AwayTeam"]].dropna()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pooled, multi-division Dixon-Coles model
# ---------------------------------------------------------------------------

class DixonColes:
    """
    Fits ONE shared attack/defense rating per team across all divisions and
    seasons supplied, plus a per-division "tempo" (goal environment) term and
    a per-division home-advantage term. Team ratings are therefore on a single
    common scale, linked via any teams that appear in more than one division
    (i.e. promoted/relegated teams) -- which is what makes cross-division
    (e.g. cup) predictions meaningful.
    """

    def __init__(self, xi: float = 0.0012):
        self.xi = xi  # time decay rate (per day)
        self.teams = None
        self.divisions = None
        self.params = None
        self.team_division = None    # team -> most recently played division
        self.team_div_weight = None  # (team, division) -> summed time-decay weight in that division
        self._fit_success = None

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def fit(self, data: pd.DataFrame):
        if "Division" not in data.columns:
            data = data.copy()
            data["Division"] = "ALL"

        self.teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
        self.divisions = sorted(data["Division"].unique())
        n = len(self.teams)
        d = len(self.divisions)
        idx = {t: i for i, t in enumerate(self.teams)}
        didx = {dv: i for i, dv in enumerate(self.divisions)}

        max_date = data["Date"].max()
        weights = np.exp(-self.xi * (max_date - data["Date"]).dt.days.values)

        home_idx = data["HomeTeam"].map(idx).values
        away_idx = data["AwayTeam"].map(idx).values
        div_idx = data["Division"].map(didx).values
        hg = data["FTHG"].values
        ag = data["FTAG"].values

        # param vector: [attack(n), defense(n), home_adv(d), tempo(d), rho]
        x0 = np.concatenate([np.zeros(n), np.zeros(n), np.full(d, 0.2), np.zeros(d), [-0.05]])

        def unpack(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n:2 * n + d]
            tempo = params[2 * n + d:2 * n + 2 * d]
            rho = params[2 * n + 2 * d]
            return attack, defense, home_adv, tempo, rho

        def neg_log_lik(params):
            attack, defense, home_adv, tempo, rho = unpack(params)
            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv[div_idx] + tempo[div_idx])
            mu = np.exp(attack[away_idx] + defense[home_idx] + tempo[div_idx])

            ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
            tau_adj = np.ones_like(ll)
            low_mask = (hg <= 1) & (ag <= 1)
            if low_mask.any():
                tau_vals = np.array([
                    self._tau(x, y, l, m, rho)
                    for x, y, l, m in zip(hg[low_mask], ag[low_mask], lam[low_mask], mu[low_mask])
                ])
                tau_vals = np.clip(tau_vals, 1e-10, None)
                tau_adj[low_mask] = tau_vals
            ll = ll + np.log(tau_adj)

            penalty = 1000 * (attack.mean()) ** 2  # soft-enforce sum(attack)=0 for identifiability
            return -np.sum(weights * ll) + penalty

        res = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-8})
        attack, defense, home_adv, tempo, rho = unpack(res.x)
        attack = attack - attack.mean()

        self.params = {
            "attack": dict(zip(self.teams, attack)),
            "defense": dict(zip(self.teams, defense)),
            "home_adv": dict(zip(self.divisions, home_adv)),
            "tempo": dict(zip(self.divisions, tempo)),
            "rho": rho,
        }
        self._fit_success = res.success

        # Per-team "most recent division" (for defaulting cup-match context),
        # and per-(team,division) accumulated weight (for the LSC calc below).
        last_div = data.sort_values("Date").groupby("HomeTeam")["Division"].last().to_dict()
        last_div_away = data.sort_values("Date").groupby("AwayTeam")["Division"].last().to_dict()
        last_date_home = data.groupby("HomeTeam")["Date"].max()
        last_date_away = data.groupby("AwayTeam")["Date"].max()
        self.team_division = {}
        for t in self.teams:
            dh = last_date_home.get(t)
            da = last_date_away.get(t)
            if dh is not None and (da is None or dh >= da):
                self.team_division[t] = last_div.get(t)
            else:
                self.team_division[t] = last_div_away.get(t)

        w_home = pd.DataFrame({"team": data["HomeTeam"], "division": data["Division"], "w": weights})
        w_away = pd.DataFrame({"team": data["AwayTeam"], "division": data["Division"], "w": weights})
        w_all = pd.concat([w_home, w_away], ignore_index=True)
        self.team_div_weight = w_all.groupby(["team", "division"])["w"].sum().to_dict()

        return self

    # -- League Strength Coefficient -----------------------------------

    def league_strength_coefficients(self) -> pd.DataFrame:
        """
        For each division, compute the time-decay-weighted average team
        "quality" (attack - defense, on the pooled global scale) of teams
        while they were playing in that division. This is the explicit,
        human-readable coefficient: how strong is a typical team in this
        division, relative to the others, on one common axis.
        """
        rows = []
        for dv in self.divisions:
            num, den = 0.0, 0.0
            for t in self.teams:
                w = self.team_div_weight.get((t, dv), 0.0)
                if w <= 0:
                    continue
                quality = self.params["attack"][t] - self.params["defense"][t]
                num += w * quality
                den += w
            avg_quality = num / den if den > 0 else np.nan
            rows.append({"Division": dv, "League": DIVISION_NAMES.get(dv, dv), "AvgTeamQuality": avg_quality})

        df = pd.DataFrame(rows).sort_values("AvgTeamQuality", ascending=False).reset_index(drop=True)
        best = df["AvgTeamQuality"].max()
        # Relative strength index: exp() converts the log-scale quality gap into
        # a goals-ratio-like multiplier, anchored at 1.00 for the strongest division.
        df["RelativeStrengthIndex"] = np.exp(df["AvgTeamQuality"] - best)
        return df

    # -- Team listing -------------------------------------------------------

    def list_teams(self, division=None) -> pd.DataFrame:
        """
        List every team available in the fitted model -- i.e. every team you
        can pass into predict_match / score_matrix / simulate_cup. Team names
        must match football-data.co.uk's exact spelling.

        Returns one row per team with:
          - CurrentDivision / League: the most recently played division
          - DivisionsPlayed: every division the team appeared in within the
            training window (promoted/relegated teams will show more than one)
          - Quality: the fitted attack-minus-defense rating on the pooled
            scale (higher = stronger), useful for a quick sanity-sort

        Pass `division` (e.g. "E0") to filter to teams currently in that division.
        """
        rows = []
        for t in self.teams:
            divs_played = sorted({dv for (tm, dv) in self.team_div_weight.keys() if tm == t})
            current = self.team_division.get(t)
            rows.append({
                "Team": t,
                "CurrentDivision": current,
                "League": DIVISION_NAMES.get(current, current),
                "DivisionsPlayed": ", ".join(divs_played),
                "Quality": self.params["attack"][t] - self.params["defense"][t],
            })
        df = pd.DataFrame(rows).sort_values(["CurrentDivision", "Team"]).reset_index(drop=True)
        if division is not None:
            df = df[df["CurrentDivision"] == division].reset_index(drop=True)
        return df

    def override_team_division(self, overrides: dict):
        """
        Manually correct the "current division" label for specific teams.

        The most common reason a team ends up mislabeled is timing, not a
        bug: `team_division` is set to whichever division a team's MOST
        RECENT recorded match was in. If a team was promoted/relegated for
        the upcoming season but that season's fixtures haven't been played
        (or football-data.co.uk hasn't posted the new season file) yet, the
        model still shows last season's division, since that's the most
        recent *result* on record. Use this to correct that manually once
        you know the real current division, e.g. right after the close
        season's promotions/relegations are confirmed:

            model.override_team_division({
                "Leicester": "E0",   # promoted back to the Prem
                "Southampton": "E1", # relegated
            })

        This updates list_teams(), and the default division context used by
        predict_match / simulate_cup when you don't pass one explicitly. It
        does NOT change the team's fitted attack/defense rating or which
        divisions count towards its DivisionsPlayed / bridging history.
        """
        for team, div in overrides.items():
            if team not in self.teams:
                raise ValueError(f'Unknown team "{team}" -- check model.list_teams() for exact spelling.')
            if div not in self.divisions:
                raise ValueError(f'Unknown division "{div}" -- must be one of {self.divisions}.')
            self.team_division[team] = div
        return self

    def find_similar_team_names(self, cutoff: float = 0.75) -> pd.DataFrame:
        """
        Flags pairs of team names in the model that are suspiciously similar
        in spelling (e.g. "Nott'm Forest" vs "Nottingham Forest", "Leeds" vs
        "Leeds United"). This is the OTHER common cause of "wrong division"
        complaints: if the same real club is spelled two different ways
        across divisions/seasons in the source data, the model silently
        treats them as two different teams -- each with its own (partial,
        and therefore possibly stale-looking) division history. Anything
        flagged here is worth checking in model.list_teams() and, if it is
        indeed a duplicate, standardising the spelling in your source data
        before refitting.

        Flags a pair if EITHER: (a) overall spelling similarity >= cutoff, or
        (b) one name's words are a subset of the other's (catches
        "Leeds"/"Leeds United"-style cases that plain string similarity
        misses).
        """
        pairs = []
        teams = self.teams
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                l1, l2 = t1.lower(), t2.lower()
                ratio = difflib.SequenceMatcher(None, l1, l2).ratio()
                words1, words2 = set(l1.split()), set(l2.split())
                token_subset = words1 and words2 and (words1 <= words2 or words2 <= words1)
                if ratio >= cutoff or token_subset:
                    pairs.append({"Team1": t1, "Team2": t2, "Similarity": round(ratio, 3),
                                  "Team1Division": self.team_division.get(t1),
                                  "Team2Division": self.team_division.get(t2)})
        if not pairs:
            return pd.DataFrame(columns=["Team1", "Team2", "Similarity", "Team1Division", "Team2Division"])
        return pd.DataFrame(pairs).sort_values("Similarity", ascending=False).reset_index(drop=True)



    def team_lambdas(self, home_team, away_team, division=None, neutral=False):
        p = self.params
        missing = [t for t in (home_team, away_team) if t not in p["attack"]]
        if missing:
            raise ValueError(f"Unknown team(s) not in training data: {missing}")

        if division is None:
            division = self.team_division.get(home_team) or self.divisions[0]
        if division not in self.divisions:
            division = self.divisions[0]

        home_adv = 0.0 if neutral else p["home_adv"][division]
        tempo = p["tempo"][division]
        lam = np.exp(p["attack"][home_team] + p["defense"][away_team] + home_adv + tempo)
        mu = np.exp(p["attack"][away_team] + p["defense"][home_team] + tempo)
        return lam, mu

    def score_matrix(self, home_team, away_team, division=None, neutral=False, max_goals=8):
        lam, mu = self.team_lambdas(home_team, away_team, division=division, neutral=neutral)
        home_pmf = poisson.pmf(np.arange(max_goals + 1), lam)
        away_pmf = poisson.pmf(np.arange(max_goals + 1), mu)
        matrix = np.outer(home_pmf, away_pmf)

        rho = self.params["rho"]
        for x in range(2):
            for y in range(2):
                matrix[x, y] *= self._tau(x, y, lam, mu, rho)

        matrix = matrix / matrix.sum()
        return matrix, lam, mu

    def predict_match(self, home_team, away_team, division=None, neutral=False, max_goals=8, top_n=5):
        matrix, lam, mu = self.score_matrix(home_team, away_team, division=division, neutral=neutral, max_goals=max_goals)
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()

        flat_idx = np.argsort(matrix, axis=None)[::-1][:top_n]
        top_scores = []
        for fi in flat_idx:
            i, j = np.unravel_index(fi, matrix.shape)
            top_scores.append({"score": f"{i}-{j}", "prob": matrix[i, j]})

        used_division = division or self.team_division.get(home_team) or self.divisions[0]
        return {
            "home_team": home_team, "away_team": away_team,
            "home_division": self.team_division.get(home_team),
            "away_division": self.team_division.get(away_team),
            "match_division_context": used_division,
            "cross_division": self.team_division.get(home_team) != self.team_division.get(away_team),
            "lambda_home": lam, "lambda_away": mu,
            "P(H)": p_home, "P(D)": p_draw, "P(A)": p_away,
            "top_scorelines": top_scores,
        }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_lsc_table(lsc: pd.DataFrame):
    print("\nLeague Strength Coefficients (pooled scale, promoted/relegated teams as bridges):")
    print("  League            AvgTeamQuality   RelativeStrengthIndex")
    for _, r in lsc.iterrows():
        print(f"  {r['League']:<16}  {r['AvgTeamQuality']:>+.3f}           {r['RelativeStrengthIndex']:.3f}")


def print_match_prediction(pred: dict):
    print(f"\n{pred['home_team']} ({DIVISION_NAMES.get(pred['home_division'], pred['home_division'])}) vs "
          f"{pred['away_team']} ({DIVISION_NAMES.get(pred['away_division'], pred['away_division'])})")
    if pred["cross_division"]:
        print(f"  Cross-division tie -- match context modeled using "
              f"{DIVISION_NAMES.get(pred['match_division_context'], pred['match_division_context'])} conditions")
    print(f"  Expected goals: {pred['home_team']} {pred['lambda_home']:.2f} - "
          f"{pred['lambda_away']:.2f} {pred['away_team']}")
    print(f"  Outcome probabilities: "
          f"H {pred['P(H)']*100:.1f}%  D {pred['P(D)']*100:.1f}%  A {pred['P(A)']*100:.1f}%")
    outcome = max([("H", pred["P(H)"]), ("D", pred["P(D)"]), ("A", pred["P(A)"])], key=lambda t: t[1])
    print(f"  Predicted result: {outcome[0]} ({outcome[1]*100:.1f}% confidence)")
    print("  Top 5 most probable scorelines:")
    for s in pred["top_scorelines"]:
        print(f"    {s['score']:>5}  {s['prob']*100:5.2f}%")


def resolve_team_name(model, name: str, cutoff: float = 0.6):
    """
    Look up a team name against the model's known teams. Returns the exact
    match if found; otherwise raises a ValueError listing close-spelling
    suggestions (e.g. "Man Utd" -> did you mean "Man United"?), since
    football-data.co.uk's exact spellings aren't always obvious.
    """
    if name in model.teams:
        return name
    suggestions = difflib.get_close_matches(name, model.teams, n=3, cutoff=cutoff)
    if suggestions:
        raise ValueError(f'Unknown team "{name}". Did you mean: {", ".join(suggestions)}?')
    raise ValueError(f'Unknown team "{name}". Not found in training data -- check model.list_teams().')


def predict_fixtures(model: "DixonColes", fixtures: pd.DataFrame, max_goals=8, verbose=True) -> pd.DataFrame:
    """
    Batch version of predict_match: given a fixtures dataframe with
    HomeTeam/AwayTeam columns (e.g. this week's games), predicts every match
    and returns a summary dataframe. If verbose, also prints the full
    per-match breakdown (same as print_match_prediction) for each row.

    Rows with an unrecognized team name are skipped (not fatal) -- a warning
    with spelling suggestions is printed, and the row is included in the
    returned dataframe with an "Error" column instead of predictions.
    """
    rows = []
    for _, r in fixtures.iterrows():
        home_raw, away_raw = str(r["HomeTeam"]).strip(), str(r["AwayTeam"]).strip()
        try:
            home = resolve_team_name(model, home_raw)
            away = resolve_team_name(model, away_raw)
        except ValueError as e:
            print(f"  [skipped] {home_raw} vs {away_raw}: {e}")
            rows.append({"HomeTeam": home_raw, "AwayTeam": away_raw, "Error": str(e)})
            continue

        pred = model.predict_match(home, away, max_goals=max_goals)
        if verbose:
            print_match_prediction(pred)

        outcome = max([("H", pred["P(H)"]), ("D", pred["P(D)"]), ("A", pred["P(A)"])], key=lambda t: t[1])
        top_score = pred["top_scorelines"][0]
        rows.append({
            "HomeTeam": home, "AwayTeam": away,
            "PredictedResult": outcome[0], "Confidence%": round(outcome[1] * 100, 1),
            "P(H)%": round(pred["P(H)"] * 100, 1), "P(D)%": round(pred["P(D)"] * 100, 1),
            "P(A)%": round(pred["P(A)"] * 100, 1),
            "xG_Home": round(pred["lambda_home"], 2), "xG_Away": round(pred["lambda_away"], 2),
            "MostLikelyScore": top_score["score"], "ScoreProb%": round(top_score["prob"] * 100, 1),
            "CrossDivision": pred["cross_division"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# League season simulation
# ---------------------------------------------------------------------------

def build_current_table(played: pd.DataFrame, teams) -> pd.DataFrame:
    table = pd.DataFrame({"Team": teams}).set_index("Team")
    for col in ["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]:
        table[col] = 0

    for _, row in played.iterrows():
        h, a, hg, ag = row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"]
        if h not in table.index or a not in table.index:
            continue
        table.loc[h, "P"] += 1
        table.loc[a, "P"] += 1
        table.loc[h, "GF"] += hg
        table.loc[h, "GA"] += ag
        table.loc[a, "GF"] += ag
        table.loc[a, "GA"] += hg
        if hg > ag:
            table.loc[h, "W"] += 1
            table.loc[h, "Pts"] += 3
            table.loc[a, "L"] += 1
        elif hg < ag:
            table.loc[a, "W"] += 1
            table.loc[a, "Pts"] += 3
            table.loc[h, "L"] += 1
        else:
            table.loc[h, "D"] += 1
            table.loc[a, "D"] += 1
            table.loc[h, "Pts"] += 1
            table.loc[a, "Pts"] += 1

    table["GD"] = table["GF"] - table["GA"]
    return table


def simulate_season(model: DixonColes, played: pd.DataFrame, fixtures: pd.DataFrame,
                     division: str, n_sims: int = 10000, max_goals: int = 8, seed: int = 42):
    """
    Simulate the remaining `fixtures` (all within a single `division`)
    n_sims times on top of the already-played `played` results.
    """
    teams = sorted(set(played["HomeTeam"]).union(played["AwayTeam"])
                   .union(fixtures["HomeTeam"]).union(fixtures["AwayTeam"]))
    base_table = build_current_table(played, teams)

    rng = np.random.default_rng(seed)
    n_teams = len(teams)
    team_pos = {t: i for i, t in enumerate(teams)}

    base_pts = base_table["Pts"].reindex(teams).fillna(0).values.astype(np.int64)
    base_gf = base_table["GF"].reindex(teams).fillna(0).values.astype(np.int64)
    base_ga = base_table["GA"].reindex(teams).fillna(0).values.astype(np.int64)

    points_acc = np.tile(base_pts, (n_sims, 1))
    gf_acc = np.tile(base_gf, (n_sims, 1))
    ga_acc = np.tile(base_ga, (n_sims, 1))

    fixture_list = list(fixtures[["HomeTeam", "AwayTeam"]].itertuples(index=False, name=None))
    for h, a in fixture_list:
        matrix, _, _ = model.score_matrix(h, a, division=division, max_goals=max_goals)
        shape = matrix.shape
        flat = matrix.flatten()
        flat = flat / flat.sum()

        choices = rng.choice(len(flat), size=n_sims, p=flat)
        hg, ag = np.unravel_index(choices, shape)

        hi, ai = team_pos[h], team_pos[a]
        gf_acc[:, hi] += hg
        gf_acc[:, ai] += ag
        ga_acc[:, hi] += ag
        ga_acc[:, ai] += hg

        home_win = hg > ag
        away_win = hg < ag
        draw = ~home_win & ~away_win
        points_acc[:, hi] += 3 * home_win + 1 * draw
        points_acc[:, ai] += 3 * away_win + 1 * draw

    gd_acc = gf_acc - ga_acc

    position_counts = np.zeros((n_teams, n_teams), dtype=int)
    for s in range(n_sims):
        order = np.lexsort((-gf_acc[s], -gd_acc[s], -points_acc[s]))
        position_counts[order, np.arange(n_teams)] += 1

    avg_pts = points_acc.mean(axis=0)
    avg_gd = gd_acc.mean(axis=0)
    avg_gf = gf_acc.mean(axis=0)
    avg_position = np.array([np.dot(np.arange(1, n_teams + 1), position_counts[i]) / n_sims
                              for i in range(n_teams)])
    title_prob = position_counts[:, 0] / n_sims
    top4_prob = position_counts[:, :4].sum(axis=1) / n_sims
    relegation_prob = position_counts[:, -3:].sum(axis=1) / n_sims

    result = pd.DataFrame({
        "Team": teams, "AvgPts": avg_pts, "AvgGD": avg_gd, "AvgGF": avg_gf,
        "AvgFinishPos": avg_position, "TitleProb%": title_prob * 100,
        "Top4Prob%": top4_prob * 100, "RelegationProb%": relegation_prob * 100,
    })
    result = result.sort_values(["AvgPts", "AvgGD", "AvgGF"], ascending=False).reset_index(drop=True)
    result.index += 1
    result.index.name = "Pos"

    # Full exact-position probability matrix (Team x Pos1..PosN, in %), reordered
    # to match `result`'s team order. Attached via .attrs so the return type/
    # signature of simulate_season doesn't change -- existing callers are
    # unaffected, but result.attrs["position_probabilities"] is available for
    # anyone who wants the complete grid (see position_probability_table()).
    pos_prob_df = pd.DataFrame(position_counts / n_sims * 100, index=teams,
                                columns=[f"Pos{i+1}" for i in range(n_teams)])
    result.attrs["position_probabilities"] = pos_prob_df.loc[result["Team"]].reset_index(drop=True)
    result.attrs["position_probabilities"].index = result.index

    return result


def position_probability_table(season_table: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls the full exact-finishing-position probability grid out of a
    simulate_season() result: one row per team, one column per league
    position (Pos1 = won the league, PosN = finished bottom), each cell the
    % probability of that team finishing in exactly that position. This is
    the "every position" companion to the summary columns already in the
    season table (TitleProb%, Top4Prob%, RelegationProb%, which are just
    specific slices/sums of this same grid).
    """
    if "position_probabilities" not in season_table.attrs:
        raise ValueError("season_table has no attached position probabilities -- "
                          "make sure it came straight from simulate_season().")
    grid = season_table.attrs["position_probabilities"].copy()
    grid.insert(0, "Team", season_table["Team"].values)
    return grid


def style_position_table(season_table: pd.DataFrame, european_spots: int = 5, relegation_spots: int = 3):
    """
    Renders the full position-probability grid as a colour-graded table for
    Jupyter display (à la the classic "season projection" probability grids):
    blue gradient over the qualification-zone columns (Pos1..european_spots),
    red gradient over the relegation-zone columns (last relegation_spots),
    plain percentage formatting elsewhere. Returns a pandas Styler -- just
    let it be the last expression in a notebook cell to render it, or call
    `.to_html()` / `.to_excel()` on it.

    Only meaningful in a Jupyter/HTML context; for scripts/CLI use
    position_probability_table() instead and print/save the plain numbers.
    """
    grid = position_probability_table(season_table)
    pos_cols = [c for c in grid.columns if c.startswith("Pos")]
    n_teams = len(pos_cols)
    euro_cols = pos_cols[:min(european_spots, n_teams)]
    releg_cols = pos_cols[max(n_teams - relegation_spots, 0):]

    styler = grid.style.format({c: "{:.1f}" for c in pos_cols}).hide(axis="index")
    if euro_cols:
        styler = styler.background_gradient(cmap="Blues", subset=euro_cols, vmin=0, vmax=100)
    if releg_cols:
        styler = styler.background_gradient(cmap="Reds", subset=releg_cols, vmin=0, vmax=100)
    middle_cols = [c for c in pos_cols if c not in euro_cols and c not in releg_cols]
    if middle_cols:
        styler = styler.background_gradient(cmap="Greys", subset=middle_cols, vmin=0, vmax=60)
    return styler


# ---------------------------------------------------------------------------
# Cup / knockout simulation
# ---------------------------------------------------------------------------

def simulate_cup(model: DixonColes, round1_ties, n_sims: int = 10000, max_goals: int = 8,
                  neutral_final: bool = True, shootout_skill_weight: float = 0.6, seed: int = 7):
    """
    Simulate a single-elimination knockout bracket, e.g. an FA-Cup-style tie
    list, drawing scorelines from the (potentially cross-division) Dixon-Coles
    model. round1_ties: list of (home_team, away_team) tuples; length must be
    a power of 2. Standard bracket pairing: winner of tie[2i] plays winner of
    tie[2i+1] in the next round. Draws are resolved via a simulated penalty
    shootout, weighted (weakly) by relative team quality.

    Returns a DataFrame with each team's probability of reaching each round
    and winning the competition.
    """
    n_teams = len(round1_ties) * 2
    if n_teams & (n_teams - 1) != 0:
        raise ValueError("Number of teams in round1_ties must be a power of 2 (e.g. 8, 16, 32, 64).")

    n_rounds = int(np.log2(n_teams))
    all_teams = sorted({t for tie in round1_ties for t in tie})
    team_idx = {t: i for i, t in enumerate(all_teams)}
    rng = np.random.default_rng(seed)

    reach_round_counts = np.zeros((len(all_teams), n_rounds + 1), dtype=int)  # +1 = won it all
    for t in all_teams:
        reach_round_counts[team_idx[t], 0] = n_sims  # everyone "reaches" round 0 (entered)

    quality = {t: model.params["attack"][t] - model.params["defense"][t] for t in all_teams}

    for s in range(n_sims):
        current_round_ties = list(round1_ties)
        for rnd in range(n_rounds):
            neutral = neutral_final and (rnd == n_rounds - 1)
            winners = []
            for home, away in current_round_ties:
                division = None  # defaults to home team's own division context
                matrix, _, _ = model.score_matrix(home, away, division=division, neutral=neutral, max_goals=max_goals)
                flat = matrix.flatten()
                flat = flat / flat.sum()
                choice = rng.choice(len(flat), p=flat)
                hg, ag = np.unravel_index(choice, matrix.shape)
                if hg > ag:
                    winner = home
                elif ag > hg:
                    winner = away
                else:
                    # Drawn -> resolve via simulated shootout, mildly skill-weighted
                    diff = quality[home] - quality[away]
                    p_home = 1 / (1 + np.exp(-shootout_skill_weight * diff))
                    winner = home if rng.random() < p_home else away
                winners.append(winner)
                reach_round_counts[team_idx[winner], rnd + 1] += 1

            # Pair up winners for next round using standard bracket seeding
            # (skip on the final round -- there is no next round to pair into)
            if rnd < n_rounds - 1:
                current_round_ties = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    round_labels = ["Entered"] + [f"Round {i+1}" for i in range(n_rounds - 1)] + ["Won Final"]
    df = pd.DataFrame(reach_round_counts / n_sims * 100, index=all_teams, columns=round_labels)
    df = df.sort_values("Won Final", ascending=False)
    df.index.name = "Team"
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Football match, season & cup predictor (pooled Dixon-Coles model)")
    ap.add_argument("--divisions", default="E0,E1,E2,E3",
                     help="comma-separated divisions to pool for training, e.g. E0,E1,E2,E3")
    ap.add_argument("--seasons", type=int, default=10, help="number of past seasons to train on, per division")
    ap.add_argument("--show-lsc", action="store_true", help="print the League Strength Coefficient table")
    ap.add_argument("--list-teams", action="store_true", help="print every team available in the fitted model")
    ap.add_argument("--predict", type=str, default=None, help='single match, e.g. "Arsenal,Chelsea"')
    ap.add_argument("--neutral", action="store_true", help="treat --predict match as a neutral venue")
    ap.add_argument("--predict-fixtures", type=str, default=None,
                     help="CSV with HomeTeam,AwayTeam columns (e.g. this week's games) -- "
                          "predicts every match and prints/saves a summary table")

    ap.add_argument("--simulate", action="store_true", help="run full league season Monte Carlo simulation")
    ap.add_argument("--division", default="E0", help="which division's season to simulate (for --simulate)")
    ap.add_argument("--fixtures", type=str, default=None,
                     help="CSV with HomeTeam,AwayTeam columns for remaining league fixtures")

    ap.add_argument("--cup", action="store_true", help="run knockout cup Monte Carlo simulation")
    ap.add_argument("--cup-fixtures", type=str, default=None,
                     help="CSV with HomeTeam,AwayTeam columns for round-1 cup ties (power-of-2 count)")

    ap.add_argument("--sims", type=int, default=10000, help="number of Monte Carlo simulations")
    ap.add_argument("--max-goals", type=int, default=8)
    ap.add_argument("--out", type=str, default="simulation_output.csv")
    # parse_known_args (not parse_args) so stray args injected by notebook/IDE
    # kernels (e.g. Jupyter's "-f kernel.json") don't crash the script.
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"(ignoring unrecognized arguments: {unknown})", file=sys.stderr)

    divisions = [d.strip().upper() for d in args.divisions.split(",")]

    print(f"Downloading last {args.seasons} season(s) for divisions: {', '.join(divisions)}...")
    data = load_multi_division_historical(divisions, args.seasons)
    print(f"Total pooled matches loaded: {len(data)}")

    print("Fitting pooled Dixon-Coles model...")
    model = DixonColes().fit(data)
    print(f"  fit converged: {model._fit_success}, rho: {model.params['rho']:.3f}")
    for dv in model.divisions:
        print(f"  {DIVISION_NAMES.get(dv, dv):<16} home_adv={model.params['home_adv'][dv]:+.3f}  "
              f"tempo={model.params['tempo'][dv]:+.3f}")

    lsc = model.league_strength_coefficients()
    if args.show_lsc or args.predict or args.predict_fixtures or args.cup:
        print_lsc_table(lsc)

    if args.list_teams:
        teams_df = model.list_teams()
        pd.set_option("display.width", 120)
        print(f"\n{len(teams_df)} teams available:\n")
        print(teams_df.to_string(index=False))
        teams_out = args.out.replace(".csv", "_teams.csv") if args.out.endswith(".csv") else "teams.csv"
        teams_df.to_csv(teams_out, index=False)
        print(f"\nSaved team list to {teams_out}")

    if args.predict:
        home, away = [t.strip() for t in args.predict.split(",")]
        pred = model.predict_match(home, away, neutral=args.neutral, max_goals=args.max_goals)
        print_match_prediction(pred)

    if args.predict_fixtures:
        fixtures_df = pd.read_csv(args.predict_fixtures)
        print(f"\nPredicting {len(fixtures_df)} fixtures from {args.predict_fixtures}...")
        summary = predict_fixtures(model, fixtures_df, max_goals=args.max_goals)
        pd.set_option("display.width", 140)
        print("\nSummary:\n")
        print(summary.to_string(index=False))
        pf_out = args.out.replace(".csv", "_predictions.csv") if args.out.endswith(".csv") else "predictions.csv"
        summary.to_csv(pf_out, index=False)
        print(f"\nSaved predictions to {pf_out}")

    if args.simulate:
        current_season_start = infer_current_start_year()
        current = load_division_seasons(args.division, 1, current_start_year=current_season_start)
        played = current[current["SeasonStartYear"] == current_season_start] if len(current) else \
            pd.DataFrame(columns=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        print(f"Matches already played this season ({args.division}): {len(played)}")

        if args.fixtures:
            fixtures = pd.read_csv(args.fixtures)
        else:
            print("No --fixtures file given; using football-data.co.uk fixtures.csv "
                  "(NOTE: typically only covers the next couple of weeks -- pass --fixtures "
                  "for a true full-season simulation).")
            fixtures = load_fixtures_from_site(args.division)
        print(f"Remaining fixtures to simulate: {len(fixtures)}")

        if len(fixtures) == 0:
            print("No fixtures to simulate. Exiting.")
        else:
            print(f"Running {args.sims} season simulations...")
            table = simulate_season(model, played, fixtures, division=args.division,
                                     n_sims=args.sims, max_goals=args.max_goals)
            pd.set_option("display.width", 120)
            print(f"\nProjected final {DIVISION_NAMES.get(args.division, args.division)} table "
                  "(ranked by expected points):\n")
            print(table.round(1).to_string())
            table.to_csv(args.out)
            print(f"\nSaved full simulation output to {args.out}")

            pos_grid = position_probability_table(table)
            print(f"\nFull finishing-position probability grid (%):\n")
            print(pos_grid.round(1).to_string(index=False))
            pos_out = args.out.replace(".csv", "_positions.csv") if args.out.endswith(".csv") else "positions.csv"
            pos_grid.to_csv(pos_out, index=False)
            print(f"\nSaved position probability grid to {pos_out}")

    if args.cup:
        if not args.cup_fixtures:
            print("--cup requires --cup-fixtures (CSV with HomeTeam,AwayTeam columns, power-of-2 rows).")
        else:
            ties_df = pd.read_csv(args.cup_fixtures)
            round1_ties = list(ties_df[["HomeTeam", "AwayTeam"]].itertuples(index=False, name=None))
            print(f"Simulating {len(round1_ties)}-tie knockout bracket, {args.sims} runs...")
            cup_result = simulate_cup(model, round1_ties, n_sims=args.sims, max_goals=args.max_goals)
            pd.set_option("display.width", 120)
            print("\nCup progression probabilities (%):\n")
            print(cup_result.round(1).to_string())
            cup_out = args.out.replace(".csv", "_cup.csv") if args.simulate else args.out
            cup_result.to_csv(cup_out)
            print(f"\nSaved cup simulation output to {cup_out}")


if __name__ == "__main__":
    main()
