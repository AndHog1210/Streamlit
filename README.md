# Football Predictor (pooled Dixon-Coles model across divisions)

## Setup
```
pip install pandas numpy scipy requests
```

## What's new in this version
- Trains on **10 seasons x 4 divisions** (Premier League, Championship, League
  One, League Two) at once, instead of one division in isolation.
- Fits **one pooled model**: every team gets a single attack/defense rating
  shared across whichever division(s) it played in. Promoted/relegated teams
  act as "bridges" that link the four divisions' scales together, so a
  Championship team's rating is directly comparable to a Premier League
  team's rating -- not just to other Championship teams.
- Reports an explicit **League Strength Coefficient (LSC)**: the average team
  quality (attack minus defense, weighted by recency) within each division,
  on that pooled scale, plus a `RelativeStrengthIndex` normalized to 1.00 for
  the strongest division. This is the "coefficient" you asked for -- it
  literally answers "how strong is a typical team in League One, relative to
  the Premier League, right now".
- Because ratings are on one common scale, the model can predict/simulate
  **cross-division matches** directly -- which is exactly what cup ties are
  (FA Cup, League Cup, etc. regularly pit teams from different tiers against
  each other). A new **knockout cup simulator** runs a full single-elimination
  bracket 10,000 times and reports each team's probability of reaching each
  round and winning it.

## How the League Strength Coefficient works (in plain terms)
Fitting each division separately can't tell you whether a good League One
team is stronger or weaker than a bad Championship team -- there's no shared
scale. But every season, ~3 teams get promoted and ~3 relegated at each
boundary. Those teams play real matches in *both* divisions, so pooling all
divisions' results into one fit and giving each team one rating (not one
rating per division) lets those transitioning teams anchor the divisions to
each other. The LSC is then just a summary: the average team rating,
per division, on that shared scale.

I validated this against synthetic data with a known ground-truth quality
scale and simulated promotion/relegation: the pooled model recovered team
quality with ~0.90 correlation to ground truth across all 80 synthetic teams,
and correctly recovered the true division ordering (top tier strongest ->
bottom tier weakest) in the LSC table.

## Usage
```bash
# Pull 10 seasons of all four English divisions and fit the pooled model,
# printing the League Strength Coefficient table
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --show-lsc

# Predict a single match -- works cross-division too (e.g. a cup tie)
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 --predict "Arsenal,Wrexham"

# Simulate the rest of one division's season 10,000 times
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 \
  --simulate --division E0 --fixtures pl_fixtures.csv --sims 10000

# Simulate a knockout cup bracket (e.g. FA Cup 3rd round, 32 or 64 ties) 10,000 times
python football_predictor.py --divisions E0,E1,E2,E3 --seasons 10 \
  --cup --cup-fixtures fa_cup_r3.csv --sims 10000
```

`--fixtures` / `--cup-fixtures` CSVs just need two columns: `HomeTeam,AwayTeam`.
For the cup, the number of ties must be a power of 2 (e.g. 8, 16, 32, 64) --
`round1_ties[0]` plays the winner of `round1_ties[1]` in round 2, and so on,
standard bracket seeding.

## Streamlit app (recommended for weekly use)

`app.py` is a Streamlit front-end built around a weekly workflow: the expensive
part (downloading results, fitting the pooled model) only happens when you
click a button, not on every page load. Once fitted, the model is cached for
the running session AND pickled to disk (`model_cache.pkl`) so restarting the
app doesn't lose it.

### Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```
`football_predictor.py` must be in the same folder as `app.py`.

### Typical weekly routine
1. Open the app. If a model was already fitted (either earlier this session
   or from a previous run, via `model_cache.pkl`), it's ready to use
   immediately -- skip to step 3.
2. **First time only / whenever you want fresh results folded in:** in the
   sidebar, pick your divisions and seasons, then click **"Fit / refresh
   model"**. This downloads data and refits -- can take a few minutes. You
   don't need to do this every week; do it whenever you want that week's
   actual results incorporated into the ratings (weekly is reasonable, but
   monthly is also fine -- ratings don't go stale that fast).
3. **Every week:** go to the *This week's fixtures* tab, upload a small CSV
   (`HomeTeam,AwayTeam`) or type the games directly into the editable table,
   click **"Predict fixtures"**. This is instant -- no retraining involved.

### Tabs
- **This week's fixtures** -- the main weekly-update tab described above.
  Predictions, H/D/A%, top scorelines, downloadable CSV.
- **Season simulation** -- upload a full remaining-fixture list (and
  optionally already-played results) for one division, runs the 10,000-sim
  Monte Carlo season projection, shows the table plus the full
  finishing-position probability heatmap.
- **Cup simulator** -- upload round-1 knockout ties (power-of-2 count),
  simulates the bracket, shows each team's probability of reaching each round.
- **Teams** -- browse/filter the team list, check for likely spelling
  duplicates, and manually fix a team's division label (e.g. right after a
  real-world promotion, before results exist yet) -- this persists to
  `model_cache.pkl` immediately.
- **League strength** -- the League Strength Coefficient table plus each
  division's fitted home-advantage/tempo terms.

### Deploying (e.g. Streamlit Community Cloud)
Push `app.py`, `football_predictor.py`, and `requirements.txt` to a GitHub
repo and point Streamlit Community Cloud at `app.py`. Note that on a free/
sleeping-instance host, `model_cache.pkl` may not persist across a full
redeploy or instance restart (ephemeral filesystem) -- if so, you'll need to
click "Fit / refresh model" again after a cold start. For a host with
persistent storage, it'll survive restarts as expected.

## Important limitations
- **Full-season fixture lists**: football-data.co.uk only publishes results,
  not a full future schedule. Their `fixtures.csv` feed covers only the next
  couple of weeks. For a true full-season simulation, supply your own
  `--fixtures` CSV (built from the official fixture list), with team names
  spelled exactly as football-data.co.uk spells them.
- **Cup draws**: real cup competitions (FA Cup especially) draw each round
  randomly rather than fixing the whole bracket in advance, so
  `simulate_cup` is built to simulate a *known* bracket (e.g. once a round's
  draw has been made, or for competitions like the Champions League league
  phase/knockout draw that fix the bracket ahead of time). It doesn't model
  the random draw process itself.
- **Two-legged ties**: not currently modeled -- each cup tie is a single
  match, with drawn matches resolved via a lightly skill-weighted simulated
  penalty shootout (approximating extra time + penalties). Two-legged
  aggregate ties would need extending `simulate_cup` to draw two matches per
  tie and sum goals -- flagged as a natural next step if you need it.
- **LSC precision** depends on how much promotion/relegation bridging data
  exists for the specific teams you're comparing. Teams that have bounced
  between divisions repeatedly are well-anchored; a club that's spent 10
  straight years in one division with zero movement has a rating anchored
  only indirectly (via opponents who *did* move). Same-division predictions
  are unaffected by this and remain the most reliable.
- This sandbox has no network access, so the full pipeline was validated
  end-to-end against synthetic multi-division data with simulated
  promotion/relegation (see notes above) rather than a live download --
  run it locally where you have internet access.
