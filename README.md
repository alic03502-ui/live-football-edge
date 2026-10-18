# Live Football Edge — v1

A Streamlit live-match dashboard using GOAL API live fixtures plus an optional local historical-results dataset.

## What this version includes

- Defensive score parsing: unknown scores stay unknown; the app pauses predictions rather than converting them to 0–0.
- API error handling and caching for live fixtures; manual refresh is available.
- Historical completed-results CSV import into local SQLite.
- Recency-weighted Poisson baseline with team attack/defence rates and shrinkage toward league averages.
- Live remaining-goals adjustment using time remaining and available shots-on-target.
- 1X2, double chance, totals, BTTS, team-goal probabilities and most likely final scores.
- SQLite storage for live snapshots.
- Expanding-window walk-forward evaluation and multiclass Brier score.
- Explicit insufficient-data / no-bet gate.
- No bookmaker odds, EV, or edge integration.

## Setup

1. Install requirements: `pip install -r requirements.txt`
2. Set `GOAL_API_KEY` in Streamlit Secrets.
3. Start: `streamlit run app.py`
4. In the sidebar, import a CSV with columns:

```csv
date,home_team,away_team,home_goals,away_goals
2025-08-01,Team A,Team B,2,1
```

Use historical results from the competition(s) you intend to analyze. Team names must match the live API names closely, otherwise the model will treat them as different teams.

## Important limitations

- The GOAL API endpoint used for live fixtures is `fixtures/live`; statistics and events use `fixtures/{id}/statistics` and `fixtures/{id}/events`, matching the existing project integration. Your API plan may not allow every endpoint.
- Historical results are imported by CSV; this version does not assume an undocumented GOAL API historical endpoint exists.
- The baseline is not automatically calibrated. A low Brier score compared with a baseline and good reliability plots over a large out-of-sample sample are needed before interpreting probabilities as dependable.
- The in-play adjustment is a transparent heuristic, not a trained live model. Red-card attribution can vary by provider, so it should be validated against actual GOAL API event payloads before being relied on.
- SQLite persists on the current filesystem. Some hosted Streamlit environments may reset local files during redeploys; back up the database or use persistent storage if needed.
- This project does not promise positive betting returns. A no-bet/data-quality rule is intentional.
