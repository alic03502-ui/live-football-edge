LIVE FOOTBALL EDGE — SAFE INCREMENTAL UPGRADE

This ZIP is an add-on, not a replacement repository. It does not touch API
fetching, Streamlit layout, GOAL_API_KEY secrets, SQLite storage, or requirements.

INSTALL:
1. Download and unzip.
2. Save a backup copy of your current app.py first.
3. In app.py, find the section "Experimental live-only Poisson model".
4. Replace ONLY the existing poisson_probabilities() and live_model() functions
   with the versions in live_model_upgrade.py. Do not replace other code.
5. Commit, wait for Streamlit Cloud to redeploy, then check a live match.
6. If anything breaks, restore those two functions from your backup.

No new packages or historical uploads are required.
Optional local test: python -m unittest -v test_live_model_upgrade.py

LIMITATIONS:
This is still an experimental heuristic, not a trained or calibrated model.
It adds a small capped adjustment when a team is trailing late and keeps
probability calculations consistent. Generic home/away baseline rates remain.
Red-card effects apply only if the caller passes real red-card counts; the
current app may not yet parse/pass those counts. SOT effects require correctly
recognized shots-on-target values. No betting accuracy or profit is guaranteed.