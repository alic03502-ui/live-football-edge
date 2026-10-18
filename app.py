from __future__ import annotations
import json
from typing import Any

import pandas as pd
import requests
import streamlit as st

from model import (
    REQUIRED_COLUMNS, prepare_history, model_status, pre_match_lambdas,
    in_play_lambdas, score_matrix, markets, top_final_scores,
    walk_forward_evaluation,
)
from storage import init_db, load_history, save_history, save_snapshot, recent_snapshots

BASE_URL = "https://api.goal-api.com/v1"
TIMEOUT = 20
init_db()

st.set_page_config(page_title="Live Football Edge", page_icon="⚽", layout="wide")
st.title("⚽ Live Football Edge")
st.caption("Live GOAL API data · Poisson baseline · historical evaluation · no bookmaker odds")
st.warning(
    "Probabilities are experimental until evaluated on your own historical data. "
    "The in-play adjustment is heuristic, not a trained or calibrated live model."
)


def get_key() -> str:
    try:
        return str(st.secrets.get("GOAL_API_KEY", "")).strip()
    except Exception:
        return ""


def api_get(endpoint: str):
    key = get_key()
    if not key:
        raise RuntimeError("GOAL_API_KEY is missing from Streamlit Secrets.")
    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=TIMEOUT,
        )
        if response.status_code == 401:
            raise RuntimeError("GOAL API rejected the key. Check Streamlit Secrets.")
        if response.status_code == 403:
            raise RuntimeError("GOAL API denied access or this endpoint is not in your plan.")
        if response.status_code == 429:
            raise RuntimeError("GOAL API quota/rate limit reached. Wait before retrying.")
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"GOAL API request failed: {exc}") from exc
    if isinstance(data, dict):
        if data.get("success") is False:
            raise RuntimeError(str(data.get("error") or data.get("message") or "API error"))
        return data.get("data", data)
    return data


def as_list(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("fixtures", "matches", "results", "events", "items", "data"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
        if any(k in payload for k in ("id", "fixtureId", "fixture_id", "match_id")):
            return [payload]
    return []


def first(obj, *keys, default=None):
    if not isinstance(obj, dict):
        return default
    for key in keys:
        if obj.get(key) is not None:
            return obj[key]
    return default


def parse_score(value):
    if isinstance(value, dict):
        value = first(value, "score", "goals", "value", "total", "current")
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(str(value).strip())
        if n >= 0 and n.is_integer():
            return int(n)
    except (TypeError, ValueError):
        pass
    return None


def score_for(item, side):
    keys = (("homeTeamScore", "home_score", "homeScore", "homeGoals", "home_goals", "match_hometeam_score")
            if side == "home" else
            ("awayTeamScore", "away_score", "awayScore", "awayGoals", "away_goals", "match_awayteam_score"))
    value = parse_score(first(item, *keys))
    if value is not None:
        return value
    containers = ("goals", "score", "scores", "currentScore")
    nested_keys = (("home", "homeScore", "home_score", "homeTeam", "homeTeamScore")
                   if side == "home" else
                   ("away", "awayScore", "away_score", "awayTeam", "awayTeamScore"))
    for name in containers:
        obj = item.get(name)
        if isinstance(obj, dict):
            value = parse_score(first(obj, *nested_keys, default=obj.get(side)))
            if value is not None:
                return value
    return None


def team_label(value, fallback):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(first(value, "name", "teamName", "team_name", "displayName", default=fallback))
    return fallback


def normalize(item):
    teams = item.get("teams") if isinstance(item.get("teams"), dict) else {}
    home = first(teams, "home", "homeTeam", "home_team",
                 default=first(item, "homeTeam", "home_team", "home", "match_hometeam_name", default="Home"))
    away = first(teams, "away", "awayTeam", "away_team",
                 default=first(item, "awayTeam", "away_team", "away", "match_awayteam_name", default="Away"))
    status_obj = first(item, "status", "fixtureStatus", default={})
    minute = first(item, "matchMinute", "matchElapsed", "match_minute", "elapsed", "minute",
                   default=first(status_obj, "elapsed", "minute", default=first(item, "match_status", default=0)))
    if isinstance(minute, str):
        import re
        found = re.search(r"\d+", minute)
        minute = int(found.group()) if found else 0
    try:
        minute = max(0, min(120, int(minute or 0)))
    except (TypeError, ValueError):
        minute = 0
    league_obj = item.get("league") if isinstance(item.get("league"), dict) else {}
    return {
        "id": str(first(item, "id", "fixtureId", "fixture_id", "match_id", default="")),
        "home_name": team_label(home, "Home"),
        "away_name": team_label(away, "Away"),
        "home_score": score_for(item, "home"),
        "away_score": score_for(item, "away"),
        "minute": minute,
        "league": str(first(league_obj, "name", "leagueName", default=first(item, "leagueName", "league_name", default="Unknown league"))),
        "status": str(status_obj),
        "raw": item,
    }


def stat_value(rows, *names):
    wanted = {n.lower() for n in names}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(first(row, "type", "name", "label", "statistic", default="")).lower()
        if label in wanted:
            value = first(row, "value", "total", "amount")
            try:
                return float(str(value).replace("%", "").strip())
            except (TypeError, ValueError):
                return None
    return None


# ---- Historical data and evaluation ----
with st.sidebar:
    st.header("Historical data")
    st.write("Upload completed results as CSV. Required columns:")
    st.code("date,home_team,away_team,home_goals,away_goals")
    uploaded = st.file_uploader("Historical matches CSV", type=["csv"])
    if uploaded is not None and st.button("Import history", use_container_width=True):
        try:
            hist = prepare_history(pd.read_csv(uploaded))
            count = save_history(hist)
            st.success(f"Imported/updated {count} rows in the local SQLite database.")
            st.rerun()
        except Exception as exc:
            st.error(f"Import failed: {exc}")

history = load_history()
with st.expander("Historical database status", expanded=history.empty):
    st.write(f"Stored completed matches: **{len(history)}**")
    if history.empty:
        st.info("No historical data yet. Upload a CSV before using pre-match team-strength predictions.")
    else:
        st.write(f"Date range: {history.date.min()} — {history.date.max()}")
        st.dataframe(history.tail(10), use_container_width=True, hide_index=True)

if not history.empty:
    st.subheader("Walk-forward historical evaluation")
    if st.button("Run expanding-window evaluation"):
        with st.spinner("Evaluating chronologically; future matches are excluded from each fit..."):
            try:
                report = walk_forward_evaluation(history)
                if report.empty:
                    st.warning("Not enough usable history to evaluate. Import more matches.")
                else:
                    st.session_state["evaluation_report"] = report
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
    if "evaluation_report" in st.session_state:
        report = st.session_state["evaluation_report"]
        brier = report["brier_1x2"].mean()
        st.metric("Mean multiclass Brier score (lower is better)", f"{brier:.4f}")
        st.caption(f"Out-of-sample predictions evaluated: {len(report)}. This is not a guarantee of future performance.")
        st.dataframe(report.tail(100), use_container_width=True, hide_index=True)

st.divider()
col1, col2 = st.columns([4, 1])
with col1:
    st.write("Choose a live fixture. The app caches API responses and avoids automatic rapid polling.")
with col2:
    refresh = st.button("Refresh fixtures", use_container_width=True)

if refresh:
    st.cache_data.clear()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live():
    return as_list(api_get("fixtures/live"))

try:
    fixtures = [normalize(x) for x in fetch_live()]
except Exception as exc:
    st.error(f"Could not load live fixtures: {exc}")
    st.stop()

fixtures = [f for f in fixtures if f["id"]]
if not fixtures:
    st.info("No live fixtures returned. This is normal when no covered matches are live.")
    st.stop()

league_options = ["All leagues"] + sorted({f["league"] for f in fixtures})
selected_league = st.selectbox("League", league_options)
if selected_league != "All leagues":
    fixtures = [f for f in fixtures if f["league"] == selected_league]

def fixture_text(f):
    hs = "—" if f["home_score"] is None else str(f["home_score"])
    aws = "—" if f["away_score"] is None else str(f["away_score"])
    return f"{f['home_name']} {hs}–{aws} {f['away_name']} · {f['minute']}' · {f['league']}"

selected = st.selectbox("Live match", fixtures, format_func=fixture_text)
st.subheader(f"{selected['home_name']} {selected['home_score'] if selected['home_score'] is not None else '—'} – "
             f"{selected['away_score'] if selected['away_score'] is not None else '—'} "
             f"{selected['away_name']}")
st.caption(f"{selected['league']} · {selected['minute']}' · Fixture ID {selected['id']}")

if selected["home_score"] is None or selected["away_score"] is None:
    st.error("One or both scores are missing/unrecognized. Predictions are paused; the app will not silently assume 0–0.")
    with st.expander("Score parser diagnostics"):
        st.write(list(selected["raw"].keys()))
        st.json(selected["raw"])
    st.stop()

try:
    stats = as_list(api_get(f"fixtures/{selected['id']}/statistics"))
    events = as_list(api_get(f"fixtures/{selected['id']}/events"))
except Exception as exc:
    st.warning(f"Live stats/events unavailable: {exc}. The app will not invent missing statistics.")
    stats, events = [], []

# Flexible stat row handling: common flat shape and team-grouped shape.
home_stats, away_stats = [], []
for row in stats:
    team = row.get("team") if isinstance(row.get("team"), dict) else {}
    label_team = str(first(team, "name", default=first(row, "teamName", "team", default=""))).lower()
    nested_stats = first(row, "statistics", "stats", "items", default=None)
    if isinstance(nested_stats, list):
        target = home_stats if "home" in label_team else away_stats if "away" in label_team else None
        if target is not None:
            target.extend(nested_stats)
    else:
        name = first(row, "type", "name", "label", "statistic")
        hv, av = first(row, "home", "homeValue", "home_value"), first(row, "away", "awayValue", "away_value")
        if name is not None:
            if hv is not None: home_stats.append({"type": name, "value": hv})
            if av is not None: away_stats.append({"type": name, "value": av})

stats_rows = []
for label, aliases in [
    ("Shots on target", ("Shots on Goal", "Shots on Target", "Shots on target")),
    ("Total shots", ("Total Shots", "Shots")),
    ("Possession", ("Ball Possession", "Possession")),
    ("Corners", ("Corner Kicks", "Corners")),
    ("xG", ("expected_goals", "Expected Goals", "xG")),
]:
    hv, av = stat_value(home_stats, *aliases), stat_value(away_stats, *aliases)
    if hv is not None or av is not None:
        stats_rows.append({"Statistic": label, selected["home_name"]: hv, selected["away_name"]: av})
if stats_rows:
    st.subheader("Available live statistics")
    st.dataframe(stats_rows, hide_index=True, use_container_width=True)
else:
    st.info("No recognized statistics were returned. Missing stats are left unknown.")

# Save the current observation, not the full raw payload.
try:
    save_snapshot(selected, json.dumps(selected["raw"], ensure_ascii=False, default=str))
except Exception:
    pass

if history.empty:
    st.warning("NO BET / INSUFFICIENT DATA: import historical results before team-strength predictions can be generated.")
    st.stop()

status = model_status(history, selected["home_name"], selected["away_name"])
if not status.ready:
    st.warning(f"NO BET / INSUFFICIENT DATA: {status.reason}")
    st.stop()

home_pre, away_pre = pre_match_lambdas(history, selected["home_name"], selected["away_name"])
home_sot = stat_value(home_stats, "Shots on Goal", "Shots on Target", "Shots on target")
away_sot = stat_value(away_stats, "Shots on Goal", "Shots on Target", "Shots on target")
# Red-card adjustment is disabled until GOAL API event team attribution is verified.
# Counting card text alone could incorrectly assign an away-team card to the home side.
home_reds = 0
away_reds = 0
home_remaining, away_remaining = in_play_lambdas(
    home_pre, away_pre, selected["minute"], selected["home_score"], selected["away_score"],
    home_sot, away_sot, home_reds, away_reds
)
matrix = score_matrix(home_remaining, away_remaining)
probs = markets(matrix, selected["home_score"], selected["away_score"])

st.subheader("Live model estimates")
st.caption("Poisson baseline fitted to imported completed results, then adjusted for time and available shots. In-play layer is not calibrated.")
c1, c2, c3 = st.columns(3)
c1.metric(f"{selected['home_name']} win", f"{probs['1']:.1%}")
c2.metric("Draw", f"{probs['X']:.1%}")
c3.metric(f"{selected['away_name']} win", f"{probs['2']:.1%}")

with st.expander("Double chance and goal markets"):
    market_rows = [{"Market": name, "Probability": f"{value:.1%}"} for name, value in probs.items()]
    st.dataframe(market_rows, hide_index=True, use_container_width=True)

st.subheader("Most likely final scores")
score_rows = [
    {"Projected final score": f"{selected['home_name']} {h}–{a} {selected['away_name']}",
     "Probability": f"{p:.1%}"}
    for p, h, a in top_final_scores(matrix, selected["home_score"], selected["away_score"])
]
st.dataframe(score_rows, hide_index=True, use_container_width=True)

st.subheader("Recent saved live snapshots")
snapshots = recent_snapshots(20)
if not snapshots.empty:
    st.dataframe(snapshots, use_container_width=True, hide_index=True)
else:
    st.caption("Snapshots will appear after a match is analyzed.")

st.divider()
st.warning("NO ODDS SOURCE IS CONNECTED. No EV/edge is calculated. Do not treat probabilities as validated until walk-forward results and calibration are reviewed.")
)