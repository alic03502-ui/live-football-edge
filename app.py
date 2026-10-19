
from __future__ import annotations

import math
import re
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

BASE_URL = "https://api.goal-api.com/v1"
TIMEOUT = 15
DB_PATH = Path("data/live_football_edge.sqlite3")

st.set_page_config(
    page_title="Live Football Edge",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Live Football Edge")
st.caption("Experimental live-only model · No historical uploads required")

st.warning(
    "TEST MODE: These are experimental estimates using generic "
    "football scoring assumptions and available live statistics. "
    "They are not historically validated or calibrated."
)


# =========================================================
# GOAL API
# =========================================================

def get_api_key():
    try:
        return str(st.secrets.get("GOAL_API_KEY", "")).strip()
    except Exception:
        return ""


def api_get(endpoint):
    key = get_api_key()

    if not key:
        raise RuntimeError(
            "GOAL_API_KEY is missing. Add it to Streamlit Secrets."
        )

    response = requests.get(
        f"{BASE_URL}/{endpoint.lstrip('/')}",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        timeout=TIMEOUT,
    )

    if response.status_code == 401:
        raise RuntimeError("API key rejected. Check Streamlit Secrets.")

    if response.status_code == 403:
        raise RuntimeError(
            "API access denied. Check whether your plan supports this endpoint."
        )

    if response.status_code == 429:
        raise RuntimeError("API rate limit reached. Wait before retrying.")

    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict):
        if payload.get("success") is False:
            raise RuntimeError(
                str(payload.get("message", payload.get("error", "API error")))
            )

        return payload.get("data", payload)

    return payload


def as_list(payload):
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in (
            "fixtures", "matches", "results", "events",
            "items", "data",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        if any(k in payload for k in (
            "id", "fixtureId", "fixture_id", "match_id"
        )):
            return [payload]

    return []


@st.cache_data(ttl=45, show_spinner=False)
def fetch_live_fixtures():
    return as_list(api_get("fixtures/live"))


def first(obj, *keys, default=None):
    if not isinstance(obj, dict):
        return default

    for key in keys:
        if obj.get(key) is not None:
            return obj[key]

    return default


def team_name(value, fallback):
    if isinstance(value, dict):
        return str(first(
            value, "name", "teamName", "team_name",
            "displayName", default=fallback
        ))

    if isinstance(value, str):
        return value

    return fallback


def parse_score(value):
    if isinstance(value, dict):
        value = first(value, "score", "goals", "value", "total")

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(str(value).strip())
        if number >= 0 and number.is_integer():
            return int(number)
    except (ValueError, TypeError):
        pass

    return None


def get_score(item, side):
    if side == "home":
        direct_keys = (
            "homeTeamScore", "home_score", "homeScore",
            "homeGoals", "home_goals", "match_hometeam_score",
        )
    else:
        direct_keys = (
            "awayTeamScore", "away_score", "awayScore",
            "awayGoals", "away_goals", "match_awayteam_score",
        )

    score = parse_score(first(item, *direct_keys))
    if score is not None:
        return score

    for container_key in ("score", "scores", "goals", "currentScore"):
        container = item.get(container_key)

        if isinstance(container, dict):
            if side in container:
                score = parse_score(container[side])
                if score is not None:
                    return score

            nested_keys = (
                ("home", "homeScore", "home_score", "homeTeam")
                if side == "home"
                else ("away", "awayScore", "away_score", "awayTeam")
            )

            score = parse_score(first(container, *nested_keys))
            if score is not None:
                return score

    return None


def normalize_fixture(item):
    teams = item.get("teams", {})
    if not isinstance(teams, dict):
        teams = {}

    home = first(
        teams, "home", "homeTeam", "home_team",
        default=first(
            item, "homeTeam", "home_team", "home",
            "match_hometeam_name", default="Home"
        )
    )

    away = first(
        teams, "away", "awayTeam", "away_team",
        default=first(
            item, "awayTeam", "away_team", "away",
            "match_awayteam_name", default="Away"
        )
    )

    status = first(item, "status", "fixtureStatus", default="")
    minute_value = first(
        item, "matchMinute", "matchElapsed", "match_minute",
        "elapsed", "minute",
        default=first(item, "match_status", default=0)
    )

    if isinstance(minute_value, dict):
        minute_value = first(
            minute_value, "elapsed", "minute", default=0
        )

    if isinstance(minute_value, str):
        match = re.search(r"\d+", minute_value)
        minute_value = int(match.group()) if match else 0

    try:
        minute = max(0, min(120, int(minute_value or 0)))
    except (TypeError, ValueError):
        minute = 0

    league = item.get("league", {})
    if not isinstance(league, dict):
        league = {}

    return {
        "id": str(first(
            item, "id", "fixtureId", "fixture_id", "match_id",
            default=""
        )),
        "home": team_name(home, "Home"),
        "away": team_name(away, "Away"),
        "home_score": get_score(item, "home"),
        "away_score": get_score(item, "away"),
        "minute": minute,
        "league": str(first(
            league, "name", "leagueName",
            default=first(
                item, "leagueName", "league_name",
                default="Unknown league"
            )
        )),
        "raw": item,
    }


# =========================================================
# Statistics parsing
# =========================================================

def stat_label(row):
    return str(first(
        row, "type", "name", "label", "statistic",
        "description", default=""
    )).strip().lower()


def numeric_value(value):
    if isinstance(value, dict):
        value = first(value, "value", "total", "amount")

    if value is None:
        return None

    try:
        return float(str(value).replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def flatten_statistics(payload):
    """Convert common API statistics response shapes into rows."""
    rows = []

    def walk(obj):
        if isinstance(obj, list):
            for entry in obj:
                walk(entry)
            return

        if not isinstance(obj, dict):
            return

        # A statistic represented as {type: ..., home: ..., away: ...}
        label = stat_label(obj)
        home_value = first(obj, "home", "homeValue", "home_value")
        away_value = first(obj, "away", "awayValue", "away_value")

        if label and (home_value is not None or away_value is not None):
            rows.append({
                "type": label,
                "home": numeric_value(home_value),
                "away": numeric_value(away_value),
            })
            return

        # A statistic represented as {type: ..., value: ...}
        if label and any(k in obj for k in ("value", "total", "amount")):
            rows.append({
                "type": label,
                "value": numeric_value(first(
                    obj, "value", "total", "amount"
                )),
            })
            return

        # Recurse through likely containers.
        for key in (
            "data", "statistics", "stats", "items",
            "response", "results", "teams", "home", "away",
        ):
            if key in obj:
                walk(obj[key])

    walk(payload)
    return rows


def get_stat(rows, side, aliases):
    aliases = {x.lower() for x in aliases}

    for row in rows:
        label = row.get("type", "")
        if label not in aliases:
            continue

        if row.get(side) is not None:
            return row[side]

        if row.get("value") is not None:
            # A value-only row has no verified team attribution.
            continue

    return None


# =========================================================
# Experimental live-only Poisson model
# =========================================================

def poisson_probabilities(lam, max_goals=10):
    lam = max(0.0, float(lam))
    probabilities = [
        math.exp(-lam) * (lam ** k) / math.factorial(k)
        for k in range(max_goals + 1)
    ]

    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]

    return probabilities


def live_model(
    minute,
    home_score,
    away_score,
    home_sot=None,
    away_sot=None,
    home_reds=0,
    away_reds=0,
):
    """
    Generic experimental model.
    It does not know team strength and is not calibrated.
    """

    # Generic baseline scoring rates per 90 minutes.
    home_rate = 1.40
    away_rate = 1.10

    # Only adjust when shots on target are actually available.
    if home_sot is not None and away_sot is not None:
        home_rate *= max(0.65, min(1.45, 0.90 + 0.06 * home_sot))
        away_rate *= max(0.65, min(1.45, 0.90 + 0.06 * away_sot))

    # Small experimental red-card adjustment, if known.
    home_rate *= max(0.55, 1.0 - 0.25 * home_reds + 0.15 * away_reds)
    away_rate *= max(0.55, 1.0 - 0.25 * away_reds + 0.15 * home_reds)

    # Use 95 minutes as the default regulation-time horizon.
    # If already beyond 95 minutes, allow up to 5 additional minutes.
    remaining = max(0, min(100 - minute, 100)) / 90.0

    home_remaining = home_rate * remaining
    away_remaining = away_rate * remaining

    home_goals_probs = poisson_probabilities(home_remaining)
    away_goals_probs = poisson_probabilities(away_remaining)

    # Matrix contains probabilities of additional goals from now.
    matrix = [
        [home_goals_probs[h] * away_goals_probs[a]
         for a in range(len(away_goals_probs))]
        for h in range(len(home_goals_probs))
    ]

    result_probs = {"1": 0.0, "X": 0.0, "2": 0.0}
    over15 = over25 = over35 = under25 = btts = 0.0
    score_candidates = []

    for add_home, row in enumerate(matrix):
        for add_away, probability in enumerate(row):
            final_home = home_score + add_home
            final_away = away_score + add_away
            total_goals = final_home + final_away

            if final_home > final_away:
                result_probs["1"] += probability
            elif final_home == final_away:
                result_probs["X"] += probability
            else:
                result_probs["2"] += probability

            if total_goals >= 2:
                over15 += probability
            if total_goals >= 3:
                over25 += probability
            if total_goals >= 4:
                over35 += probability
            if total_goals <= 2:
                under25 += probability
            if final_home >= 1 and final_away >= 1:
                btts += probability

            score_candidates.append((
                probability, final_home, final_away
            ))

    # Normalize probabilities to handle the tiny truncated Poisson tail.
    total = sum(result_probs.values())
    if total > 0:
        result_probs = {
            key: value / total
            for key, value in result_probs.items()
        }

    markets = {
        "Home win": result_probs["1"],
        "Draw": result_probs["X"],
        "Away win": result_probs["2"],
        "Over 1.5 goals": over15,
        "Under 1.5 goals": 1.0 - over15,
        "Over 2.5 goals": over25,
        "Under 2.5 goals": under25,
        "Over 3.5 goals": over35,
        "Under 3.5 goals": 1.0 - over35,
        "Both teams to score — Yes": btts,
        "Both teams to score — No": 1.0 - btts,
        "Home or draw (1X)": result_probs["1"] + result_probs["X"],
        "Away or draw (X2)": result_probs["X"] + result_probs["2"],
    }

    score_candidates.sort(reverse=True)
    top_scores = score_candidates[:5]

    return markets, top_scores, home_remaining, away_remaining


# =========================================================
# Local snapshot storage
# =========================================================

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT,
                fixture_id TEXT,
                home_team TEXT,
                away_team TEXT,
                minute INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                payload TEXT
            )
        """)


def save_snapshot(fixture):
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO live_snapshots
            (captured_at, fixture_id, home_team, away_team,
             minute, home_score, away_score, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            fixture["id"],
            fixture["home"],
            fixture["away"],
            fixture["minute"],
            fixture["home_score"],
            fixture["away_score"],
            json.dumps(fixture["raw"], ensure_ascii=False, default=str),
        ))


# =========================================================
# Sidebar and fixture refresh
# =========================================================

with st.sidebar:
    st.header("Model settings")
    st.write("Mode: **Live-only experimental**")
    st.write("Historical CSV: **Not required**")
    st.caption(
        "This model uses generic team scoring rates. "
        "It does not estimate the real strength of individual teams."
    )

if st.button("Refresh fixtures", use_container_width=True):
    fetch_live_fixtures.clear()

try:
    raw_fixtures = fetch_live_fixtures()
except Exception as exc:
    st.error(f"Could not load live fixtures: {exc}")
    st.stop()

fixtures = [
    normalize_fixture(item)
    for item in raw_fixtures
]

fixtures = [
    fixture for fixture in fixtures
    if fixture["id"]
]

if not fixtures:
    st.info("No live fixtures were returned by the API right now.")
    st.stop()

leagues = ["All leagues"] + sorted({
    fixture["league"] for fixture in fixtures
})

selected_league = st.selectbox("League", leagues)

if selected_league != "All leagues":
    fixtures = [
        f for f in fixtures if f["league"] == selected_league
    ]

if not fixtures:
    st.info("No live matches in this league.")
    st.stop()


def fixture_label(f):
    hs = "?" if f["home_score"] is None else f["home_score"]
    aws = "?" if f["away_score"] is None else f["away_score"]

    return (
        f"{f['home']} {hs}–{aws} {f['away']} "
        f"· {f['minute']}' · {f['league']}"
    )


selected = st.selectbox(
    "Live match",
    fixtures,
    format_func=fixture_label,
)

if (
    selected["home_score"] is None
    or selected["away_score"] is None
):
    st.error(
        "The API did not provide both scores in a recognized format. "
        "Predictions are paused rather than assuming 0–0."
    )

    with st.expander("Fixture response diagnostics"):
        st.json(selected["raw"])

    st.stop()


st.subheader(
    f"{selected['home']} {selected['home_score']} – "
    f"{selected['away_score']} {selected['away']}"
)
st.caption(
    f"{selected['league']} · {selected['minute']}' "
    f"· Fixture ID {selected['id']}"
)


# =========================================================
# Fetch statistics, but keep the model usable if unavailable
# =========================================================

stats_rows = []
stats_error = None

try:
    stats_payload = api_get(
        f"fixtures/{selected['id']}/statistics"
    )
    stats_rows = flatten_statistics(stats_payload)
except Exception as exc:
    stats_error = str(exc)

home_sot = get_stat(
    stats_rows,
    "home",
    {"shots on target", "shots on goal", "shots on target"},
)
away_sot = get_stat(
    stats_rows,
    "away",
    {"shots on target", "shots on goal", "shots on target"},
)

if stats_error:
    st.warning(
        f"Live statistics could not be loaded: {stats_error}. "
        "The model will use generic assumptions without shot adjustments."
    )

if home_sot is None or away_sot is None:
    st.info(
        "Shots on target are unavailable or not recognized. "
        "Predictions will use generic scoring rates."
    )


# =========================================================
# Generate predictions immediately — no historical data
# =========================================================

markets, top_scores, home_remaining, away_remaining = live_model(
    minute=selected["minute"],
    home_score=selected["home_score"],
    away_score=selected["away_score"],
    home_sot=home_sot,
    away_sot=away_sot,
)

st.subheader("Live match predictions")

st.caption(
    "Experimental estimates only. Generic scoring assumptions are used; "
    "team-specific strength and historical calibration are absent."
)

c1, c2, c3 = st.columns(3)
c1.metric("Home win", f"{markets['Home win']:.1%}")
c2.metric("Draw", f"{markets['Draw']:.1%}")
c3.metric("Away win", f"{markets['Away win']:.1%}")

st.subheader("Goal and double-chance markets")

market_rows = [
    {"Market": name, "Probability": f"{probability:.1%}"}
    for name, probability in markets.items()
]

st.dataframe(
    market_rows,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Most likely final scores")

score_rows = [
    {
        "Projected final score": f"{selected['home']} {h}–{a} {selected['away']}",
        "Probability": f"{probability:.1%}",
    }
    for probability, h, a in top_scores
]

st.dataframe(
    score_rows,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Estimated remaining goals: {selected['home']} "
    f"{home_remaining:.2f} · {selected['away']} {away_remaining:.2f}"
)


# =========================================================
# Show raw stats and save snapshot
# =========================================================

if stats_rows:
    with st.expander("Recognized statistics (debug view)"):
        st.json(stats_rows)

with st.expander("Raw API response (debug view)"):
    st.json(selected["raw"])

try:
    save_snapshot(selected)
except Exception as exc:
    st.warning(f"Could not save snapshot: {exc}")

st.divider()
st.caption(
    "This version is intended for testing model behaviour and data parsing. "
    "Do not interpret these probabilities as proven betting edges."
)