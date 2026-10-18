
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://api.goal-api.com/v1"
REQUEST_TIMEOUT = 20

st.set_page_config(
    page_title="Live Football Edge",
    page_icon="⚽",
    layout="wide",
)

# ============================================================
# GOAL API CONNECTION
# ============================================================

def get_api_key() -> str:
    try:
        return str(st.secrets.get("GOAL_API_KEY", "")).strip()
    except Exception:
        return ""


def api_get(endpoint: str) -> Any:
    key = get_api_key()

    if not key:
        raise RuntimeError(
            "GOAL_API_KEY is missing from Streamlit Secrets. "
            "Add it under your Streamlit app settings."
        )

    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to GOAL API: {exc}"
        ) from exc

    if response.status_code == 401:
        raise RuntimeError(
            "GOAL API rejected the key. Check GOAL_API_KEY in Streamlit Secrets."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "GOAL API denied access. Check your plan and endpoint permissions."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "GOAL API rate limit reached. Wait for the quota to reset."
        )

    if response.status_code >= 500:
        raise RuntimeError(
            f"GOAL API server error ({response.status_code}). Try again later."
        )

    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            f"GOAL API returned an invalid response: {exc}"
        ) from exc

    if isinstance(payload, dict) and payload.get("success") is False:
        error = payload.get("error") or payload.get("message") or payload
        raise RuntimeError(f"GOAL API error: {error}")

    if isinstance(payload, dict):
        return payload.get("data", payload)

    return payload


def list_from_payload(payload: Any) -> list:
    """Extract a list from common API response structures."""
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "fixtures",
            "matches",
            "results",
            "events",
            "statistics",
            "data",
            "items",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        if any(
            key in payload
            for key in ("id", "fixtureId", "fixture_id", "match_id")
        ):
            return [payload]

    return []


@st.cache_data(ttl=60, show_spinner=False)
def get_live_fixtures() -> list:
    payload = api_get("fixtures/live")
    return list_from_payload(payload)


@st.cache_data(ttl=90, show_spinner=False)
def get_fixture_details(fixture_id: str) -> Any:
    return api_get(f"fixtures/{fixture_id}")


@st.cache_data(ttl=90, show_spinner=False)
def get_fixture_statistics(fixture_id: str) -> list:
    payload = api_get(f"fixtures/{fixture_id}/statistics")
    return list_from_payload(payload)


@st.cache_data(ttl=90, show_spinner=False)
def get_fixture_events(fixture_id: str) -> list:
    payload = api_get(f"fixtures/{fixture_id}/events")
    return list_from_payload(payload)


# ============================================================
# FLEXIBLE DATA PARSING
# ============================================================

def first_value(obj: Any, *keys: str, default=None):
    if not isinstance(obj, dict):
        return default

    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value

    return default


def nested(obj: Any, *keys: str) -> dict:
    value = first_value(obj, *keys, default={})
    return value if isinstance(value, dict) else {}


def as_number(value):
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None

    if isinstance(value, str):
        cleaned = value.replace("%", "").strip()
        try:
            number = float(cleaned)
            return number if math.isfinite(number) else None
        except ValueError:
            return None

    return None


def safe_int(value, default=0) -> int:
    number = as_number(value)
    return int(number) if number is not None else default


def parse_score(value):
    """
    Return an integer score, or None if the score is genuinely unavailable.
    Never convert a missing or unrecognized score to zero.
    """
    if isinstance(value, dict):
        value = first_value(
            value,
            "score",
            "goals",
            "value",
            "total",
            "current",
        )

    number = as_number(value)

    if number is None or number < 0:
        return None

    return int(number)


def extract_score(item: dict, side: str):
    """
    Recognize multiple common score formats, including GOAL API's
    homeTeamScore and awayTeamScore fields.
    """
    if side == "home":
        direct_keys = (
            "homeTeamScore",
            "home_score",
            "homeScore",
            "homeGoals",
            "home_goals",
            "match_hometeam_score",
            "scoreHome",
        )
        nested_keys = (
            "home",
            "homeScore",
            "home_score",
            "homeTeam",
            "homeTeamScore",
        )
    else:
        direct_keys = (
            "awayTeamScore",
            "away_score",
            "awayScore",
            "awayGoals",
            "away_goals",
            "match_awayteam_score",
            "scoreAway",
        )
        nested_keys = (
            "away",
            "awayScore",
            "away_score",
            "awayTeam",
            "awayTeamScore",
        )

    # First: explicit score fields on the fixture.
    direct = first_value(item, *direct_keys)
    score = parse_score(direct)

    if score is not None:
        return score

    # Second: common nested score containers.
    for container_key in ("goals", "score", "scores", "currentScore"):
        container = item.get(container_key)

        if not isinstance(container, dict):
            continue

        candidate = first_value(container, *nested_keys)
        score = parse_score(candidate)

        if score is not None:
            return score

        # Some APIs use nested structures such as {"home": {"score": 2}}.
        if side in container:
            score = parse_score(container[side])
            if score is not None:
                return score

    # Third: nested fixture structures.
    fixture = nested(item, "fixture")
    for container_key in ("goals", "score", "scores"):
        container = fixture.get(container_key)

        if isinstance(container, dict):
            candidate = first_value(container, *nested_keys)
            score = parse_score(candidate)
            if score is not None:
                return score

    return None


def team_name(team: Any, fallback: str) -> str:
    if isinstance(team, str):
        return team

    if isinstance(team, dict):
        return str(
            first_value(
                team,
                "name",
                "teamName",
                "team_name",
                "displayName",
                default=fallback,
            )
        )

    return fallback


def team_id(team: Any):
    if isinstance(team, dict):
        return first_value(team, "id", "teamId", "team_id")
    return None


def normalise_fixture(item: dict) -> dict:
    """
    Normalize fixture fields. A missing score remains None instead of
    being incorrectly displayed as 0.
    """
    fixture = nested(item, "fixture")
    teams = nested(item, "teams")
    league = nested(item, "league")
    status = nested(item, "status", "fixtureStatus")

    home = first_value(
        teams,
        "home",
        "homeTeam",
        "home_team",
        default=first_value(
            item,
            "homeTeam",
            "home_team",
            "home",
            default=first_value(
                item,
                "match_hometeam_name",
                "homeTeamName",
                default="Home",
            ),
        ),
    )

    away = first_value(
        teams,
        "away",
        "awayTeam",
        "away_team",
        default=first_value(
            item,
            "awayTeam",
            "away_team",
            "away",
            default=first_value(
                item,
                "match_awayteam_name",
                "awayTeamName",
                default="Away",
            ),
        ),
    )

    home_name = team_name(home, "Home")
    away_name = team_name(away, "Away")

    # FIX: recognize homeTeamScore / awayTeamScore and other formats.
    home_score = extract_score(item, "home")
    away_score = extract_score(item, "away")

    minute = first_value(
        item,
        "matchMinute",
        "matchElapsed",
        "match_minute",
        "elapsed",
        "minute",
        default=first_value(
            status,
            "elapsed",
            "minute",
            default=first_value(item, "match_status"),
        ),
    )

    if isinstance(minute, str):
        digits = ""
        for char in minute:
            if char.isdigit():
                digits += char
            elif digits:
                break
        minute = safe_int(digits, 0)

    fixture_id = first_value(
        item,
        "id",
        "fixtureId",
        "fixture_id",
        "match_id",
        default=first_value(
            fixture,
            "id",
            "fixtureId",
            "fixture_id",
        ),
    )

    league_name = first_value(
        league,
        "name",
        "leagueName",
        "league_name",
        default=first_value(
            item,
            "leagueName",
            "league_name",
            default="Unknown league",
        ),
    )

    return {
        "id": str(fixture_id) if fixture_id is not None else "",
        "home_name": home_name,
        "away_name": away_name,
        "home_id": team_id(home),
        "away_id": team_id(away),
        "home_score": home_score,
        "away_score": away_score,
        "minute": max(0, min(130, safe_int(minute))),
        "league": str(league_name),
        "status": status,
        "raw": item,
    }


def normalise_stat_rows(payload: list) -> tuple[list, list]:
    home_stats = []
    away_stats = []

    for row in payload or []:
        if not isinstance(row, dict):
            continue

        team = nested(row, "team")
        stats = first_value(
            row,
            "statistics",
            "stats",
            "items",
            default=[],
        )

        if isinstance(stats, list):
            name = team_name(team, "")
            target = None

            if name.lower() in ("home", "home team"):
                target = home_stats
            elif name.lower() in ("away", "away team"):
                target = away_stats

            if target is not None:
                target.extend(stats)

            continue

        stat_name = first_value(
            row,
            "type",
            "name",
            "statistic",
            "label",
        )
        home_value = first_value(
            row,
            "home",
            "homeValue",
            "home_value",
        )
        away_value = first_value(
            row,
            "away",
            "awayValue",
            "away_value",
        )

        if stat_name is not None:
            if home_value is not None:
                home_stats.append({
                    "type": stat_name,
                    "value": home_value,
                })
            if away_value is not None:
                away_stats.append({
                    "type": stat_name,
                    "value": away_value,
                })

    return home_stats, away_stats


def stat_value(stat_list, *names):
    wanted = {name.lower() for name in names}

    for item in stat_list or []:
        if not isinstance(item, dict):
            continue

        label = str(
            first_value(
                item,
                "type",
                "name",
                "label",
                "statistic",
                default="",
            )
        ).lower()

        if label in wanted:
            return as_number(
                first_value(item, "value", "total", "amount")
            )

    return None


def count_red_cards(events: list, team_identifier=None, team_label=None) -> int:
    count = 0

    for event in events or []:
        if not isinstance(event, dict):
            continue

        event_team = nested(event, "team")
        event_team_id = team_id(event_team)
        event_team_name = team_name(event_team, "")

        if team_identifier is not None and event_team_id is not None:
            if str(event_team_id) != str(team_identifier):
                continue
        elif team_label and event_team_name:
            if event_team_name.lower() != team_label.lower():
                continue

        detail = str(
            first_value(
                event,
                "detail",
                "description",
                "card",
                default="",
            )
        ).lower()

        event_type = str(
            first_value(event, "type", "event", default="")
        ).lower()

        if "red" in detail or "red card" in event_type:
            count += 1

    return count


def event_rows(events: list) -> list:
    rows = []

    for event in events or []:
        if not isinstance(event, dict):
            continue

        time_info = nested(event, "time", "clock")
        event_minute = first_value(
            event,
            "matchMinute",
            "minute",
            default=first_value(
                time_info,
                "elapsed",
                "minute",
            ),
        )

        extra = first_value(
            time_info,
            "extra",
            default=first_value(event, "extraTime"),
        )

        player = first_value(
            event,
            "player",
            "playerName",
            "player_name",
            default="",
        )

        if isinstance(player, dict):
            player = first_value(
                player,
                "name",
                "displayName",
                default="",
            )

        team = first_value(
            event,
            "team",
            "teamName",
            default="",
        )

        if isinstance(team, dict):
            team = team_name(team, "")

        rows.append({
            "Minute": f"{event_minute or '?'}"
                      + (f"+{extra}" if extra else "") + "'",
            "Team": team,
            "Event": first_value(event, "type", "event", default=""),
            "Detail": first_value(event, "detail", "description", default=""),
            "Player": player,
        })

    return rows


# ============================================================
# EXPERIMENTAL LIVE MODEL
# ============================================================

def poisson_probs(lam: float, max_goals: int = 12):
    lam = max(0.0, float(lam))
    probs = [math.exp(-lam)]

    for k in range(1, max_goals + 1):
        probs.append(probs[-1] * lam / k)

    tail = max(0.0, 1.0 - sum(probs))
    return probs, tail


def probability_pct(value: float) -> str:
    return f"{100 * max(0.0, min(1.0, value)):.1f}%"


def estimate_remaining_goals(
    minute: int,
    team_stats: list,
    opponent_stats: list,
    red_cards_for: int,
    red_cards_against: int,
):
    """Experimental heuristic, NOT a trained or calibrated model."""
    remaining = max(0, 90 - minute)

    if remaining <= 0:
        return 0.0, "No regular time remaining"

    lam = 1.35 * remaining / 90.0
    evidence = []

    shots = stat_value(team_stats, "Total Shots", "Shots")
    opponent_shots = stat_value(opponent_stats, "Total Shots", "Shots")

    shots_on_target = stat_value(
        team_stats,
        "Shots on Goal",
        "Shots on Target",
        "Shots on target",
        "Shots on goal",
    )

    opponent_sot = stat_value(
        opponent_stats,
        "Shots on Goal",
        "Shots on Target",
        "Shots on target",
        "Shots on goal",
    )

    if shots is not None and opponent_shots is not None:
        shot_diff = max(-8.0, min(8.0, shots - opponent_shots))
        lam *= math.exp(0.025 * shot_diff)
        evidence.append("shot volume")

    if shots_on_target is not None:
        lam *= math.exp(0.07 * min(5.0, shots_on_target))
        evidence.append("shots on target")

    if opponent_sot is not None:
        lam *= math.exp(-0.035 * min(5.0, opponent_sot))
        evidence.append("opponent shots on target")

    if red_cards_for:
        lam *= max(0.45, 0.78 ** red_cards_for)
        evidence.append("team red card")

    if red_cards_against:
        lam *= min(1.8, 1.25 ** red_cards_against)
        evidence.append("opponent red card")

    lam = max(0.0, min(2.5, lam))
    return lam, ", ".join(evidence) or "generic prior only"


def build_score_matrix(home_lam: float, away_lam: float, max_goals: int = 10):
    home_probs, _ = poisson_probs(home_lam, max_goals)
    away_probs, _ = poisson_probs(away_lam, max_goals)

    matrix = [
        (hp * ap, h, a)
        for h, hp in enumerate(home_probs)
        for a, ap in enumerate(away_probs)
    ]

    total = sum(prob for prob, _, _ in matrix)

    if total > 0:
        matrix = [
            (prob / total, h, a)
            for prob, h, a in matrix
        ]

    return matrix


def summarize_outcomes(matrix, home_score: int, away_score: int):
    home_win = draw = away_win = 0.0

    for prob, add_home, add_away in matrix:
        h = home_score + add_home
        a = away_score + add_away

        if h > a:
            home_win += prob
        elif h == a:
            draw += prob
        else:
            away_win += prob

    total = home_win + draw + away_win

    if total <= 0:
        return 0.0, 0.0, 0.0

    return home_win / total, draw / total, away_win / total


def over_probability(matrix, home_score: int, away_score: int, line: float):
    return sum(
        prob
        for prob, add_home, add_away in matrix
        if home_score + away_score + add_home + add_away > line
    )


def btts_probability(matrix, home_score: int, away_score: int):
    return sum(
        prob
        for prob, add_home, add_away in matrix
        if home_score + add_home > 0 and away_score + add_away > 0
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title("⚽ Live Football Edge")
st.caption(
    "GOAL API live data · Experimental remaining-goals model · "
    "Not validated for betting"
)

st.info(
    "The predictions are experimental estimates, not verified betting edges. "
    "Do not use them as a substitute for a validated model."
)

col_info, col_refresh = st.columns([4, 1])

with col_info:
    st.write(
        "Choose a live match to inspect its current score, statistics, "
        "events and experimental remaining-match predictions."
    )

with col_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        get_live_fixtures.clear()
        get_fixture_details.clear()
        get_fixture_statistics.clear()
        get_fixture_events.clear()
        st.rerun()


# ============================================================
# LOAD LIVE FIXTURES
# ============================================================

try:
    with st.spinner("Loading live fixtures..."):
        raw_fixtures = get_live_fixtures()
except Exception as exc:
    st.error(f"Could not load live fixtures: {exc}")
    st.stop()

fixtures = [
    normalise_fixture(item)
    for item in raw_fixtures
    if isinstance(item, dict)
]

fixtures = [item for item in fixtures if item["id"]]

if not fixtures:
    st.info(
        "GOAL API returned no live fixtures. This can be normal when "
        "no covered matches are currently live. If you expect live matches, "
        "check your API plan and account dashboard."
    )
    st.stop()


# ============================================================
# LEAGUE FILTER AND MATCH SELECTION
# ============================================================

league_names = sorted({item["league"] for item in fixtures})

selected_league = st.selectbox(
    "Filter by league",
    ["All leagues"] + league_names,
)

filtered = [
    item
    for item in fixtures
    if selected_league == "All leagues"
    or item["league"] == selected_league
]

if not filtered:
    st.info("No live matches found for that league.")
    st.stop()


def fixture_label(item: dict) -> str:
    home_score = (
        str(item["home_score"])
        if item["home_score"] is not None
        else "—"
    )
    away_score = (
        str(item["away_score"])
        if item["away_score"] is not None
        else "—"
    )

    return (
        f"{item['home_name']} {home_score}–"
        f"{away_score} {item['away_name']} · "
        f"{item['minute']}' · {item['league']}"
    )


labels = {
    f"{fixture_label(item)} · ID {item['id']}": item
    for item in filtered
}

selected_label = st.selectbox(
    "Select a live match",
    list(labels.keys()),
)

match = labels[selected_label]

fixture_id = match["id"]
home_name = match["home_name"]
away_name = match["away_name"]
home_id = match["home_id"]
away_id = match["away_id"]
home_score = match["home_score"]
away_score = match["away_score"]
minute = match["minute"]

st.divider()

score_col1, score_col2, score_col3 = st.columns([2, 1, 2])

with score_col1:
    st.subheader(home_name)
    st.metric(
        "Home goals",
        home_score if home_score is not None else "—",
    )

with score_col2:
    st.subheader(f"{minute}'")
    st.caption(match["league"])

with score_col3:
    st.subheader(away_name)
    st.metric(
        "Away goals",
        away_score if away_score is not None else "—",
    )


# ============================================================
# GUARD AGAINST MISSING SCORES
# ============================================================

if home_score is None or away_score is None:
    st.warning(
        "The API returned this fixture, but the app could not identify "
        "both current scores. The score is shown as — instead of assuming "
        "0–0. Predictions are paused to avoid calculations using incorrect "
        "score data. Try Refresh; if the issue continues, the API response "
        "format for this fixture needs to be inspected."
    )

    with st.expander("Score parsing diagnostics"):
        raw = match.get("raw", {})
        st.write("Available top-level response fields:")
        st.code(", ".join(str(key) for key in raw.keys()))

        for key in (
            "homeTeamScore",
            "awayTeamScore",
            "goals",
            "score",
            "scores",
            "currentScore",
        ):
            if key in raw:
                st.write(f"**{key}:**")
                st.json(raw[key])

    st.stop()


# ============================================================
# LOAD MATCH DATA
# ============================================================

if "selected_fixture_id" not in st.session_state:
    st.session_state.selected_fixture_id = None

if st.button(
    "📊 Load statistics, events & predictions",
    type="primary",
):
    st.session_state.selected_fixture_id = fixture_id

if st.session_state.selected_fixture_id != fixture_id:
    st.info("Press the button above to analyze this match.")
    st.stop()

try:
    with st.spinner("Loading match data..."):
        stats_payload = get_fixture_statistics(fixture_id)
        events = get_fixture_events(fixture_id)
except Exception as exc:
    st.error(f"Could not load statistics or events: {exc}")
    st.stop()

home_stats, away_stats = normalise_stat_rows(stats_payload)


# ============================================================
# LIVE STATISTICS
# ============================================================

st.subheader("Live match statistics")

stats_to_show = [
    ("Shots on target", ("Shots on Goal", "Shots on Target", "Shots on target")),
    ("Total shots", ("Total Shots", "Shots")),
    ("Shots off target", ("Shots off Goal", "Shots off target")),
    ("Blocked shots", ("Blocked Shots",)),
    ("Possession", ("Ball Possession", "Possession")),
    ("Corners", ("Corner Kicks", "Corners")),
    ("Yellow cards", ("Yellow Cards",)),
    ("Red cards", ("Red Cards",)),
    ("Expected goals (xG)", ("expected_goals", "xG", "Expected Goals")),
]

rows = []

for label, names in stats_to_show:
    hv = stat_value(home_stats, *names)
    av = stat_value(away_stats, *names)

    if hv is not None or av is not None:
        if label == "Possession":
            h_text = f"{hv:g}%" if hv is not None else "N/A"
            a_text = f"{av:g}%" if av is not None else "N/A"
        else:
            h_text = f"{hv:g}" if hv is not None else "N/A"
            a_text = f"{av:g}" if av is not None else "N/A"

        rows.append({
            "Statistic": label,
            home_name: h_text,
            away_name: a_text,
        })

if rows:
    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
    )
else:
    st.warning(
        "The API response did not match the statistics formats currently "
        "recognized by this app. Predictions will use a generic prior until "
        "we adjust the parser to the actual response."
    )


# ============================================================
# PREDICTIONS
# ============================================================

home_reds = count_red_cards(events, home_id, home_name)
away_reds = count_red_cards(events, away_id, away_name)

st.subheader("Experimental remaining-goals estimates")

home_lam, home_basis = estimate_remaining_goals(
    minute,
    home_stats,
    away_stats,
    home_reds,
    away_reds,
)

away_lam, away_basis = estimate_remaining_goals(
    minute,
    away_stats,
    home_stats,
    away_reds,
    home_reds,
)

home_p, _ = poisson_probs(home_lam)
away_p, _ = poisson_probs(away_lam)

matrix = build_score_matrix(home_lam, away_lam)

p_home, p_draw, p_away = summarize_outcomes(
    matrix,
    home_score,
    away_score,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    f"{home_name}: expected remaining goals",
    f"{home_lam:.2f}",
)
c2.metric(
    f"{home_name}: chance to score again",
    probability_pct(1 - home_p[0]),
)
c3.metric(
    f"{away_name}: expected remaining goals",
    f"{away_lam:.2f}",
)
c4.metric(
    f"{away_name}: chance to score again",
    probability_pct(1 - away_p[0]),
)

with st.expander("What influenced these estimates?"):
    st.write(f"**{home_name}:** {home_basis}")
    st.write(f"**{away_name}:** {away_basis}")
    st.write(
        "The model uses a generic scoring-rate prior, adjusted by available "
        "shots and red-card counts. It has not been trained or calibrated "
        "on historical in-play data."
    )


# ============================================================
# PROJECTED FINAL RESULT
# ============================================================

st.subheader("Projected final result")

st.caption(
    "These are approximate model probabilities. They assume independent "
    "Poisson-distributed remaining goals and regular time up to 90 minutes."
)

c1, c2, c3 = st.columns(3)

c1.metric(f"{home_name} win", probability_pct(p_home))
c2.metric("Draw", probability_pct(p_draw))
c3.metric(f"{away_name} win", probability_pct(p_away))


# ============================================================
# MOST LIKELY FINAL SCORES
# ============================================================

st.subheader("Most likely final scores")

ranked_scores = sorted(matrix, reverse=True)[:5]

score_rows = [
    {
        "Projected final score": (
            f"{home_name} {home_score + add_home}–"
            f"{away_score + add_away} {away_name}"
        ),
        "Probability within displayed score range": probability_pct(prob),
    }
    for prob, add_home, add_away in ranked_scores
]

st.dataframe(
    score_rows,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# OVER / UNDER MARKETS
# ============================================================

st.subheader("Total goals markets")

market_rows = []

for line in (0.5, 1.5, 2.5, 3.5, 4.5):
    over_p = over_probability(
        matrix,
        home_score,
        away_score,
        line,
    )

    market_rows.append({
        "Market": f"Over {line}",
        "Probability": probability_pct(over_p),
    })

    market_rows.append({
        "Market": f"Under {line}",
        "Probability": probability_pct(1 - over_p),
    })

st.dataframe(
    market_rows,
    hide_index=True,
    use_container_width=True,
)


# ============================================================
# BOTH TEAMS TO SCORE
# ============================================================

st.subheader("Both teams to score")

btts_p = btts_probability(
    matrix,
    home_score,
    away_score,
)

btts_c1, btts_c2 = st.columns(2)

btts_c1.metric("BTTS — Yes", probability_pct(btts_p))
btts_c2.metric("BTTS — No", probability_pct(1 - btts_p))


# ============================================================
# RECENT MATCH EVENTS
# ============================================================

st.subheader("Recent match events")

rows = event_rows(events)

if rows:
    st.dataframe(
        list(reversed(rows[-15:])),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No match events were returned for this fixture.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.warning(
    "EXPERIMENTAL MODEL — These outputs are not validated betting edges. "
    "Statistics may be missing or differently formatted, and the scoring "
    "heuristic has not been calibrated against historical in-play results. "
    "Treat all probabilities as provisional."
)

st.caption(
    f"Data provider: GOAL API · Fixture ID: {fixture_id} · "
    f"Calculated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)
st.caption(
    f"Data provider: GOAL API · Fixture ID: {fixture_id} · "
    f"Calculated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)