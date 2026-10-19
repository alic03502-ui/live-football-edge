
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import requests
import streamlit as st

from model import (
    prepare_history,
    model_status,
    pre_match_lambdas,
    in_play_lambdas,
    score_matrix,
    markets,
    top_final_scores,
    walk_forward_evaluation,
)
from storage import (
    init_db,
    load_history,
    save_history,
    save_snapshot,
    recent_snapshots,
)

BASE_URL = "https://api.goal-api.com/v1"
TIMEOUT = 20

init_db()

st.set_page_config(
    page_title="Live Football Edge",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Live Football Edge")
st.caption(
    "Live GOAL API data · Optional historical model · "
    "No bookmaker odds required for live monitoring"
)

st.info(
    "Live-only mode is supported. Without reliable historical results, "
    "the app displays live data but does not generate betting probabilities."
)


# --------------------------------------------------
# GOAL API connection
# --------------------------------------------------

def get_key() -> str:
    try:
        return str(st.secrets.get("GOAL_API_KEY", "")).strip()
    except Exception:
        return ""


def api_get(endpoint: str):
    key = get_key()

    if not key:
        raise RuntimeError(
            "GOAL_API_KEY is missing from Streamlit Secrets."
        )

    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            },
            timeout=TIMEOUT,
        )

        if response.status_code == 401:
            raise RuntimeError(
                "GOAL API rejected the key. Check Streamlit Secrets."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "GOAL API denied access or this endpoint is not in your plan."
            )

        if response.status_code == 429:
            raise RuntimeError(
                "GOAL API quota/rate limit reached. Wait before retrying."
            )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"GOAL API request failed: {exc}"
        ) from exc

    if isinstance(data, dict):
        if data.get("success") is False:
            raise RuntimeError(
                str(
                    data.get("error")
                    or data.get("message")
                    or "GOAL API error"
                )
            )

        return data.get("data", data)

    return data


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live():
    return as_list(api_get("fixtures/live"))


def as_list(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [
            item for item in payload
            if isinstance(item, dict)
        ]

    if isinstance(payload, dict):
        for key in (
            "fixtures",
            "matches",
            "results",
            "events",
            "items",
            "data",
        ):
            if isinstance(payload.get(key), list):
                return [
                    item for item in payload[key]
                    if isinstance(item, dict)
                ]

        if any(
            key in payload
            for key in (
                "id",
                "fixtureId",
                "fixture_id",
                "match_id",
            )
        ):
            return [payload]

    return []


def first(obj, *keys, default=None):
    if not isinstance(obj, dict):
        return default

    for key in keys:
        if obj.get(key) is not None:
            return obj[key]

    return default


# --------------------------------------------------
# Fixture and score parsing
# --------------------------------------------------

def parse_score(value):
    if isinstance(value, dict):
        value = first(
            value,
            "score",
            "goals",
            "value",
            "total",
            "current",
        )

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(str(value).strip())

        if number >= 0 and number.is_integer():
            return int(number)

    except (TypeError, ValueError):
        pass

    return None


def score_for(item, side):
    if side == "home":
        keys = (
            "homeTeamScore",
            "home_score",
            "homeScore",
            "homeGoals",
            "home_goals",
            "match_hometeam_score",
        )
        nested_keys = (
            "home",
            "homeScore",
            "home_score",
            "homeTeam",
            "homeTeamScore",
        )
    else:
        keys = (
            "awayTeamScore",
            "away_score",
            "awayScore",
            "awayGoals",
            "away_goals",
            "match_awayteam_score",
        )
        nested_keys = (
            "away",
            "awayScore",
            "away_score",
            "awayTeam",
            "awayTeamScore",
        )

    value = parse_score(first(item, *keys))

    if value is not None:
        return value

    for container_name in (
        "goals",
        "score",
        "scores",
        "currentScore",
    ):
        container = item.get(container_name)

        if isinstance(container, dict):
            value = parse_score(
                first(
                    container,
                    *nested_keys,
                    default=container.get(side),
                )
            )

            if value is not None:
                return value

    return None


def team_label(value, fallback):
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return str(
            first(
                value,
                "name",
                "teamName",
                "team_name",
                "displayName",
                default=fallback,
            )
        )

    return fallback


def normalize(item):
    teams = (
        item.get("teams")
        if isinstance(item.get("teams"), dict)
        else {}
    )

    home = first(
        teams,
        "home",
        "homeTeam",
        "home_team",
        default=first(
            item,
            "homeTeam",
            "home_team",
            "home",
            "match_hometeam_name",
            default="Home",
        ),
    )

    away = first(
        teams,
        "away",
        "awayTeam",
        "away_team",
        default=first(
            item,
            "awayTeam",
            "away_team",
            "away",
            "match_awayteam_name",
            default="Away",
        ),
    )

    status_obj = first(
        item,
        "status",
        "fixtureStatus",
        default={},
    )

    minute = first(
        item,
        "matchMinute",
        "matchElapsed",
        "match_minute",
        "elapsed",
        "minute",
        default=first(
            status_obj,
            "elapsed",
            "minute",
            default=first(item, "match_status", default=0),
        ),
    )

    if isinstance(minute, str):
        match = re.search(r"\d+", minute)
        minute = int(match.group()) if match else 0

    try:
        minute = max(0, min(120, int(minute or 0)))
    except (TypeError, ValueError):
        minute = 0

    league_obj = (
        item.get("league")
        if isinstance(item.get("league"), dict)
        else {}
    )

    return {
        "id": str(
            first(
                item,
                "id",
                "fixtureId",
                "fixture_id",
                "match_id",
                default="",
            )
        ),
        "home_name": team_label(home, "Home"),
        "away_name": team_label(away, "Away"),
        "home_score": score_for(item, "home"),
        "away_score": score_for(item, "away"),
        "minute": minute,
        "league": str(
            first(
                league_obj,
                "name",
                "leagueName",
                default=first(
                    item,
                    "leagueName",
                    "league_name",
                    default="Unknown league",
                ),
            )
        ),
        "status": str(status_obj),
        "raw": item,
    }


# --------------------------------------------------
# Statistics parsing
# --------------------------------------------------

def stat_value(rows, *names):
    wanted = {name.lower() for name in names}

    for row in rows:
        if not isinstance(row, dict):
            continue

        label = str(
            first(
                row,
                "type",
                "name",
                "label",
                "statistic",
                default="",
            )
        ).lower()

        if label not in wanted:
            continue

        value = first(row, "value", "total", "amount")

        try:
            return float(
                str(value).replace("%", "").strip()
            )
        except (TypeError, ValueError):
            return None

    return None


# --------------------------------------------------
# Optional historical data
# No historical dataset is bundled or required.
# --------------------------------------------------

with st.sidebar:
    st.header("Historical data (optional)")

    st.caption(
        "The live dashboard works without CCV historical data. "
        "Import completed results only if you already have a reliable dataset."
    )

    with st.expander(
        "Import completed results",
        expanded=False,
    ):
        st.write("Required CSV columns:")

        st.code(
            "date,home_team,away_team,home_goals,away_goals"
        )

        uploaded = st.file_uploader(
            "Optional historical CSV",
            type=["csv"],
        )

        if uploaded is not None:
            if st.button(
                "Import history",
                use_container_width=True,
            ):
                try:
                    historical = prepare_history(
                        pd.read_csv(uploaded)
                    )

                    count = save_history(historical)

                    st.success(
                        f"Imported/updated {count} historical rows."
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"History import failed: {exc}"
                    )


history = load_history()

with st.expander(
    "Historical database status",
    expanded=False,
):
    st.write(
        f"Stored completed matches: **{len(history)}**"
    )

    if history.empty:
        st.info(
            "No historical dataset is installed. Live monitoring "
            "is available; betting predictions and historical "
            "evaluation are disabled."
        )
    else:
        st.write(
            f"Date range: {history.date.min()} — {history.date.max()}"
        )

        st.dataframe(
            history.tail(10),
            use_container_width=True,
            hide_index=True,
        )


if not history.empty:
    st.subheader("Historical evaluation")

    if st.button("Run walk-forward evaluation"):
        with st.spinner("Evaluating historical matches..."):
            try:
                report = walk_forward_evaluation(history)

                if report.empty:
                    st.warning(
                        "Not enough usable history to evaluate."
                    )
                else:
                    st.session_state["evaluation_report"] = report

            except Exception as exc:
                st.error(
                    f"Evaluation failed: {exc}"
                )

    if "evaluation_report" in st.session_state:
        report = st.session_state["evaluation_report"]

        if "brier_1x2" in report.columns:
            brier = report["brier_1x2"].mean()

            st.metric(
                "Mean multiclass Brier score (lower is better)",
                f"{brier:.4f}",
            )

        st.dataframe(
            report.tail(100),
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# Load live fixtures
# --------------------------------------------------

st.divider()

col1, col2 = st.columns([4, 1])

with col1:
    st.write(
        "Choose a live fixture. Responses are cached for 60 seconds "
        "to reduce API requests."
    )

with col2:
    refresh = st.button(
        "Refresh fixtures",
        use_container_width=True,
    )

if refresh:
    fetch_live.clear()

try:
    fixtures = [
        normalize(item)
        for item in fetch_live()
    ]

except Exception as exc:
    st.error(
        f"Could not load live fixtures: {exc}"
    )
    st.stop()


fixtures = [
    fixture for fixture in fixtures
    if fixture["id"]
]

if not fixtures:
    st.info(
        "No live fixtures were returned. This can happen when "
        "no matches covered by the API are currently live."
    )
    st.stop()


league_options = ["All leagues"] + sorted(
    {
        fixture["league"]
        for fixture in fixtures
    }
)

selected_league = st.selectbox(
    "League",
    league_options,
)

if selected_league != "All leagues":
    fixtures = [
        fixture for fixture in fixtures
        if fixture["league"] == selected_league
    ]

if not fixtures:
    st.info("No fixtures are available for this league.")
    st.stop()


def fixture_text(fixture):
    home_score = (
        "—"
        if fixture["home_score"] is None
        else str(fixture["home_score"])
    )

    away_score = (
        "—"
        if fixture["away_score"] is None
        else str(fixture["away_score"])
    )

    return (
        f"{fixture['home_name']} {home_score}–{away_score} "
        f"{fixture['away_name']} · {fixture['minute']}' "
        f"· {fixture['league']}"
    )


selected = st.selectbox(
    "Live match",
    fixtures,
    format_func=fixture_text,
)


home_score_display = (
    "—"
    if selected["home_score"] is None
    else str(selected["home_score"])
)

away_score_display = (
    "—"
    if selected["away_score"] is None
    else str(selected["away_score"])
)

st.subheader(
    f"{selected['home_name']} {home_score_display} – "
    f"{away_score_display} {selected['away_name']}"
)

st.caption(
    f"{selected['league']} · {selected['minute']}' "
    f"· Fixture ID {selected['id']}"
)


# --------------------------------------------------
# Stop on unrecognized scores.
# Never silently assume 0–0.
# --------------------------------------------------

if (
    selected["home_score"] is None
    or selected["away_score"] is None
):
    st.error(
        "One or both scores are missing or unrecognized. "
        "Predictions are paused rather than assuming 0–0."
    )

    with st.expander("Score parser diagnostics"):
        st.write(list(selected["raw"].keys()))
        st.json(selected["raw"])

    st.stop()


# --------------------------------------------------
# Fetch live statistics and events
# --------------------------------------------------

try:
    stats = as_list(
        api_get(
            f"fixtures/{selected['id']}/statistics"
        )
    )

    # Events are fetched for future inspection, but red-card
    # adjustments are not applied because team attribution has
    # not been verified for this API response.
    events = as_list(
        api_get(
            f"fixtures/{selected['id']}/events"
        )
    )

except Exception as exc:
    st.warning(
        f"Live stats/events unavailable: {exc}. "
        "Unavailable statistics remain unknown."
    )

    stats = []
    events = []


home_stats = []
away_stats = []

for row in stats:
    team_obj = (
        row.get("team")
        if isinstance(row.get("team"), dict)
        else {}
    )

    team_name = str(
        first(
            team_obj,
            "name",
            default=first(
                row,
                "teamName",
                "team",
                default="",
            ),
        )
    ).lower()

    nested_stats = first(
        row,
        "statistics",
        "stats",
        "items",
        default=None,
    )

    if isinstance(nested_stats, list):
        target = (
            home_stats
            if "home" in team_name
            else away_stats
            if "away" in team_name
            else None
        )

        if target is not None:
            target.extend(nested_stats)

    else:
        name = first(
            row,
            "type",
            "name",
            "label",
            "statistic",
        )

        home_value = first(
            row,
            "home",
            "homeValue",
            "home_value",
        )

        away_value = first(
            row,
            "away",
            "awayValue",
            "away_value",
        )

        if name is not None:
            if home_value is not None:
                home_stats.append(
                    {
                        "type": name,
                        "value": home_value,
                    }
                )

            if away_value is not None:
                away_stats.append(
                    {
                        "type": name,
                        "value": away_value,
                    }
                )


# --------------------------------------------------
# Display available live statistics
# --------------------------------------------------

stats_rows = []

stat_definitions = [
    (
        "Shots on target",
        (
            "Shots on Goal",
            "Shots on Target",
            "Shots on target",
        ),
    ),
    (
        "Total shots",
        ("Total Shots", "Shots"),
    ),
    (
        "Possession",
        ("Ball Possession", "Possession"),
    ),
    (
        "Corners",
        ("Corner Kicks", "Corners"),
    ),
    (
        "xG",
        (
            "expected_goals",
            "Expected Goals",
            "xG",
        ),
    ),
]

for label, aliases in stat_definitions:
    home_value = stat_value(
        home_stats,
        *aliases,
    )

    away_value = stat_value(
        away_stats,
        *aliases,
    )

    if (
        home_value is not None
        or away_value is not None
    ):
        stats_rows.append(
            {
                "Statistic": label,
                selected["home_name"]: home_value,
                selected["away_name"]: away_value,
            }
        )


if stats_rows:
    st.subheader("Available live statistics")

    st.dataframe(
        stats_rows,
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info(
        "No recognized live statistics were returned by the API."
    )


# --------------------------------------------------
# Save live snapshot
# --------------------------------------------------

try:
    save_snapshot(
        selected,
        json.dumps(
            selected["raw"],
            ensure_ascii=False,
            default=str,
        ),
    )

except Exception as exc:
    st.warning(
        f"Could not save this snapshot: {exc}"
    )


# --------------------------------------------------
# Predictions: require reliable historical input
# --------------------------------------------------

if history.empty:
    st.info(
        "LIVE-ONLY MODE: Live fixtures, scores, recognized statistics, "
        "and snapshots are available. No betting probabilities are "
        "generated because historical team-strength data is missing."
    )

else:
    status = model_status(
        history,
        selected["home_name"],
        selected["away_name"],
    )

    if not status.ready:
        st.warning(
            f"NO PREDICTION / INSUFFICIENT DATA: {status.reason}"
        )

    else:
        home_pre, away_pre = pre_match_lambdas(
            history,
            selected["home_name"],
            selected["away_name"],
        )

        home_sot = stat_value(
            home_stats,
            "Shots on Goal",
            "Shots on Target",
            "Shots on target",
        )

        away_sot = stat_value(
            away_stats,
            "Shots on Goal",
            "Shots on Target",
            "Shots on target",
        )

        # Red cards are deliberately not inferred from event text
        # until the API's event team attribution is verified.
        home_reds = 0
        away_reds = 0

        home_remaining, away_remaining = in_play_lambdas(
            home_pre,
            away_pre,
            selected["minute"],
            selected["home_score"],
            selected["away_score"],
            home_sot,
            away_sot,
            home_reds,
            away_reds,
        )

        matrix = score_matrix(
            home_remaining,
            away_remaining,
        )

        probs = markets(
            matrix,
            selected["home_score"],
            selected["away_score"],
        )

        st.subheader("Experimental live model estimates")

        st.caption(
            "The model uses optional imported historical results and "
            "a heuristic in-play adjustment. These estimates have not "
            "been independently validated or calibrated."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            f"{selected['home_name']} win",
            f"{probs['1']:.1%}",
        )

        col2.metric(
            "Draw",
            f"{probs['X']:.1%}",
        )

        col3.metric(
            f"{selected['away_name']} win",
            f"{probs['2']:.1%}",
        )

        with st.expander("Double chance and goal markets"):
            market_rows = [
                {
                    "Market": name,
                    "Probability": f"{value:.1%}",
                }
                for name, value in probs.items()
            ]

            st.dataframe(
                market_rows,
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("Most likely final scores")

        score_rows = [
            {
                "Projected final score": (
                    f"{selected['home_name']} {home_goals}–"
                    f"{away_goals} {selected['away_name']}"
                ),
                "Probability": f"{probability:.1%}",
            }
            for probability, home_goals, away_goals
            in top_final_scores(
                matrix,
                selected["home_score"],
                selected["away_score"],
            )
        ]

        st.dataframe(
            score_rows,
            hide_index=True,
            use_container_width=True,
        )


# --------------------------------------------------
# Recent snapshots
# --------------------------------------------------

st.subheader("Recent saved live snapshots")

snapshots = recent_snapshots(20)

if not snapshots.empty:
    st.dataframe(
        snapshots,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption(
        "Snapshots will appear here after the app saves observations."
    )


# --------------------------------------------------
# Limitations
# --------------------------------------------------

st.divider()

st.warning(
    "NO BOOKMAKER ODDS SOURCE IS CONNECTED. No EV/edge is calculated. "
    "Live-only mode does not generate betting probabilities. "
    "When historical results are imported, model estimates remain "
    "experimental and should not be treated as validated betting advice."
)