
import streamlit as st
import requests
from datetime import datetime
from typing import Any

# ---------------------------------------------------------
# Live Football Edge — API-Football only
# ---------------------------------------------------------

BASE_URL = "https://v3.football.api-sports.io"
REQUEST_TIMEOUT = 15

st.set_page_config(
    page_title="Live Football Edge",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Live Football Edge")
st.caption("Live match data and statistics | API-Football")

# Read the key from Streamlit Secrets.
try:
    API_KEY = st.secrets.get("API_FOOTBALL_KEY", "")
except Exception:
    API_KEY = ""

if not API_KEY:
    st.error(
        "API-Football key not configured. "
        "Open Streamlit Settings → Secrets and add "
        "API_FOOTBALL_KEY."
    )
    st.code('API_FOOTBALL_KEY = "your_actual_api_key"', language="toml")
    st.stop()


def api_get(endpoint: str, params: dict | None = None) -> list:
    """Request an API-Football endpoint and validate its response."""
    response = requests.get(
        f"{BASE_URL}/{endpoint.lstrip('/')}",
        headers={"x-apisports-key": API_KEY},
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 401:
        raise RuntimeError(
            "API key rejected (401). Check the key in Streamlit Secrets."
        )
    if response.status_code == 403:
        raise RuntimeError(
            "Access forbidden (403). Check your plan and endpoint access."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "API rate limit reached. Wait before making more requests."
        )

    response.raise_for_status()
    payload = response.json()

    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"API-Football reported: {errors}")

    results = payload.get("response")
    if not isinstance(results, list):
        raise RuntimeError("Unexpected API response format.")

    return results


def get_live_fixtures() -> list:
    return api_get("fixtures", {"live": "all"})


def get_fixture_statistics(fixture_id: int) -> list:
    return api_get("fixtures/statistics", {"fixture": fixture_id})


def get_fixture_events(fixture_id: int) -> list:
    return api_get("fixtures/events", {"fixture": fixture_id})


def safe_number(value: Any) -> str:
    return str(value) if value is not None else "—"


# ---------------------------------------------------------
# Sidebar and live match list
# ---------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    refresh = st.button("🔄 Refresh live matches", type="primary")
    st.caption(
        "Refresh manually to help preserve your API request quota."
    )

try:
    with st.spinner("Fetching live fixtures..."):
        fixtures = get_live_fixtures()
except requests.Timeout:
    st.error("The API request timed out. Please try again shortly.")
    st.stop()
except requests.RequestException as exc:
    st.error(f"Network/API request failed: {exc}")
    st.stop()
except (RuntimeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.metric("Live fixtures returned by API", len(fixtures))

if not fixtures:
    st.info(
        "No live fixtures were returned right now. "
        "This does not necessarily mean your API key is invalid."
    )
    st.stop()

# Display matches in a predictable order.
league_names = sorted(
    {
        item.get("league", {}).get("name", "Unknown league")
        for item in fixtures
    }
)

selected_league = st.selectbox(
    "Filter by league",
    ["All leagues"] + league_names,
)

if selected_league != "All leagues":
    fixtures = [
        item for item in fixtures
        if item.get("league", {}).get("name", "Unknown league")
        == selected_league
    ]

match_labels = {}
for item in fixtures:
    fixture = item.get("fixture", {})
    teams = item.get("teams", {})
    goals = item.get("goals", {})
    league = item.get("league", {})
    status = fixture.get("status", {})

    fixture_id = fixture.get("id")
    home = teams.get("home", {}).get("name", "Home")
    away = teams.get("away", {}).get("name", "Away")
    home_goals = safe_number(goals.get("home"))
    away_goals = safe_number(goals.get("away"))
    minute = status.get("elapsed")
    clock = f"{minute}′" if minute is not None else status.get("long", "—")

    label = (
        f"{home} {home_goals} - {away_goals} {away} "
        f"| {clock} | {league.get('name', 'Unknown league')}"
    )
    match_labels[label] = fixture_id

st.subheader("Live matches")

for label in match_labels:
    st.write(f"• {label}")

st.divider()
st.subheader("Match details")

selected_match = st.selectbox(
    "Choose a match to inspect",
    list(match_labels.keys()),
)

selected_fixture_id = match_labels[selected_match]

if st.button("Load match statistics and events"):
    try:
        with st.spinner("Loading match details..."):
            statistics = get_fixture_statistics(selected_fixture_id)
            events = get_fixture_events(selected_fixture_id)

        left, right = st.columns(2)

        with left:
            st.markdown("### Match statistics")
            if not statistics:
                st.info("Statistics are not available for this match yet.")
            else:
                for team_data in statistics:
                    team_name = team_data.get("team", {}).get(
                        "name", "Team"
                    )
                    st.markdown(f"**{team_name}**")
                    rows = team_data.get("statistics", [])
                    if rows:
                        st.dataframe(
                            [
                                {
                                    "Statistic": row.get("type", "Unknown"),
                                    "Value": safe_number(row.get("value")),
                                }
                                for row in rows
                            ],
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.write("No statistics supplied.")

        with right:
            st.markdown("### Match events")
            if not events:
                st.info("No events supplied for this match yet.")
            else:
                for event in events:
                    elapsed = event.get("time", {}).get("elapsed")
                    extra = event.get("time", {}).get("extra")
                    minute = f"{elapsed}′" if elapsed is not None else "—"
                    if extra:
                        minute += f"+{extra}"

                    team = event.get("team", {}).get("name", "Unknown team")
                    player = event.get("player", {}).get("name", "")
                    event_type = event.get("type", "")
                    detail = event.get("detail", "")

                    st.write(
                        f"**{minute}** — {team}: "
                        f"{event_type} — {detail}"
                        + (f" ({player})" if player else "")
                    )

    except requests.Timeout:
        st.error("The details request timed out. Please try again.")
    except requests.RequestException as exc:
        st.error(f"Could not retrieve match details: {exc}")
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))

st.divider()
st.caption(
    "Data availability varies by competition and subscription. "
    "This version displays provider data; it does not yet generate "
    "predictions or claim betting value."
)