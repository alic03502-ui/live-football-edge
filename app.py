
from __future__ import annotations

import math
from datetime import datetime, timezone

import requests
import streamlit as st

BASE_URL = "https://v3.football.api-sports.io"
REQUEST_TIMEOUT = 15

st.set_page_config(
    page_title="Live Football Edge",
    page_icon="⚽",
    layout="wide",
)

# ---------- API ----------

def get_api_key() -> str:
    try:
        return str(st.secrets.get("API_FOOTBALL_KEY", "")).strip()
    except Exception:
        return ""


def api_get(endpoint: str, params: dict | None = None):
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "API_FOOTBALL_KEY is missing from Streamlit Secrets."
        )

    response = requests.get(
        f"{BASE_URL}/{endpoint.lstrip('/')}",
        headers={"x-apisports-key": key},
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code in (401, 403):
        raise RuntimeError(
            "API-Football rejected the key or access to this endpoint."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "API request limit reached. Wait before trying again."
        )

    response.raise_for_status()
    payload = response.json()

    if payload.get("errors"):
        raise RuntimeError(f"API-Football error: {payload['errors']}")

    return payload.get("response", [])


@st.cache_data(ttl=25, show_spinner=False)
def get_live_fixtures():
    return api_get("fixtures", {"live": "all"})


@st.cache_data(ttl=30, show_spinner=False)
def get_fixture_statistics(fixture_id: int):
    return api_get("fixtures/statistics", {"fixture": fixture_id})


@st.cache_data(ttl=30, show_spinner=False)
def get_fixture_events(fixture_id: int):
    return api_get("fixtures/events", {"fixture": fixture_id})


# ---------- Data helpers ----------

def as_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def stat_value(stat_list, *names):
    wanted = {name.lower() for name in names}
    for item in stat_list or []:
        label = str(item.get("type", "")).lower()
        if label in wanted:
            return as_number(item.get("value"))
    return None


def unpack_stats(api_rows, home_id, away_id):
    by_team = {}
    for row in api_rows or []:
        team = row.get("team") or {}
        team_id = team.get("id")
        if team_id is not None:
            by_team[team_id] = row.get("statistics") or []

    return (
        by_team.get(home_id, []),
        by_team.get(away_id, []),
    )


def count_red_cards(events, team_id):
    count = 0
    for event in events or []:
        if (event.get("team") or {}).get("id") != team_id:
            continue
        detail = str(event.get("detail") or "").lower()
        event_type = str(event.get("type") or "").lower()
        if "red card" in detail or (
            event_type == "card" and "red" in detail
        ):
            count += 1
    return count


def match_minute(fixture):
    elapsed = (fixture.get("fixture") or {}).get("status") or {}
    value = elapsed.get("elapsed")
    extra = elapsed.get("extra")
    if value is None:
        return 0
    try:
        return max(0, min(130, int(value) + int(extra or 0)))
    except (TypeError, ValueError):
        return 0


def poisson_probs(lam, max_goals=12):
    """Return probabilities for 0..max_goals and a tail bucket."""
    lam = max(0.0, float(lam))
    probs = [math.exp(-lam)]
    for k in range(1, max_goals + 1):
        probs.append(probs[-1] * lam / k)
    tail = max(0.0, 1.0 - sum(probs))
    return probs, tail


def poisson_at_least(probs, tail, threshold):
    if threshold <= 0:
        return 1.0
    if threshold >= len(probs):
        return tail
    return min(1.0, sum(probs[threshold:]) + tail)


def probability_pct(value):
    return f"{100 * max(0.0, min(1.0, value)):.1f}%"


def estimate_remaining_goals(
    minute,
    team_stats,
    opponent_stats,
    red_cards_for,
    red_cards_against,
):
    """
    Experimental heuristic, not a trained or validated model.

    Starts with a generic remaining-goal rate and adjusts it using
    available shots, shots on target and red-card counts.
    """
    remaining = max(0, 90 - minute)
    if remaining <= 0:
        return 0.0, "No regular time remaining"

    # Generic prior: 1.35 goals per team over a full 90 minutes.
    lam = 1.35 * remaining / 90.0
    evidence = []

    shots = stat_value(team_stats, "Total Shots")
    opponent_shots = stat_value(opponent_stats, "Total Shots")
    shots_on_target = stat_value(
        team_stats, "Shots on Goal", "Shots on Target"
    )
    opponent_sot = stat_value(
        opponent_stats, "Shots on Goal", "Shots on Target"
    )

    # Adjust conservatively; API statistics may be incomplete.
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

    # Avoid extreme outputs from a small number of live statistics.
    lam = max(0.01, min(2.5, lam))
    return lam, ", ".join(evidence) or "generic prior only"


def build_score_matrix(home_lam, away_lam, max_goals=10):
    home_probs, home_tail = poisson_probs(home_lam, max_goals)
    away_probs, away_tail = poisson_probs(away_lam, max_goals)

    # Matrix covers 0..max_goals. Tail is excluded from exact-score
    # ranking and 1X2 calculations, so those calculations are approximate.
    matrix = []
    for h, hp in enumerate(home_probs):
        for a, ap in enumerate(away_probs):
            matrix.append((hp * ap, h, a))

    total = sum(p for p, _, _ in matrix)
    if total > 0:
        matrix = [(p / total, h, a) for p, h, a in matrix]

    return matrix, home_tail, away_tail


def summarize_outcomes(matrix, home_score, away_score):
    home_win = draw = away_win = 0.0
    for p, add_home, add_away in matrix:
        h = home_score + add_home
        a = away_score + add_away
        if h > a:
            home_win += p
        elif h == a:
            draw += p
        else:
            away_win += p
    total = home_win + draw + away_win
    if total:
        return home_win / total, draw / total, away_win / total
    return 0.0, 0.0, 0.0


def total_goals_probability(matrix, home_score, away_score, line):
    return sum(
        p for p, add_home, add_away in matrix
        if home_score + away_score + add_home + add_away > line
    )


def btts_probability(matrix, home_score, away_score):
    return sum(
        p for p, add_home, add_away in matrix
        if home_score + add_home > 0 and away_score + add_away > 0
    )


def next_goal_probabilities(home_lam, away_lam):
    total = home_lam + away_lam
    if total <= 0:
        return 0.0, 0.0, 1.0
    # Conditional on another goal being scored.
    home_next = home_lam / total
    away_next = away_lam / total
    # Approximate chance no further goal is scored.
    no_more = math.exp(-total)
    return (
        home_next * (1 - no_more),
        away_next * (1 - no_more),
        no_more,
    )


# ---------- User interface ----------

st.title("⚽ Live Football Edge")
st.caption(
    "API-Football live data · Experimental remaining-goals model · "
    "Not yet validated for betting"
)

if "selected_fixture_id" not in st.session_state:
    st.session_state.selected_fixture_id = None

top_left, top_right = st.columns([3, 1])
with top_left:
    st.write(
        "Choose a live fixture, load its statistics and inspect "
        "experimental estimates for the rest of the match."
    )
with top_right:
    if st.button("🔄 Refresh live data", use_container_width=True):
        get_live_fixtures.clear()
        get_fixture_statistics.clear()
        get_fixture_events.clear()
        st.rerun()

try:
    fixtures = get_live_fixtures()
except Exception as exc:
    st.error(f"Could not load live fixtures: {exc}")
    st.stop()

if not fixtures:
    st.info("No live fixtures are currently returned by the API.")
    st.stop()

league_names = sorted({
    (item.get("league") or {}).get("name", "Unknown league")
    for item in fixtures
})
selected_league = st.selectbox("Filter by league", ["All leagues"] + league_names)

filtered = [
    item for item in fixtures
    if selected_league == "All leagues"
    or (item.get("league") or {}).get("name", "Unknown league")
    == selected_league
]

def fixture_label(item):
    teams = item.get("teams") or {}
    home = (teams.get("home") or {}).get("name", "Home")
    away = (teams.get("away") or {}).get("name", "Away")
    goals = item.get("goals") or {}
    home_goals = goals.get("home")
    away_goals = goals.get("away")
    status = (item.get("fixture") or {}).get("status") or {}
    elapsed = status.get("elapsed")
    minute = f"{elapsed}'" if elapsed is not None else status.get("short", "Live")
    league = (item.get("league") or {}).get("name", "Unknown league")
    return (
        f"{home} {home_goals if home_goals is not None else '-'}–"
        f"{away_goals if away_goals is not None else '-'} {away} · "
        f"{minute} · {league}"
    )


labels = {fixture_label(item): item for item in filtered}
if not labels:
    st.info("No fixtures match that league filter.")
    st.stop()

chosen_label = st.selectbox("Select a live match", list(labels.keys()))
match = labels[chosen_label]
fixture_info = match.get("fixture") or {}
teams = match.get("teams") or {}
home_team = teams.get("home") or {}
away_team = teams.get("away") or {}
home_id = home_team.get("id")
away_id = away_team.get("id")
home_name = home_team.get("name", "Home")
away_name = away_team.get("name", "Away")
goals = match.get("goals") or {}
home_score = int(goals.get("home") or 0)
away_score = int(goals.get("away") or 0)
minute = match_minute(match)
fixture_id = int(fixture_info.get("id"))

st.divider()
score_col1, score_col2, score_col3 = st.columns([2, 1, 2])
with score_col1:
    st.subheader(home_name)
    st.metric("Home goals", home_score)
with score_col2:
    st.subheader(f"{minute}'")
    st.caption((fixture_info.get("status") or {}).get("long", "Live"))
with score_col3:
    st.subheader(away_name)
    st.metric("Away goals", away_score)

if st.button("📊 Load stats, events & predictions", type="primary"):
    st.session_state.selected_fixture_id = fixture_id

if st.session_state.selected_fixture_id != fixture_id:
    st.info("Press **Load stats, events & predictions** to analyze this match.")
    st.stop()

try:
    with st.spinner("Loading match statistics and events..."):
        stats_rows = get_fixture_statistics(fixture_id)
        events = get_fixture_events(fixture_id)
except Exception as exc:
    st.error(f"Could not load match statistics/events: {exc}")
    st.stop()

home_stats, away_stats = unpack_stats(stats_rows, home_id, away_id)
home_reds = count_red_cards(events, home_id)
away_reds = count_red_cards(events, away_id)

st.subheader("Live match statistics")
stats_to_show = [
    ("Shots on target", ("Shots on Goal", "Shots on Target")),
    ("Total shots", ("Total Shots",)),
    ("Shots off target", ("Shots off Goal",)),
    ("Blocked shots", ("Blocked Shots",)),
    ("Possession", ("Ball Possession",)),
    ("Corners", ("Corner Kicks",)),
    ("Yellow cards", ("Yellow Cards",)),
    ("Red cards", ("Red Cards",)),
    ("Expected goals (xG)", ("expected_goals", "xG")),
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
        rows.append({"Statistic": label, home_name: h_text, away_name: a_text})

if rows:
    st.dataframe(rows, hide_index=True, use_container_width=True)
else:
    st.warning(
        "The provider returned no usable statistics for this fixture. "
        "Predictions will rely on a generic prior and are especially uncertain."
    )

st.subheader("Experimental remaining-goals estimates")

home_lam, home_basis = estimate_remaining_goals(
    minute, home_stats, away_stats, home_reds, away_reds
)
away_lam, away_basis = estimate_remaining_goals(
    minute, away_stats, home_stats, away_reds, home_reds
)

home_p, home_tail = poisson_probs(home_lam)
away_p, away_tail = poisson_probs(away_lam)
matrix, _, _ = build_score_matrix(home_lam, away_lam)
p_home, p_draw, p_away = summarize_outcomes(matrix, home_score, away_score)

home_c1, home_c2, away_c1, away_c2 = st.columns(4)
home_c1.metric(f"{home_name}: expected remaining goals", f"{home_lam:.2f}")
home_c2.metric(
    f"{home_name}: chance to score again",
    probability_pct(1 - home_p[0])
)
away_c1.metric(f"{away_name}: expected remaining goals", f"{away_lam:.2f}")
away_c2.metric(
    f"{away_name}: chance to score again",
    probability_pct(1 - away_p[0])
)

with st.expander("What influenced the estimate?"):
    st.write(f"**{home_name}:** {home_basis}")
    st.write(f"**{away_name}:** {away_basis}")
    st.write(
        "These estimates use a generic scoring-rate prior adjusted by "
        "available live shots and red cards. They do not yet use a trained "
        "historical model or verified pre-match team strengths."
    )

st.subheader("Projected final result")
st.caption(
    "The exact-score list and market probabilities below are model estimates, "
    "not guaranteed outcomes. They assume independent Poisson-distributed "
    "remaining goals and use regular time up to 90 minutes."
)

c1, c2, c3 = st.columns(3)
c1.metric(f"{home_name} win", probability_pct(p_home))
c2.metric("Draw", probability_pct(p_draw))
c3.metric(f"{away_name} win", probability_pct(p_away))

ranked_scores = sorted(matrix, reverse=True)[:5]
score_rows = []
for prob, add_home, add_away in ranked_scores:
    score_rows.append({
        "Projected final score": (
            f"{home_name} {home_score + add_home}–"
            f"{away_score + add_away} {away_name}"
        ),
        "Probability within displayed score range": probability_pct(prob),
    })
st.dataframe(score_rows, hide_index=True, use_container_width=True)

st.subheader("Total goals markets")
market_rows = []
for line in (0.5, 1.5, 2.5, 3.5, 4.5):
    over_p = total_goals_probability(
        matrix, home_score, away_score, line
    )
    market_rows.append({
        "Market": f"Over {line}",
        "Probability": probability_pct(over_p),
    })
    market_rows.append({
        "Market": f"Under {line}",
        "Probability": probability_pct(1 - over_p),
    })

st.dataframe(market_rows, hide_index=True, use_container_width=True)

btts_p = btts_probability(matrix, home_score, away_score)
btts_c1, btts_c2 = st.columns(2)
btts_c1.metric("BTTS — Yes", probability_pct(btts_p))
btts_c2.metric("BTTS — No", probability_pct(1 - btts_p))

st.subheader("Next-goal estimates")
home_next, away_next, no_more = next_goal_probabilities(
    home_lam, away_lam
)
next_c1, next_c2, next_c3 = st.columns(3)
next_c1.metric(f"{home_name} scores next*", probability_pct(home_next))
next_c2.metric(f"{away_name} scores next*", probability_pct(away_next))
next_c3.metric("No more goals*", probability_pct(no_more))
st.caption(
    "*Approximation from remaining-goal rates; these are not validated "
    "next-goal market probabilities."
)

st.subheader("Recent match events")
if events:
    event_rows = []
    for event in events[-15:]:
        time_info = event.get("time") or {}
        team = (event.get("team") or {}).get("name", "")
        player = (event.get("player") or {}).get("name", "")
        assist = (event.get("assist") or {}).get("name", "")
        event_rows.append({
            "Minute": (
                f"{time_info.get('elapsed', '?')}'"
                + (
                    f"+{time_info.get('extra')}"
                    if time_info.get("extra") else ""
                )
            ),
            "Team": team,
            "Event": event.get("type", ""),
            "Detail": event.get("detail", ""),
            "Player": player,
            "Assist": assist,
        })
    st.dataframe(
        list(reversed(event_rows)),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No events were returned for this fixture.")

st.divider()
st.warning(
    "EXPERIMENTAL MODEL — Do not treat these outputs as verified betting "
    "edges. The model has not yet been trained or calibrated on historical "
    "in-play data. It can be wrong, especially when statistics are missing, "
    "matches have unusual red-card situations, or stoppage time is significant."
)
st.caption(
    f"Fixture ID: {fixture_id} · "
    f"Last page calculation: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)