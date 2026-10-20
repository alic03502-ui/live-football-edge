
from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

BASE_URL = "https://api.goal-api.com/v1"
TIMEOUT = 20
DB_PATH = Path("data/live_football_edge.sqlite3")

# Two recent completed seasons plus the current season to date.
HISTORY_FROM = "2024-07-01"
HISTORY_TO = date.today().isoformat()

PAGE_SIZE = 500
MAX_HISTORY_PAGES = 100  # safety cap: at most 50,000 results per sync

st.set_page_config(
    page_title="Live Football Edge",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Live Football Edge")
st.caption("Live predictions informed by historical match results")

st.warning(
    "EXPERIMENTAL MODEL: Historical results inform team strengths, "
    "but probabilities are not yet calibrated or proven profitable. "
    "Missing history falls back toward generic scoring assumptions."
)


# =========================================================
# GOAL API
# =========================================================

def get_api_key():
    try:
        return str(st.secrets.get("GOAL_API_KEY", "")).strip()
    except Exception:
        return ""


def api_get_envelope(endpoint, params=None):
    """Return the full API response, including pagination metadata."""
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
        params=params or {},
        timeout=TIMEOUT,
    )

    if response.status_code == 401:
        raise RuntimeError("API key rejected. Check Streamlit Secrets.")
    if response.status_code == 402:
        raise RuntimeError(
            "This endpoint requires a higher API plan."
        )
    if response.status_code == 403:
        raise RuntimeError(
            "API access denied. Check your plan and endpoint permissions."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "API rate limit reached. Wait before retrying."
        )

    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(
            str(
                payload.get(
                    "message",
                    payload.get("error", "API error"),
                )
            )
        )

    return payload


def api_get(endpoint, params=None):
    payload = api_get_envelope(endpoint, params)

    if isinstance(payload, dict):
        return payload.get("data", payload)

    return payload


def as_list(payload):
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in (
            "fixtures", "matches", "results", "events", "items", "data"
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        if any(
            key in payload
            for key in ("id", "fixtureId", "fixture_id", "match_id")
        ):
            return [payload]

    return []


@st.cache_data(ttl=45, show_spinner=False)
def fetch_live_fixtures():
    return as_list(api_get("fixtures/live"))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_available_leagues():
    return as_list(
        api_get("leagues", {"isActive": True, "limit": 100, "offset": 0})
    )


def first(obj, *keys, default=None):
    if not isinstance(obj, dict):
        return default

    for key in keys:
        if obj.get(key) is not None:
            return obj[key]

    return default


def object_id(value):
    if isinstance(value, dict):
        value = first(value, "id", "teamId", "team_id", "leagueId")

    if value is None:
        return ""

    return str(value).strip()


def team_name(value, fallback):
    if isinstance(value, dict):
        return str(
            first(
                value, "name", "teamName", "team_name", "displayName",
                default=fallback,
            )
        )

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
    direct_keys = (
        (
            "homeTeamScore", "home_score", "homeScore", "homeGoals",
            "home_goals", "match_hometeam_score",
        )
        if side == "home"
        else (
            "awayTeamScore", "away_score", "awayScore", "awayGoals",
            "away_goals", "match_awayteam_score",
        )
    )

    score = parse_score(first(item, *direct_keys))
    if score is not None:
        return score

    for key in ("score", "scores", "goals", "currentScore"):
        container = item.get(key)
        if not isinstance(container, dict):
            continue

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


def extract_teams(item):
    teams = item.get("teams", {})
    if not isinstance(teams, dict):
        teams = {}

    home = first(
        teams, "home", "homeTeam", "home_team",
        default=first(
            item, "homeTeam", "home_team", "home",
            "match_hometeam_name", default="Home",
        ),
    )
    away = first(
        teams, "away", "awayTeam", "away_team",
        default=first(
            item, "awayTeam", "away_team", "away",
            "match_awayteam_name", default="Away",
        ),
    )

    return home, away


def extract_league(item):
    league = item.get("league", {})
    if not isinstance(league, dict):
        league = {}

    return (
        object_id(
            first(item, "leagueId", "league_id", default=league)
        ),
        str(
            first(
                league, "name", "leagueName",
                default=first(
                    item, "leagueName", "league_name",
                    default="Unknown league",
                ),
            )
        ),
    )


def normalize_fixture(item):
    home, away = extract_teams(item)

    minute_value = first(
        item, "matchElapsed", "matchMinute", "match_minute",
        "elapsed", "minute", "match_status", default=0,
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

    league_id, league_name = extract_league(item)

    return {
        "id": str(
            first(
                item, "id", "fixtureId", "fixture_id", "match_id",
                default="",
            )
        ),
        "home": team_name(home, "Home"),
        "away": team_name(away, "Away"),
        "home_id": object_id(home),
        "away_id": object_id(away),
        "home_score": get_score(item, "home"),
        "away_score": get_score(item, "away"),
        "minute": minute,
        "league": league_name,
        "league_id": league_id,
        "raw": item,
    }


# =========================================================
# Database
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_matches (
                match_id TEXT PRIMARY KEY,
                match_date TEXT,
                league_id TEXT,
                league_name TEXT,
                home_id TEXT,
                away_id TEXT,
                home_team TEXT,
                away_team TEXT,
                home_goals INTEGER NOT NULL,
                away_goals INTEGER NOT NULL,
                payload TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS history_sync (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                synced_at TEXT,
                date_from TEXT,
                date_to TEXT,
                fetched_count INTEGER DEFAULT 0,
                saved_count INTEGER DEFAULT 0,
                pages INTEGER DEFAULT 0,
                note TEXT
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


def parse_match_date(item):
    value = first(
        item,
        "date", "matchDate", "match_date", "fixtureDate",
        "kickoff", "kickoffTime", "startTime", "dateTime",
    )

    if isinstance(value, dict):
        value = first(value, "date", "datetime", "timestamp")

    if value is None:
        return ""

    text = str(value).strip()

    # ISO date or datetime.
    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        ).date().isoformat()
    except ValueError:
        pass

    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else ""


def normalize_historical_match(item):
    home, away = extract_teams(item)
    home_goals = get_score(item, "home")
    away_goals = get_score(item, "away")

    # Never train on incomplete or non-final scores.
    if home_goals is None or away_goals is None:
        return None

    match_date = parse_match_date(item)
    if not match_date:
        return None

    if not (HISTORY_FROM <= match_date <= HISTORY_TO):
        return None

    match_id = str(
        first(
            item, "id", "fixtureId", "fixture_id", "match_id",
            "matchApiId", default="",
        )
    ).strip()

    if not match_id:
        return None

    league_id, league_name = extract_league(item)

    return {
        "match_id": match_id,
        "match_date": match_date,
        "league_id": league_id,
        "league_name": league_name,
        "home_id": object_id(home),
        "away_id": object_id(away),
        "home_team": team_name(home, "Home"),
        "away_team": team_name(away, "Away"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "payload": json.dumps(item, ensure_ascii=False, default=str),
    }


def sync_historical_results():
    """
    Sync all leagues available from the global results endpoint.
    Results are deduplicated by match ID in SQLite.
    """
    init_db()

    fetched = 0
    saved = 0
    pages = 0
    offset = 0
    has_more = True

    with st.spinner(
        "Downloading historical results from GOAL API. "
        "This can take several requests..."
    ):
        while has_more and pages < MAX_HISTORY_PAGES:
            envelope = api_get_envelope(
                "results",
                {
                    "from": HISTORY_FROM,
                    "to": HISTORY_TO,
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
            )

            rows = as_list(envelope)
            if not rows:
                break

            pages += 1
            fetched += len(rows)
            normalized = [
                row for row in
                (normalize_historical_match(item) for item in rows)
                if row is not None
            ]

            with sqlite3.connect(DB_PATH) as conn:
                before = conn.total_changes

                conn.executemany("""
                    INSERT OR IGNORE INTO historical_matches
                    (match_id, match_date, league_id, league_name,
                     home_id, away_id, home_team, away_team,
                     home_goals, away_goals, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        row["match_id"], row["match_date"],
                        row["league_id"], row["league_name"],
                        row["home_id"], row["away_id"],
                        row["home_team"], row["away_team"],
                        row["home_goals"], row["away_goals"],
                        row["payload"],
                    )
                    for row in normalized
                ])

                saved += conn.total_changes - before

            pagination = (
                envelope.get("pagination", {})
                if isinstance(envelope, dict)
                else {}
            )

            if isinstance(pagination, dict) and "hasMore" in pagination:
                has_more = bool(pagination["hasMore"])
            else:
                # Fallback for APIs that omit hasMore.
                has_more = len(rows) >= PAGE_SIZE

            offset += len(rows)

            if len(rows) < PAGE_SIZE:
                has_more = False

    note = (
        "Safety cap reached; run sync again to continue."
        if has_more and pages >= MAX_HISTORY_PAGES
        else "Sync finished."
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO history_sync
            (id, synced_at, date_from, date_to, fetched_count,
             saved_count, pages, note)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                synced_at=excluded.synced_at,
                date_from=excluded.date_from,
                date_to=excluded.date_to,
                fetched_count=excluded.fetched_count,
                saved_count=excluded.saved_count,
                pages=excluded.pages,
                note=excluded.note
        """, (
            datetime.now(timezone.utc).isoformat(),
            HISTORY_FROM, HISTORY_TO, fetched, saved, pages, note,
        ))

    return {
        "fetched": fetched,
        "saved": saved,
        "pages": pages,
        "note": note,
    }


def history_summary():
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM historical_matches"
        ).fetchone()[0]

        leagues = conn.execute("""
            SELECT COUNT(DISTINCT league_id)
            FROM historical_matches
            WHERE league_id != ''
        """).fetchone()[0]

        teams = conn.execute("""
            SELECT COUNT(DISTINCT team_key)
            FROM (
                SELECT home_id AS team_key FROM historical_matches
                WHERE home_id != ''
                UNION
                SELECT away_id AS team_key FROM historical_matches
                WHERE away_id != ''
            )
        """).fetchone()[0]

        sync = conn.execute("""
            SELECT synced_at, date_from, date_to, fetched_count,
                   saved_count, pages, note
            FROM history_sync WHERE id=1
        """).fetchone()

    return {
        "matches": count,
        "leagues": leagues,
        "teams": teams,
        "sync": sync,
    }


# =========================================================
# Historical team attack and defence
# =========================================================

def historical_strengths(league_id, home_id, away_id):
    """
    Calculate team attack/defence factors from finished results.

    Each team's rates are shrunk toward its league's scoring average.
    This avoids treating a team with only one or two matches as certain.
    """
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        if league_id:
            rows = conn.execute("""
                SELECT match_date, league_id, home_id, away_id,
                       home_team, away_team, home_goals, away_goals
                FROM historical_matches
                WHERE league_id = ?
                ORDER BY match_date DESC
            """, (str(league_id),)).fetchall()
        else:
            rows = conn.execute("""
                SELECT match_date, league_id, home_id, away_id,
                       home_team, away_team, home_goals, away_goals
                FROM historical_matches
                ORDER BY match_date DESC
            """).fetchall()

    # A stable league-wide fallback when league ID is absent.
    league_matches = [
        row for row in rows
        if row[6] is not None and row[7] is not None
    ]

    if not league_matches:
        return None

    league_home_avg = sum(r[6] for r in league_matches) / len(league_matches)
    league_away_avg = sum(r[7] for r in league_matches) / len(league_matches)
    overall_avg = (league_home_avg + league_away_avg) / 2.0

    # If the league has too few results, avoid extreme factors.
    overall_avg = max(0.6, min(2.2, overall_avg))

    def team_stats(team_id, team_label):
        goals_for = []
        goals_against = []

        for row in rows:
            _, _, h_id, a_id, h_name, a_name, hg, ag = row

            if hg is None or ag is None:
                continue

            same_home = (
                (team_id and str(team_id) == str(h_id))
                or (not team_id and team_label == h_name)
            )
            same_away = (
                (team_id and str(team_id) == str(a_id))
                or (not team_id and team_label == a_name)
            )

            if same_home:
                goals_for.append(hg)
                goals_against.append(ag)
            elif same_away:
                goals_for.append(ag)
                goals_against.append(hg)

        n = len(goals_for)
        if n == 0:
            return {
                "matches": 0,
                "attack": 1.0,
                "defence": 1.0,
                "gf": None,
                "ga": None,
            }

        # Prior equivalent to six matches at league-average strength.
        prior_matches = 6.0
        gf = (
            sum(goals_for) + prior_matches * overall_avg
        ) / (n + prior_matches)
        ga = (
            sum(goals_against) + prior_matches * overall_avg
        ) / (n + prior_matches)

        attack = gf / overall_avg
        defence = overall_avg / max(0.25, ga)

        # Guard against wild estimates from sparse or unusual samples.
        attack = max(0.65, min(1.45, attack))
        defence = max(0.65, min(1.45, defence))

        return {
            "matches": n,
            "attack": attack,
            "defence": defence,
            "gf": gf,
            "ga": ga,
        }

    return {
        "league_matches": len(league_matches),
        "league_avg": overall_avg,
        "home": team_stats(home_id, None),
        "away": team_stats(away_id, None),
    }


# =========================================================
# Statistics parsing
# =========================================================

def stat_label(row):
    return str(
        first(
            row, "type", "name", "label", "statistic", "description",
            default="",
        )
    ).strip().lower()


def numeric_value(value):
    if isinstance(value, dict):
        value = first(value, "value", "total", "amount")

    if value is None:
        return None

    try:
        number = float(str(value).replace("%", "").strip())
        if math.isfinite(number):
            return number
    except (ValueError, TypeError):
        pass

    return None


def flatten_statistics(payload):
    rows = []

    def walk(obj):
        if isinstance(obj, list):
            for entry in obj:
                walk(entry)
            return

        if not isinstance(obj, dict):
            return

        label = stat_label(obj)
        home_value = first(obj, "home", "homeValue", "home_value")
        away_value = first(obj, "away", "awayValue", "away_value")

        if label and (
            home_value is not None or away_value is not None
        ):
            rows.append({
                "type": label,
                "home": numeric_value(home_value),
                "away": numeric_value(away_value),
            })
            return

        if label and any(k in obj for k in ("value", "total", "amount")):
            rows.append({
                "type": label,
                "value": numeric_value(
                    first(obj, "value", "total", "amount")
                ),
            })
            return

        for key in (
            "data", "statistics", "stats", "items", "response",
            "results", "teams", "home", "away",
        ):
            if key in obj:
                walk(obj[key])

    walk(payload)
    return rows


def get_stat(rows, side, aliases):
    aliases = {x.lower() for x in aliases}

    for row in rows:
        if row.get("type", "") not in aliases:
            continue

        if row.get(side) is not None:
            return row[side]

    return None


# =========================================================
# Poisson model
# =========================================================

def poisson_probabilities(lam, max_goals=10):
    lam = max(0.0, float(lam))
    max_goals = max(0, int(max_goals))

    if lam == 0:
        return [1.0] + [0.0] * max_goals

    probabilities = [math.exp(-lam)]

    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * lam / goals)

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
    historical=None,
):
    minute = max(0, min(120, int(minute or 0)))
    home_score = max(0, int(home_score or 0))
    away_score = max(0, int(away_score or 0))
    home_reds = max(0, int(home_reds or 0))
    away_reds = max(0, int(away_reds or 0))

    # Generic baselines.
    home_rate = 1.40
    away_rate = 1.10

    history_used = False

    # Historical attack/defence adjustment.
    if historical:
        h = historical["home"]
        a = historical["away"]
        league_avg = historical["league_avg"]

        if h["matches"] >= 3 or a["matches"] >= 3:
            # Use the league's average scoring environment as the base.
            home_rate = max(
                0.55,
                min(
                    2.5,
                    league_avg * h["attack"] * a["defence"] * 1.06,
                ),
            )
            away_rate = max(
                0.45,
                min(
                    2.3,
                    league_avg * a["attack"] * h["defence"] * 0.94,
                ),
            )
            history_used = True

    if home_sot is not None:
        home_rate *= max(
            0.80, min(1.25, 0.96 + 0.035 * max(0, float(home_sot)))
        )

    if away_sot is not None:
        away_rate *= max(
            0.80, min(1.25, 0.96 + 0.035 * max(0, float(away_sot)))
        )

    home_rate *= max(0.55, 1.0 - 0.22 * home_reds + 0.12 * away_reds)
    away_rate *= max(0.55, 1.0 - 0.22 * away_reds + 0.12 * home_reds)

    if minute >= 55 and home_score != away_score:
        if home_score < away_score:
            home_rate *= 1.10
        else:
            away_rate *= 1.10

    remaining_minutes = max(0, min(100 - minute, 100))
    fraction = remaining_minutes / 90.0

    home_remaining = home_rate * fraction
    away_remaining = away_rate * fraction

    hp = poisson_probabilities(home_remaining)
    ap = poisson_probabilities(away_remaining)

    result = {"1": 0.0, "X": 0.0, "2": 0.0}
    over15 = over25 = over35 = btts = 0.0
    score_candidates = []

    for add_home, ph in enumerate(hp):
        for add_away, pa in enumerate(ap):
            probability = ph * pa
            final_home = home_score + add_home
            final_away = away_score + add_away
            total_goals = final_home + final_away

            if final_home > final_away:
                result["1"] += probability
            elif final_home == final_away:
                result["X"] += probability
            else:
                result["2"] += probability

            if total_goals >= 2:
                over15 += probability
            if total_goals >= 3:
                over25 += probability
            if total_goals >= 4:
                over35 += probability
            if final_home >= 1 and final_away >= 1:
                btts += probability

            score_candidates.append(
                (probability, final_home, final_away)
            )

    total = sum(sum(row) for row in [
        [hp[h] * ap[a] for a in range(len(ap))]
        for h in range(len(hp))
    ])

    if total > 0:
        result = {key: value / total for key, value in result.items()}
        over15 /= total
        over25 /= total
        over35 /= total
        btts /= total
        score_candidates = [
            (p / total, home, away)
            for p, home, away in score_candidates
        ]

    markets = {
        "Home win": result["1"],
        "Draw": result["X"],
        "Away win": result["2"],
        "Over 1.5 goals": over15,
        "Under 1.5 goals": 1.0 - over15,
        "Over 2.5 goals": over25,
        "Under 2.5 goals": 1.0 - over25,
        "Over 3.5 goals": over35,
        "Under 3.5 goals": 1.0 - over35,
        "Both teams to score — Yes": btts,
        "Both teams to score — No": 1.0 - btts,
        "Home or draw (1X)": result["1"] + result["X"],
        "Away or draw (X2)": result["X"] + result["2"],
    }

    score_candidates.sort(key=lambda item: item[0], reverse=True)

    return markets, score_candidates[:5], home_remaining, away_remaining, history_used


# =========================================================
# Sidebar: history management
# =========================================================

init_db()

with st.sidebar:
    st.header("Historical database")
    st.write(f"Date range: **{HISTORY_FROM} → {HISTORY_TO}**")
    st.caption(
        "All available leagues · Historical results are stored in SQLite. "
        "Sync only when you want to refresh the database."
    )

    if st.button("Sync historical results", use_container_width=True):
        try:
            result = sync_historical_results()
            st.success(
                f"Fetched {result['fetched']:,} results; "
                f"added {result['saved']:,} new matches "
                f"across {result['pages']} API page(s)."
            )
            st.info(result["note"])
        except Exception as exc:
            st.error(f"Historical sync failed: {exc}")

    summary = history_summary()
    st.metric("Stored matches", f"{summary['matches']:,}")
    st.metric("Leagues represented", f"{summary['leagues']:,}")
    st.metric("Teams represented", f"{summary['teams']:,}")

    if summary["sync"]:
        sync = summary["sync"]
        st.caption(f"Last sync: {sync[0]}")
        st.caption(
            f"Last sync fetched {sync[3]:,}; "
            f"newly saved {sync[4]:,}."
        )
        if sync[6]:
            st.caption(sync[6])

    st.divider()
    st.header("Live model")
    st.write("Mode: **Live + historical strengths**")
    st.write("CSV uploads: **Not required**")


# =========================================================
# Live fixtures
# =========================================================

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
fixtures = [fixture for fixture in fixtures if fixture["id"]]

if not fixtures:
    st.info("No live fixtures were returned by the API right now.")
    st.stop()

leagues = ["All leagues"] + sorted({
    fixture["league"] for fixture in fixtures
})

selected_league = st.selectbox("League", leagues)

if selected_league != "All leagues":
    fixtures = [
        fixture for fixture in fixtures
        if fixture["league"] == selected_league
    ]

if not fixtures:
    st.info("No live matches in this league.")
    st.stop()


def fixture_label(fixture):
    home_score = "?" if fixture["home_score"] is None else fixture["home_score"]
    away_score = "?" if fixture["away_score"] is None else fixture["away_score"]

    return (
        f"{fixture['home']} {home_score}–{away_score} "
        f"{fixture['away']} · {fixture['minute']}' · {fixture['league']}"
    )


selected = st.selectbox(
    "Live match",
    fixtures,
    format_func=fixture_label,
)

if selected["home_score"] is None or selected["away_score"] is None:
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
# Live statistics
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
    stats_rows, "home", {"shots on target", "shots on goal"}
)
away_sot = get_stat(
    stats_rows, "away", {"shots on target", "shots on goal"}
)

if stats_error:
    st.warning(
        f"Live statistics could not be loaded: {stats_error}. "
        "The model will continue without shot adjustments."
    )

if home_sot is None or away_sot is None:
    st.info(
        "Shots on target are unavailable or not recognized. "
        "Generic assumptions will be used for that adjustment."
    )


# =========================================================
# Historical strengths and predictions
# =========================================================

historical = historical_strengths(
    selected["league_id"],
    selected["home_id"],
    selected["away_id"],
)

if historical:
    h = historical["home"]
    a = historical["away"]

    st.subheader("Historical team profile")
    history_cols = st.columns(3)

    history_cols[0].metric(
        "League matches in database",
        f"{historical['league_matches']:,}",
    )
    history_cols[1].metric(
        f"{selected['home']} matches",
        f"{h['matches']:,}",
    )
    history_cols[2].metric(
        f"{selected['away']} matches",
        f"{a['matches']:,}",
    )

    with st.expander("Attack and defence estimates"):
        st.write(
            "Attack factor above 1.00 means above the league scoring "
            "average; defence factor above 1.00 means fewer goals conceded "
            "than the league average. Estimates are shrunk toward the "
            "league average when the sample is small."
        )
        st.dataframe([
            {
                "Team": selected["home"],
                "Matches": h["matches"],
                "Adjusted goals scored/match": (
                    round(h["gf"], 2) if h["gf"] is not None else "N/A"
                ),
                "Adjusted goals conceded/match": (
                    round(h["ga"], 2) if h["ga"] is not None else "N/A"
                ),
                "Attack factor": round(h["attack"], 2),
                "Defence factor": round(h["defence"], 2),
            },
            {
                "Team": selected["away"],
                "Matches": a["matches"],
                "Adjusted goals scored/match": (
                    round(a["gf"], 2) if a["gf"] is not None else "N/A"
                ),
                "Adjusted goals conceded/match": (
                    round(a["ga"], 2) if a["ga"] is not None else "N/A"
                ),
                "Attack factor": round(a["attack"], 2),
                "Defence factor": round(a["defence"], 2),
            },
        ], use_container_width=True, hide_index=True)
else:
    st.info(
        "No usable historical results found for this league yet. "
        "Press **Sync historical results** in the sidebar. "
        "Until then, the model uses generic scoring assumptions."
    )

(
    markets,
    top_scores,
    home_remaining,
    away_remaining,
    history_used,
) = live_model(
    minute=selected["minute"],
    home_score=selected["home_score"],
    away_score=selected["away_score"],
    home_sot=home_sot,
    away_sot=away_sot,
    historical=historical,
)

st.subheader("Live match predictions")

if history_used:
    st.caption(
        "Historical attack/defence estimates and available live statistics "
        "are influencing the goal rates. This is still experimental."
    )
else:
    st.caption(
        "Generic scoring rates are being used because sufficient historical "
        "team data is not available."
    )

c1, c2, c3 = st.columns(3)
c1.metric("Home win", f"{markets['Home win']:.1%}")
c2.metric("Draw", f"{markets['Draw']:.1%}")
c3.metric("Away win", f"{markets['Away win']:.1%}")

st.subheader("Goal and double-chance markets")
st.dataframe(
    [
        {"Market": name, "Probability": f"{probability:.1%}"}
        for name, probability in markets.items()
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Most likely final scores")
st.dataframe(
    [
        {
            "Projected final score": (
                f"{selected['home']} {home}–{away} {selected['away']}"
            ),
            "Probability": f"{probability:.1%}",
        }
        for probability, home, away in top_scores
    ],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Estimated remaining goals: {selected['home']} "
    f"{home_remaining:.2f} · {selected['away']} "
    f"{away_remaining:.2f}"
)

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
    "Historical results are not the same as historical calibration. "
    "Probabilities still require out-of-sample evaluation before being "
    "treated as reliable betting estimates."
)