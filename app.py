import os
import requests
import streamlit as st

API_BASE = "https://api.odds-data-api.com"

st.set_page_config(page_title="Live Football Edge", page_icon="⚽", layout="wide")
st.title("⚽ Live Football Edge")
st.caption("Live match monitor — data freshness must be verified before predictions.")

with st.sidebar:
    st.header("Data connection")
    st.write("Provider: Odds Data API")
    refresh = st.button("Refresh live matches", type="primary")


def load_live_matches(token):
    response = requests.get(
        f"{API_BASE}/v3/events/inplay",
        params={"sport_id": 1, "token": token},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("success") != 1:
        raise RuntimeError(payload.get("error", "Provider returned an unsuccessful response"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Unexpected API response format")
    return results


try:
    token = st.secrets.get("ODDS_DATA_API_TOKEN", os.getenv("ODDS_DATA_API_TOKEN", ""))
except Exception:
    token = os.getenv("ODDS_DATA_API_TOKEN", "")

if not token:
    st.warning("API token not configured yet. Add ODDS_DATA_API_TOKEN to Streamlit Secrets after obtaining provider access.")
    st.info("No live matches or predictions are fabricated while the feed is disconnected.")
else:
    try:
        with st.spinner("Loading live matches..."):
            matches = load_live_matches(token)
        st.metric("Live matches returned", len(matches))
        if not matches:
            st.info("The provider returned no live matches at this moment.")
        for match in matches:
            home = match.get("home", {})
            away = match.get("away", {})
            if isinstance(home, dict):
                home = home.get("name", "Home team")
            if isinstance(away, dict):
                away = away.get("name", "Away team")
            league = match.get("league", {})
            if isinstance(league, dict):
                league = league.get("name", "League unavailable")
            score = match.get("ss") or "Score unavailable"
            timer = match.get("timer") or {}
            if isinstance(timer, dict):
                minute = timer.get("tm") or timer.get("minute")
                clock = f"{minute}′" if minute is not None else "Clock unavailable"
            else:
                clock = str(timer) if timer else "Clock unavailable"
            with st.container(border=True):
                st.caption(str(league))
                st.subheader(f"{home}  {score}  {away}")
                st.write(f"Match clock: {clock}")
                st.caption(f"Provider event ID: {match.get('id', 'unavailable')}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the live data provider: {exc}")
    except (ValueError, RuntimeError) as exc:
        st.error(f"Live data could not be validated: {exc}")