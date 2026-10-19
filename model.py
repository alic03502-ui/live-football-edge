"""Transparent Poisson football model with explicit data-quality gates.

Expected CSV columns: date, home_team, away_team, home_goals, away_goals.
This is a baseline model, not a claim of profitability or calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals"}


@dataclass
class ModelStatus:
    ready: bool
    reason: str
    league_matches: int
    home_team_matches: int
    away_team_matches: int


def prepare_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize uploaded historical match results."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("CSV missing columns: " + ", ".join(sorted(missing)))

    out = frame[list(REQUIRED_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out["home_team"] = out["home_team"].astype("string").str.strip()
    out["away_team"] = out["away_team"].astype("string").str.strip()
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=list(REQUIRED_COLUMNS))
    out = out[
        (out["home_team"] != "") & (out["away_team"] != "") &
        (out["home_goals"] >= 0) & (out["away_goals"] >= 0) &
        (out["home_goals"] % 1 == 0) & (out["away_goals"] % 1 == 0) &
        (out["home_team"] != out["away_team"])
    ].copy()
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)
    out = out.sort_values("date").drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
    ).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid completed matches found in the CSV.")
    return out


def _weighted_mean(values: pd.Series, dates: pd.Series, reference: pd.Timestamp,
                   half_life_days: float = 240.0) -> float:
    age_days = (reference - dates).dt.total_seconds().clip(lower=0) / 86400
    weights = (0.5 ** (age_days / half_life_days)).astype(float)
    if weights.sum() <= 0:
        return float(values.mean())
    return float((values.astype(float) * weights).sum() / weights.sum())


def model_status(history: pd.DataFrame, home_team: str, away_team: str,
                 minimum_league_matches: int = 100,
                 minimum_team_matches: int = 8) -> ModelStatus:
    home_rows = history[(history.home_team == home_team) | (history.away_team == home_team)]
    away_rows = history[(history.home_team == away_team) | (history.away_team == away_team)]
    league_n = len(history)
    home_n, away_n = len(home_rows), len(away_rows)
    if league_n < minimum_league_matches:
        reason = f"Only {league_n} historical matches; need at least {minimum_league_matches}."
        return ModelStatus(False, reason, league_n, home_n, away_n)
    if home_n < minimum_team_matches or away_n < minimum_team_matches:
        reason = (f"Team history too small: {home_team} has {home_n}, "
                  f"{away_team} has {away_n}; need at least {minimum_team_matches} each.")
        return ModelStatus(False, reason, league_n, home_n, away_n)
    return ModelStatus(True, "Minimum history checks passed; probabilities remain uncalibrated until evaluated.",
                       league_n, home_n, away_n)


def _team_scored_conceded(history: pd.DataFrame, team: str,
                          reference: pd.Timestamp) -> tuple[float, float, int]:
    home = history[history.home_team == team]
    away = history[history.away_team == team]
    scored = pd.concat([
        pd.DataFrame({"g": home.home_goals, "date": home.date}),
        pd.DataFrame({"g": away.away_goals, "date": away.date}),
    ], ignore_index=True)
    conceded = pd.concat([
        pd.DataFrame({"g": home.away_goals, "date": home.date}),
        pd.DataFrame({"g": away.home_goals, "date": away.date}),
    ], ignore_index=True)
    if scored.empty:
        return 1.25, 1.25, 0
    return (_weighted_mean(scored.g, scored.date, reference),
            _weighted_mean(conceded.g, conceded.date, reference), len(scored))


def pre_match_lambdas(history: pd.DataFrame, home_team: str, away_team: str) -> tuple[float, float]:
    """Estimate expected goals using recency-weighted attack/defence rates."""
    hist = prepare_history(history)
    reference = hist.date.max()
    league_home = _weighted_mean(hist.home_goals, hist.date, reference)
    league_away = _weighted_mean(hist.away_goals, hist.date, reference)
    h_scored, h_conceded, _ = _team_scored_conceded(hist, home_team, reference)
    a_scored, a_conceded, _ = _team_scored_conceded(hist, away_team, reference)

    # Shrink team rates toward league means to reduce small-sample extremes.
    h_n = max(1, int(((hist.home_team == home_team) | (hist.away_team == home_team)).sum()))
    a_n = max(1, int(((hist.home_team == away_team) | (hist.away_team == away_team)).sum()))
    shrink = 8.0
    h_attack = (h_scored * h_n + league_home * shrink) / (h_n + shrink)
    h_defence = (h_conceded * h_n + league_away * shrink) / (h_n + shrink)
    a_attack = (a_scored * a_n + league_away * shrink) / (a_n + shrink)
    a_defence = (a_conceded * a_n + league_home * shrink) / (a_n + shrink)

    home_lam = league_home * (h_attack / max(0.2, league_home)) * (a_defence / max(0.2, league_home))
    away_lam = league_away * (a_attack / max(0.2, league_away)) * (h_defence / max(0.2, league_away))
    return max(0.05, min(4.5, home_lam)), max(0.05, min(4.5, away_lam))


def poisson_probs(lam: float, max_goals: int = 10) -> list[float]:
    lam = max(0.0, float(lam))
    probs = [math.exp(-lam)]
    for k in range(1, max_goals + 1):
        probs.append(probs[-1] * lam / k)
    total = sum(probs)
    return [p / total for p in probs] if total else [1.0] + [0.0] * max_goals


def score_matrix(home_lam: float, away_lam: float, max_goals: int = 10):
    hp, ap = poisson_probs(home_lam, max_goals), poisson_probs(away_lam, max_goals)
    return [(ph * pa, h, a) for h, ph in enumerate(hp) for a, pa in enumerate(ap)]


def markets(matrix: list[tuple[float, int, int]], home_score: int = 0,
            away_score: int = 0) -> dict[str, float]:
    result = {"1": 0.0, "X": 0.0, "2": 0.0, "BTTS Yes": 0.0}
    totals = {line: 0.0 for line in (0.5, 1.5, 2.5, 3.5, 4.5)}
    team_home = {line: 0.0 for line in (0.5, 1.5, 2.5)}
    team_away = {line: 0.0 for line in (0.5, 1.5, 2.5)}
    for prob, gh, ga in matrix:
        h, a = home_score + gh, away_score + ga
        result["1" if h > a else "X" if h == a else "2"] += prob
        if h > 0 and a > 0:
            result["BTTS Yes"] += prob
        for line in totals:
            if h + a > line:
                totals[line] += prob
        for line in team_home:
            if h > line:
                team_home[line] += prob
            if a > line:
                team_away[line] += prob
    out = dict(result)
    out["1X"] = result["1"] + result["X"]
    out["X2"] = result["X"] + result["2"]
    out["12"] = result["1"] + result["2"]
    out["BTTS No"] = 1 - result["BTTS Yes"]
    for line, prob in totals.items():
        out[f"Over {line}"] = prob
        out[f"Under {line}"] = 1 - prob
    for line, prob in team_home.items():
        out[f"Home team over {line}"] = prob
    for line, prob in team_away.items():
        out[f"Away team over {line}"] = prob
    return out


def in_play_lambdas(home_pre: float, away_pre: float, minute: int,
                    home_score: int, away_score: int,
                    home_shots_on_target: float | None = None,
                    away_shots_on_target: float | None = None,
                    home_red_cards: int = 0, away_red_cards: int = 0) -> tuple[float, float]:
    """Conservative live adjustment. This is a heuristic, not trained live calibration."""
    remaining = max(0.0, min(1.0, (90 - max(0, min(120, minute))) / 90))
    h = home_pre * remaining
    a = away_pre * remaining
    if home_shots_on_target is not None and away_shots_on_target is not None:
        diff = max(-5.0, min(5.0, home_shots_on_target - away_shots_on_target))
        h *= math.exp(0.035 * diff)
        a *= math.exp(-0.025 * diff)
    h *= max(0.5, 0.78 ** max(0, away_red_cards))
    a *= max(0.5, 0.78 ** max(0, home_red_cards))
    return max(0.0, min(3.0, h)), max(0.0, min(3.0, a))


def top_final_scores(matrix, home_score: int = 0, away_score: int = 0, n: int = 5):
    return sorted(
        [(p, home_score + h, away_score + a) for p, h, a in matrix],
        reverse=True
    )[:n]


def brier_score_1x2(prob_rows: list[dict[str, Any]], actual_results: list[str]) -> float | None:
    """Multiclass Brier score; each row has probabilities under keys 1, X, 2."""
    if not prob_rows or len(prob_rows) != len(actual_results):
        return None
    scores = []
    for probs, actual in zip(prob_rows, actual_results):
        if actual not in {"1", "X", "2"}:
            continue
        scores.append(sum((float(probs.get(k, 0.0)) - (1.0 if k == actual else 0.0)) ** 2
                          for k in ("1", "X", "2")))
    return sum(scores) / len(scores) if scores else None


def walk_forward_evaluation(history: pd.DataFrame, minimum_train: int = 100) -> pd.DataFrame:
    """Chronological expanding-window evaluation; no future rows enter each fit."""
    hist = prepare_history(history).sort_values("date").reset_index(drop=True)
    rows = []
    for idx in range(minimum_train, len(hist)):
        train = hist.iloc[:idx].copy()
        match = hist.iloc[idx]
        home, away = str(match.home_team), str(match.away_team)
        status = model_status(train, home, away, minimum_league_matches=minimum_train,
                              minimum_team_matches=1)
        if not status.ready:
            continue
        lh, la = pre_match_lambdas(train, home, away)
        matrix = score_matrix(lh, la)
        p = markets(matrix)
        actual = "1" if match.home_goals > match.away_goals else "X" if match.home_goals == match.away_goals else "2"
        rows.append({
            "date": match.date, "home_team": home, "away_team": away,
            "actual": actual, "p_home": p["1"], "p_draw": p["X"], "p_away": p["2"],
            "brier_1x2": sum((p[k] - (1.0 if k == actual else 0.0)) ** 2 for k in ("1", "X", "2")),
        })
    return pd.DataFrame(rows)
