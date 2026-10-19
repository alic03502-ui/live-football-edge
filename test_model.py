import unittest
import pandas as pd

from model import (
    prepare_history, model_status, pre_match_lambdas, score_matrix, markets,
    top_final_scores, in_play_lambdas, walk_forward_evaluation
)


def sample_history(n=140):
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    teams = ["Alpha", "Bravo", "Charlie", "Delta"]
    for i, date in enumerate(dates):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        if home == away:
            away = teams[(i + 2) % 4]
        rows.append({
            "date": date, "home_team": home, "away_team": away,
            "home_goals": (i % 3), "away_goals": ((i + 1) % 2)
        })
    return pd.DataFrame(rows)


class ModelTests(unittest.TestCase):
    def test_invalid_rows_are_removed(self):
        frame = pd.DataFrame([
            {"date": "2025-01-01", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0},
            {"date": "not-a-date", "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 0},
            {"date": "2025-01-03", "home_team": "A", "away_team": "A", "home_goals": 0, "away_goals": 0},
        ])
        cleaned = prepare_history(frame)
        self.assertEqual(len(cleaned), 1)

    def test_history_gate_rejects_small_dataset(self):
        history = prepare_history(sample_history(20))
        status = model_status(history, "Alpha", "Bravo")
        self.assertFalse(status.ready)

    def test_probabilities_sum_to_one(self):
        matrix = score_matrix(1.4, 1.0)
        self.assertAlmostEqual(sum(p for p, _, _ in matrix), 1.0, places=8)
        p = markets(matrix)
        self.assertAlmostEqual(p["1"] + p["X"] + p["2"], 1.0, places=8)
        self.assertAlmostEqual(p["Over 2.5"] + p["Under 2.5"], 1.0, places=8)

    def test_pre_match_lambdas_are_bounded(self):
        history = prepare_history(sample_history(140))
        home, away = pre_match_lambdas(history, "Alpha", "Bravo")
        self.assertTrue(0.05 <= home <= 4.5)
        self.assertTrue(0.05 <= away <= 4.5)

    def test_in_play_remaining_goals_fall_to_zero_at_full_time(self):
        home, away = in_play_lambdas(1.5, 1.2, 90, 1, 0)
        self.assertEqual(home, 0.0)
        self.assertEqual(away, 0.0)

    def test_top_scores_are_ranked(self):
        rows = top_final_scores(score_matrix(1.3, 0.9), n=5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows, sorted(rows, reverse=True))

    def test_walk_forward_does_not_return_future_date_as_training(self):
        history = prepare_history(sample_history(110))
        report = walk_forward_evaluation(history, minimum_train=100)
        if not report.empty:
            self.assertTrue((report["date"] >= history.iloc[100]["date"]).all())


if __name__ == "__main__":
    unittest.main()
