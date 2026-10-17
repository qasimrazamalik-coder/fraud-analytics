import unittest

import numpy as np

from fraudlab import cost, model


class BestCut(unittest.TestCase):
    # costs by k alerts: k=0 1500, k=1 600, k=2 700, k=3 300, k=4 400  (review cost 100)
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    a = np.array([1000.0, 50.0, 500.0, 10.0])

    def test_hand_computed_optimum(self):
        c = cost.best_cut(self.y, self.s, self.a, 100.0)
        self.assertEqual(c["flagged"], 3)
        self.assertAlmostEqual(c["threshold"], 0.7)
        self.assertAlmostEqual(c["cost"], 300.0)

    def test_expensive_reviews_flag_nothing(self):
        c = cost.best_cut(self.y, self.s, self.a, 10_000.0)
        self.assertEqual(c["flagged"], 0)
        self.assertEqual(c["threshold"], float("inf"))
        self.assertAlmostEqual(c["cost"], 1500.0)

    def test_ties_are_flagged_together(self):
        y, s, a = np.array([1, 0, 0]), np.array([0.5, 0.5, 0.1]), np.array([1000.0, 10.0, 10.0])
        c = cost.best_cut(y, s, a, 1.0)
        self.assertEqual(c["flagged"], 2)  # k=1 would split the tie and is not allowed
        self.assertAlmostEqual(c["cost"], 2.0)

    def test_equal_cost_prefers_fewer_alerts(self):
        y, s, a = np.array([1, 0]), np.array([0.9, 0.1]), np.array([100.0, 10.0])
        c = cost.best_cut(y, s, a, 100.0)  # k=0 costs 100, k=1 costs 100
        self.assertEqual(c["flagged"], 0)

    def test_optimum_beats_every_cut_by_brute_force(self):
        rng = np.random.default_rng(3)
        y = (rng.random(200) < 0.1).astype(int)
        s = rng.random(200)
        a = rng.uniform(10, 1000, 200)
        best = cost.best_cut(y, s, a, 40.0)["cost"]
        for thr in np.unique(s):
            self.assertLessEqual(best, cost.apply(y, s, a, thr, 40.0)["cost"] + 1e-9)
        self.assertLessEqual(best, a[y == 1].sum() + 1e-9)


class Apply(unittest.TestCase):
    def test_hand_computed(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.1])
        a = np.array([1000.0, 50.0, 500.0, 10.0])
        r = cost.apply(y, s, a, 0.75, 100.0)  # flags the first two
        self.assertEqual(r["alerts"], 2)
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["fraud_amount_caught"], 1000.0)
        self.assertAlmostEqual(r["cost"], 500.0 + 200.0)
        self.assertAlmostEqual(r["saving"], 1500.0 - 700.0)

    def test_infinite_threshold_equals_no_model(self):
        y, s, a = np.array([1, 0]), np.array([0.9, 0.1]), np.array([100.0, 10.0])
        r = cost.apply(y, s, a, float("inf"), 5.0)
        self.assertEqual(r["alerts"], 0)
        self.assertAlmostEqual(r["saving"], 0.0)


class RecallAtRate(unittest.TestCase):
    def test_by_hand(self):
        y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        s = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1, 0.25, 0.05])
        self.assertAlmostEqual(model.recall_at_alert_rate(y, s, 0.10), 0.5)  # top 1 holds one of two
        self.assertAlmostEqual(model.recall_at_alert_rate(y, s, 0.20), 0.5)  # top 2: idx0 (.9), idx3 (.3)
        self.assertAlmostEqual(model.recall_at_alert_rate(y, s, 0.30), 1.0)  # top 3 adds idx8 (.25)
        self.assertLess(model.recall_at_alert_rate(y, -s, 0.10), 0.5)


if __name__ == "__main__":
    unittest.main()
