import json
import tempfile
import unittest
from pathlib import Path

from fraudlab import report

RESULT = json.loads((Path(__file__).resolve().parents[1] / "reports/metrics.json").read_text())


class HourRates(unittest.TestCase):
    def test_hand_computed(self):
        res = {"fraud_by_hour": {0: {"fraud": 1, "n": 4}, 5: {"fraud": 0, "n": 10}}}
        r = report.hour_rates(res)
        self.assertEqual(len(r), 24)
        self.assertAlmostEqual(r[0], 25.0)
        self.assertEqual(r[5], 0.0)
        self.assertEqual(r[7], 0.0)  # hour absent from the data

    def test_hour_with_zero_transactions_is_zero_not_a_crash(self):
        r = report.hour_rates({"fraud_by_hour": {9: {"fraud": 0, "n": 0}}})
        self.assertEqual(r[9], 0.0)

    def test_int_and_string_keys_agree(self):
        as_int = {"fraud_by_hour": {3: {"fraud": 2, "n": 8}}}
        as_str = json.loads(json.dumps(as_int))
        self.assertEqual(report.hour_rates(as_int), report.hour_rates(as_str))

    def test_synthetic_night_is_riskier_than_afternoon(self):
        r = report.hour_rates(RESULT)
        self.assertGreater(sum(r[0:7]) / 7, 2 * sum(r[12:19]) / 7)
        self.assertGreater(max(r), 0)  # the chart cannot be empty


class Render(unittest.TestCase):
    def test_functions_write_nonempty_pngs(self):
        out = Path(tempfile.mkdtemp())
        report.fraud_by_hour(RESULT, out / "h.png")
        self.assertGreater((out / "h.png").stat().st_size, 5_000)


if __name__ == "__main__":
    unittest.main()
