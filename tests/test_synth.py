import unittest

from fraudlab import synth


class Synth(unittest.TestCase):
    def test_deterministic(self):
        self.assertTrue(synth.generate(3000, seed=5).equals(synth.generate(3000, seed=5)))

    def test_seed_changes_data(self):
        self.assertFalse(synth.generate(3000, seed=5).equals(synth.generate(3000, seed=6)))

    def test_schema_and_size(self):
        d = synth.generate(4000, seed=1)
        self.assertEqual(list(d.columns), synth.COLUMNS)
        self.assertEqual(len(d), 4000)
        self.assertTrue(d["step"].between(1, synth.HOURS).all())
        self.assertTrue(d["step"].is_monotonic_increasing)

    def test_fraud_count_and_scope(self):
        d = synth.generate(20_000, seed=2)
        self.assertEqual(int(d["isFraud"].sum()), 50)
        self.assertTrue(set(d.loc[d["isFraud"] == 1, "type"]) <= {"TRANSFER", "CASH_OUT"})

    def test_fraud_is_mostly_a_large_drain(self):
        d = synth.generate(60_000, seed=3)
        f = d[d["isFraud"] == 1]
        self.assertGreater((f["amount"] >= 0.3 * f["oldbalanceOrg"]).mean(), 0.99)
        self.assertGreater((f["newbalanceOrig"] == 0).mean(), 0.5)

    def test_ledger_artefact_exists_and_is_label_dependent(self):
        d = synth.generate(60_000, seed=4)
        tr = d[d["type"] == "TRANSFER"]
        uncredited = tr["newbalanceDest"] == tr["oldbalanceDest"]
        self.assertGreater(uncredited[tr["isFraud"] == 1].mean(), 0.5)
        self.assertEqual(uncredited[tr["isFraud"] == 0].sum(), 0)

    def test_flag_rule(self):
        d = synth.generate(20_000, seed=6)
        expect = (d["type"] == "TRANSFER") & (d["amount"] > synth.RULE_AMOUNT)
        self.assertTrue((d["isFlaggedFraud"] == expect.astype(int)).all())

    def test_money_is_never_negative(self):
        d = synth.generate(20_000, seed=7)
        for c in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
            self.assertTrue((d[c] >= 0).all(), c)


if __name__ == "__main__":
    unittest.main()
