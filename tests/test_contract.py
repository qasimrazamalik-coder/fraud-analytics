import unittest

import pandas as pd

from fraudlab import contract, synth


def good():
    return synth.generate(n=2000, seed=1)


class Contract(unittest.TestCase):
    def test_synthetic_passes(self):
        contract.validate(good())

    def test_missing_column(self):
        self.assertIn("missing columns", contract.violations(good().drop(columns=["amount"]))[0])

    def test_nulls(self):
        d = good()
        d.loc[0, "nameDest"] = None
        self.assertTrue(any("nameDest: 1 null" in v for v in contract.violations(d)))

    def test_unknown_type(self):
        d = good()
        d.loc[0, "type"] = "WIRE"
        self.assertTrue(any("WIRE" in v for v in contract.violations(d)))

    def test_negative_money(self):
        d = good()
        d.loc[0, "amount"] = -1.0
        self.assertTrue(any("amount: 1 negative" in v for v in contract.violations(d)))

    def test_step_must_start_at_one(self):
        d = good()
        d.loc[0, "step"] = 0
        self.assertTrue(any("step" in v for v in contract.violations(d)))

    def test_label_domain(self):
        d = good()
        d.loc[0, "isFraud"] = 2
        self.assertTrue(any("isFraud must be 0 or 1" in v for v in contract.violations(d)))

    def test_no_fraud_rows(self):
        d = good()
        d["isFraud"] = 0
        self.assertTrue(any("no fraud rows" in v for v in contract.violations(d)))

    def test_empty(self):
        self.assertEqual(contract.violations(good().iloc[0:0]), ["table is empty"])

    def test_all_violations_reported_together(self):
        d = good()
        d.loc[0, "type"] = "WIRE"
        d.loc[1, "amount"] = -5.0
        with self.assertRaises(contract.DataContractError) as ctx:
            contract.validate(d)
        self.assertIn("WIRE", str(ctx.exception))
        self.assertIn("negative", str(ctx.exception))

    def test_fraud_outside_scope_is_counted_not_fatal(self):
        d = good()
        i = d.index[(d["type"] == "PAYMENT")][0]
        d.loc[i, "isFraud"] = 1
        contract.validate(d)
        self.assertEqual(contract.fraud_outside_scope(d), 1)
        self.assertIsInstance(d, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
