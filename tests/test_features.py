import unittest

import pandas as pd

from fraudlab import features


def tx(step, orig, dest, amount=100.0, typ="TRANSFER", fraud=0, old_org=1000.0, old_dest=0.0, new_dest=None):
    return dict(
        step=step, type=typ, amount=amount, nameOrig=orig, oldbalanceOrg=old_org,
        newbalanceOrig=old_org - amount, nameDest=dest, oldbalanceDest=old_dest,
        newbalanceDest=old_dest + amount if new_dest is None else new_dest, isFraud=fraud,
    )  # fmt: skip


class HandComputed(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                tx(1, "A", "D", amount=500.0, old_org=1000.0),  # first sight of A and D
                tx(2, "A", "E"),  # A's second tx; E new
                tx(25, "B", "D"),  # D's second inbound; hour_of_day = 0 (step 25 -> 24th hour -> 0)
                tx(26, "A", "D"),  # A's third, D's third
            ]
        )
        self.f = features.build(self.df)

    def test_counts_are_strictly_prior(self):
        self.assertEqual(self.f["orig_tx_count_prior"].tolist(), [0, 1, 0, 2])
        self.assertEqual(self.f["dest_prior_inbound"].tolist(), [0, 0, 1, 2])
        self.assertEqual(self.f["dest_is_new"].tolist(), [1, 1, 0, 0])

    def test_amount_to_balance_and_hour(self):
        self.assertAlmostEqual(self.f.loc[0, "amount_to_balance"], 500 / 1001)
        self.assertEqual(self.f["hour_of_day"].tolist(), [0, 1, 0, 1])

    def test_amount_to_balance_is_capped(self):
        f = features.build(pd.DataFrame([tx(1, "A", "D", amount=5000.0, old_org=0.0)]))
        self.assertEqual(f.loc[0, "amount_to_balance"], 2.0)

    def test_error_balance_dest_is_zero_when_credited_and_positive_when_not(self):
        df = pd.DataFrame([tx(1, "A", "D", amount=100.0), tx(2, "B", "E", amount=100.0, new_dest=0.0)])
        f = features.build(df)
        self.assertEqual(f["error_balance_dest"].tolist(), [0.0, 100.0])


class ConfirmationDelay(unittest.TestCase):
    """A fraud at step 10 into D becomes known to the system at step 58 (10 + 48)."""

    def frame(self):
        return pd.DataFrame(
            [
                tx(10, "V", "D", fraud=1),
                tx(57, "X1", "D"),  # one hour early
                tx(58, "X2", "D"),  # exactly at the delay
                tx(100, "X3", "D"),
                tx(100, "X4", "OTHER"),
            ]
        )

    def test_boundary(self):
        f = features.build(self.frame())
        got = dict(zip(f["nameOrig"], f["dest_confirmed_fraud_prior"], strict=True))
        self.assertEqual(got["V"], 0)  # a fraud never confirms itself
        self.assertEqual(got["X1"], 0)
        self.assertEqual(got["X2"], 1)
        self.assertEqual(got["X3"], 1)
        self.assertEqual(got["X4"], 0)

    def test_custom_delay(self):
        f = features.build(self.frame(), label_delay=10)
        got = dict(zip(f["nameOrig"], f["dest_confirmed_fraud_prior"], strict=True))
        self.assertEqual(got["X1"], 1)

    def test_nonpositive_delay_rejected(self):
        with self.assertRaises(ValueError):
            features.build(self.frame(), label_delay=0)

    def test_counts_multiple_confirmed_frauds(self):
        df = pd.DataFrame([tx(1, "V1", "D", fraud=1), tx(5, "V2", "D", fraud=1), tx(60, "X", "D")])
        f = features.build(df)
        self.assertEqual(int(f.loc[f["nameOrig"] == "X", "dest_confirmed_fraud_prior"].iloc[0]), 2)

    def test_no_fraud_anywhere(self):
        df = pd.DataFrame([tx(1, "A", "D"), tx(60, "B", "D")])
        self.assertEqual(features.build(df)["dest_confirmed_fraud_prior"].sum(), 0)


class PointInTime(unittest.TestCase):
    def frame(self):
        rows = []
        for i in range(60):
            rows.append(tx(1 + i * 5, f"O{i % 7}", f"D{i % 5}", amount=50.0 + i, fraud=int(i % 11 == 0)))
        return pd.DataFrame(rows)

    def test_future_rows_do_not_change_past_features(self):
        full = features.build(self.frame())
        cut = 150
        base = self.frame()
        part = features.build(base[base["step"] <= cut])
        cols = features.NUMERIC
        a = full[full["step"] <= cut][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, part[cols], check_dtype=False)

    def test_unconfirmed_label_flip_changes_nothing_inside_the_delay(self):
        df = self.frame()
        flip_idx = df.index[(df["isFraud"] == 1) & (df["step"] > 100)][0]
        flip_step = int(df.loc[flip_idx, "step"])
        df2 = df.copy()
        df2.loc[flip_idx, "isFraud"] = 0
        a, b = features.build(df), features.build(df2)
        cutoff = flip_step + features.LABEL_DELAY_HOURS  # the label is first usable AT this step
        pd.testing.assert_frame_equal(
            a[a["step"] < cutoff][features.NUMERIC], b[b["step"] < cutoff][features.NUMERIC]
        )


class Guard(unittest.TestCase):
    def test_leaky_columns_rejected(self):
        for col in ("newbalanceDest", "error_balance_dest", "isFraud"):
            with self.assertRaises(ValueError):
                features.assert_no_leakage(["log_amount", col])

    def test_feature_set_is_clean(self):
        features.assert_no_leakage(features.FEATURES)

    def test_leak_demo_columns_are_all_declared_leaky(self):
        self.assertTrue(set(features.LEAK_DEMO) <= features.LEAKY)


if __name__ == "__main__":
    unittest.main()
