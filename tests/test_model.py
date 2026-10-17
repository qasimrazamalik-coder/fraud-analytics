import unittest

from fraudlab import features, model, synth

_CACHE: dict = {}


def feat(seed=11, n=60_000):
    if (seed, n) not in _CACHE:
        _CACHE[(seed, n)] = features.build(synth.generate(n, seed=seed, fraud_rate=0.01))
    return _CACHE[(seed, n)].copy()


class EndToEnd(unittest.TestCase):
    def test_finds_real_signal(self):
        r = model.evaluate(feat())
        best = r["models"][r["selected_model"]]
        self.assertGreater(best["pr_auc"], r["test_prevalence"] * 5)
        self.assertGreater(best["roc_auc"], 0.85)

    def test_negative_control_collapses_to_prevalence(self):
        """Shuffled labels carry no signal. If PR-AUC stays high, something leaks the label."""
        r = model.evaluate(feat(), shuffle_labels=True)
        for name, v in r["models"].items():
            self.assertLess(v["pr_auc"], r["test_prevalence"] * 3, name)
            self.assertLessEqual(v["pr_auc_ci"][0], r["test_prevalence"] * 2, name)

    def test_ledger_artefact_inflates_the_score(self):
        r = model.evaluate(feat())
        self.assertGreater(r["ablation_with_ledger_artefact"]["pr_auc"], r["models"]["gradient_boosting"]["pr_auc"])

    def test_deterministic(self):
        f = feat()
        a, b = model.evaluate(f), model.evaluate(f)
        a.pop("curves"), b.pop("curves")
        self.assertEqual(a, b)

    def test_model_saves_money_against_doing_nothing_and_against_the_rule(self):
        d = model.evaluate(feat())["decision"]["at_review_cost"]
        self.assertLess(d["cost"], d["cost_without_model"])
        self.assertLess(d["cost"], d["rule_cost"])

    def test_threshold_is_set_on_validation_not_test(self):
        """Rewriting test-block labels must leave the chosen threshold and validation score unchanged."""
        base = feat()
        r1 = model.evaluate(base)
        changed = base.copy()
        ts = model.split_points(changed)[1]
        in_test = (changed["step"] >= ts) & (changed["isFraud"] == 1)
        drop = changed.index[in_test][::2]
        changed.loc[drop, "isFraud"] = 0
        r2 = model.evaluate(changed)
        k = str(int(r1["decision"]["review_cost"]))
        self.assertEqual(r1["decision"]["sensitivity"][k]["threshold"], r2["decision"]["sensitivity"][k]["threshold"])
        for name in r1["models"]:
            self.assertEqual(r1["models"][name]["val_pr_auc"], r2["models"][name]["val_pr_auc"])
        self.assertNotEqual(r1["fraud_counts"]["test"], r2["fraud_counts"]["test"])

    def test_block_without_fraud_is_refused(self):
        f = feat()
        f.loc[f["step"] >= model.split_points(f)[1], "isFraud"] = 0
        with self.assertRaises(ValueError):
            model.evaluate(f)

    def test_scope_is_transfer_and_cash_out_only(self):
        r = model.evaluate(feat())
        n_scope = int(feat()["type"].isin(features.ELIGIBLE_TYPES).sum())
        s = r["split"]
        self.assertEqual(s["n_fit"] + s["n_val"] + s["n_test"] + s["n_purged"], n_scope)


if __name__ == "__main__":
    unittest.main()
