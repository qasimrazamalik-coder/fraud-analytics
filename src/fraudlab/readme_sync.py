"""Keep the README results block generated from reports/metrics.json, never typed by hand."""

from __future__ import annotations

import json
import re
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def _money(x: float) -> str:
    return f"{x:,.0f}"


def render_block(r: dict) -> str:
    d, s, dec = r["data"], r["split"], r["decision"]
    L = [
        START,
        f"**Data: {d['provenance'].upper()}** ({d['n_transactions']:,} transactions, "
        f"{d['n_in_scope']:,} in scope, {d['n_fraud']:,} fraud = "
        f"{_pct(d['fraud_rate_in_scope'], 2)} of in-scope)",
        "",
        f"Split by time, label delay {s['label_delay_hours']}h: fit {s['n_fit']:,} rows "
        f"({r['fraud_counts']['fit']} fraud), validation {s['n_val']:,} ({r['fraud_counts']['val']}), "
        f"test {s['n_test']:,} ({r['fraud_counts']['test']}); "
        f"{s['n_purged']:,} rows purged at the two boundaries.",
        "",
        "| Model (test, out of time) | PR-AUC | 95% interval | ROC-AUC | Fraud in top 1% by score |",
        "|---|---|---|---|---|",
    ]
    for name, v in r["models"].items():
        mark = " (selected on validation)" if name == r["selected_model"] else ""
        L.append(
            f"| {name.replace('_', ' ')}{mark} | {v['pr_auc']:.3f} | "
            f"{v['pr_auc_ci'][0]:.3f} to {v['pr_auc_ci'][1]:.3f} | {v['roc_auc']:.3f} | "
            f"{_pct(v['recall_at_1pct_alerts'])} |"
        )
    L += [
        "",
        f"Test prevalence {_pct(r['test_prevalence'], 2)}: a model with no skill scores a PR-AUC near that.",
        "",
        "Operating point: threshold chosen on validation only; the middle row is the default "
        f"review cost ({dec['review_cost']:,.0f} per alert).",
        "",
        "| Review cost | Alerts | Alert rate | Recall | Precision | Cost with model "
        "| Cost, no model | Cost, amount rule | Saving |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for k in sorted(dec["sensitivity"], key=float):
        v = dec["sensitivity"][k]
        L.append(
            f"| {float(k):,.0f} | {v['alerts']:,} | {_pct(v['alert_rate'], 2)} | {_pct(v['recall'])} | "
            f"{_pct(v['precision'])} | {_money(v['cost'])} | {_money(v['cost_without_model'])} | "
            f"{_money(v['rule_cost'])} | {_money(v['saving'])} |"
        )
    a = r["ablation_with_ledger_artefact"]
    best = r["models"][r["selected_model"]]
    L += [
        "",
        "**Leak trap.** Adding the post-event ledger column (`error_balance_dest`) lifts "
        f"gradient boosting from PR-AUC {r['models']['gradient_boosting']['pr_auc']:.3f} "
        f"to {a['pr_auc']:.3f}. "
        "That gain is the generator's artefact, not skill, and it is excluded from every number above "
        f"(selected model: PR-AUC {best['pr_auc']:.3f}).",
    ]
    if r["fraud_outside_scope"]:
        n_out = r["fraud_outside_scope"]
        L += ["", f"{n_out} fraud rows fall outside TRANSFER/CASH_OUT and are unreachable by design."]
    L += [END]
    return "\n".join(L)


def sync(readme: Path, results_path: Path, check: bool = False) -> int:
    results = json.loads(results_path.read_text())
    text = readme.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        print(f"{readme}: results markers not found")
        return 2
    updated = pattern.sub(lambda _m: render_block(results), text)
    if check:
        if updated != text:
            print("README results block is stale: run `make readme`")
            return 1
        print("README results block is in sync")
        return 0
    readme.write_text(updated)
    print("README results block updated")
    return 0
