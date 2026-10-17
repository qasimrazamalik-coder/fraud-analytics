"""python -m fraudlab.cli {generate,run,sync-readme}

`run` reads a PaySim-layout CSV, enforces the contract, builds point-in-time features, evaluates out of
time and writes reports/metrics.json plus figures. The provenance flag is mandatory so a synthetic
result can never be mistaken for a real one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import contract, features, model, synth


def analyse(raw: pd.DataFrame, provenance: str, source: str) -> tuple[dict, dict]:
    contract.validate(raw)
    feat = features.build(raw)
    res = model.evaluate(feat)
    curves = res.pop("curves")
    el = feat[feat["type"].isin(features.ELIGIBLE_TYPES)]
    by_hour = el.groupby("hour_of_day")["isFraud"].agg(["sum", "count"])
    res["data"] = {
        "provenance": provenance,
        "source": source,
        "n_transactions": int(len(feat)),
        "n_in_scope": int(len(el)),
        "n_fraud": int(feat["isFraud"].sum()),
        "fraud_rate_in_scope": float(el["isFraud"].mean()),
    }
    res["fraud_by_hour"] = {
        int(h): {"fraud": int(r["sum"]), "n": int(r["count"])} for h, r in by_hour.iterrows()
    }
    return res, curves


def _generate(a: argparse.Namespace) -> int:
    n = synth.write(a.out_dir, n=a.n, seed=a.seed)
    print(f"wrote {n:,} synthetic PaySim-layout rows to {a.out_dir}/paysim_like.csv")
    return 0


def _run(a: argparse.Namespace) -> int:
    from . import report

    path = Path(a.csv)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make data` or place the real PaySim CSV there.")
    raw = pd.read_csv(path)
    if "isFlaggedFraud" not in raw.columns:
        raw["isFlaggedFraud"] = 0
    res, curves = analyse(raw, a.provenance, str(path))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=float))
    report.render_all(res, curves, Path(a.img_dir))
    print(f"wrote {out / 'metrics.json'} and figures to {a.img_dir} (data={a.provenance})")
    return 0


def _sync(a: argparse.Namespace) -> int:
    from .readme_sync import sync

    return sync(Path(a.readme), Path(a.results), check=a.check)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fraudlab")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="write the seeded synthetic PaySim-layout CSV")
    g.add_argument("--out-dir", default="data/raw")
    g.add_argument("--n", type=int, default=300_000)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=_generate)
    r = sub.add_parser("run", help="validate, build features, evaluate, write metrics + figures")
    r.add_argument("--csv", default="data/raw/paysim_like.csv")
    r.add_argument("--provenance", choices=["synthetic", "real"], required=True)
    r.add_argument("--out-dir", default="reports")
    r.add_argument("--img-dir", default="docs/img")
    r.set_defaults(func=_run)
    s = sub.add_parser("sync-readme", help="regenerate (or --check) the README results block")
    s.add_argument("--readme", default="README.md")
    s.add_argument("--results", default="reports/metrics.json")
    s.add_argument("--check", action="store_true")
    s.set_defaults(func=_sync)
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except contract.DataContractError as exc:
        print(f"DATA CONTRACT VIOLATION: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
