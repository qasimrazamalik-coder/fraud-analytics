"""Figures for the README. Palette: blue and orange categorical slots, ink and grid tokens."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _fig(w=8.0, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, ax, path: Path, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def pr_curves(res: dict, curves: dict, path: Path) -> None:
    y = np.array(curves["y_test"])
    fig, ax = _fig()
    for key, label, color, ls in (
        ("score_clean", "clean features (what the model may know)", BLUE, "-"),
        ("score_leaky", "plus post-event ledger column (leak)", ORANGE, "--"),
    ):
        p, r, _ = precision_recall_curve(y, np.array(curves[key]))
        ax.plot(r, p, color=color, lw=2, ls=ls, label=label)
    ax.axhline(res["test_prevalence"], color=INK2, lw=1, ls=":", label="no skill (prevalence)")
    ax.set_xlabel("recall", color=INK2)
    ax.set_ylabel("precision", color=INK2)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    _save(fig, ax, path, "Out-of-time precision-recall: the leak looks like skill")


def cost_curve(res: dict, curves: dict, path: Path) -> None:
    """Total cost on the test block as the number of alerts grows, at the chosen review cost."""
    y, s, a = (np.array(curves[k]) for k in ("y_test", "score_clean", "amount_test"))
    rc = res["decision"]["review_cost"]
    order = np.argsort(-s, kind="stable")
    caught = np.concatenate([[0.0], np.cumsum(np.where(y[order] == 1, a[order], 0.0))])
    k = np.arange(len(s) + 1)
    total = (a[y == 1].sum() - caught) + k * rc
    step = max(1, len(s) // 1500)
    fig, ax = _fig()
    ax.plot(k[::step] / len(s) * 100, total[::step], color=BLUE, lw=2, label="cost by alert volume")
    op = res["decision"]["at_review_cost"]
    ax.scatter([op["alert_rate"] * 100], [op["cost"]], color=ORANGE, s=60, zorder=3,
               label="threshold picked on validation")  # fmt: skip
    ax.axhline(op["cost_without_model"], color=INK2, lw=1, ls=":", label="no model: every fraud lands")
    ax.set_xlim(0, min(100, max(5, op["alert_rate"] * 100 * 8)))
    ax.set_xlabel("share of transactions flagged (%)", color=INK2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.0f}M")
    ax.set_ylabel("fraud loss + review cost (currency units)", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _save(fig, ax, path, f"Test cost at review cost {rc:,.0f} per alert")


def hour_rates(res: dict) -> list[float]:
    """Fraud rate (%) for each hour 0-23. Keys are ints in memory and strings after a JSON round trip."""
    h = {int(k): v for k, v in res["fraud_by_hour"].items()}
    return [h[x]["fraud"] / h[x]["n"] * 100 if x in h and h[x]["n"] else 0.0 for x in range(24)]


def fraud_by_hour(res: dict, path: Path) -> None:
    hours = list(range(24))
    rate = hour_rates(res)
    fig, ax = _fig()
    ax.bar(hours, rate, color=BLUE, width=0.7)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("hour of day", color=INK2)
    ax.set_ylabel("fraud rate in scope (%)", color=INK2)
    _save(fig, ax, path, "Fraud rate by hour of day")


def render_all(res: dict, curves: dict, img_dir: Path) -> None:
    pr_curves(res, curves, img_dir / "pr_curves.png")
    cost_curve(res, curves, img_dir / "cost_curve.png")
    fraud_by_hour(res, img_dir / "fraud_by_hour.png")
