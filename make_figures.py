"""Figures, paired mechanism-check statistics and LaTeX tables from results/.

Usage:  python make_figures.py [--res results] [--fig ../report/figures] [--tab ../report/tables]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from boundaries.metrics import mean_ci

LABELS = {
    "B0_broad": "B0 Broad permission", "B1_always_ask": "B1 Always confirm",
    "A_rules": "A Scoped rules", "A_no_ask_tier": "A $-$ ask tier",
    "B_full": "B Boundary-aware model", "B_no_boundaries": "B $-$ boundaries",
    "B_no_ask_band": "B $-$ ask band", "B_soft_deny": "B $-$ hard deny",
}
MAIN = ["B0_broad", "B1_always_ask", "A_rules", "B_full"]
ABL = ["A_no_ask_tier", "B_no_boundaries", "B_no_ask_band", "B_soft_deny"]
COLORS = {"B0_broad": "#9e9e9e", "B1_always_ask": "#616161", "A_rules": "#e08a2c", "B_full": "#2b6cb0",
          "A_no_ask_tier": "#f2c08f", "B_no_boundaries": "#90b4dd", "B_no_ask_band": "#5a8fcf", "B_soft_deny": "#1d4577"}


def pipeline_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 46); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec="#333333", lw=1.2, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, c="#333333", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2, linestyle=ls))

    hc = "#dbe8f7"
    box(1, 30, 15, 9, "User task\n(goal in\nnatural language)")
    box(21, 30, 16, 9, "Agent planner\n(simulated)\nproposes tool call")
    box(1, 12, 15, 11, "User-defined\nboundaries\nallow / ask / deny\nper (tool, data)", fc=hc, ec="#2b6cb0", lw=1.8, bold=True)
    box(42, 30, 17, 9, "Action features\ntool, data category,\nrecipient, scope,\nreversible, relevance")
    box(42, 12, 17, 11, "Boundary features\nrule for this pair,\nuser deny/allow share,\nsame-tool / same-data", fc=hc, ec="#2b6cb0", lw=1.8)
    box(63, 21, 15, 12, "Risk model\nGBDT + isotonic\ncalibration\n$\\hat p$ = P(approve)", fc=hc, ec="#2b6cb0", lw=1.8, bold=True)
    box(82, 30, 16, 9, "Gate\n$\\hat p\\geq\\tau_{hi}$: execute\n$\\tau_{lo}<\\hat p<\\tau_{hi}$: ASK\n$\\hat p\\leq\\tau_{lo}$: block", fc=hc, ec="#2b6cb0", lw=1.8)
    box(82, 12, 16, 11, "Hard constraints\ndeny rule $\\to$ block\nallow rule never\nsilently blocked", fc=hc, ec="#2b6cb0", lw=1.8)
    box(62, 1, 36, 7, "Outcome: execute / ask user / block\n$\\to$ UER, FBR, ask rate, expected cost", fs=8)
    box(20, 1, 36, 7, "Design A (rules): stated rule $\\to$ direct lookup;\nunruled pair $\\to$ static tier\n(irreversible or external $\\to$ ask)",
        fc="#fdf0e2", ec="#a35c12", lw=1.4, fs=7.8)
    arrow(16, 34.5, 21, 34.5); arrow(37, 34.5, 42, 34.5); arrow(16, 17.5, 42, 17.5)
    arrow(59, 34.5, 63, 29); arrow(59, 17.5, 63, 25); arrow(78, 27, 82, 33)
    arrow(90, 30, 90, 23); arrow(90, 12, 90, 8.6)
    arrow(10, 12, 20, 6, c="#a35c12", ls="--"); arrow(56, 4.5, 62, 4.5, c="#a35c12", ls="--")
    ax.text(1, 44, "Design B (proposed): boundary-aware calibrated gating", fontsize=10, fontweight="bold", color="#2b6cb0")
    ax.text(99, 44, "blue = human-centred mechanism", fontsize=8, color="#2b6cb0", style="italic", ha="right")
    ax.text(1, 41.2, "Baselines on the same proposals: B0 executes every proposal; B1 asks before every proposal.", fontsize=7.8, color="#555555")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def tradeoff(summary, path):
    s = summary[summary.habituation == 0.15].set_index("policy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, x in zip(axes, ["ask_rate", "FBR"]):
        for p in MAIN + ABL:
            r = s.loc[p]
            ax.errorbar(r[x], r["UER"], xerr=r[x + "_ci"], yerr=r["UER_ci"], fmt="o" if p in MAIN else "s",
                        color=COLORS[p], ms=7 if p in MAIN else 5, capsize=2, label=LABELS[p])
        ax.set_xlabel({"ask_rate": "Confirmation rate (share of proposals)", "FBR": "False-block rate  P(not executed | wanted)"}[x])
        ax.set_ylabel("Unintended execution rate (UER)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def coverage_plot(cov, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, m, lab in zip(axes, ["cost", "UER"], ["Expected cost per action", "Unintended execution rate"]):
        for p in ["A_rules", "B_full", "B_no_boundaries"]:
            g = cov[cov.policy == p].sort_values("coverage")
            ax.errorbar(g.coverage, g[m], yerr=g[m + "_ci"], marker="o", capsize=2, color=COLORS[p], label=LABELS[p])
        ax.set_xlabel("Rule coverage (share of (tool, data) pairs with a stated rule)")
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def subgroup_plot(sub, path):
    sub = sub[(sub.group == "archetype") & sub.policy.isin(["A_rules", "B_full", "B_no_boundaries"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    arche = ["conservative", "balanced", "delegator"]
    for ax, m, lab in zip(axes, ["UER", "FBR"], ["Unintended execution rate", "False-block rate"]):
        for i, p in enumerate(["A_rules", "B_full", "B_no_boundaries"]):
            g = sub[sub.policy == p].set_index("value").loc[arche]
            ax.bar(np.arange(3) + (i - 1) * 0.26, g[m], 0.26, yerr=g[m + "_ci"], capsize=2, color=COLORS[p], label=LABELS[p])
        ax.set_xticks(range(3)); ax.set_xticklabels(arche); ax.set_ylabel(lab); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def paired(per_seed, ref, others, metrics, h=0.15):
    """Paired (same seed, same test users) differences other - ref with 95% t-CI."""
    d = per_seed[per_seed.habituation == h].set_index(["seed", "policy"])
    rows = []
    for o in others:
        row = {"policy": o}
        for m in metrics:
            diff = d.xs(o, level="policy")[m] - d.xs(ref, level="policy")[m]
            row[m], row[m + "_ci"] = mean_ci(diff)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mu, ci, pct=True, signed=False):
    if pct:
        mu, ci = 100 * mu, 100 * ci
    s = f"{mu:+.1f}" if signed else f"{mu:.1f}"
    return f"{s} $\\pm$ {ci:.1f}" if pct else (f"{mu:+.3f}" if signed else f"{mu:.3f}") + f" $\\pm$ {ci:.3f}"


def tables(res, tab):
    s = pd.read_csv(res / "summary.csv")
    s15 = s[s.habituation == 0.15].set_index("policy")
    lines = []
    for p in MAIN + ABL:
        r = s15.loc[p]
        cells = [fmt(r[m], r[m + "_ci"]) for m in ["UER", "FBR", "ask_rate", "autonomy", "mistake_exec"]]
        cells.append(fmt(r["cost"], r["cost_ci"], pct=False))
        if p == ABL[0]:
            lines.append("\\midrule")
        lines.append(LABELS[p] + " & " + " & ".join(cells) + " \\\\")
    (tab / "main.tex").write_text("\n".join(lines) + "\n")

    per_seed = pd.read_csv(res / "per_seed.csv")
    pr = paired(per_seed, "B_full", ["B_no_boundaries", "B_no_ask_band", "B_soft_deny"], ["UER", "FBR", "ask_rate", "autonomy", "cost"])
    pa = paired(per_seed, "A_rules", ["A_no_ask_tier"], ["UER", "FBR", "ask_rate", "autonomy", "cost"])
    lines = []
    for _, r in pd.concat([pr, pa]).iterrows():
        cells = [fmt(r[m], r[m + "_ci"], signed=True) for m in ["UER", "FBR", "ask_rate", "autonomy"]]
        cells.append(fmt(r["cost"], r["cost_ci"], pct=False, signed=True))
        ref = "A" if r.policy.startswith("A") else "B"
        lines.append(f"{LABELS[r.policy]} vs.\\ {ref} & " + " & ".join(cells) + " \\\\")
    (tab / "ablation.tex").write_text("\n".join(lines) + "\n")
    pd.concat([pr, pa]).to_csv(res / "paired_ablation.csv", index=False)

    sub = pd.read_csv(res / "subgroups.csv")
    lines = []
    order = [("archetype", "conservative"), ("archetype", "balanced"), ("archetype", "delegator"),
             ("irreversible", "irreversible"), ("irreversible", "reversible"),
             ("rule_present", "ruled pair"), ("rule_present", "unruled pair"),
             ("mistake", "none"), ("mistake", "off_task"), ("mistake", "overbroad_scope"), ("mistake", "wrong_recipient")]
    for g, v in order:
        rr = sub[(sub.group == g) & (sub.value == v)].set_index("policy")
        n = int(round(rr.loc["B_full", "n"]))
        cells = []
        for p in ["A_rules", "B_full", "B_no_boundaries"]:
            cells += [fmt(rr.loc[p, "UER"], rr.loc[p, "UER_ci"]), fmt(rr.loc[p, "FBR"], rr.loc[p, "FBR_ci"])]
        lines.append(f"{v.replace('_', ' ')} & {n} & " + " & ".join(cells) + " \\\\")
    (tab / "subgroups.tex").write_text("\n".join(lines) + "\n")

    hab = s[s.policy.isin(["B1_always_ask", "A_rules", "B_full"])]
    lines = []
    for p in ["B1_always_ask", "A_rules", "B_full"]:
        cells = []
        for h in (0.0, 0.15, 0.30):
            r = hab[(hab.policy == p) & (hab.habituation == h)].iloc[0]
            cells += [fmt(r["UER"], r["UER_ci"]), fmt(r["cost"], r["cost_ci"], pct=False)]
        lines.append(LABELS[p] + " & " + " & ".join(cells) + " \\\\")
    (tab / "habituation.tex").write_text("\n".join(lines) + "\n")

    cal = pd.read_csv(res / "calibration.csv")
    lines = []
    for _, r in cal.iterrows():
        lines.append(f"{LABELS[r.model]} & {'isotonic' if r.calibrated else 'raw'} & "
                     + " & ".join(fmt(r[m], r[m + '_ci'], pct=False) for m in ["AUROC", "Brier", "ECE"]) + " \\\\")
    (tab / "calibration.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--fig", default="../report/figures")
    ap.add_argument("--tab", default="../report/tables")
    a = ap.parse_args()
    res, fig, tab = Path(a.res), Path(a.fig), Path(a.tab)
    fig.mkdir(parents=True, exist_ok=True); tab.mkdir(parents=True, exist_ok=True)
    pipeline_diagram(fig / "pipeline.pdf"); pipeline_diagram(fig / "pipeline.png")
    tradeoff(pd.read_csv(res / "summary.csv"), fig / "tradeoff.pdf")
    coverage_plot(pd.read_csv(res / "coverage.csv"), fig / "coverage.pdf")
    subgroup_plot(pd.read_csv(res / "subgroups.csv"), fig / "subgroups.pdf")
    tables(res, tab)
    print("figures and tables written")


if __name__ == "__main__":
    main()
