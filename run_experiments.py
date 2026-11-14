"""Main experiment: baselines, Design A, Design B and mechanism-check ablations.

Usage:  python run_experiments.py [--seeds 10] [--out results]
Writes per-seed metrics, summary tables with 95% CIs, subgroup tables, a coverage sweep,
a habituation sensitivity analysis and case-level examples to the output folder.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boundaries.metrics import evaluate, expected_cost_fn, mean_ci, prob_metrics
from boundaries.policies import (BoundaryRiskModel, policy_always_ask, policy_broad,
                                 policy_rules, EXECUTE, ASK, BLOCK)
from boundaries.sim import SimConfig, describe, generate

SPLIT = (0.6, 0.2, 0.2)  # train / validation / test, split by USER (test users are unseen)


def split_users(df, seed):
    users = np.random.default_rng(seed + 999).permutation(df["user_id"].unique())
    n = len(users)
    a, b = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
    pick = lambda u: df[df["user_id"].isin(u)].reset_index(drop=True)
    return pick(users[:a]), pick(users[a:b]), pick(users[b:])


def fit_models(train, val, rules, seed):
    cost_fn = expected_cost_fn()
    m = {
        "B_full": BoundaryRiskModel(True, True, seed).fit(train, val, rules).tune_thresholds(val, rules, cost_fn),
        "B_no_boundaries": BoundaryRiskModel(False, False, seed).fit(train, val, rules).tune_thresholds(val, rules, cost_fn),
        "B_soft_deny": BoundaryRiskModel(True, False, seed).fit(train, val, rules).tune_thresholds(val, rules, cost_fn),
    }
    nb = BoundaryRiskModel(True, True, seed)
    nb.raw, nb.cal = m["B_full"].raw, m["B_full"].cal   # same model, only the ask band removed
    m["B_no_ask_band"] = nb.tune_thresholds(val, rules, cost_fn, band=False)
    return m


def all_decisions(test, rules, models):
    d = {
        "B0_broad": policy_broad(test),
        "B1_always_ask": policy_always_ask(test),
        "A_rules": policy_rules(test, with_ask=True),
        "A_no_ask_tier": policy_rules(test, with_ask=False),
    }
    for name, mdl in models.items():
        d[name] = mdl.decide(test, mdl.predict_proba(test, rules))
    return d


def run_seed(seed, cfg, habituations=(0.15,)):
    df, rules = generate(cfg, seed)
    train, val, test = split_users(df, seed)
    models = fit_models(train, val, rules, seed)
    dec = all_decisions(test, rules, models)

    rows, sub_rows = [], []
    for h in habituations:
        for name, d in dec.items():
            r = evaluate(test, d, np.random.default_rng(seed), habituation=h)
            rows.append({"seed": seed, "policy": name, "habituation": h, **r})
    # subgroup results at the default habituation
    groups = {
        "archetype": test["archetype"],
        "irreversible": test["reversible"].map({0: "irreversible", 1: "reversible"}),
        "rule_present": pd.Series(np.where(test["rule"] == "none", "unruled pair", "ruled pair")),
        "mistake": test["mistake"],
    }
    for gname, g in groups.items():
        for gval in pd.unique(g):
            mask = (g == gval).values
            sub = test[mask].reset_index(drop=True)
            for name, d in dec.items():
                r = evaluate(sub, d[mask], np.random.default_rng(seed))
                sub_rows.append({"seed": seed, "group": gname, "value": gval, "policy": name,
                                 "n": int(mask.sum()), **r})

    prob = []
    for name in ("B_full", "B_no_boundaries"):
        for calibrated in (False, True):
            p = models[name].predict_proba(test, rules, calibrated)
            prob.append({"seed": seed, "model": name, "calibrated": calibrated,
                         **prob_metrics(test["approve"].values, p)})
    thresholds = {k: (m.tau_lo, m.tau_hi) for k, m in models.items()}
    base = {"seed": seed, "test_actions": len(test), "test_users": test["user_id"].nunique(),
            "approve_rate": test["approve"].mean(), "mistake_rate": (test["mistake"] != "none").mean(),
            "rule_coverage_obs": (test["rule"] != "none").mean(), "thresholds": thresholds}
    return rows, sub_rows, prob, base, (test, dec, models)


def coverage_sweep(seeds, coverages):
    rows = []
    for cov in coverages:
        for s in seeds:
            cfg = SimConfig(rule_coverage=cov)
            df, rules = generate(cfg, s)
            train, val, test = split_users(df, s)
            cost_fn = expected_cost_fn()
            pol = {"A_rules": policy_rules(test)}
            for name, ub in (("B_full", True), ("B_no_boundaries", False)):
                m = BoundaryRiskModel(ub, ub, s).fit(train, val, rules).tune_thresholds(val, rules, cost_fn)
                pol[name] = m.decide(test, m.predict_proba(test, rules))
            for name, d in pol.items():
                rows.append({"coverage": cov, "seed": s, "policy": name,
                             **evaluate(test, d, np.random.default_rng(s))})
    return pd.DataFrame(rows)


def cases(test, dec, models, k=4):
    """Representative successes and failures of Design B versus Design A (seed 0)."""
    t = test.copy()
    names = {EXECUTE: "execute", ASK: "ask", BLOCK: "block"}
    t["A"] = [names[x] for x in dec["A_rules"]]
    t["B"] = [names[x] for x in dec["B_full"]]
    t["desc"] = t.apply(describe, axis=1)
    cols = ["desc", "archetype", "rule", "relevance", "mistake", "approve", "A", "B"]
    sel = {
        "B_catches_A_misses": t[(t.approve == 0) & (t.A == "execute") & (t.B != "execute")],
        "B_executes_unwanted": t[(t.approve == 0) & (t.B == "execute")],
        "B_saves_prompt": t[(t.approve == 1) & (t.A == "ask") & (t.B == "execute")],
        "B_overblocks": t[(t.approve == 1) & (t.B == "block")],
    }
    out = {name: v[cols].head(k).round(2).to_dict(orient="records") for name, v in sel.items()}
    out.update({f"n_{name}": int(len(v)) for name, v in sel.items()})
    return out


def summarise(df, keys, metrics):
    out = []
    for key, g in df.groupby(keys):
        row = dict(zip(keys if isinstance(keys, list) else [keys], key if isinstance(key, tuple) else (key,)))
        for m in metrics:
            mu, h = mean_ci(g[m])
            row[m] = mu
            row[m + "_ci"] = h
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    cfg = SimConfig()
    metrics = ["UER", "FBR", "ask_rate", "autonomy", "mistake_catch", "mistake_exec", "cost", "block_rate"]

    all_rows, all_sub, all_prob, bases = [], [], [], []
    for s in range(args.seeds):
        rows, sub, prob, base, payload = run_seed(s, cfg, habituations=(0.0, 0.15, 0.30))
        all_rows += rows; all_sub += sub; all_prob += prob; bases.append(base)
        if s == 0:
            json.dump(cases(*payload), open(out / "cases_seed0.json", "w"), indent=2, default=str)
        print(f"seed {s} done")

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summarise(per_seed, ["habituation", "policy"], metrics).to_csv(out / "summary.csv", index=False)
    summarise(pd.DataFrame(all_sub), ["group", "value", "policy"], metrics + ["n"]).to_csv(out / "subgroups.csv", index=False)
    summarise(pd.DataFrame(all_prob), ["model", "calibrated"], ["AUROC", "Brier", "ECE"]).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(bases).to_json(out / "data_stats.json", orient="records", indent=2)

    cov = coverage_sweep(range(min(args.seeds, 5)), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cov.to_csv(out / "coverage_per_seed.csv", index=False)
    summarise(cov, ["coverage", "policy"], ["UER", "ask_rate", "cost", "FBR"]).to_csv(out / "coverage.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
