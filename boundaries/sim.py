"""Synthetic scenario generator for agent actions, user preferences and user-defined boundaries.

Every simulated user has a latent preference function that decides whether they would
approve a proposed agent action (ground truth). Users also write an explicit, coarse and
incomplete set of boundary rules over (tool, data category) pairs. The agent only sees the
action description and the stated rules, never the latent preferences.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# tool -> (reversible, external, base risk)
TOOLS = {
    "read_file":    (True,  False, 0.0),
    "create_event": (True,  False, 0.2),
    "write_file":   (True,  False, 0.4),
    "delete_event": (False, False, 0.8),
    "delete_file":  (False, False, 1.2),
    "share_file":   (True,  True,  1.0),
    "send_email":   (False, True,  1.0),
    "make_payment": (False, True,  2.0),
}
TOOL_FREQ = np.array([0.24, 0.14, 0.14, 0.07, 0.08, 0.10, 0.17, 0.06])

CATEGORIES = {"public": 0.0, "work": 0.3, "personal": 0.8, "health": 1.5, "finance": 1.5}
CAT_FREQ = np.array([0.15, 0.40, 0.25, 0.08, 0.12])

RECIPIENTS = {"none": 0.0, "self": 0.0, "team": 0.3, "external_known": 0.8, "external_unknown": 1.8}
EXTERNAL_RECIP = ["self", "team", "external_known", "external_unknown"]
EXTERNAL_RECIP_FREQ = np.array([0.15, 0.40, 0.35, 0.10])

SCOPES = {"single": 0.0, "few": 0.3, "bulk": 1.0}
SCOPE_FREQ = np.array([0.70, 0.22, 0.08])

ARCHETYPES = {
    # name: (base permissiveness, sensitivity scale, population share)
    "conservative": (2.0, 1.4, 0.30),
    "balanced":     (3.0, 1.0, 0.45),
    "delegator":    (4.0, 0.7, 0.25),
}

MISTAKE_TYPES = ["off_task", "wrong_recipient", "overbroad_scope"]
RULE_MARGIN = 1.0          # |typical score| above which a user writes allow/deny instead of ask
RULE_FLIP_PROB = 0.05      # probability that a stated rule is misspecified


@dataclass
class SimConfig:
    n_users: int = 300
    actions_per_user: int = 60
    rule_coverage: float = 0.6   # share of (tool, category) pairs for which a user writes a rule
    mistake_rate: float = 0.15   # share of agent proposals that contain an injected agent error


def _sample_user(rng, uid):
    names = list(ARCHETYPES)
    arche = names[rng.choice(len(names), p=[ARCHETYPES[n][2] for n in names])]
    base, scale, _ = ARCHETYPES[arche]
    return {
        "user_id": uid,
        "archetype": arche,
        "base": base + rng.normal(0, 0.3),
        "w_tool": scale * rng.lognormal(0, 0.25),
        "w_cat": {c: scale * rng.lognormal(0, 0.4) for c in CATEGORIES},
        "w_recip": scale * rng.lognormal(0, 0.3),
        "w_scope": scale * rng.lognormal(0, 0.3),
        # idiosyncratic per-pair preference not explained by the shared risk structure
        "pair_offset": {(t, c): rng.normal(0, 0.6) for t in TOOLS for c in CATEGORIES},
    }


def _score(user, tool, cat, recip, scope):
    """Deterministic part of a user's approval score (positive = would approve)."""
    return (user["base"]
            - user["w_tool"] * TOOLS[tool][2]
            - user["w_cat"][cat] * CATEGORIES[cat]
            - user["w_recip"] * RECIPIENTS[recip]
            - user["w_scope"] * SCOPES[scope]
            + user["pair_offset"][(tool, cat)])


def _typical_recipient(tool):
    return "team" if TOOLS[tool][1] else "none"


def _write_rules(rng, user, coverage):
    """The user's stated boundaries: coarse, incomplete and occasionally misspecified."""
    rules = {}
    for t in TOOLS:
        for c in CATEGORIES:
            if rng.random() >= coverage:
                continue
            m = _score(user, t, c, _typical_recipient(t), "single")
            rule = "allow" if m > RULE_MARGIN else "deny" if m < -RULE_MARGIN else "ask"
            if rng.random() < RULE_FLIP_PROB:
                rule = rng.choice([r for r in ("allow", "ask", "deny") if r != rule])
            rules[(t, c)] = rule
    return rules


def _sample_action(rng, user, mistake_rate):
    tool = list(TOOLS)[rng.choice(len(TOOLS), p=TOOL_FREQ)]
    cat = list(CATEGORIES)[rng.choice(len(CATEGORIES), p=CAT_FREQ)]
    recip = EXTERNAL_RECIP[rng.choice(4, p=EXTERNAL_RECIP_FREQ)] if TOOLS[tool][1] else "none"
    scope = list(SCOPES)[rng.choice(3, p=SCOPE_FREQ)]
    mistake = "none"
    penalty = 0.0
    if rng.random() < mistake_rate:
        mistake = MISTAKE_TYPES[rng.integers(3)]
        if mistake == "wrong_recipient" and not TOOLS[tool][1]:
            mistake = "overbroad_scope"
        if mistake == "off_task":
            penalty = 3.0
        elif mistake == "wrong_recipient":
            recip, penalty = "external_unknown", 2.0
        else:
            scope, penalty = "bulk", 1.5
    # noisy task-relevance signal the agent can observe (e.g. from its own plan)
    relevance = rng.beta(2, 5) if mistake == "off_task" else rng.beta(5, 2)
    score = _score(user, tool, cat, recip, scope) - penalty + rng.normal(0, 0.5)
    return {
        "tool": tool, "category": cat, "recipient": recip, "scope": scope,
        "reversible": int(TOOLS[tool][0]), "external": int(TOOLS[tool][1]),
        "relevance": relevance, "mistake": mistake, "latent_score": score,
        "approve": int(score > 0),
    }


def generate(cfg: SimConfig, seed: int):
    """Return (actions DataFrame, rules dict user_id -> {(tool, cat): rule})."""
    rng = np.random.default_rng(seed)
    rows, all_rules = [], {}
    for uid in range(cfg.n_users):
        user = _sample_user(rng, uid)
        all_rules[uid] = _write_rules(rng, user, cfg.rule_coverage)
        for _ in range(cfg.actions_per_user):
            a = _sample_action(rng, user, cfg.mistake_rate)
            a.update(user_id=uid, archetype=user["archetype"])
            a["rule"] = all_rules[uid].get((a["tool"], a["category"]), "none")
            rows.append(a)
    return pd.DataFrame(rows), all_rules


def describe(row):
    """Readable one-line description of an action, used for case-level error analysis."""
    tgt = f" to {row['recipient']}" if row["recipient"] != "none" else ""
    return f"{row['tool']}({row['category']}, {row['scope']}){tgt}"
