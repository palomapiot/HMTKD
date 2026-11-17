"""Evaluation metrics. Stdlib only, so they run anywhere.

Every metric takes prediction rows joined with their corpus record:
    {"record": <corpus record>, "label": "hate"|"offensive"|"normal"|None, "analysis": dict | None}
An unparsed output has label None and counts as an error.
"""

from collections import defaultdict

from . import scoring
from . import taxonomy as tx


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def f1_per_class(gold, pred, classes):
    out = {}
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def macro_f1(gold, pred, classes):
    """Macro-F1 over the classes present in the gold labels (None = always wrong)."""
    present = [c for c in classes if c in gold]
    if not present:
        return None
    return round(sum(f1_per_class(gold, pred, present).values()) / len(present), 4)


def classification(rows):
    """Set accuracy, 3-way macro-F1 (single-label posts) and hate-vs-rest macro-F1."""
    out = {"n": len(rows),
           "parse_rate": mean(r["label"] is not None for r in rows),
           "set_accuracy": mean(r["label"] in r["record"]["label_set"] for r in rows)}
    single = [r for r in rows if len(r["record"]["label_set"]) == 1]
    out["n_single_label"] = len(single)
    out["macro_f1_3way"] = macro_f1([r["record"]["label_set"][0] for r in single],
                                    [r["label"] for r in single], tx.LABELS)

    decided = [(tx.hate_gold(r["record"]["label_set"]), r) for r in rows]
    decided = [(g, r) for g, r in decided if g is not None]
    gold = [g for g, _ in decided]
    pred = [None if r["label"] is None else int(r["label"] == "hate") for _, r in decided]
    out["hate_vs_rest"] = {"n": len(decided), "macro_f1": macro_f1(gold, pred, [0, 1]),
                           "f1_positive": round(f1_per_class(gold, pred, [1])[1], 4) if decided else None}
    return out


def analyses(rows):
    """Agreement of generated rationales with the gold annotation (rationale outputs only)."""
    scored = [scoring.facet_scores(r["analysis"], r["record"]) for r in rows if r.get("analysis")]
    if not scored:
        return None
    sev = [s["severity_dist"] for s in scored if s["severity_dist"] is not None]
    return {
        "n": len(scored),
        "consistency_rate": mean(s["consistent"] for s in scored),
        "target_f1": mean(s["target_f1"] for s in scored),
        "severity_in_range": mean(d == 0 for d in sev) if sev else None,
        "severity_distance": mean(sev) if sev else None,
        "stance_accuracy": mean(s["stance_ok"] for s in scored),
        "evidence_verbatim": mean(s["evidence_verbatim"] for s in scored),
        "explanation_words": mean(s["n_words"] for s in scored),
        "explanation_in_band": mean(scoring.ACCEPT["min_words"] <= s["n_words"] <= scoring.ACCEPT["max_words"]
                                    for s in scored),
    }


def by_target(rows):
    """Hate detection per gold target group (posts with annotated targets)."""
    out = {}
    for g in tx.TARGET_GROUPS:
        sub = [r for r in rows if g in (r["record"]["targets"] or [])]
        pairs = [(y, r["label"] == "hate") for y, r in
                 ((tx.hate_gold(r["record"]["label_set"]), r) for r in sub) if y is not None]
        neg = [p for y, p in pairs if y == 0]
        out[g] = {"n": len(pairs),
                  "false_positive_rate": mean(neg) if neg else None,
                  "hate_f1": round(f1_per_class([y for y, _ in pairs], [int(p) for _, p in pairs], [1])[1], 4)
                  if pairs else None}
    return out


def evaluate(rows):
    """All metrics, per dataset."""
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["record"]["dataset"]].append(r)
    return {ds: {"classification": classification(rs), "analyses": analyses(rs),
                 "by_target": by_target(rs) if any(r["record"]["targets"] for r in rs) else None}
            for ds, rs in sorted(by_ds.items())}
