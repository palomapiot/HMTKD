"""Stage III rewards (paper Section 3.4, Equation 3), computed against the human annotation.

Each reward maps one completion and its post to a value in [0, 1] (label: [-2, 2]), or None
when the gold component it needs is not annotated: None is left out of the weighted sum
(the indicator I[G_c != empty] of Equation 3). The checks reuse hmtkd/scoring.py, so a
completion is rewarded on exactly what Stage I accepted teacher rationales on.

    fmt   parses x explanation-length factor (1 inside 80-150 words, linear to 0 at 40 and 300 words)
    lbl   hate or not hate: "offensive" and "normal" are the same answer. Weight
          w = max(0.2, 2 * agreement - 1) (annotator agreement); +2w(1 - p_y) if correct, -2w p_y
          otherwise, where p_y is the (agreement-weighted) share of training posts of class y, so
          always answering one class earns 0 on average. None where the gold set holds hate and
          another label.
    cons  1 if the label equals the decision rule applied to the completion's own fields
    tgt   target F1 with the gold targets
    sev   1 - distance to the gold severity range / 4
    stn   1 if the stance equals the gold stance
    evd   share of evidence spans copied verbatim (0 if a non-normal label cites none, or if the
          spans cover > 80% of a post of 10+ tokens)
    sim   max cosine between the explanation and the post's accepted teacher explanations
"""

from . import scoring
from . import taxonomy as tx
from .io import agreement

WEIGHTS = {"lbl": 1.0, "cons": 0.5, "tgt": 0.5, "sev": 0.5, "stn": 0.5, "evd": 0.5, "sim": 0.5,
           "fmt": 0.25}


def label_priors(records):
    """Agreement-weighted share of the posts with a binary gold that are hate / not hate."""
    decided = [r for r in records if hate_gold(r) is not None]
    total = sum(label_weight(r) for r in decided)
    p = sum(label_weight(r) for r in decided if hate_gold(r) == 1) / total
    return {"hate": p, "not": 1 - p}


def hate_gold(record):
    """1 hate, 0 not hate, None when the gold set holds hate and another label."""
    return tx.hate_gold(record["label_set"])


def label_reward(label, record, priors):
    """Label reward of one prediction (see the module docstring)."""
    gold = hate_gold(record)
    if gold is None:
        return None
    w = label_weight(record)
    if label is None:
        return -w
    pred = "hate" if label == "hate" else "not"
    return 2 * w * (1 - priors[pred]) if (pred == "hate") == (gold == 1) else -2 * w * priors[pred]


def label_weight(record):
    a = agreement(record)
    return 1.0 if a is None else max(0.2, 2 * a - 1)


def length_factor(n_words, lo=scoring.ACCEPT["min_words"], hi=scoring.ACCEPT["max_words"]):
    """1 inside [lo, hi], linear to 0 at lo/2 and at 2*hi (paper, format reward)."""
    if n_words < lo:
        return max(0.0, (n_words - lo / 2) / (lo / 2))
    if n_words > hi:
        return max(0.0, (2 * hi - n_words) / hi)
    return 1.0


def component_rewards(text, record, priors, references=None, embed=None):
    """dict reward name -> value or None for one completion."""
    a = scoring.parse_analysis(text)
    if a is None:   # unparsed: no format credit and a wrong label; the fields cannot be scored
        lbl = None if hate_gold(record) is None else -label_weight(record)
        return {"fmt": 0.0, "lbl": lbl, "cons": 0.0, "tgt": None, "sev": None,
                "stn": None, "evd": 0.0, "sim": 0.0 if references else None}
    s = scoring.facet_scores(a, record)
    evd = s["evidence_verbatim"]
    if a["label"] != "normal" and not a["evidence"]:
        evd = 0.0
    elif evd is None:                      # normal label without evidence: nothing to check
        evd = 1.0
    if s["n_post_tokens"] >= scoring.ACCEPT["coverage_min_post_tokens"] and \
            s["evidence_coverage"] > scoring.ACCEPT["max_evidence_coverage"]:
        evd = 0.0
    sim = None
    if references and embed is not None:
        sim = max(0.0, max(embed(a["explanation"], ref) for ref in references))
    return {
        "fmt": length_factor(s["n_words"]),
        "lbl": label_reward(a["label"], record, priors),
        "cons": float(s["consistent"]),
        "tgt": s["target_f1"],
        "sev": None if s["severity_dist"] is None else 1 - s["severity_dist"] / 4,
        "stn": None if s["stance_ok"] is None else float(s["stance_ok"]),
        "evd": evd,
        "sim": sim,
    }


def total(components, weights=WEIGHTS):
    """Equation 3: weighted sum over the components whose gold is available."""
    return sum(weights[k] * v for k, v in components.items() if v is not None and k in weights)
