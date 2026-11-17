"""Parse a rationale and score it against the human annotation of a record.

Shared by the teacher acceptance checks (Stage I) and the GRPO rewards (Stage III), so a
rationale is accepted under exactly the same checks the student is rewarded on.
Every score is None when the gold field is not annotated.
"""

import json
import re

from . import taxonomy as tx

FIELDS = ["evidence", "targets", "stance", "severity", "explanation", "label"]


# Placeholders some teachers write inside the list instead of leaving it empty (Gemma 4: ["[]"]).
NO_EVIDENCE = {"", "[]", "none", "None", "N/A", "n/a"}


def parse_analysis(text):
    """JSON analysis -> normalised dict, or None if malformed."""
    if text is None:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict) or any(f not in raw for f in FIELDS):
        return None
    try:
        severity = int(raw["severity"])
    except (TypeError, ValueError):
        return None
    if (raw["label"] not in tx.LABELS or raw["stance"] not in tx.STANCES
            or severity not in tx.SEVERITY_LEVELS
            or not isinstance(raw["evidence"], list) or not isinstance(raw["targets"], list)
            or not isinstance(raw["explanation"], str)):
        return None
    return {
        "evidence": [str(e).strip() for e in raw["evidence"] if str(e).strip() not in NO_EVIDENCE],
        "targets": sorted({t for t in raw["targets"] if t in tx.TARGET_GROUPS}),
        "stance": raw["stance"],
        "severity": severity,
        "explanation": raw["explanation"].strip(),
        "label": raw["label"],
    }



def parse_label(text):
    """Label of a direct-label answer ({"label": ...}), or None if malformed."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        label = json.loads(match.group(0)).get("label") if match else None
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in tx.LABELS else None

def _norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def _tokens(s):
    return re.findall(r"\w+", s.lower())


def evidence_verbatim(evidence, text):
    """Share of evidence spans found word for word in the post (None if no spans)."""
    if not evidence:
        return None
    post = _norm(text)
    return sum(_norm(e) in post for e in evidence) / len(evidence)


def evidence_coverage(evidence, text):
    """Fraction of the post's tokens covered by the evidence (to catch copy-the-post)."""
    post = _tokens(text)
    if not post:
        return 0.0
    return min(1.0, sum(len(_tokens(e)) for e in evidence) / len(post))


def rationale_f1(evidence, rationale):
    """Token F1 between evidence spans and human rationale spans (HateXplain)."""
    if not rationale:
        return None
    pred = set(t for e in evidence for t in _tokens(e))
    gold = set(t for r in rationale for t in _tokens(r))
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def target_f1(pred, gold):
    if gold is None:
        return None
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def severity_distance(pred, lo, hi):
    """0 inside the gold range, otherwise levels away from it."""
    if lo is None:
        return None
    return max(lo - pred, 0, pred - hi)


def facet_scores(analysis, record):
    """All facet scores of one parsed analysis against one corpus record."""
    a = analysis
    return {
        "label_ok": a["label"] in record["label_set"],
        "consistent": a["label"] == tx.decision_rule(a["stance"], a["targets"], a["severity"]),
        "target_f1": target_f1(a["targets"], record["targets"]),
        "severity_dist": severity_distance(a["severity"], record["severity_lo"], record["severity_hi"]),
        "stance_ok": None if record["stance"] is None else a["stance"] == record["stance"],
        "evidence_verbatim": evidence_verbatim(a["evidence"], record["text"]),
        "evidence_coverage": evidence_coverage(a["evidence"], record["text"]),
        "rationale_f1": rationale_f1(a["evidence"], record.get("rationale")),
        "n_words": len(a["explanation"].split()),
        "n_post_tokens": len(_tokens(record["text"])),
        "n_evidence": len(a["evidence"]),
    }


ACCEPT = {
    "min_target_f1": 0.5,
    "max_severity_dist": 1,          # within one level of the human range
    "max_words": 150,
    "min_words": 80,
    "max_evidence": 3,
    "max_evidence_coverage": 0.8,
    "coverage_min_post_tokens": 10,   # short posts may legitimately be all evidence
}


def accept(scores, analysis, cfg=ACCEPT, require_label=True):
    """Checks a teacher trace must pass. Returns (accepted, list of failed checks)."""
    failed = []
    if require_label and not scores["label_ok"]:
        failed.append("label")
    if not scores["consistent"]:
        failed.append("consistency")
    if scores["target_f1"] is not None and scores["target_f1"] < cfg["min_target_f1"]:
        failed.append("targets")
    if scores["severity_dist"] is not None and scores["severity_dist"] > cfg["max_severity_dist"]:
        failed.append("severity")
    if scores["stance_ok"] is False:
        failed.append("stance")
    if analysis["label"] != "normal" and not analysis["evidence"]:
        failed.append("no_evidence")
    if scores["evidence_verbatim"] is not None and scores["evidence_verbatim"] < 1.0:
        failed.append("evidence_not_verbatim")
    if (scores["n_evidence"] > cfg["max_evidence"]
            or (scores["n_post_tokens"] >= cfg["coverage_min_post_tokens"]
                and scores["evidence_coverage"] > cfg["max_evidence_coverage"])):
        failed.append("evidence_too_long")
    if not cfg["min_words"] <= scores["n_words"] <= cfg["max_words"]:
        failed.append("explanation_length")
    return not failed, failed


def quality(scores):
    """Ranks candidates of the same post (higher is better); used to pick one trace."""
    def val(x, default):
        return default if x is None else float(x)
    return (2 * scores["label_ok"] + scores["consistent"]
            + val(scores["target_f1"], 1) + val(scores["stance_ok"], 1)
            - 0.5 * val(scores["severity_dist"], 0)
            + val(scores["evidence_verbatim"], 1) + val(scores["rationale_f1"], 0))

