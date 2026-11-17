"""Shared label space of HMTKD.

Every dataset is mapped into these fields so that the teacher prompt, the
GRPO rewards and the evaluation all speak the same vocabulary.

Task: 3-way classification into LABELS. Gold is a *set* of accepted labels,
because datasets draw the hate/offensive line in different places; where the
source cannot separate two classes, both are accepted.

Decision rule (the paper's rho):
    hate       <=>  stance == "attack" and a PROTECTED target and severity >= 2
    offensive  <=>  not hate and severity >= 1
    normal     <=>  severity == 0
"""

LABELS = ["normal", "offensive", "hate"]

TARGET_GROUPS = [
    "race_ethnicity",
    "religion",
    "national_origin",   # nationality, immigration / refugee status
    "gender",
    "gender_identity",   # trans, non-binary
    "sexual_orientation",
    "age",
    "disability",
    "politics",          # political ideology: NOT protected
    "other",             # individuals, occupations, class, ...: NOT protected
]
PROTECTED_TARGETS = [g for g in TARGET_GROUPS if g not in ("politics", "other")]

# Ordinal severity of the author's own attack, independent of who is attacked.
SEVERITY_LEVELS = {
    0: "none",         # no attack
    1: "offensive",    # insult, profanity, disrespect
    2: "derogatory",   # demeaning, stereotyping, inferiority claims, slurs
    3: "dehumanizing or demonizing",  # animals / vermin / disease / monsters, denying humanity
    4: "violent",      # threats, incitement, glorification of violence, genocide
}
HATE_MIN_SEVERITY = 2

STANCES = [
    "attack",              # the author attacks / demeans the target
    "support_or_counter",  # the author defends the target or counters hate
    "neutral",             # mentions a group without attacking or defending it
]


def decision_rule(stance, targets, severity) -> str:
    if (stance == "attack" and any(t in PROTECTED_TARGETS for t in targets)
            and severity >= HATE_MIN_SEVERITY):
        return "hate"
    return "offensive" if severity >= 1 else "normal"


def hate_gold(label_set):
    """Binary gold: 1 hate, 0 not hate, None when the set holds hate and another label."""
    s = set(label_set)
    return 1 if s == {"hate"} else 0 if "hate" not in s else None
