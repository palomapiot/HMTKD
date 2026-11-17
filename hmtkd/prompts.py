"""Prompts and output schema shared by the teachers and the students.

The rationale is generated field by field in FIELDS order (evidence -> targets ->
stance -> severity -> explanation -> label), so the label comes last and can be
checked against the decision rule applied to the facets before it.
"""

from typing import List, Literal

from pydantic import BaseModel

from . import taxonomy as tx

SYSTEM = (
    "You are an expert content-moderation analyst. You analyse social media posts for "
    "hate speech and offensive language, and you justify every decision with evidence "
    "taken from the post."
)

GUIDELINES = """Classify the post as hate, offensive or normal.

Definitions
- hate: the author attacks or demeans people because of a protected characteristic (race or ethnicity, religion, national origin or immigration status, gender, gender identity, sexual orientation, age, disability), with at least derogatory severity.
- offensive: insulting, profane, demeaning or threatening language that is not hate speech: it targets an individual or a non-protected group (e.g. a political ideology or a profession), or it is too mild.
- normal: neither.

Fill in these fields, in this order:
1. evidence: up to 3 short spans copied word for word from the post that support your decision. Use [] only if nothing in the post is abusive.
2. targets: the groups the post is directed at or about, chosen from: race_ethnicity, religion, national_origin, gender, gender_identity, sexual_orientation, age, disability, politics (political ideology; not protected), other (individuals, professions and other non-protected groups). Use [] if the post is not about any person or group.
3. stance: "attack" if the author attacks or demeans the target; "support_or_counter" if the author defends the target or counters hate (quoting or reporting hate in order to criticise it counts here); "neutral" otherwise.
4. severity of the author's own attack: 0 none; 1 offensive (insult, profanity, disrespect); 2 derogatory (demeaning, stereotyping, claims of inferiority, slurs); 3 dehumanizing or demonizing (comparing people to animals, vermin, disease or monsters, or denying their humanity); 4 violent (threats, incitement, glorification of violence, genocide).
5. explanation: 4 to 6 complete sentences (100 to 150 words) that justify the decision by referring to the evidence, target, stance and severity. Do not repeat the post.
6. label: "hate" if stance is attack, a protected group is targeted and severity is 2 or more; otherwise "offensive" if severity is 1 or more; otherwise "normal".

Notes: slurs reclaimed by the group itself, hateful content quoted in order to criticise it, and negated statements ("I don't hate X") are not attacks. Coded or implicit attacks (dog whistles, stereotypes phrased as questions or jokes) are attacks."""

TEACHER_USER = """{guidelines}

{hint}Post:
\"\"\"{text}\"\"\"

Answer with the JSON object only."""

HINT = ("Human annotators judged this post as: {labels}. Your analysis must be consistent "
        "with that judgement.\n\n")


class Analysis(BaseModel):
    """Field order = generation order under guided decoding."""
    evidence: List[str]
    targets: List[Literal[tuple(tx.TARGET_GROUPS)]]
    stance: Literal[tuple(tx.STANCES)]
    severity: Literal[tuple(tx.SEVERITY_LEVELS)]
    explanation: str
    label: Literal[tuple(tx.LABELS)]


def teacher_messages(text, label_set=None, system_extra=""):
    """Chat messages for the teacher; `label_set` adds the gold hint (hinted round)."""
    hint = ""
    if label_set:
        hint = HINT.format(labels=" or ".join(label_set))
    return [
        {"role": "system", "content": f"{SYSTEM}\n{system_extra}".strip()},
        {"role": "user", "content": TEACHER_USER.format(
            guidelines=GUIDELINES, hint=hint, text=text)},
    ]



# --------------------------------------------------------------------------
# Label-only prompt (baselines without a rationale: zero-shot label only, label-only SFT)
# --------------------------------------------------------------------------

DEFINITIONS = GUIDELINES.split("\n\nFill in these fields")[0]

LABEL_USER = """{definitions}

Post:
\"\"\"{text}\"\"\"

Answer with the JSON object {{"label": "hate" | "offensive" | "normal"}} only."""


class LabelOnly(BaseModel):
    label: Literal[tuple(tx.LABELS)]


def label_messages(text, system_extra=""):
    """Chat messages asking for the label only, with the same definitions as the full prompt."""
    return [
        {"role": "system", "content": f"{SYSTEM}\n{system_extra}".strip()},
        {"role": "user", "content": LABEL_USER.format(definitions=DEFINITIONS, text=text)},
    ]
