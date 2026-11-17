"""Small helpers shared by the scripts (stdlib only)."""

import json


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows, mode="w"):
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def agreement(record):
    """Share of annotators agreeing with the gold decision, or None if the record cannot tell."""
    soft = record.get("soft") or {}
    if record["label"] in soft:
        return soft[record["label"]]
    if "hate" in soft and record["hate"] is not None:
        return soft["hate"] if record["hate"] else 1 - soft["hate"]
    if "abusive" in soft and record["abusive"] is not None:
        return soft["abusive"] if record["abusive"] else 1 - soft["abusive"]
    return None


def difficulty(record):
    """Annotator agreement bucket of a corpus record: high / medium / low / unknown."""
    agree = agreement(record)
    if agree is None:
        return "unknown"
    return "high" if agree >= 0.9 else "medium" if agree >= 0.67 else "low"
