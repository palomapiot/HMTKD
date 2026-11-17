"""Choose the GRPO checkpoint on the MHS development split.

For one GRPO run, reads the metrics of every checkpoint on the development split
(results/metrics-dev/<run>-step<N>-analysis.json, from 03_zero_shot.py and 06_evaluate.py) and
selects the step with the highest hate-vs-not macro F1 (ties: the earlier step).
Step 0 (the SFT warm-up the run starts from) is printed for reference but cannot be selected.

Writes results/models/<run>/selected.json and the symlink results/models/<run>/selected -> the
chosen checkpoint. No test set is read.

Stdlib only:  python3 scripts/05b_select_checkpoint.py grpo-llama8b-s3407
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hate_f1(path):
    return json.loads(path.read_text())["datasets"]["mhs"]["classification"]["hate_vs_rest"]["macro_f1"]


def main():
    run = sys.argv[1]
    mdir, model = ROOT / "results" / "metrics-dev", ROOT / "results" / "models" / run
    rows = []
    for p in sorted(mdir.glob(f"{run}-step*-analysis.json")):
        m = re.fullmatch(rf"{re.escape(run)}-step(\d+)-analysis\.json", p.name)
        if m:
            rows.append({"step": int(m.group(1)), "mhs_dev": hate_f1(p)})
    rows.sort(key=lambda r: r["step"])
    cands = [r for r in rows if r["step"] > 0]
    if not cands:
        sys.exit(f"❌ no checkpoint of {run} evaluated on the development split in {mdir}")
    best = max(cands, key=lambda r: (round(r["mhs_dev"], 4), -r["step"]))
    for r in rows:
        mark = "  <- selected" if r is best else ("  (SFT warm-up)" if r["step"] == 0 else "")
        print(f"step {r['step']:5d}: MHS dev hate-vs-not macro F1 {r['mhs_dev']:.3f}{mark}")
    target = model / "checkpoints" / f"checkpoint-{best['step']}"
    if not target.exists():
        sys.exit(f"❌ {target} not found")
    link = model / "selected"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.relative_to(model))
    (model / "selected.json").write_text(json.dumps({
        "selected_step": best["step"], "criterion": "hate-vs-not macro F1 on MHS dev", "checkpoints": rows},
        indent=2))
    print(f"✅ {run}: step {best['step']} -> {link}")


if __name__ == "__main__":
    main()
