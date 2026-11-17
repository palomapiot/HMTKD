"""Metrics for every prediction run, per test set.

Reads results/predictions/<run>/predictions.jsonl (from 03_zero_shot.py or any trained
model run through it), joins it with data/processed/test.jsonl and writes
results/metrics/<run>.json. A run that does not cover every test post is reported
as incomplete and skipped, unless --allow_partial.

Stdlib only:  python3 scripts/06_evaluate.py                 (every run)
              python3 scripts/06_evaluate.py llama8b-analysis  (some runs)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmtkd import metrics  # noqa: E402
from hmtkd.io import read_jsonl  # noqa: E402


def parse_arguments():
    p = argparse.ArgumentParser(description="Evaluate prediction runs.")
    p.add_argument("runs", nargs="*", help="run folders under results/predictions (default: all)")
    p.add_argument("--pred_dir", type=Path, default=ROOT / "results" / "predictions")
    p.add_argument("--out_dir", type=Path, default=ROOT / "results" / "metrics")
    p.add_argument("--split", default="test")
    p.add_argument("--allow_partial", action="store_true", help="evaluate runs on the posts they cover")
    return p.parse_args()


def main():
    args = parse_arguments()
    gold = {r["uid"]: r for r in read_jsonl(ROOT / "data" / "processed" / f"{args.split}.jsonl")}
    runs = args.runs or sorted(d.name for d in args.pred_dir.iterdir()
                               if (d / "predictions.jsonl").exists())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        path = args.pred_dir / run / "predictions.jsonl"
        if not path.exists():
            print(f"❌ {run}: {path} not found")
            continue
        preds = {p["uid"]: p for p in read_jsonl(path)}
        unknown = set(preds) - set(gold)
        missing = set(gold) - set(preds)
        if unknown:
            print(f"⚠️  {run}: {len(unknown)} predictions for posts not in the {args.split} split; ignored")
        if missing and not args.allow_partial:
            print(f"⏭️  {run}: incomplete, {len(missing)} of {len(gold)} posts missing "
                  f"({dict(Counter(gold[u]['dataset'] for u in missing))}); skipped")
            continue
        rows = [{"record": gold[u], "label": p["label"], "analysis": p.get("analysis")}
                for u, p in preds.items() if u in gold]
        result = {"run": run, "n_posts": len(rows), "complete": not missing,
                  "datasets": metrics.evaluate(rows)}
        (args.out_dir / f"{run}.json").write_text(json.dumps(result, indent=2))

        print(f"\n{run}" + ("" if not missing else f"  (partial: {len(rows)} posts)"))
        print(f"  {'dataset':12s}{'set acc':>9}{'F1-3way':>9}{'F1 hate':>9}{'consist':>9}")
        for ds, m in result["datasets"].items():
            c, a = m["classification"], m["analyses"] or {}
            def fmt(x):
                return f"{x:.3f}" if isinstance(x, float) else "  —  "
            print(f"  {ds:12s}{fmt(c['set_accuracy']):>9}{fmt(c['macro_f1_3way']):>9}"
                  f"{fmt(c['hate_vs_rest']['macro_f1']):>9}"
                  f"{fmt(a.get('consistency_rate')):>9}")
        print(f"  -> {args.out_dir / (run + '.json')}")


if __name__ == "__main__":
    main()
