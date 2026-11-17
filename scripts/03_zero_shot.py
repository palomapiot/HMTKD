"""Predictions on a split: zero-shot students and teachers, or a trained student with --lora.

One greedy output per post, under the same JSON schema as the teachers:
    --mode analysis   the structured rationale (evidence ... label), as HMTKD produces it
    --mode label      the label only ({"label": ...}), without a rationale

Inputs   data/processed/<split>.jsonl                   posts with texts (see README)
Outputs  results/predictions/<name>-<mode>/predictions.jsonl   resumable
         (with --lora: a trained student, e.g. --name sft-llama8b-label-s3407)
         results/predictions/<name>-<mode>/config.json

Usage
    python3 scripts/03_zero_shot.py --model meta-llama/Llama-3.1-8B-Instruct --name llama8b --mode analysis
    add --limit 50 for a smoke test (50 posts per test set)
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmtkd import prompts, scoring  # noqa: E402
from hmtkd.io import read_jsonl, write_jsonl  # noqa: E402


def parse_arguments():
    p = argparse.ArgumentParser(description="Predictions on a split.")
    p.add_argument("--model", "--teacher", dest="model", required=True, help="HF model id or local path")
    p.add_argument("--name", required=True, help="short model name for the output folder")
    p.add_argument("--mode", choices=["analysis", "label"], default="analysis")
    p.add_argument("--data_dir", type=Path, default=ROOT / "data")
    p.add_argument("--out_dir", type=Path, default=ROOT / "results" / "predictions")
    p.add_argument("--split", default="test")
    p.add_argument("--datasets", nargs="+", default=None, help="default: every dataset in the split")
    p.add_argument("--limit", type=int, default=None, help="posts per dataset, for smoke tests")
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--gpu_mem", type=float, default=0.90)
    p.add_argument("--quantization", default=None)
    p.add_argument("--text_only", action="store_true")
    p.add_argument("--max_num_seqs", type=int, default=None)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--backend", default="mp", choices=["mp", "ray"])
    p.add_argument("--no_thinking", action="store_true")
    p.add_argument("--system_extra", default="")
    p.add_argument("--lora", type=Path, default=None,
                   help="LoRA adapter of a trained student (results/models/<run>), applied to --model")
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def load_posts(args):
    path = args.data_dir / "processed" / f"{args.split}.jsonl"
    if not path.exists():
        sys.exit(f"❌ {path} not found: add the post texts to the splits first (README).")
    posts = [r for r in read_jsonl(path) if not args.datasets or r["dataset"] in args.datasets]
    if args.limit:
        by_ds = {}
        for r in posts:
            by_ds.setdefault(r["dataset"], []).append(r)
        posts = [r for ds in sorted(by_ds)
                 for r in random.Random(args.seed).sample(by_ds[ds], min(args.limit, len(by_ds[ds])))]
    return posts


def build_llm(args, schema):
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        structured = {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        structured = {"guided_decoding": GuidedDecodingParams(json=schema)}
    extra = {}
    if args.text_only:
        extra["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
    if args.max_num_seqs:
        extra["max_num_seqs"] = args.max_num_seqs
    if args.enforce_eager:
        extra["enforce_eager"] = True
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, quantization=args.quantization, seed=args.seed,
              trust_remote_code=True, enable_prefix_caching=True,
              **({"enable_lora": True, "max_lora_rank": args.max_lora_rank} if args.lora else {}),
              distributed_executor_backend=args.backend if args.tp > 1 else None, **extra)
    return llm, SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens, **structured)


def main():
    args = parse_arguments()
    out_dir = args.out_dir / f"{args.name}-{args.mode}"
    # resume only predictions of the same adapter: if the run name now points to another adapter,
    # its old predictions are moved aside, not reused
    old = out_dir / "config.json"
    if old.exists() and json.loads(old.read_text()).get("lora") != str(args.lora):
        prev = json.loads(old.read_text()).get("lora")
        stale = out_dir.with_name(f"{out_dir.name}-stale-{time.strftime('%Y%m%d-%H%M%S')}")
        out_dir.rename(stale)
        print(f"⚠️  {out_dir.name} was made with the adapter {prev}, now {args.lora}: moved to {stale.name}",
              flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    path = out_dir / "predictions.jsonl"

    posts = load_posts(args)
    done = {json.loads(line)["uid"] for line in open(path)} if path.exists() else set()
    todo = [p for p in posts if p["uid"] not in done]
    print(f"🔍 {len(posts)} posts ({dict(Counter(p['dataset'] for p in posts))}), "
          f"{len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        print(f"✅ nothing to do: {path}")
        return

    schema = (prompts.Analysis if args.mode == "analysis" else prompts.LabelOnly).model_json_schema()
    llm, greedy = build_llm(args, schema)
    lora = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest(args.name, 1, str(args.lora))
    chat_kwargs = {"enable_thinking": False} if args.no_thinking else {}

    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        if args.mode == "analysis":
            messages = [prompts.teacher_messages(p["text"], None, args.system_extra) for p in chunk]
        else:
            messages = [prompts.label_messages(p["text"], args.system_extra) for p in chunk]
        t0 = time.time()
        outputs = llm.chat(messages, sampling_params=greedy, use_tqdm=True, lora_request=lora,
                           **({"chat_template_kwargs": chat_kwargs} if chat_kwargs else {}))
        raws = [o.outputs[0].text for o in outputs]
        rows = []
        for post, raw in zip(chunk, raws):
            analysis = scoring.parse_analysis(raw) if args.mode == "analysis" else None
            label = analysis["label"] if analysis else (scoring.parse_label(raw) if args.mode == "label" else None)
            rows.append({"uid": post["uid"], "dataset": post["dataset"], "raw": raw,
                         "analysis": analysis, "label": label})
        write_jsonl(path, rows, mode="a")
        ok = sum(r["label"] in p["label_set"] for r, p in zip(rows, chunk))
        print(f"   {start + len(chunk)}/{len(todo)} posts, set accuracy {ok / len(chunk):.3f}, "
              f"parsed {sum(r['label'] is not None for r in rows)}/{len(rows)}, {time.time() - t0:.0f}s",
              flush=True)
        if start == 0:
            print(f"   example output:\n{raws[0]}", flush=True)
    print(f"✅ {path}")


if __name__ == "__main__":
    main()
