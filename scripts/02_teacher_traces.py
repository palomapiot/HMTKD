"""Stage I: teacher rationales for the SFT warm-up.

The teacher never sees the gold label in the blind rounds. Every sample is
parsed and checked against the human annotation (hmtkd/scoring.py:accept); the
same checks become GRPO rewards in Stage III.

Rounds (one model load, resumable; choose with --rounds):
    blind        k samples per post, no label
    blind_extra  k_extra more samples for posts with no accepted trace yet
    hinted       k_hint samples with the gold label set in the prompt, only for
                 posts still unresolved; flagged so SFT can include or exclude them

Multi-teacher flow:
    1. every teacher runs the blind rounds (default --rounds blind blind_extra)
    2. scripts/02b_merge_teachers.py pools them and writes the posts no teacher solved
    3. one teacher runs --rounds hinted --only_uids <that list>, then merge again

Inputs
    data/processed/{train,dev}.jsonl            posts with texts (see README)
Outputs (data/traces/)
    pool_{train,dev}.jsonl                      sampled posts, shared by all teachers
    {name}/candidates_{round}.jsonl             every sample with its scores
    {name}/traces.jsonl                         one selected trace per post + status
    {name}/summary.json                         acceptance by dataset / label set / difficulty / round

Usage
    python3 scripts/02_teacher_traces.py --teacher meta-llama/Llama-3.3-70B-Instruct --name llama70b \
        --quantization fp8 --pool_name pool-mhs
    python3 scripts/02_teacher_traces.py --teacher Qwen/Qwen3.8-27B --name qwen38 --no_thinking --text_only \
        --pool_name pool-mhs
    add --limit 20 for a smoke test
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmtkd import prompts, scoring  # noqa: E402
from hmtkd.io import difficulty, read_jsonl, write_jsonl  # noqa: E402


def parse_arguments():
    p = argparse.ArgumentParser(description="Generate and filter teacher rationales.")
    p.add_argument("--teacher", required=True, help="HF model id")
    p.add_argument("--name", required=True, help="short teacher name for the output folder")
    p.add_argument("--data_dir", type=Path, default=ROOT / "data")
    p.add_argument("--datasets", nargs="+", default=["mhs"], help="datasets the pool is drawn from")
    p.add_argument("--pool_name", default="pool",
                   help="pool file prefix under data/traces/; a new pool for a new dataset choice")
    p.add_argument("--n_train_per_dataset", type=int, default=15000)
    p.add_argument("--n_dev_per_dataset", type=int, default=1500)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--k_extra", type=int, default=12)
    p.add_argument("--k_hint", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size (GPUs)")
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--gpu_mem", type=float, default=0.90)
    p.add_argument("--quantization", default=None, help="e.g. fp8")
    p.add_argument("--text_only", action="store_true",
                   help="disable the image/audio/video encoders of multimodal models (skips their warm-up)")
    p.add_argument("--max_num_seqs", type=int, default=None,
                   help="cap on concurrent sequences (hybrid Mamba models have a fixed cache)")
    p.add_argument("--enforce_eager", action="store_true", help="skip torch.compile / CUDA graphs")
    p.add_argument("--backend", default="mp", choices=["mp", "ray"],
                   help="multi-GPU executor; mp avoids attaching to another job's Ray cluster")
    p.add_argument("--no_thinking", action="store_true", help="disable thinking via the chat template")
    p.add_argument("--system_extra", default="", help="text appended to the system prompt, e.g. /no_think")
    p.add_argument("--chunk", type=int, default=1000, help="posts per generation call")
    p.add_argument("--seed", type=int, default=3407, help="generation seed")
    p.add_argument("--pool_seed", type=int, default=3407,
                   help="seed of the pool and --limit subset; keep it fixed across teachers")
    p.add_argument("--limit", type=int, default=None, help="posts per split, for smoke tests")
    p.add_argument("--rounds", nargs="+", default=["blind", "blind_extra"],
                   choices=["blind", "blind_extra", "hinted"])
    p.add_argument("--only_uids", type=Path, default=None,
                   help="restrict the pool to these uids (one per line), e.g. the posts no teacher solved")
    return p.parse_args()


# --------------------------------------------------------------------------
# Pool: a stratified sample of the train/dev posts, shared by every teacher
# --------------------------------------------------------------------------

def sample_pool(records, datasets, n_per_dataset, seed):
    """Per dataset, sample n posts keeping the label-set proportions."""
    rng = random.Random(seed)
    pool = []
    for ds in datasets:
        rows = [r for r in records if r["dataset"] == ds]
        if len(rows) <= n_per_dataset:
            pool += rows
            continue
        strata = defaultdict(list)
        for r in rows:
            strata["|".join(r["label_set"])].append(r)
        for key in sorted(strata):
            group = strata[key]
            rng.shuffle(group)
            pool += group[:round(n_per_dataset * len(group) / len(rows))]
    return pool


def load_pool(args, split, n_per_dataset):
    path = args.data_dir / "traces" / f"{args.pool_name}_{split}.jsonl"
    if path.exists():
        # same posts, but gold facets refreshed from the current corpus
        current = {r["uid"]: r for r in read_jsonl(args.data_dir / "processed" / f"{split}.jsonl")}
        pool = [current[r["uid"]] for r in read_jsonl(path) if r["uid"] in current]
        missing = sum(1 for _ in open(path)) - len(pool)
        if missing:
            print(f"⚠️  {missing} pool posts no longer in the {split} corpus; skipped.", flush=True)
    else:
        source = args.data_dir / "processed" / f"{split}.jsonl"
        if not source.exists():
            sys.exit(f"❌ {source} not found: add the post texts to the splits first (README).")
        pool = sample_pool(read_jsonl(source), args.datasets, n_per_dataset, args.pool_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, pool)
        print(f"📦 Sampled {len(pool)} {split} posts -> {path}", flush=True)
    if args.limit:   # smoke test: a small random subset across datasets
        pool = random.Random(args.pool_seed).sample(pool, min(args.limit, len(pool)))
    return pool


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def build_llm(args):
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import StructuredOutputsParams
        structured = {"structured_outputs": StructuredOutputsParams(json=prompts.Analysis.model_json_schema())}
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams
        structured = {"guided_decoding": GuidedDecodingParams(json=prompts.Analysis.model_json_schema())}

    extra = {}
    if args.text_only:
        extra["limit_mm_per_prompt"] = {"image": 0, "video": 0, "audio": 0}
    if args.max_num_seqs:
        extra["max_num_seqs"] = args.max_num_seqs
    if args.enforce_eager:
        extra["enforce_eager"] = True
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, quantization=args.quantization,
              seed=args.seed, trust_remote_code=True,
              distributed_executor_backend=args.backend if args.tp > 1 else None, **extra)

    def sampling(n):
        return SamplingParams(n=n, temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_tokens, **structured)
    return llm, sampling


def run_round(llm, sampling, args, out_dir, name, posts, n, hinted):
    """Generate n samples for each post, score them, append to candidates_{name}.jsonl."""
    path = out_dir / f"candidates_{name}.jsonl"
    done = {json.loads(line)["uid"] for line in open(path)} if path.exists() else set()
    todo = [p for p in posts if p["uid"] not in done]
    print(f"\n▶️  round {name}: {len(posts)} posts, {len(done & {p['uid'] for p in posts})} done, "
          f"{len(todo)} to generate, {n} samples each", flush=True)
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else {}

    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        messages = [prompts.teacher_messages(p["text"], p["label_set"] if hinted else None,
                                             args.system_extra)
                    for p in chunk]
        t0 = time.time()
        outputs = llm.chat(messages, sampling_params=sampling(n), use_tqdm=True, **kwargs)
        rows = []
        for post, out in zip(chunk, outputs):
            for j, sample in enumerate(out.outputs):
                analysis = scoring.parse_analysis(sample.text)
                row = {"uid": post["uid"], "round": name, "sample": j, "raw": sample.text,
                       "analysis": analysis, "accepted": False, "failed": ["parse"], "scores": None}
                if analysis:
                    row["scores"] = scoring.facet_scores(analysis, post)
                    row["accepted"], row["failed"] = scoring.accept(row["scores"], analysis)
                rows.append(row)
        write_jsonl(path, rows, mode="a")
        n_acc = len({r["uid"] for r in rows if r["accepted"]})
        print(f"   {start + len(chunk)}/{len(todo)} posts, {n_acc}/{len(chunk)} with an accepted "
              f"sample, {time.time() - t0:.0f}s", flush=True)
        if start == 0:
            print(f"   example output:\n{rows[0]['raw']}", flush=True)
    return [json.loads(line) for line in open(path)] if path.exists() else []


# --------------------------------------------------------------------------
# Selection and report
# --------------------------------------------------------------------------

def select(posts, candidates):
    """One trace per post: best blind accepted > best hinted accepted > best hinted (unresolved)."""
    by_uid = defaultdict(list)
    for c in candidates:
        if c["analysis"]:
            by_uid[c["uid"]].append(c)
    traces = []
    for post in posts:
        cands = by_uid.get(post["uid"], [])
        blind = [c for c in cands if c["accepted"] and c["round"] != "hinted"]
        hinted_ok = [c for c in cands if c["accepted"] and c["round"] == "hinted"]
        hinted_all = [c for c in cands if c["round"] == "hinted"]
        for status, pool in (("blind", blind), ("hinted", hinted_ok), ("unresolved", hinted_all)):
            if pool:
                best = max(pool, key=lambda c: scoring.quality(c["scores"]))
                break
        else:
            status, best = "failed", None
        traces.append({
            "uid": post["uid"], "dataset": post["dataset"], "split": post["split"],
            "text": post["text"], "label_set": post["label_set"],
            "difficulty": difficulty(post), "status": status,
            "hinted": status in ("hinted", "unresolved"),
            "analysis": best["analysis"] if best else None,
            "scores": best["scores"] if best else None,
            "round": best["round"] if best else None,
            "n_accepted_blind": len(blind),
        })
    return traces


def rate_table(traces, key):
    table = defaultdict(Counter)
    for t in traces:
        table[key(t)][t["status"]] += 1
    return {k: {**dict(v), "n": sum(v.values()),
                "blind_rate": round(v["blind"] / sum(v.values()), 4)}
            for k, v in sorted(table.items())}


def summarise(args, traces, candidates):
    blind_first = [c for c in candidates if c["round"] == "blind"]
    return {
        "teacher": args.teacher, "config": {k: str(v) for k, v in vars(args).items()},
        "status": dict(Counter(t["status"] for t in traces)),
        "by_dataset": rate_table(traces, lambda t: t["dataset"]),
        "by_label_set": rate_table(traces, lambda t: "|".join(t["label_set"])),
        "by_difficulty": rate_table(traces, lambda t: t["difficulty"]),
        "blind_first_round": {
            "samples": len(blind_first),
            "accepted": sum(c["accepted"] for c in blind_first),
            "failed_checks": dict(Counter(f for c in blind_first for f in c["failed"]).most_common()),
        },
        "rounds": dict(Counter(c["round"] for c in candidates)),
    }


def main():
    args = parse_arguments()
    out_dir = args.data_dir / "traces" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    posts = (load_pool(args, "train", args.n_train_per_dataset)
             + load_pool(args, "dev", args.n_dev_per_dataset))
    if args.only_uids:
        keep = set(args.only_uids.read_text().split())
        posts = [p for p in posts if p["uid"] in keep]
    print(f"🔍 {len(posts)} posts: {dict(Counter((p['dataset'], p['split']) for p in posts))}", flush=True)

    llm, sampling = build_llm(args)

    candidates = []
    for name, n, hinted in (("blind", args.k, False), ("blind_extra", args.k_extra, False),
                            ("hinted", args.k_hint, True)):
        if name not in args.rounds:
            continue
        accepted = {c["uid"] for c in candidates if c["accepted"]}
        todo = posts if name == "blind" else [p for p in posts if p["uid"] not in accepted]
        candidates += run_round(llm, sampling, args, out_dir, name, todo, n, hinted)

    traces = select(posts, candidates)
    write_jsonl(out_dir / "traces.jsonl", traces)
    summary = summarise(args, traces, candidates)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + json.dumps({k: summary[k] for k in
                             ("status", "by_dataset", "by_difficulty", "blind_first_round")}, indent=2))
    print(f"\n✅ {len(traces)} traces -> {out_dir / 'traces.jsonl'}")


if __name__ == "__main__":
    main()
