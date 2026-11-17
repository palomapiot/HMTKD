"""Stage III: GRPO of a student (TRL GRPOTrainer, LoRA, vLLM generation).

The student starts from its Stage II adapter (--init results/models/sft-<s>-warmup-s<seed>) and
is optimised with the structured-rationale prompt and the rewards of hmtkd/rewards.py.

Posts: the whole Stage I pool (train split, 15,000 MHS posts), including those without an
accepted rationale: the rewards need only the human annotation. Posts that no teacher solved
blind are repeated --oversample_hard times, so the hard posts are not under-represented.
The similarity reward uses the post's accepted teacher explanations; posts without any get
no similarity term (None), as in Equation 3.

The SFT adapter keeps training (no second adapter). With --beta > 0, TRL keeps a frozen copy of
the --init adapter and computes the KL penalty against it. The adapter is saved to
results/models/<run>/ and is evaluated by scripts/03_zero_shot.py --lora like every other student.

Reward ablation: --no_rewards drops rewards, e.g. every reward but the target reward:
    --no_rewards lbl cons sev stn evd sim fmt

Usage
    python3 scripts/05_grpo.py --model meta-llama/Llama-3.1-8B-Instruct --name llama8b \
        --init results/models/sft-llama8b-warmup-s3407 --keep_checkpoints
    add --limit 64 --max_steps 3 for a smoke test
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmtkd import prompts, rewards  # noqa: E402
from hmtkd.io import read_jsonl  # noqa: E402


def parse_arguments():
    p = argparse.ArgumentParser(description="GRPO of a student with the rationale rewards.")
    p.add_argument("--model", "--teacher", dest="model", required=True, help="HF id of the base student")
    p.add_argument("--name", required=True, help="short student name, e.g. llama8b")
    p.add_argument("--init", required=True, help="Stage II adapter to start from")
    p.add_argument("--data_dir", type=Path, default=ROOT / "data")
    p.add_argument("--pool_name", default="pool-mhs")
    p.add_argument("--traces", default="pooled-len150", help="folder under data/traces/ (similarity references)")
    p.add_argument("--out_dir", type=Path, default=ROOT / "results" / "models")
    p.add_argument("--run_name", default=None, help="default: grpo-<name>[-no<reward>...]-s<seed>")
    p.add_argument("--oversample_hard", type=int, default=2)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--resume", action="store_true",
                   help="continue the run from its last checkpoint (optimizer, data order and step included)")
    p.add_argument("--keep_checkpoints", action="store_true",
                   help="keep every checkpoint (to check them on the development split)")
    p.add_argument("--no_rewards", nargs="*", default=[], choices=sorted(rewards.WEIGHTS),
                   help="ablation: drop these rewards")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16, help="completions per device step")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient to the --init adapter")
    p.add_argument("--vllm_gpu_memory", type=float, default=0.35)
    p.add_argument("--no_thinking", action="store_true")
    p.add_argument("--system_extra", default="")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--limit", type=int, default=None, help="posts, for smoke tests")
    return p.parse_args()


def load_posts(args):
    """Pool posts (with gold) and, per uid, the accepted teacher explanations."""
    pool = read_jsonl(args.data_dir / "traces" / f"{args.pool_name}_train.jsonl")
    refs, solved_blind = {}, set()
    path = args.data_dir / "traces" / args.traces / "traces.jsonl"
    if not path.exists():
        sys.exit(f"❌ {path} not found: the similarity reward needs the Stage I rationales.")
    for t in read_jsonl(path):
        if t.get("status") in ("blind", "hinted"):
            refs[t["uid"]] = [tr["analysis"]["explanation"] for tr in t["traces"] if tr.get("analysis")]
        if t.get("status") == "blind":
            solved_blind.add(t["uid"])
    return pool, refs, solved_blind


def main():
    args = parse_arguments()
    abl = "".join(f"-no{r}" for r in args.no_rewards)
    run = args.run_name or f"grpo-{args.name}{abl}-s{args.seed}"
    out = args.out_dir / run
    rng = random.Random(args.seed)

    pool, refs, solved_blind = load_posts(args)
    priors = rewards.label_priors(pool)   # from the whole pool, before --limit
    print(f"⚖️  label reward class shares {priors}", flush=True)
    if args.limit:
        pool = rng.sample(pool, min(args.limit, len(pool)))
    gold = {p["uid"]: p for p in pool}
    rows = []
    for p in pool:
        rows += [p] * (1 if p["uid"] in solved_blind else args.oversample_hard)
    rng.shuffle(rows)
    print(f"🔍 {run}: {len(pool)} posts, {len(rows)} prompts after oversampling the "
          f"{len(pool) - len(solved_blind & set(gold))} posts no teacher solved blind; "
          f"{sum(u in refs for u in gold)} with teacher explanations", flush=True)

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer
    from hmtkd import training

    # Gemma 4: TRL copies every parameter into vLLM after each update, including the frozen
    # vision/audio towers, and vLLM names one of them differently -> ValueError. Those towers
    # never change (no LoRA there), so they are left out of the copy.
    from trl.generation import vllm_generation
    _iter = vllm_generation.VLLMGeneration._iter_named_params
    _frozen = re.compile(training.LORA_EXCLUDE)

    def _iter_language_params(self):
        for name, param in _iter(self):
            if not _frozen.match(name):
                yield name, param
    vllm_generation.VLLMGeneration._iter_named_params = _iter_language_params

    tokenizer = training.load_tokenizer(args.model)
    model = training.load_student(args.model)
    # with beta > 0, GRPOTrainer itself copies this adapter into a frozen adapter named "ref" and
    # computes the KL against it, so the KL anchors the run to the warm-up, not to the base model
    model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    print(f"🔗 continuing the adapter {args.init}; KL (beta {args.beta}) to a frozen copy of it", flush=True)

    kw = training.chat_kwargs(args.no_thinking)

    def render(post):
        msgs = prompts.teacher_messages(post["text"], None, args.system_extra)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]     # the tokenizer adds it again
        return text

    data = Dataset.from_list([{"prompt": render(p), "uid": p["uid"]} for p in rows])

    # similarity reward: cosine between sentence embeddings (cached per text)
    embed = None
    if "sim" not in args.no_rewards:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.embedder, device="cuda")
        cache = {}

        def vec(t):
            # lone surrogates (e.g. a half emoji escape "\ud83d" in the policy's JSON) pass json.loads
            # but the tokenizer rejects them
            t = t.encode("utf-8", "replace").decode("utf-8")
            if t not in cache:
                cache[t] = encoder.encode(t, convert_to_tensor=True, normalize_embeddings=True)
            return cache[t]

        def embed(a, b):
            return float(torch.dot(vec(a), vec(b)))

    names = [k for k in rewards.WEIGHTS if k not in args.no_rewards]
    memo = {}

    def components(text, uid):
        key = (uid, text)
        if key not in memo:
            if len(memo) > 50000:
                memo.clear()
            memo[key] = rewards.component_rewards(text, gold[uid], priors, refs.get(uid), embed)
        return memo[key]

    def make_reward(name):
        def fn(prompts, completions, uid, **_):
            return [components(c if isinstance(c, str) else c[0]["content"], u)[name]
                    for c, u in zip(completions, uid)]
        fn.__name__ = f"r_{name}"
        return fn

    config = GRPOConfig(
        output_dir=str(out / "checkpoints"), max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup", warmup_steps=0.03,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations, max_completion_length=args.max_completion_length,
        temperature=args.temperature, beta=args.beta, loss_type="grpo", scale_rewards="group",
        mask_truncated_completions=False,
        reward_weights=[rewards.WEIGHTS[n] for n in names],
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_gpu_memory,
        # prompt (~1k tokens) + answer (<= 512): without a cap vLLM reserves the model's full context
        # (Mistral NeMo: 131k tokens), which does not fit in its memory share
        vllm_max_model_length=3072,
        bf16=True, gradient_checkpointing=True, optim="adamw_bnb_8bit",
        logging_steps=5, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=None if args.keep_checkpoints else 1,
        save_only_model=False,   # optimizer and trainer state too, so --resume can continue
        seed=args.seed, report_to="none")
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=config, train_dataset=data,
                          reward_funcs=[make_reward(n) for n in names])
    trainer.model.print_trainable_parameters()
    stats = trainer.train(resume_from_checkpoint=True if args.resume else None)

    trainer.model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    history = [h for h in trainer.state.log_history if "reward" in h or any(k.startswith("rewards/") for k in h)]
    (out / "run.json").write_text(json.dumps({
        "run": run, "base_model": args.model, "init": args.init,
        "rewards": names, "beta": args.beta, "label_priors": priors,
        "n_posts": len(pool), "n_prompts": len(rows),
        "args": {k: str(v) for k, v in vars(args).items()}, "metrics": stats.metrics,
        "log": history}, indent=2))
    print(f"✅ adapter -> {out}", flush=True)


if __name__ == "__main__":
    main()
