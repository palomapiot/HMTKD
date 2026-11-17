"""Supervised fine-tuning of a student with LoRA (PEFT + TRL).

    --target rationale   Stage II warm-up (--epochs 1) and the Rationale SFT baseline (--epochs 2):
                         the student learns to write the accepted teacher rationales
                         (evidence ... label) from data/traces/<traces>/traces.jsonl
    --target label       label-only SFT: the student learns to answer {"label": ...} from the
                         human label of each Stage I post, without a rationale

Both use the same posts (the Stage I pool, train split) and the same prompts as inference
(hmtkd/prompts.py), so the two conditions differ only in what the student writes.

Per-post weighting (Equation 1 of the paper): every post weighs 1 in total. A post with
several accepted rationales (or a label set with two labels) contributes one of them per
epoch, drawn at random, which gives each the expected weight 1/|A_x|.

The loss covers the answer only, not the prompt. The LoRA adapter is saved to
results/models/<run>/ and is loaded by scripts/03_zero_shot.py --lora for the test sets.

Usage
    python3 scripts/04_sft.py --model meta-llama/Llama-3.1-8B-Instruct --name llama8b --target rationale \
        --epochs 1 --run_name sft-llama8b-warmup-s3407
    python3 scripts/04_sft.py --model meta-llama/Llama-3.1-8B-Instruct --name llama8b --target label
    add --limit 64 --max_steps 5 for a smoke test
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmtkd import prompts, scoring  # noqa: E402
from hmtkd.io import read_jsonl  # noqa: E402


def parse_arguments():
    p = argparse.ArgumentParser(description="LoRA SFT of a student.")
    p.add_argument("--model", "--teacher", dest="model", required=True, help="HF id of the base student")
    p.add_argument("--name", required=True, help="short student name, e.g. llama8b")
    p.add_argument("--target", choices=["label", "rationale"], required=True)
    p.add_argument("--data_dir", type=Path, default=ROOT / "data")
    p.add_argument("--pool_name", default="pool-mhs")
    p.add_argument("--traces", default="pooled-len150", help="folder under data/traces/ (rationale target)")
    p.add_argument("--out_dir", type=Path, default=ROOT / "results" / "models")
    p.add_argument("--run_name", default=None, help="default: sft-<name>-<target>-s<seed>")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--no_thinking", action="store_true", help="render prompts with enable_thinking=False")
    p.add_argument("--system_extra", default="")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--limit", type=int, default=None, help="posts, for smoke tests")
    p.add_argument("--max_steps", type=int, default=-1, help="for smoke tests")
    p.add_argument("--traces_path", type=Path, default=None,
                   help="explicit traces.jsonl (overrides --traces), e.g. the smoke-test pool")
    return p.parse_args()


# --------------------------------------------------------------------------
# Training examples: (post, answer) pairs, one draw per post and epoch
# --------------------------------------------------------------------------

def answer_text(analysis):
    """The rationale as the student writes it: the JSON object, fields in generation order."""
    return json.dumps({f: analysis[f] for f in scoring.FIELDS}, ensure_ascii=False)


def candidates_per_post(args):
    """uid -> (post, [answer, ...]) for the chosen target."""
    if args.target == "label":
        posts = read_jsonl(args.data_dir / "traces" / f"{args.pool_name}_train.jsonl")
        return {p["uid"]: (p, [json.dumps({"label": lbl}) for lbl in p["label_set"]]) for p in posts}

    path = args.traces_path or args.data_dir / "traces" / args.traces / "traces.jsonl"
    if not path.exists():
        sys.exit(f"❌ {path} not found: run scripts/02b_merge_teachers.py first.")
    out = {}
    for t in read_jsonl(path):
        if t.get("split") != "train" or t.get("status") not in ("blind", "hinted"):
            continue        # "unresolved" posts keep their best failed rationale: never train on it
        traces = [tr for tr in t.get("traces", []) if tr.get("analysis")]
        if traces:
            out[t["uid"]] = (t, [answer_text(tr["analysis"]) for tr in traces])
    return out


def build_examples(args, per_post, rng):
    """List of (post, answer) for all epochs, shuffled."""
    examples = []
    for _ in range(args.epochs):
        for post, answers in per_post.values():
            examples.append((post, rng.choice(answers)))
    rng.shuffle(examples)
    return examples


def render(tokenizer, args, post, answer):
    """Prompt and completion strings exactly as the chat template writes them."""
    if args.target == "label":
        messages = prompts.label_messages(post["text"], args.system_extra)
    else:
        messages = prompts.teacher_messages(post["text"], None, args.system_extra)
    kw = {"enable_thinking": False} if args.no_thinking else {}
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    try:
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                             tokenize=False, add_generation_prompt=False, **kw)
        end = full[full.rfind(answer) + len(answer):]      # end-of-turn marker(s) of the template
    except Exception:   # mistral-common tokenizers refuse a conversation ending with the assistant
        end = tokenizer.eos_token
    if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
        prompt = prompt[len(tokenizer.bos_token):]          # the tokenizer adds it again
    return {"prompt": prompt, "completion": answer + end}


def main():
    args = parse_arguments()
    run = args.run_name or f"sft-{args.name}-{args.target}-s{args.seed}"
    out = args.out_dir / run
    rng = random.Random(args.seed)

    per_post = candidates_per_post(args)
    if args.limit:
        keep = rng.sample(sorted(per_post), min(args.limit, len(per_post)))
        per_post = {u: per_post[u] for u in keep}
    n_answers = Counter(len(a) for _, a in per_post.values())
    print(f"🔍 {run}: {len(per_post)} posts; answers per post {dict(sorted(n_answers.items()))}", flush=True)

    import torch  # GPU needed from here on
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template is None:          # multimodal checkpoints keep it in the processor
        from transformers import AutoProcessor
        tokenizer.chat_template = AutoProcessor.from_pretrained(args.model).chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Multimodal checkpoints (Gemma 4, Qwen 3.5) are loaded whole, so the adapter's module names
    # (model.language_model.layers...) match the checkpoint that vLLM loads; with the text-only
    # class the names differ and vLLM silently ignores the adapter.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model)
    if getattr(config, "vision_config", None) is not None:
        from transformers import AutoModelForImageTextToText as AutoModelClass
    else:
        AutoModelClass = AutoModelForCausalLM
    model = AutoModelClass.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    print(f"🧩 {type(model).__name__}", flush=True)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "qkv_proj", "gate_up_proj"],            # Phi-4 fuses q/k/v and gate/up
        exclude_modules=r".*(vision|visual|audio|multi_modal|mm_projector|embed_vision|embed_audio).*")

    examples = build_examples(args, per_post, rng)
    data = Dataset.from_list([render(tokenizer, args, p, a) for p, a in examples])
    print(f"📚 {len(data)} training examples ({args.epochs} epochs drawn)\n--- example ---\n"
          f"{data[0]['prompt']}{data[0]['completion']}\n---------------", flush=True)

    config = SFTConfig(
        output_dir=str(out / "checkpoints"), num_train_epochs=1, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.01,   # <1 = ratio
        optim="adamw_bnb_8bit", bf16=True, gradient_checkpointing=True,
        logging_steps=10, save_strategy="no", seed=args.seed, report_to="none",
        max_length=args.max_seq_len, completion_only_loss=True, packing=False)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=data, args=config,
                         peft_config=lora)
    trainer.model.print_trainable_parameters()
    stats = trainer.train()

    trainer.model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "run.json").write_text(json.dumps({
        "run": run, "base_model": args.model, "target": args.target,
        "n_posts": len(per_post), "n_examples": len(data), "args": {k: str(v) for k, v in vars(args).items()},
        "train_loss": stats.training_loss, "metrics": stats.metrics}, indent=2))
    print(f"✅ adapter -> {out}", flush=True)


if __name__ == "__main__":
    main()
