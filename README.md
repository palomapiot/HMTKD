# HMTKD: Human-guided Multi-Teacher Knowledge Distillation

Code, prompts and data splits for explainable hate speech detection with HMTKD. A student model learns to write a structured rationale (evidence, targets, stance, severity, explanation, label) in three stages:

1. **Teacher rationales.** Three teachers (Gemma 4 31B, Llama 3.3 70B, Qwen 3.8 27B) each write four rationales per training post without seeing the gold label. A rationale is kept if it passes the acceptance checks. A teacher with no accepted rationale for a post writes twelve more. For posts that no teacher solves, Qwen 3.8 27B writes four more with the gold label in the prompt.
2. **Supervised fine-tuning.** The student is trained with LoRA for one epoch on the accepted rationales.
3. **GRPO.** The student is optimised with rewards for label, consistency, target, severity, stance, evidence, similarity and format.

## Contents

```
hmtkd/      package: prompts, acceptance checks, rewards, metrics, training helpers
scripts/    pipeline, in order of use
prompts/    the prompts as plain text ({post} marks where the post goes)
splits/     post ids and annotations, without the texts
```

| Script | Step |
|---|---|
| `02_teacher_traces.py` | teacher rationales (blind and label-conditioned rounds) |
| `02b_merge_teachers.py` | pool the accepted rationales of the three teachers |
| `04_sft.py` | supervised fine-tuning (LoRA): Stage II, Rationale SFT and label-only SFT |
| `05_grpo.py` | GRPO |
| `03_zero_shot.py` | inference: zero-shot, or a trained adapter with `--lora` |
| `06_evaluate.py` | metrics |
| `05b_select_checkpoint.py` | choose the GRPO checkpoint on the MHS development split |

## Splits

| File | Posts | Use |
|---|---|---|
| `splits/train.jsonl` | 15,000 MHS | teacher rationales, SFT, GRPO |
| `splits/stage1_dev.jsonl` | 1,499 MHS | development posts of the teacher-rationale pool |
| `splits/dev.jsonl` | 3,949 MHS | GRPO checkpoint selection |
| `splits/test.jsonl` | 25,874 | MHS (3,953), HateXplain (1,921), Toraman English (10,000), Toraman Turkish (10,000) |

Each line gives `uid`, `dataset`, `split`, `label_set` (all accepted labels), `label`, `hate` (1, 0, or null when the label set holds `hate` and another label), `abusive`, `soft` (share of annotators per answer, used for the agreement weight of the label reward), `targets`, `target_subgroups`, `severity_lo`, `severity_hi` and `stance`. In `train` and `stage1_dev`, `teacher_rationale` is `blind` when a teacher rationale was accepted without the label, `hinted` when it was accepted with the label, and `null` when none was accepted. SFT uses the 10,774 training posts with a rationale. GRPO uses all 15,000, and the 5,057 posts without a blind rationale appear twice.

The `uid` is the dataset name followed by the original id: `comment_id` for MHS, `post_id` for HateXplain and the tweet id for the English and Turkish tweets of Toraman et al. (2022, second release). Get the texts from the original datasets and add a `text` field to each record.

The scripts read the posts, with texts, from:

```
data/traces/pool-mhs_train.jsonl   <- splits/train.jsonl
data/traces/pool-mhs_dev.jsonl     <- splits/stage1_dev.jsonl
data/processed/train.jsonl         <- splits/train.jsonl
data/processed/dev.jsonl           <- splits/dev.jsonl
data/processed/test.jsonl          <- splits/test.jsonl
```

## Running

Tested with torch 2.13, transformers 5.17, vLLM 0.31, TRL 1.15, PEFT 0.21, datasets 5.1 and sentence-transformers 6.1 (`requirements.txt`). Seed 3407. Example for one student (`MODEL=meta-llama/Llama-3.1-8B-Instruct`, `S=llama8b`):

```bash
# 1. Teacher rationales (repeat for each teacher), pooling, label-conditioned round, final pooling
python scripts/02_teacher_traces.py --teacher google/gemma-4-31B-it --text_only \
    --pool_name pool-mhs --datasets mhs --name gemma4-len150
#    ... same for meta-llama/Llama-3.3-70B-Instruct (--quantization fp8, name llama70b-len150)
#    and Qwen/Qwen3.8-27B (--no_thinking --text_only, name qwen38-len150)
python scripts/02b_merge_teachers.py --teachers llama70b-len150 qwen38-len150 gemma4-len150 \
    --pool_name pool-mhs --out_name pooled-len150
python scripts/02_teacher_traces.py --teacher Qwen/Qwen3.8-27B --no_thinking --text_only \
    --pool_name pool-mhs --datasets mhs --name hinted-qwen38-len150 \
    --rounds hinted --only_uids data/traces/pooled-len150/unresolved_uids.txt
python scripts/02b_merge_teachers.py --teachers llama70b-len150 qwen38-len150 gemma4-len150 \
    --hinted hinted-qwen38-len150 --pool_name pool-mhs --out_name pooled-len150

# 2. Supervised fine-tuning, one epoch
python scripts/04_sft.py --model $MODEL --name $S --target rationale --epochs 1 \
    --run_name sft-$S-warmup-s3407 --seed 3407

# 3. GRPO (1000 steps, KL penalty 0.04 to the warm-up)
python scripts/05_grpo.py --model $MODEL --name $S --init results/models/sft-$S-warmup-s3407 \
    --run_name grpo-$S-s3407 --seed 3407 --keep_checkpoints

# Checkpoint selection on MHS dev: predict with every checkpoint, evaluate, select
python scripts/03_zero_shot.py --model $MODEL --lora results/models/grpo-$S-s3407/checkpoints/checkpoint-<N> \
    --split dev --datasets mhs --out_dir results/predictions-dev --name grpo-$S-s3407-step<N> --mode analysis
python scripts/06_evaluate.py grpo-$S-s3407-step<N>-analysis --split dev \
    --pred_dir results/predictions-dev --out_dir results/metrics-dev
python scripts/05b_select_checkpoint.py grpo-$S-s3407

# Test
python scripts/03_zero_shot.py --model $MODEL --lora results/models/grpo-$S-s3407/selected \
    --name grpo-$S-s3407 --mode analysis
python scripts/06_evaluate.py
```

Baselines and ablation:

```bash
# Zero-shot, label only / with explanation (students and teachers)
python scripts/03_zero_shot.py --model $MODEL --name $S --mode label
python scripts/03_zero_shot.py --model $MODEL --name $S --mode analysis

# Rationale SFT: two epochs on the accepted rationales (seeds 3407, 1, 2)
python scripts/04_sft.py --model $MODEL --name $S --target rationale --epochs 2 --seed 3407
python scripts/03_zero_shot.py --model $MODEL --lora results/models/sft-$S-rationale-s3407 \
    --name sft-$S-rationale-s3407 --mode analysis

# Label-only SFT: two epochs on the human labels
python scripts/04_sft.py --model $MODEL --name $S --target label --epochs 2 --seed 3407
python scripts/03_zero_shot.py --model $MODEL --lora results/models/sft-$S-label-s3407 \
    --name sft-$S-label-s3407 --mode label

# Reward ablation (Llama 3.1 8B): one reward at a time, e.g. the target reward alone
python scripts/05_grpo.py --model $MODEL --name $S --init results/models/sft-$S-warmup-s3407 \
    --run_name grpo-$S-onlytgt-s3407 --keep_checkpoints --no_rewards lbl cons sev stn evd sim fmt
```

Run the scripts from the repository root. Each script's `--help` lists its options.
