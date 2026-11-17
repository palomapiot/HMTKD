"""Model loading shared by SFT (scripts/04_sft.py) and GRPO (scripts/05_grpo.py). Needs a GPU image."""

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "qkv_proj", "gate_up_proj"]                       # Phi-4 fuses q/k/v and gate/up
LORA_EXCLUDE = r".*(vision|visual|audio|multi_modal|mm_projector|embed_vision|embed_audio).*"


def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.chat_template is None:          # multimodal checkpoints keep it in the processor
        from transformers import AutoProcessor
        tokenizer.chat_template = AutoProcessor.from_pretrained(model_id).chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_student(model_id):
    """Base student in bf16. Multimodal checkpoints (Gemma 4, Qwen 3.5) are loaded whole, so the
    adapter's module names (model.language_model.layers...) match the checkpoint that vLLM loads;
    with the text-only class the names differ and vLLM silently ignores the adapter."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "vision_config", None) is not None:
        from transformers import AutoModelForImageTextToText as AutoModelClass
    else:
        AutoModelClass = AutoModelForCausalLM
    model = AutoModelClass.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
    print(f"🧩 {type(model).__name__}", flush=True)
    return model


def chat_kwargs(no_thinking):
    return {"enable_thinking": False} if no_thinking else {}
