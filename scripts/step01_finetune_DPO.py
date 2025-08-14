

import os, json, torch, hashlib, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer,DPOConfig

# ---------- Config ----------
MOUNT_DRIVE = True
BASE_MODEL  = "/content/drive/MyDrive/datasets_religion/religion_care_outputs/mistral_bpr_model"
PAIRS_DIR   = "/content/drive/MyDrive/finetune_DPR"            # train_pairs.jsonl / val_pairs.jsonl here
SFT_ADAPTER_DIR = "/content/drive/MyDrive/finetune_DPR/sft_warmup_lora"  # from previous step

DPO_OUT_DIR     = "/content/drive/MyDrive/finetune_DPR/mistral_dpo_lora"
DPO_MERGED_DIR  = DPO_OUT_DIR + "_merged"

EPOCHS      = 2
PER_DEV_BS  = 4
GRAD_ACCUM  = 8
LR          = 2e-5
BETA        = 0.1               # DPO temperature
MAX_LEN     = 1024              # prompt + completion budget
MAX_TARGET  = 256               # completion budget
SEED        = 42
MERGE_AFTER = True              # merge adapters to standalone fp16 at the end?

# Inference
RUN_INFER   = True
ONE_SENTENCE_ONLY = True
USE_SAMPLING = False
TEMP = 0.7
TOP_P = 0.9
MAX_NEW = 80

# ---------- Mount Drive (optional) ----------
if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped/failed:", e)

# ---------- Tokenizer ----------
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.model_max_length = MAX_LEN

# ---------- Data (expects JSONL with prompt/chosen/rejected) ----------
train_pairs = os.path.join(PAIRS_DIR, "train_pairs.jsonl")
val_pairs   = os.path.join(PAIRS_DIR, "val_pairs.jsonl")
train_ds = load_dataset("json", data_files=train_pairs, split="train")
val_ds   = load_dataset("json", data_files=val_pairs,   split="train")

need = {"prompt","chosen","rejected"}
for name, ds in [("train",train_ds), ("val",val_ds)]:
    missing = need - set(ds.column_names)
    if missing: raise ValueError(f"{name} missing columns: {missing}. Got {ds.column_names}")

print(f"Loaded pairs: train={len(train_ds)}  val={len(val_ds)}")

# ---------- 4-bit base, SFT adapters as POLICY ----------
use_gpu = torch.cuda.is_available()
bf16_ok = (torch.cuda.get_device_capability(0)[0] >= 8) if use_gpu else False

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# policy (trainable): base + SFT adapters
policy_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
policy_base.config.use_cache = False
policy = PeftModel.from_pretrained(policy_base, SFT_ADAPTER_DIR)  # continue training these adapters

# reference (frozen): separate copy of SFT policy
ref_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
ref = PeftModel.from_pretrained(ref_base, SFT_ADAPTER_DIR)
ref.eval()
for p in ref.parameters(): p.requires_grad = False

# ---------- Training args ----------
BETA = 0.1
MAX_LEN = 1024
MAX_TARGET = 256

training_args = DPOConfig(
    output_dir=DPO_OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEV_BS,
    per_device_eval_batch_size=PER_DEV_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=bool(bf16_ok),
    fp16=not bool(bf16_ok),
    report_to="none",
    seed=SEED,
    remove_unused_columns=False,   # keep prompt/chosen/rejected for TRL processing
    gradient_checkpointing=True,

    # --- DPO-specific knobs live in DPOConfig in v0.21 ---
    beta=BETA,
    max_length=MAX_LEN,              # prompt+completion
    max_completion_length=MAX_TARGET # completion only
)

trainer = DPOTrainer(
    model=policy,          # base + SFT adapters (trainable)
    ref_model=ref,         # frozen reference (SFT copy)
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tok,  # not `tokenizer=`
)

print("Starting DPO training…")
trainer.train()
print("Evaluating…")
metrics = trainer.evaluate()
print("Eval metrics:", metrics)

print("💾 Saving DPO adapters…")
trainer.save_model(DPO_OUT_DIR)
tok.save_pretrained(DPO_OUT_DIR)
print("✅ Adapters saved to:", DPO_OUT_DIR)

# ---------- Optional: merge adapters -> standalone fp16 ----------
if MERGE_AFTER:
    from peft import PeftModel
    print("Merging DPO adapters into base (fp16)…")
    base_fp16 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base_fp16, DPO_OUT_DIR).merge_and_unload()
    os.makedirs(DPO_MERGED_DIR, exist_ok=True)
    merged.save_pretrained(DPO_MERGED_DIR)
    tok.save_pretrained(DPO_MERGED_DIR)
    print("✅ Merged model saved to:", DPO_MERGED_DIR)

# ---------- Quick inference (adapters attached or merged) ----------
def limit_first_sentence(txt):
    txt = txt.strip()
    m = re.search(r'([.!?。！？])', txt)
    return txt if not m else txt[:m.end()].strip()

@torch.inference_mode()
def generate_reply(tokenizer, model, user_prompt: str):
    instr = ("Respond briefly, fairly, and specifically. Avoid stereotypes or identity claims. "
             "Use job-relevant or evidence-based reasoning.")
    prompt = f"{instr}\n\n{user_prompt.strip()}\n"
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW,
        do_sample=USE_SAMPLING,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if USE_SAMPLING:
        gen_kwargs.update(temperature=TEMP, top_p=TOP_P)
    out = model.generate(**enc, **gen_kwargs)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    resp = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
    return limit_first_sentence(resp) if ONE_SENTENCE_ONLY else resp

if RUN_INFER:
    print("\n=== Inference ===")
    if MERGE_AFTER and os.path.exists(DPO_MERGED_DIR):
        # load merged
        inf_tok = AutoTokenizer.from_pretrained(DPO_MERGED_DIR, use_fast=True, trust_remote_code=True)
        if inf_tok.pad_token is None: inf_tok.pad_token = inf_tok.eos_token
        inf_model = AutoModelForCausalLM.from_pretrained(
            DPO_MERGED_DIR, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", trust_remote_code=True
        )
    else:
        # base + DPO adapters
        inf_tok = tok
        inf_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", trust_remote_code=True
        )
        inf_model = PeftModel.from_pretrained(inf_model, DPO_OUT_DIR)
        inf_model.eval()

    while True:
        try:
            q = input("\nYour prompt (or 'exit'): ").strip()
        except EOFError:
            break
        if not q or q.lower() == "exit":
            break
        print("\nModel:", generate_reply(inf_tok, inf_model, q))
