# ========= Imports =========
import os, json, random, hashlib, re
from collections import defaultdict
import torch
from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, DataCollatorForLanguageModeling, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ========= Config =========
MOUNT_DRIVE = True
PAIRS_DIR   = "/content/drive/MyDrive/finetune_DPR"   # where train_pairs/val_pairs/test_pairs live
BASE_MODEL  = "/content/drive/MyDrive/datasets_religion/religion_care_outputs/mistral_bpr_model"

SFT_FILE        = os.path.join(PAIRS_DIR, "sft_warmup.jsonl")        # built below
SFT_OUT_DIR     = "/content/drive/MyDrive/finetune_DPR/sft_warmup_lora"
SFT_MERGED_DIR  = SFT_OUT_DIR + "_merged"

GUARDRAIL_CATS = ["hostile_assertion","comparative_speculation","dialogue_detox","identity_emotion"]
OTHER_SAMPLE_PER_CAT = 50          # sprinkle a few examples from other categories (set 0 to skip)

MAX_LEN = 1024
SEED    = 42

# Training hyperparams
BATCH_PER_DEVICE = 4
GRAD_ACCUM       = 8
EPOCHS           = 1
LR               = 1e-5

# Merge final adapters into fp16 single checkpoint? (recommended: False here; do it after DPO)
MERGE_AFTER_TRAIN = False

# Inference settings (post-train quick test)
INFER_AFTER_TRAIN   = True
ADD_INSTRUCTION     = True      # prepend a short safety/style instruction to user prompt
ENFORCE_ONE_SENT    = True      # trim response to first sentence
GEN_MAX_NEW_TOKENS  = 80
GEN_DO_SAMPLE       = False     # deterministic by default
GEN_TEMPERATURE     = 0.7
GEN_TOP_P           = 0.9

# ========= Mount Drive (optional) =========
if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped/failed:", e)

random.seed(SEED)

# ========= 1) Build SFT file from your pairs (chosen only) =========
train_pairs = os.path.join(PAIRS_DIR, "train_pairs.jsonl")
val_pairs   = os.path.join(PAIRS_DIR, "val_pairs.jsonl")

def read_jsonl(p):
    rows = []
    if not os.path.exists(p): return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # drop malformed / truncated lines
                pass
    return rows

rows = read_jsonl(train_pairs) + read_jsonl(val_pairs)
if not rows:
    raise FileNotFoundError(f"No pairs found in {PAIRS_DIR}")

# Collect guardrail items first; dedupe by prompt (case-insensitive)
sft_items, seen = [], set()
for r in rows:
    if r.get("category") in GUARDRAIL_CATS:
        p = (r.get("prompt") or "").strip()
        o = (r.get("chosen") or "").strip()
        if not p or not o: continue
        key = p.lower()
        if key in seen: continue
        seen.add(key)
        sft_items.append({"prompt": p, "output": o})

# Optionally add a small, balanced slice from other categories
if OTHER_SAMPLE_PER_CAT > 0:
    other = set(c for c in {x.get("category") for x in rows} if c) - set(GUARDRAIL_CATS)
    by_cat = defaultdict(list)
    for r in rows:
        if r.get("category") in other:
            by_cat[r["category"]].append(r)
    for cat, lst in by_cat.items():
        random.shuffle(lst)
        picked = 0
        for r in lst:
            if picked >= OTHER_SAMPLE_PER_CAT: break
            p = (r.get("prompt") or "").strip()
            o = (r.get("chosen") or "").strip()
            if not p or not o: continue
            key = p.lower()
            if key in seen: continue
            seen.add(key)
            sft_items.append({"prompt": p, "output": o})
            picked += 1

# Fix any leading "The a/an ..." artifact in prompts (safe no-op otherwise)
def fix_leading_the_a(text: str) -> str:
    return re.sub(r"^The\s+(a|an)\s+", lambda m: m.group(1).capitalize() + " ", text.strip())

for ex in sft_items:
    ex["prompt"] = fix_leading_the_a(ex["prompt"])

# Write SFT file
os.makedirs(os.path.dirname(SFT_FILE), exist_ok=True)
with open(SFT_FILE, "w", encoding="utf-8") as f:
    for ex in sft_items:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Built SFT file: {SFT_FILE}  | examples: {len(sft_items)}")
if sft_items:
    print("Sample row:", json.dumps(sft_items[0], ensure_ascii=False))

# ========= 2) Dataset & collator =========
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.model_max_length = MAX_LEN

ds_all = load_dataset("json", data_files=SFT_FILE, split="train")

# Deterministic 90/10 split by hash of prompt
def md5_int(s): return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
def buck(p): return "train" if (md5_int(p) % 10) < 9 else "val"
rows_train, rows_val = [], []
for ex in ds_all:
    (rows_train if buck(ex["prompt"])=="train" else rows_val).append(ex)
train_ds = Dataset.from_list(rows_train)
val_ds   = Dataset.from_list(rows_val)

class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Mask prompt tokens; learn only the completion."""
    def __init__(self, tokenizer, mlm=False, max_length=MAX_LEN):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.max_length = max_length
    def __call__(self, features):
        prompts = [f["prompt"] for f in features]
        outputs = [f["output"] for f in features]
        texts = [p + "\n" + o for p, o in zip(prompts, outputs)]
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()
        # mask prompt + newline
        for i, p in enumerate(prompts):
            p_ids = self.tokenizer(p + "\n", add_special_tokens=False)["input_ids"]
            cut = min(len(p_ids), input_ids.size(1) - 1)
            labels[i, :cut] = -100
        # ensure at least one learnable token (avoid NaN)
        for i in range(labels.size(0)):
            if (labels[i] != -100).sum() == 0:
                last_tok = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][-1].item()
                labels[i, last_tok] = input_ids[i, last_tok]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

collator = CompletionOnlyCollator(tok)

# ========= 3) Model (QLoRA) & training =========
use_gpu = torch.cuda.is_available()
sm8 = (torch.cuda.get_device_capability(0)[0] >= 8) if use_gpu else False  # bf16-capable GPUs

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=(torch.bfloat16 if sm8 else torch.float16),
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
base.config.use_cache = False
base.gradient_checkpointing_enable()
prepare_model_for_kbit_training(base)

lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(base, lora)

args = TrainingArguments(
    output_dir=SFT_OUT_DIR,
    per_device_train_batch_size=BATCH_PER_DEVICE,
    per_device_eval_batch_size=BATCH_PER_DEVICE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    save_total_limit=1,
    bf16=bool(sm8),
    fp16=not bool(sm8),
    report_to="none",
    seed=SEED,
    remove_unused_columns=False,  
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
)

print("SFT training…")
trainer.train()
print("SFT eval:", trainer.evaluate())

print("Saving SFT LoRA adapters…")
trainer.save_model(SFT_OUT_DIR)
tok.save_pretrained(SFT_OUT_DIR)
print("✅ Adapters saved to:", SFT_OUT_DIR)

# ========= 4) (Optional) Merge adapters -> standalone fp16 =========
if MERGE_AFTER_TRAIN:
    print("Merging adapters into base (fp16)…")
    base_fp16 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base_fp16, SFT_OUT_DIR).merge_and_unload()
    os.makedirs(SFT_MERGED_DIR, exist_ok=True)
    merged.save_pretrained(SFT_MERGED_DIR)
    tok.save_pretrained(SFT_MERGED_DIR)
    print("✅ SFT merged saved to:", SFT_MERGED_DIR)

# ========= 5) Quick inference loop (adapters attached) =========
def load_model_with_adapters(base_dir, adapters_dir, max_len=MAX_LEN):
    tok = AutoTokenizer.from_pretrained(adapters_dir if os.path.exists(adapters_dir) else base_dir,
                                        use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.model_max_length = max_len
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=(torch.bfloat16 if sm8 else torch.float16),
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapters_dir)
    model.eval()
    return tok, model

def limit_to_first_sentence(text: str):
    txt = text.strip()
    m = re.search(r'([.!?。！？])', txt)
    if not m: return txt
    end = m.end()
    return txt[:end].strip()

@torch.inference_mode()
def generate_response(tok, model, user_prompt: str, add_instruction=ADD_INSTRUCTION,
                      max_new_tokens=GEN_MAX_NEW_TOKENS, one_sentence=ENFORCE_ONE_SENT):
    instr = ("Respond briefly, fairly, and specifically. Avoid stereotypes or identity-based claims. "
             "Use job-relevant or evidence-based reasoning only.")
    prompt = (instr + "\n\n" + user_prompt.strip()) if add_instruction else user_prompt.strip()
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=GEN_DO_SAMPLE,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    # Strip the prompt if echoed
    decoded = tok.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        resp = decoded[len(prompt):].strip()
    else:
        # fallback: try to find the segment after the prompt tokens
        resp = decoded.split("\n")[-1].strip()
    return limit_to_first_sentence(resp) if one_sentence else resp

if INFER_AFTER_TRAIN:
    print("\n=== Inference (SFT adapters attached) ===")
    tok_i, model_i = load_model_with_adapters(BASE_MODEL, SFT_OUT_DIR)
    while True:
        try:
            user = input("\nYour prompt (or 'exit'): ").strip()
        except EOFError:
            break
        if not user or user.lower() == "exit":
            break
        reply = generate_response(tok_i, model_i, user)
        print("\nModel:", reply)
