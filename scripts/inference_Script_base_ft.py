# ==== Minimal Inference: Base OR Base+Adapters ====
import re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from google.colab import drive

# --- Paths ---
drive.mount('/content/drive', force_remount=False)
BASE_DIR    = "/content/drive/MyDrive/datasets_religion/religion_care_outputs/mistral_bpr_model"
ADAPTER_DIR = "/content/drive/MyDrive/finetune_DPR/mistral_dpo_lora"   # your SFT/DPO adapters

USE_ADAPTERS = True   # True = finetuned (base + adapters), False = base-only

# --- Load ---
tok = AutoTokenizer.from_pretrained(ADAPTER_DIR if USE_ADAPTERS else BASE_DIR, use_fast=True, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_DIR,
    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
)
if USE_ADAPTERS:
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

model.eval()
model.config.use_cache = True

# --- Generation config (pick ONE) ---
# Deterministic (good for eval):
# GEN = dict(max_new_tokens=80, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)

# Helpful paragraphs (good for users):
GEN = dict(
    max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.92,
    no_repeat_ngram_size=3, repetition_penalty=1.05,
    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
)

INSTR = "Write 3–5 sentences. Be specific, fair, and avoid identity-based claims."

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

@torch.inference_mode()
def reply(user_text: str) -> str:
    prompt = f"{INSTR}\n\nPrompt: {user_text.strip()}\nAnswer:"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, **GEN)
    dec = tok.decode(out[0], skip_special_tokens=True)
    return _clean(dec[len(prompt):] if dec.startswith(prompt) else dec)

# --- Interactive ---
print(f"Ready. Mode = {'FINETUNED (adapters)' if USE_ADAPTERS else 'BASE ONLY'}. Type 'exit' to quit.")
while True:
    try:
        q = input("Your text: ").strip()
    except EOFError:
        break
    if not q or q.lower() == "exit":
        break
    print("\nModel:", reply(q), "\n" + "-"*80)
