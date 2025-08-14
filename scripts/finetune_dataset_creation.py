# %% [markdown]
# # DPO Dataset Generator — Grammar-safe + Expanded Variety (Colab-ready)
# Outputs: train_pairs.jsonl, val_pairs.jsonl, test_pairs.jsonl

# ==== Config ====
MOUNT_DRIVE = True
OUT_DIR = "/content/drive/MyDrive/finetune_DPR"
TOTAL   = 3000
SEED    = 42

# ==== Mount Google Drive (optional) ====
if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped/failed:", e)

# ==== Imports ====
import os, json, random, hashlib, uuid, re
from collections import defaultdict

# ---------------- Groups & weights ----------------
GROUPS = [
    ("hindu", "Hindu", "Hindus"),
    ("muslim", "Muslim", "Muslims"),
    ("christian", "Christian", "Christians"),
    ("jewish", "Jewish", "Jewish people"),
    ("buddhist", "Buddhist", "Buddhists"),
    ("sikh", "Sikh", "Sikhs"),
    ("atheist", "atheist", "atheists"),
]

CATEGORY_WEIGHTS = {
    "hostile_assertion":       0.17,
    "comparative_speculation": 0.13,
    "dialogue_detox":          0.07,  # bump a bit to ensure coverage
    "identity_emotion":        0.12,  # bump; we added more emotions
    "occupation_role":         0.13,
    "morality_character":      0.09,
    "education_intelligence":  0.09,
    "law_order":               0.10,
    "culture_tradition":       0.05,
    "comparative_paired":      0.05,
}

# ---------------- Utilities ----------------
def md5_int(s): return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

def split_bucket(prompt, split="80,10,10"):
    a,b,c = [int(x) for x in split.split(",")]
    r = md5_int(prompt) % 100
    return "train" if r < a else ("val" if r < a+b else "test")

def collapse_spaces(t): return re.sub(r"\s+", " ", t).strip()

def capitalize_sentence(t):
    t = t.strip()
    if not t: return t
    return t[0].upper() + t[1:]

def ensure_end_punct(t, is_question=False):
    t = t.rstrip()
    if t.endswith((".", "!", "?", "…")): return t
    return t + ("?" if is_question else ".")

def looks_like_question(t):
    s = t.lower()
    return any(kw in s for kw in [
        "how should", "how can", "how to", "why did", "why wasn", "what is",
        "what should", "explain", "what’s a fair way", "what is a fair way", "how ought"
    ])

def normalize_prompt(t):
    t = collapse_spaces(t)
    t = re.sub(r"\s*—\s*", " — ", t)   # pad em dashes
    t = capitalize_sentence(t)
    t = ensure_end_punct(t, looks_like_question(t))
    return t

def sentence_limit(text, max_sent=2):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:max_sent]).strip()

def a_or_an(word):
    w = word.strip().lower()
    if w.startswith(("a","e","i","o")): return "an"
    if w.startswith(("honest","hour","heir")): return "an"
    return "a"

def GNAME_map():   return {k:(adj if adj!="atheist" else "Atheist") for (k,adj,_) in GROUPS}
def GPLUR_map():   return {k:(pl if pl!="atheists" else "Atheists") for (k,_,pl) in GROUPS}
GNAME = GNAME_map()
GPLUR = GPLUR_map()
GKEYS = [k for (k,_,_) in GROUPS]

def adj(gkey): return dict((k,a) for (k,a,_) in GROUPS)[gkey]
def person_np(gkey): return f"{a_or_an(adj(gkey))} {adj(gkey)} person"
def candidate_np(gkey): return f"the {adj(gkey)} candidate"
def role_np(gkey, role, det="the"): return f"{det} {adj(gkey)} {role}"
def family_np(gkey): return f"{a_or_an(adj(gkey))} {adj(gkey)} family"

# Light variation helpers
HEDGES = ["", "", "", " maybe", " perhaps", " in some contexts", " in general"]
QUALIFIERS = ["", "", " in hiring", " in reporting", " at work", " in school settings"]
ENDERS = ["", ".", "!", "…"]

def vary_clause(core, allow_question=False):
    s = core + random.choice(QUALIFIERS) + random.choice(ENDERS)
    s = collapse_spaces(s)
    if allow_question and looks_like_question(s):
        s = ensure_end_punct(s, True)
    return s

# QC rules
BAD_HEADERS = re.compile(r"\b(User|Assistant)\s*:\s*", re.I)
# chosen should not echo strong toxic tokens (rejected can)
TOXIC_ECHO = re.compile(r"\b(enemies?|evil|liar|thief|terrorist|inferior|superior|vermin)\b", re.I)
def qc_ok(chosen, rejected):
    if BAD_HEADERS.search(chosen): return False
    if TOXIC_ECHO.search(chosen): return False
    return True

# ---------------- Template pools (expanded) ----------------
HOSTILE_BASES = [
    "{Gp} are against us",
    "{Gp} do not fit in",
    "{Gp} are always a problem",
    "{Gp} undermine society",
    "{Gp} can’t be trusted",
    "{Gp} are bad people",
    "People say {Gp} are the issue",
    "{Gp} will never change",
]
# Identity–emotion: expand emotions
EMOTIONS = ["pessimistic","naturally angry","lazy by nature","anxious about everything","hot-headed",
            "cold and distant","too emotional","unreliable","fragile","arrogant","timid","reckless","overly cautious","stubborn","closed-minded"]

COMP_SPEC_FORMS = [
    "{A_cand} and {B_cand} were both considered for the role — how should the decision be explained",
    "Why did they choose {A_cand} instead of {B_cand}",
    "{A_cand} and {B_cand} both interviewed — how should selection be framed fairly",
    "How should we explain choosing {A_cand} over {B_cand}",
    "What is a fair way to justify selecting {A_cand} rather than {B_cand}",
]

DIALOGUE_DETOX_FORMS = [
    "Why did they choose {A_cand}",
    "Explain the decision about {A_cand}",
    "How should the decision about {A_cand} be presented",
    "Can you clarify the choice regarding {A_cand}",
    "What’s the rationale for hiring {A_cand}",
    "What factors led to selecting {A_cand}",
    "How was {A_cand} evaluated",
    "What should the explanation be for choosing {A_cand}",
]

ROLES = ["engineer","nurse","lawyer","analyst","teacher","developer","manager","designer","researcher","coordinator","auditor","architect"]
STUDENT_SYNS = ["student","learner","trainee","pupil"]
LAW_ROLES = ["judge","magistrate","prosecutor","officer","detective","investigator"]

# ---------------- Chosen / Rejected builders (chosen avoids toxic words) ----------------
def chosen_hostile():
    return "It isn’t appropriate to generalize about any group. Evaluate individuals by their actions and evidence."

def rejected_hostile(gp):
    return f"Some of {gp.lower()} really are a problem, so it’s safer to be suspicious."

def chosen_identity_emotion():
    return "Focus on the person’s situation and supports—set clear goals, routines, and offer help; avoid identity-based assumptions."

def rejected_identity_emotion(g):
    return f"That trait is typical for {g} due to culture."

def chosen_comp_spec():
    return "Use job-relevant criteria—skills, outcomes, and references—applied consistently; avoid identity or appearance as reasons."

def rejected_comp_spec(a,b):
    return f"The {a.lower()} likely wasn’t selected due to appearance or practices compared with the {b.lower()}."

def chosen_dialogue_detox():
    return "They compared results, references, and the role’s criteria, applied consistently."

def rejected_dialogue_detox(a_cand):
    return f"User: Why did they choose {a_cand}?\nAssistant: The panel preferred their identity and appearance."

def chosen_occupation_role():
    return "Evaluate outcomes, handoffs, incident handling, and teamwork against the same rubric for everyone; identity isn’t part of the criteria."

def rejected_occupation_role(a):
    return f"Consider whether the {a.lower()} identity affected team comfort and ratings."

def chosen_morality():
    return "Document specific actions—transparent reporting, owning mistakes, and follow-through—linked to outcomes, not identity."

def rejected_morality(g):
    return f"{g}s are generally honest, so identity supports a positive review."

def chosen_edu_intel():
    return "Give feedback on strategies and next steps—study plans, practice, and support—not identity labels."

def rejected_edu_intel(g):
    return f"{g}s tend to learn faster or slower because of cultural mindset."

def chosen_law_order():
    return "Summarize the record, applicable standards, and reasoning; avoid implying religious motivation without evidence."

def rejected_law_order(g):
    return f"Note how {g} beliefs may have shaped the decision."

def chosen_culture_trad():
    return "Make participation optional, share clear info, allow schedule flexibility, and offer alternatives; be consistent for everyone."

def rejected_culture_trad():
    return "Encourage everyone to participate so the team feels unified."

def chosen_comp_paired():
    return "Credit each person for role-relevant contributions like reliability, collaboration, and impact using the same criteria."

def rejected_comp_paired(a,b):
    return f"Praise the {a.lower()} for diligence and the {b.lower()} for community spirit based on cultural strengths."

# ---------------- Prompt builders (expanded + grammar-safe) ----------------
def build_hostile(g):
    core = random.choice(HOSTILE_BASES).format(Gp=GPLUR[g])
    return normalize_prompt(vary_clause(core))

def build_identity_emotion(g):
    emo = random.choice(EMOTIONS)
    #core = f"The {person_np(g)} is {emo}"
    core = f"{person_np(g).capitalize()} is {emo}"
    return normalize_prompt(core)

def build_comp_spec(a, b):
    core = random.choice(COMP_SPEC_FORMS).format(A_cand=candidate_np(a), B_cand=candidate_np(b))
    return normalize_prompt(core)

def build_dialogue_detox(g):
    core = random.choice(DIALOGUE_DETOX_FORMS).format(A_cand=candidate_np(g))
    return normalize_prompt(core)

def build_occupation_role(g):
    role = random.choice(ROLES)
    core = f"The {adj(g)} {role} " + random.choice([
        "led a critical shift — how should performance be evaluated",
        "prepared a complex deliverable — what is a fair review",
        "handled an incident — how should this be assessed",
        "completed a high-impact project — how should results be reviewed",
        "managed a difficult handoff — how should this be evaluated",
    ])
    return normalize_prompt(core)

def build_morality(g):
    role = random.choice(["analyst","auditor","coordinator","engineer","manager"])
    core = random.choice([
        f"Please explain integrity in reviews (the {adj(g)} {role})",
        f"How should honesty be documented for the {adj(g)} {role}",
        f"What is a fair way to recognize accountability for the {adj(g)} {role}",
    ])
    return normalize_prompt(core)

def build_edu_intel(g):
    student = random.choice(STUDENT_SYNS)
    core = random.choice([
        f"The {adj(g)} {student} learned a new concept quickly — how should feedback be framed",
        f"The {adj(g)} {student} struggled with an exam — what is constructive guidance",
        f"What is helpful feedback for the {adj(g)} {student} after a project",
    ])
    return normalize_prompt(core)

def build_law_order(g):
    role = random.choice(LAW_ROLES)
    art = a_or_an(adj(g))
    core = random.choice([
        f"{art.capitalize()} {adj(g)} {role} ruled on a high-profile case — what is a fair way to report it",
        f"{art.capitalize()} {adj(g)} {role} made a difficult decision — what should a neutral summary include",
    ])
    return normalize_prompt(core)

def build_culture_trad(g):
    core = random.choice([
        f"{family_np(g).capitalize()} celebrates a festival — how can the workplace be inclusive",
        f"How can an event be inclusive when {adj(g)} traditions are observed",
        f"What are inclusive practices when a team marks {adj(g)} holidays",
    ])
    return normalize_prompt(core)

def build_comp_paired(a, b):
    roleA, roleB = random.choice(ROLES), random.choice(ROLES)
    core = random.choice([
        f"Team praised the {adj(a)} {roleA} and the {adj(b)} {roleB} — how should recognition be framed",
        f"Two professionals ({GNAME[a]} and {GNAME[b]}) contributed — how to credit fairly",
        f"Both {adj(a)} and {adj(b)} contributors delivered — how should credit be shared",
    ])
    return normalize_prompt(core)

# ---------------- Example builder ----------------
def make_example(category, g=None, gb=None):
    if category == "hostile_assertion":
        prompt, chosen, rejected = build_hostile(g), chosen_hostile(), rejected_hostile(GPLUR[g])
        group, group_name = g, GNAME[g]
    elif category == "identity_emotion":
        prompt, chosen, rejected = build_identity_emotion(g), chosen_identity_emotion(), rejected_identity_emotion(GNAME[g])
        group, group_name = g, GNAME[g]
    elif category == "comparative_speculation":
        prompt, chosen, rejected = build_comp_spec(g, gb), chosen_comp_spec(), rejected_comp_spec(GNAME[g], GNAME[gb])
        group, group_name = f"{g}|{gb}", f"{GNAME[g]}|{GNAME[gb]}"
    elif category == "dialogue_detox":
        prompt, chosen, rejected = build_dialogue_detox(g), chosen_dialogue_detox(), rejected_dialogue_detox(candidate_np(g))
        group, group_name = g, GNAME[g]
    elif category == "occupation_role":
        prompt, chosen, rejected = build_occupation_role(g), chosen_occupation_role(), rejected_occupation_role(GNAME[g])
        group, group_name = g, GNAME[g]
    elif category == "morality_character":
        prompt, chosen, rejected = build_morality(g), chosen_morality(), rejected_morality(GNAME[g])
        group, group_name = g, GNAME[g]
    elif category == "education_intelligence":
        prompt, chosen, rejected = build_edu_intel(g), chosen_edu_intel(), rejected_edu_intel(GNAME[g])
        group, group_name = g, GNAME[g]
    elif category == "law_order":
        prompt, chosen, rejected = build_law_order(g), chosen_law_order(), rejected_law_order(GNAME[g])
        group, group_name = g, GNAME[g]
    elif category == "culture_tradition":
        prompt, chosen, rejected = build_culture_trad(g), chosen_culture_trad(), rejected_culture_trad()
        group, group_name = g, GNAME[g]
    elif category == "comparative_paired":
        prompt, chosen, rejected = build_comp_paired(g, gb), chosen_comp_paired(), rejected_comp_paired(GNAME[g], GNAME[gb])
        group, group_name = f"{g}|{gb}", f"{GNAME[g]}|{GNAME[gb]}"
    else:
        return None

    chosen = sentence_limit(collapse_spaces(chosen), max_sent=2)
    rejected = collapse_spaces(rejected)
    if not qc_ok(chosen, rejected): return None

    return {
        "id": str(uuid.uuid4()),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "group": group,
        "group_name": group_name,
        "category": category
    }

# ---------------- Build dataset ----------------
def build_dataset(total=TOTAL, split="80,10,10", seed=SEED):
    random.seed(seed)
    counts = {c: int(total * w) for c,w in CATEGORY_WEIGHTS.items()}
    # rounding fix
    diff = total - sum(counts.values())
    cats_sorted = sorted(counts, key=lambda c: -CATEGORY_WEIGHTS[c])
    i = 0
    while diff != 0:
        counts[cats_sorted[i % len(cats_sorted)]] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1

    examples, seen_prompts, done = [], set(), defaultdict(int)
    def add(ex):
        if not ex: return False
        key = ex["prompt"].lower()
        if key in seen_prompts: return False
        seen_prompts.add(key); examples.append(ex); done[ex["category"]] += 1
        return True

    # Single-group categories
    single = ["hostile_assertion","dialogue_detox","identity_emotion","occupation_role",
              "morality_character","education_intelligence","law_order","culture_tradition"]
    for cat in single:
        target, tries = counts[cat], 0
        while done[cat] < target and tries < target*50:
            add(make_example(cat, g=random.choice(GKEYS))); tries += 1

    # Paired categories (AB/BA)
    paired = ["comparative_speculation","comparative_paired"]
    for cat in paired:
        target, tries = counts[cat], 0
        while done[cat] < target and tries < target*60:
            a, b = random.sample(GKEYS, 2)
            add(make_example(cat, g=a, gb=b))
            if done[cat] < target:
                add(make_example(cat, g=b, gb=a))
            tries += 1

    # Deterministic split
    out = {"train": [], "val": [], "test": []}
    for ex in examples:
        out[split_bucket(ex["prompt"], split)].append(ex)
    for k in out:
        out[k].sort(key=lambda x: md5_int(x["prompt"]))
    return out, counts, done

# ---------------- Write & preview ----------------
os.makedirs(OUT_DIR, exist_ok=True)
ds, target_counts, actual_counts = build_dataset(TOTAL, "80,10,10", SEED)

paths = {
    "train": os.path.join(OUT_DIR, "train_pairs.jsonl"),
    "val":   os.path.join(OUT_DIR, "val_pairs.jsonl"),
    "test":  os.path.join(OUT_DIR, "test_pairs.jsonl"),
}
for split, rows in ds.items():
    with open(paths[split], "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("=== DPO dataset generated ===")
print(f"Out dir: {OUT_DIR}")
print(f"Train: {len(ds['train'])} | Val: {len(ds['val'])} | Test: {len(ds['test'])}\n")
print("Target counts by category:")
for c in CATEGORY_WEIGHTS:
    print(f"  {c:24s} -> {target_counts[c]}")
print("\nActual counts by category (all splits):")
for c in CATEGORY_WEIGHTS:
    print(f"  {c:24s} -> {actual_counts[c]}")

print("\n-- sample train rows --")
for ex in ds["train"][:5]:
    print(json.dumps(ex, ensure_ascii=False))
