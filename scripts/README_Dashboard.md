# ReligionCARE — BPR Dashboard (PRE vs POST)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://religioncare-zu7pxeurx8pdxjxinjkikf.streamlit.app)

This document covers the dashboard component of the project.

---

## 🔗 Live App

👉 **Open the dashboard:** https://religioncare-zu7pxeurx8pdxjxinjkikf.streamlit.app

No install needed. Works in any modern browser.

---

## 💡 What the app shows

- **BPR comparisons** (PRE vs POST) with Δ (POST − PRE).
- **Perspective Toxicity metrics**: TOXICITY, SEVERE_TOXICITY, INSULT, THREAT, IDENTITY_ATTACK, PROFANITY  
  - View **average score** or **% ≥ threshold**.
- Filter by **Religion**, **Axis**, and **Prompt Group (G1/G2/G3)**.

---

## 📥 Using the app

1. Open the link above.
2. Use the **sidebar** to:
   - Keep default Google Drive paths, or  
   - **Upload files directly** (CSV/JSON/JSONL).

**BPR files (any of):**
- `bpr_full_results_log.csv`
- `bpr_outputs_log.json`
- `bpr_summary_by_religion_axis.csv`
- `bpr_summary_by_religion_group.csv`
- `bpr_summary_by_religion.csv`

**Toxicity files (any of):**
- `Perspective_outputs_with_base_model.csv`
- `Perspective_outputs_with_finetuned_model.csv`
- `perspective_outputs.csv`

> Expected toxicity columns (case-insensitive):  
> `TOXICITY, SEVERE_TOXICITY, INSULT, THREAT, IDENTITY_ATTACK, PROFANITY`

---

## 🗂 Optional Google Drive layout

