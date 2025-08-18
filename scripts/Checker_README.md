# Toxicity & Rationale Checker

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<YOUR-APP>.streamlit.app)

This tool lets you paste any text and get **two independent safety assessments**:

1) **Perspective API** scores for: `TOXICITY`, `INSULT`, `IDENTITY_ATTACK`, `THREAT`, `PROFANITY`.  
2) A **GPT judge** verdict with a structured explanation:
   - `toxicity_label`: `toxic` | `non_toxic` | `unclear`  
   - `rationale`: short, concrete reasoning  
   - `evidence_spans`: 1–3 **direct quotes** pulled from the text  
   - `suggested_rewrite`: present only when the text is judged toxic

> Optional: a lightweight **sentiment** score (if the HuggingFace `transformers` pipeline is available).

---

## 🔗 Live app

👉 **Open the checker:** https://<YOUR-APP>.streamlit.app  
_No install needed. Works in any modern browser._

---

## 🧠 How it works (aligned with the code)

- **Perspective API**  
  The app POSTs the input text to Google’s Comment Analyzer (`/v1alpha1/comments:analyze`) with attributes:  
  `TOXICITY`, `INSULT`, `IDENTITY_ATTACK`, `THREAT`, `PROFANITY`.  
  It returns a dict of summary scores (0–1) for whichever attributes are present.

- **GPT judge (structured JSON)**  
  The app calls the **OpenAI Chat Completions** endpoint with your selected model (e.g., `gpt-4.1`, `gpt-4o-mini`, `gpt-5`) and a strict **JSON schema**:
  ```json
  {
    "toxicity_label": "toxic|non_toxic|unclear",
    "rationale": "string",
    "evidence_spans": ["string", "..."],
    "suggested_rewrite": "string"
  }
