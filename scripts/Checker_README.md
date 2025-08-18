# Toxicity & Rationale Checker

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://religioncare-eekb6uctjxuhugjashz743.streamlit.app/)

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

👉 **Open the checker:** https://religioncare-eekb6uctjxuhugjashz743.streamlit.app/  
_No install needed. Works in any modern browser._

---

## 🧠 How it works

- **Perspective API**  
  Sends the text to Google’s Comment Analyzer (`/v1alpha1/comments:analyze`) requesting scores for  
  `TOXICITY`, `INSULT`, `IDENTITY_ATTACK`, `THREAT`, `PROFANITY`, and returns 0–1 summary scores.

- **GPT judge (structured JSON)**  
  Calls the OpenAI Chat Completions API with your selected model (`gpt-4.1`, `gpt-4o-mini`, or `gpt-5`), asking for
  a JSON object matching this schema:
  ```json
  {
    "toxicity_label": "toxic|non_toxic|unclear",
    "rationale": "string",
    "evidence_spans": ["string", "..."],
    "suggested_rewrite": "string"
  }
