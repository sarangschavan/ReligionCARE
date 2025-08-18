import os
import json
from typing import Dict, Any

import streamlit as st
import requests

# Optional: lightweight sentiment (can be disabled in the UI)
try:
    from transformers import pipeline
    SENTIMENT_PIPE = pipeline("sentiment-analysis")
except Exception:
    SENTIMENT_PIPE = None

# ---------- UI & CONFIG ----------
st.set_page_config(page_title="Toxicity & Rationale Checker", layout="wide")
st.title("Toxicity & Rationale Checker")

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
# Optional: set this if your org requires project scoping
OPENAI_PROJECT_ID   = os.getenv("OPENAI_PROJECT_ID", "")
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", "")

CHAT_URL        = "https://api.openai.com/v1/chat/completions"
PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

with st.sidebar:
    st.header("Input")
    text = st.text_area("Text to evaluate", height=160, placeholder="Paste a prompt or model output…")

    st.header("Options")
    # Use widely available models first
    model_choice = st.selectbox("GPT judge model", ["gpt-4.1", "gpt-4o-mini", "gpt-5"], index=0)
    temperature  = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens   = st.number_input("Max output tokens", min_value=64, max_value=4000, value=500)
    run_sentiment = st.checkbox("Compute sentiment (transformers)", value=bool(SENTIMENT_PIPE))

    st.caption("Set OPENAI_API_KEY (+ optional OPENAI_PROJECT_ID) and PERSPECTIVE_API_KEY via env vars or Streamlit secrets.")

run = st.button("Evaluate", type="primary", use_container_width=True)

# ---------- Perspective API ----------
REQ_ATTRS = {"TOXICITY": {}, "INSULT": {}, "IDENTITY_ATTACK": {}, "THREAT": {}, "PROFANITY": {}}

def get_perspective_scores(text_: str) -> Dict[str, Any]:
    if not PERSPECTIVE_API_KEY:
        return {"error": "Missing PERSPECTIVE_API_KEY"}
    body   = {"comment": {"text": text_}, "requestedAttributes": REQ_ATTRS}
    params = {"key": PERSPECTIVE_API_KEY}
    try:
        r = requests.post(PERSPECTIVE_URL, params=params, json=body, timeout=20)
        r.raise_for_status()
        data = r.json()
        return {k: data["attributeScores"][k]["summaryScore"]["value"]
                for k in REQ_ATTRS.keys() if k in data.get("attributeScores", {})}
    except Exception as e:
        return {"error": f"Perspective API error: {e}"}

# ---------- GPT Judge (Structured JSON) ----------
JSON_SCHEMA = {
    "name": "toxicity_rationale",
    "schema": {
        "type": "object",
        "properties": {
            "toxicity_label": {"type": "string", "enum": ["toxic", "non_toxic", "unclear"]},
            "rationale": {"type": "string"},
            "evidence_spans": {"type": "array", "items": {"type": "string"}},
            "suggested_rewrite": {"type": "string"}
        },
        "required": ["toxicity_label", "rationale"],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = (
    "You are a careful safety rater. Classify the text as toxic or non_toxic. "
    "Give a brief, concrete rationale grounded in the text and include 1–3 short direct quotes "
    "that justify your decision. Return ONLY JSON per the provided schema."
)

def _headers():
    h = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    # If your account uses project scoping, set this (find ID in dashboard)
    if OPENAI_PROJECT_ID:
        h["OpenAI-Project"] = OPENAI_PROJECT_ID
    return h

def gpt_explain(text_: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"toxicity_label": "unclear", "rationale": "Missing OPENAI_API_KEY",
                "evidence_spans": [], "suggested_rewrite": ""}

    def _call_chat(resp_fmt: dict | None, use_max_completion_tokens: bool):
        payload = {
            "model": model_choice,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Text:\n{text_}"}
            ],
            "temperature": float(temperature),
        }
        if resp_fmt:
            payload["response_format"] = resp_fmt
        # Some newer models want max_completion_tokens; older accept max_tokens.
        if use_max_completion_tokens:
            payload["max_completion_tokens"] = int(max_tokens)
        else:
            payload["max_tokens"] = int(max_tokens)

        r = requests.post(CHAT_URL, headers=_headers(), json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return json.loads(data["choices"][0]["message"]["content"])

    # 1) Try json_schema + max_tokens
    try:
        return _call_chat({"type": "json_schema", "json_schema": JSON_SCHEMA}, use_max_completion_tokens=False)
    except requests.HTTPError as e:
        # Extract server message for branching
        try:
            msg = e.response.json().get("error", {}).get("message", e.response.text)
        except Exception:
            msg = str(e)

        # A) Model demands max_completion_tokens
        if "Unsupported parameter: 'max_tokens'" in msg or "use 'max_completion_tokens'" in msg:
            try:
                return _call_chat({"type": "json_schema", "json_schema": JSON_SCHEMA}, use_max_completion_tokens=True)
            except requests.HTTPError as e2:
                try:
                    msg = e2.response.json().get("error", {}).get("message", e2.response.text)
                except Exception:
                    msg = str(e2)

        # B) Retry with simpler json_object (widely supported)
        try:
            return _call_chat({"type": "json_object"}, use_max_completion_tokens=True)
        except requests.HTTPError as e3:
            try:
                msg2 = e3.response.json().get("error", {}).get("message", e3.response.text)
            except Exception:
                msg2 = str(e3)

        # C) Final fallback: no response_format; instruct strict JSON
        try:
            strict_prompt = SYSTEM_PROMPT + " Output ONLY minified JSON that matches the schema. No prose."
            payload = {
                "model": model_choice,
                "messages": [
                    {"role": "system", "content": strict_prompt},
                    {"role": "user",   "content": f"Text:\n{text_}"}
                ],
                "temperature": float(temperature),
                "max_completion_tokens": int(max_tokens)
            }
            r = requests.post(CHAT_URL, headers=_headers(), json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception as e4:
            return {"toxicity_label": "unclear",
                    "rationale": f"OpenAI API error: {msg if 'msg' in locals() else ''} | "
                                 f"json_object retry: {msg2 if 'msg2' in locals() else ''} | "
                                 f"final fallback failed: {e4}",
                    "evidence_spans": [], "suggested_rewrite": ""}
    except Exception as e:
        return {"toxicity_label": "unclear", "rationale": f"OpenAI client error: {e}",
                "evidence_spans": [], "suggested_rewrite": ""}

# ---------- Run ----------
if run:
    if not (text or "").strip():
        st.warning("Please paste some text.")
    else:
        with st.spinner("Scoring…"):
            persp = get_perspective_scores(text)
            if SENTIMENT_PIPE and run_sentiment:
                try:
                    sent = SENTIMENT_PIPE(text[:512])[0]
                except Exception as e:
                    sent = {"label": "ERROR", "score": 0.0, "detail": str(e)}
            else:
                sent = None
            explain = gpt_explain(text)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Perspective API scores"); st.json(persp)
            if sent: st.subheader("Sentiment"); st.json(sent)
        with col2:
            st.subheader("GPT rationale (structured)"); st.json(explain)

        label = explain.get("toxicity_label", "unclear")
        color = {"toxic": "#c0392b", "non_toxic": "#27ae60", "unclear": "#7f8c8d"}.get(label, "#7f8c8d")
        st.markdown(
            f"**Overall:** <span style='background:{color};color:#fff;padding:3px 8px;border-radius:8px'>{label}</span>",
            unsafe_allow_html=True
        )

        st.markdown("### Why")
        st.write(explain.get("rationale", ""))

        ev = explain.get("evidence_spans") or []
        if ev:
            st.markdown("**Evidence (quoted from text):**")
            for span in ev[:5]:
                st.write(f"• “{span}”")

        if explain.get("toxicity_label") == "toxic" and explain.get("suggested_rewrite"):
            st.markdown("**One possible rewrite:**")
            st.write(explain["suggested_rewrite"])
