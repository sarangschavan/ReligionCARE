import os, re, json, glob, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="ReligionCARE — BPR Dashboard", layout="wide")
st.title("🔎 ReligionCARE — BPR Dashboard (PRE vs POST)")

# ---------- Mount Drive (Colab) ----------
if 'drive_mounted' not in st.session_state:
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive', force_remount=False)
        st.session_state['drive_mounted'] = True
    except Exception:
        st.session_state['drive_mounted'] = False

# ---------- Helpers ----------
CANDIDATE_BPR = [
    "bpr_full_results_log.csv",
    "bpr_outputs_log.json",
    "bpr_summary_by_religion_axis.csv",
    "bpr_summary_by_religion_group.csv",
    "bpr_summary_by_religion.csv",
]

CANDIDATE_TOX = [
    "Perspective_outputs_with_base_model.csv",
    "Perspective_outputs_with_finetuned_model.csv",
    "perspective_outputs.csv",
]

PERSPECTIVE_METRICS = {"TOXICITY","SEVERE_TOXICITY","INSULT","THREAT","IDENTITY_ATTACK","PROFANITY"}

def _read_textlike(uploaded):
    data = uploaded.read()
    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("utf-8", errors="ignore")
    try:
        uploaded.seek(0)
    except Exception:
        pass
    return text

def read_tabular(src):
    name = getattr(src, "name", str(src)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(src)
    if name.endswith((".jsonl",".ndjson")):
        if hasattr(src, "read"):
            return pd.read_json(io.StringIO(_read_textlike(src)), lines=True)
        return pd.read_json(src, lines=True)
    if name.endswith(".json"):
        if hasattr(src, "read"):
            data = json.loads(_read_textlike(src))
        else:
            with open(src, "r") as fh:
                data = json.load(fh)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return pd.json_normalize(data)
    st.warning(f"Unsupported file type for: {name}")
    return None

def coalesce_columns(df, mapping):
    if df is None or df.empty:
        return df
    df = df.copy()
    lower = {c.lower(): c for c in df.columns}
    for canon, choices in mapping.items():
        for c in choices:
            if c.lower() in lower:
                df.rename(columns={lower[c.lower()]: canon}, inplace=True)
                break
        if canon not in df.columns:
            df[canon] = np.nan
    return df

def compute_bpr_from_rows(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "is_biased" in df.columns:
        df["is_biased"] = pd.to_numeric(df["is_biased"], errors="coerce").fillna(0).clip(0,1)
        if "is_unanswered" in df.columns:
            df["is_unanswered"] = pd.to_numeric(df["is_unanswered"], errors="coerce").fillna(0).clip(0,1)
        else:
            df["is_unanswered"] = 0.0
        anti = 1 - df["is_biased"] - df["is_unanswered"]
        df["_biased_hit"] = (df["is_biased"] > 0).astype(int)
        df["_anti_hit"]   = anti.clip(lower=0).astype(int)
        df["_answered"]   = 1 - (df["is_unanswered"] > 0).astype(int)
    elif "completion" in df and (("biased_word" in df) or ("anti_word" in df)):
        def hit(text, word):
            if pd.isna(text) or pd.isna(word) or str(word).strip()=="":
                return False
            return re.search(rf"\b{re.escape(str(word))}\b", str(text), flags=re.IGNORECASE) is not None
        df["_biased_hit"] = df.apply(lambda r: hit(r.get("completion",""), r.get("biased_word","")), axis=1)
        df["_anti_hit"]   = df.apply(lambda r: hit(r.get("completion",""), r.get("anti_word","")), axis=1)
        df["_answered"]   = df["_biased_hit"] | df["_anti_hit"]
    else:
        label_col = next((c for c in df.columns if df[c].astype(str).str.lower().isin(["biased","anti"]).any()), None)
        if label_col is None:
            return pd.DataFrame()
        s = df[label_col].astype(str).str.lower()
        df["_biased_hit"] = s.eq("biased").astype(int)
        df["_anti_hit"]   = s.eq("anti").astype(int)
        df["_answered"]   = 1
    gcols = [c for c in ["religion","axis","g_type"] if c in df.columns] or ["religion"]
    agg = (df[df["_answered"] == 1]
           .groupby(gcols, dropna=False)
           .agg(samples=("_answered","count"),
                biased_hits=("_biased_hit","sum"),
                anti_hits=("_anti_hit","sum"))
           .reset_index())
    denom = (agg["biased_hits"] + agg["anti_hits"]).replace(0,np.nan)
    agg["BPR"] = (agg["biased_hits"] / denom).clip(0,1)
    return agg

def compute_bpr_from_counts(df):
    df = df.copy()
    for c in ["biased_hits","anti_hits","samples"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "BPR" not in df.columns and {"biased_hits","anti_hits"}.issubset(df.columns):
        denom = (df["biased_hits"] + df["anti_hits"]).replace(0,np.nan)
        df["BPR"] = (df["biased_hits"] / denom).clip(0,1)
    gcols = [c for c in ["religion","axis","g_type"] if c in df.columns] or ["religion"]
    keep = gcols + [c for c in ["samples","biased_hits","anti_hits","BPR"] if c in df.columns]
    return df[keep]

def load_bpr_from_any(src):
    raw = read_tabular(src)
    if raw is None or raw.empty:
        return None, "empty-or-unreadable"
    with st.expander(f"📄 Detected columns ({getattr(src,'name',str(src))})", expanded=False):
        st.write(list(raw.columns)); st.write("Shape:", raw.shape); st.dataframe(raw.head(8))
    mapping = {
        "religion": ["religion","group_name","identity"],
        "axis": ["axis","context_axis","category"],
        "g_type": ["g_type","g","prompt_type","group"],
        "completion": ["completion","output","response","text"],
        "biased_word": ["biased_word","biased","target_word","biased_term"],
        "anti_word": ["anti_word","anti","counter_word","anti_term"],
        "is_biased": ["is_biased","biased_flag"],
        "is_unanswered": ["is_unanswered","unanswered","no_answer"],
        "biased_hits": ["biased_hits","biased_count","biased_total"],
        "anti_hits": ["anti_hits","anti_count","anti_total"],
        "samples": ["samples","n_samples","count","n"],
        "label": ["label","class","choice","selected","hit","bpr_label"]
    }
    df = coalesce_columns(raw, mapping)
    agg = compute_bpr_from_rows(df)
    if agg is not None and not agg.empty:
        return agg, "per-row"
    agg = compute_bpr_from_counts(df)
    if agg is not None and not agg.empty:
        return agg, "summary"
    return None, "could-not-infer"

def load_toxic_from_any(src):
    raw = read_tabular(src)
    if raw is None or raw.empty:
        return None, []
    mapping = {
        "religion": ["religion","group_name","identity"],
        "axis": ["axis","context_axis","category"],
        "g_type": ["g_type","g","prompt_type","group"],
        "samples": ["samples","n","count"]
    }
    df = coalesce_columns(raw, mapping)
    # find available perspective metrics
    avail = []
    for c in raw.columns:
        u = str(c).upper()
        if u in PERSPECTIVE_METRICS:
            df[u] = pd.to_numeric(raw[c], errors="coerce")
            avail.append(u)
    return (df if avail else None), avail

def auto_outputs_root():
    for p in ["/content/drive/My Drive/Capstone/Dashboard/Outputs",
              "/content/drive/MyDrive/Capstone/Dashboard/Outputs"]:
        if os.path.isdir(p):
            return p
    return "/content/drive/My Drive/Capstone/Dashboard/Outputs"

def find_any(folder, candidates, glob_pattern=None):
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    if glob_pattern:
        matches = sorted(glob.glob(os.path.join(folder, glob_pattern)), key=os.path.getmtime, reverse=True)
        return matches[0] if matches else None
    return None

def apply_filters(df, f_rel, f_axis, f_gt):
    if df is None:
        return None
    out = df.copy()
    if f_rel != "All" and "religion" in out:
        out = out[out["religion"] == f_rel]
    if f_axis != "All" and "axis" in out:
        out = out[out["axis"] == f_axis]
    if f_gt   != "All" and "g_type" in out:
        out = out[out["g_type"] == f_gt]
    return out

def reaggregate_bpr(df, dims):
    if df is None or df.empty:
        return df
    dims = [d for d in dims if d in df.columns] or ["religion"]
    cols = [c for c in ["samples","biased_hits","anti_hits"] if c in df.columns]
    if cols:
        agg = df.groupby(dims, dropna=False)[cols].sum().reset_index()
        denom = (agg["biased_hits"] + agg["anti_hits"]).replace(0,np.nan)
        agg["BPR"] = (agg["biased_hits"] / denom).clip(0,1)
        return agg
    if "BPR" in df.columns:
        if "samples" in df.columns:
            tmp = df.copy(); tmp["w"] = tmp["samples"].fillna(0)
            agg = tmp.groupby(dims, dropna=False).apply(
                lambda g: np.average(g["BPR"], weights=g["w"]) if g["w"].sum()>0 else g["BPR"].mean()
            ).reset_index(name="BPR")
        else:
            agg = df.groupby(dims, dropna=False)["BPR"].mean().reset_index()
        return agg
    return df

def reaggregate_tox(df, dims, metric, mode, thr):
    """mode: 'avg' or 'rate' (>= thr). Returns df with VALUE column."""
    if df is None or df.empty:
        return df
    dims = [d for d in dims if d in df.columns] or ["religion"]
    m = metric
    tmp = df.copy()
    if mode == "avg":
        agg = tmp.groupby(dims, dropna=False)[m].mean().reset_index().rename(columns={m:"VALUE"})
    else:  # rate
        flag = (tmp[m] >= thr).astype(int)
        agg = tmp.assign(_f=flag).groupby(dims, dropna=False)["_f"].mean().reset_index().rename(columns={"_f":"VALUE"})
    return agg

def show_plot(fig, static):
    if static:
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})
    else:
        st.plotly_chart(fig, use_container_width=True)

def bar(df, title, ycol, value_col="BPR", static=False):
    fig = px.bar(
        df.sort_values(value_col, ascending=True),
        x=value_col, y=ycol,
        color=df.get("g_type") if "g_type" in df.columns else None,
        orientation="h", title=title
    )
    show_plot(fig, static)

# New: toxicity-specific bar with correct axis labels
def bar_tox(df, title, ycol, mode, thr, static=False):
    """
    mode: 'avg' (average score) or 'rate' (% ≥ threshold)
    df must contain a 'VALUE' column
    """
    v = "VALUE"
    fig = px.bar(
        df.sort_values(v, ascending=True),
        x=v, y=ycol, orientation="h",
        title=title,
        color=df.get("g_type") if "g_type" in df.columns else None,
    )
    if mode == "rate":
        fig.update_layout(xaxis_tickformat=".0%")
        fig.update_xaxes(title_text=f"% ≥ {thr:.2f}")
    else:
        fig.update_layout(xaxis_tickformat=".2f")
        fig.update_xaxes(title_text="Average score")
    show_plot(fig, static)

# ---------- Sidebar (paths & uploads) ----------
with st.sidebar:
    st.header("Folders on Drive")
root = auto_outputs_root()
pre_dir  = st.text_input("PRE folder",  value=os.path.join(root, "Base_Model")).strip()
post_dir = st.text_input("POST folder", value=os.path.join(root, "Fine_Tuned_Model")).strip()
st.caption(f"Base Outputs: `{root}`")

st.sidebar.markdown("---")
st.sidebar.caption("Optional: upload files directly")
up_pre_bpr  = st.sidebar.file_uploader("Upload PRE BPR",  type=["csv","json","jsonl","ndjson"], key="up_pre_bpr")
up_post_bpr = st.sidebar.file_uploader("Upload POST BPR", type=["csv","json","jsonl","ndjson"], key="up_post_bpr")
up_pre_tox  = st.sidebar.file_uploader("Upload PRE Toxicity CSV",  type=["csv"], key="up_pre_tox")
up_post_tox = st.sidebar.file_uploader("Upload POST Toxicity CSV", type=["csv"], key="up_post_tox")

# Display/perf controls shared
st.sidebar.markdown("---")
st.sidebar.header("Display")
group_options = {
    "Religion": ["religion"],
    "Religion + Axis": ["religion","axis"],
    "Axis": ["axis"],
}
group_key = st.sidebar.selectbox("Group by", list(group_options.keys()), index=0)
top_n = st.sidebar.slider("Show top N rows", 10, 200, 50, 10)
sort_choice = st.sidebar.radio("Sort", ["Highest (worst)", "Lowest (best)", "Most samples"], horizontal=False)
split_by_g = st.sidebar.checkbox("Show separate panels for G1/G2/G3", value=True)

# ---------- Load BPR ----------
pre_bpr, pre_bpr_src, post_bpr, post_bpr_src = None, None, None, None
if up_pre_bpr is not None:
    pre_bpr, _ = load_bpr_from_any(up_pre_bpr); pre_bpr_src = f"Uploaded: {up_pre_bpr.name}"
elif os.path.isdir(pre_dir):
    p = find_any(pre_dir, CANDIDATE_BPR, glob_pattern="bpr_*")
    if p:
        pre_bpr, _ = load_bpr_from_any(p); pre_bpr_src = p

if up_post_bpr is not None:
    post_bpr, _ = load_bpr_from_any(up_post_bpr); post_bpr_src = f"Uploaded: {up_post_bpr.name}"
elif os.path.isdir(post_dir):
    p = find_any(post_dir, CANDIDATE_BPR, glob_pattern="bpr_*")
    if p:
        post_bpr, _ = load_bpr_from_any(p); post_bpr_src = p

domain = pre_bpr if pre_bpr is not None and not pre_bpr.empty else post_bpr

# ---------- Filters ----------
rels  = ["All"] + (sorted(list(domain["religion"].dropna().unique())) if (domain is not None and "religion" in domain) else [])
axes  = ["All"] + (sorted(list(domain["axis"].dropna().unique()))     if (domain is not None and "axis" in domain)     else [])
gts   = ["All"] + (sorted(list(domain["g_type"].dropna().unique()))   if (domain is not None and "g_type" in domain)   else [])
if not rels: rels = ["All"]
if not axes: axes = ["All"]
if not gts:  gts  = ["All"]

with st.sidebar:
    st.header("Filters")
    f_rel  = st.selectbox("Religion",    rels, index=0)
    f_axis = st.selectbox("Axis",        axes, index=0)
    f_gt   = st.selectbox("Prompt Type", gts,  index=0)

def filtered(df):
    return apply_filters(df, f_rel, f_axis, f_gt)

# ---------- Overview ----------
col1, col2 = st.columns(2)
if pre_bpr is not None and not pre_bpr.empty:
    col1.metric("PRE BPR groups", f"{pre_bpr.shape[0]}"); col1.caption(f"{pre_bpr_src}")
if post_bpr is not None and not post_bpr.empty:
    col2.metric("POST BPR groups", f"{post_bpr.shape[0]}"); col2.caption(f"{post_bpr_src}")

# ---------- BPR section (3 charts, with optional G tabs) ----------
def prep_show_bpr(df, dims):
    agg = reaggregate_bpr(df, dims)
    if agg is None or agg.empty:
        return agg
    if sort_choice == "Highest (worst)":
        return agg.sort_values(["BPR","samples" if "samples" in agg else agg.columns[0]], ascending=[False,False]).head(top_n)
    elif sort_choice == "Lowest (best)":
        return agg.sort_values(["BPR","samples" if "samples" in agg else agg.columns[0]], ascending=[True,False]).head(top_n)
    else:
        return agg.sort_values("samples", ascending=False).head(top_n) if "samples" in agg else agg.head(top_n)

def three_bpr(pre_df, post_df, dims, title_suffix=""):
    pre_show  = prep_show_bpr(pre_df, dims)  if pre_df  is not None else None
    post_show = prep_show_bpr(post_df, dims) if post_df is not None else None
    delta_show = None
    if (pre_show is not None and not pre_show.empty) and (post_show is not None and not post_show.empty):
        keys = [d for d in dims if d in post_show.columns and d in pre_show.columns]
        joined = post_show.merge(pre_show, on=keys, how="left", suffixes=("_post","_pre"))
        if {"BPR_post","BPR_pre"}.issubset(joined.columns):
            joined["ΔBPR"] = joined["BPR_post"] - joined["BPR_pre"]
            joined["Δ%"]   = (joined["ΔBPR"] / joined["BPR_pre"]).replace([np.inf,-np.inf], np.nan)
            delta_show = joined.sort_values("ΔBPR", ascending=True).head(top_n)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.subheader(f"PRE {title_suffix}")
        if pre_show is None or pre_show.empty:
            st.info("No PRE data.")
        else:
            bar(pre_show, "PRE BPR (lower = fewer biased hits)", dims[0], "BPR", static=len(pre_show)>80)
    with c2:
        st.subheader(f"POST {title_suffix}")
        if post_show is None or post_show.empty:
            st.info("No POST data.")
        else:
            bar(post_show, "POST BPR (lower = fewer biased hits)", dims[0], "BPR", static=len(post_show)>80)
    with c3:
        st.subheader(f"Δ (POST−PRE) {title_suffix}")
        if delta_show is None or delta_show.empty:
            st.info("Δ unavailable.")
        else:
            fig = px.bar(delta_show.sort_values("ΔBPR"), x="ΔBPR", y=dims[0], orientation="h",
                         title="ΔBPR (negative = improvement)")
            show_plot(fig, static=len(delta_show)>80)

st.header("BPR")
dims_base = group_options[group_key]
if split_by_g and (("g_type" in (pre_bpr.columns if pre_bpr is not None else [])) or ("g_type" in (post_bpr.columns if post_bpr is not None else []))):
    g_vals = []
    if pre_bpr is not None: g_vals += list(pre_bpr["g_type"].dropna().unique())
    if post_bpr is not None: g_vals += list(post_bpr["g_type"].dropna().unique())
    g_vals = sorted(sorted(set(g_vals))) or ["G1","G2","G3"]
    tabs = st.tabs(["All G"] + [f"{g}" for g in g_vals])
    with tabs[0]:
        three_bpr(filtered(pre_bpr), filtered(post_bpr), dims_base, "(All G)")
    for i,g in enumerate(g_vals, start=1):
        with tabs[i]:
            pre_f = filtered(pre_bpr)
            post_f = filtered(post_bpr)
            if pre_f is not None:  pre_f  = pre_f[pre_f["g_type"]==g]
            if post_f is not None: post_f = post_f[post_f["g_type"]==g]
            three_bpr(pre_f, post_f, dims_base, f"— {g}")
else:
    three_bpr(filtered(pre_bpr), filtered(post_bpr), dims_base)

# ---------- TOXICITY section ----------
st.markdown("---")
st.header("Toxicity (Perspective)")

# Load toxicity files
pre_tox, post_tox, pre_tox_src, post_tox_src = None, None, None, None
if up_pre_tox is not None:
    pre_tox, pre_avail = load_toxic_from_any(up_pre_tox); pre_tox_src = f"Uploaded: {up_pre_tox.name}"
elif os.path.isdir(pre_dir):
    p = find_any(pre_dir, CANDIDATE_TOX, glob_pattern="*Perspective*csv")
    if p:
        pre_tox, pre_avail = load_toxic_from_any(p); pre_tox_src = p
else:
    pre_avail = []

if up_post_tox is not None:
    post_tox, post_avail = load_toxic_from_any(up_post_tox); post_tox_src = f"Uploaded: {up_post_tox.name}"
elif os.path.isdir(post_dir):
    p = find_any(post_dir, CANDIDATE_TOX, glob_pattern="*Perspective*csv")
    if p:
        post_tox, post_avail = load_toxic_from_any(p); post_tox_src = p
else:
    post_avail = []

available_metrics = sorted(set((pre_avail or [])) | set((post_avail or [])))
if not available_metrics:
    st.info("No toxicity CSVs found yet. Expected columns include TOXICITY, SEVERE_TOXICITY, INSULT, THREAT, IDENTITY_ATTACK, PROFANITY.")
else:
    c1,c2 = st.columns(2)
    c1.caption(f"PRE source: {pre_tox_src or '—'}")
    c2.caption(f"POST source: {post_tox_src or '—'}")

    # Metric switcher (horizontal)
    metric = st.radio(
        "Metric",
        available_metrics,
        index=(available_metrics.index("TOXICITY") if "TOXICITY" in available_metrics else 0),
        horizontal=True,
    )

    mode_label = st.radio("Show", ["Average score", "Percent ≥ threshold"], horizontal=True)
    mode_key   = "avg" if mode_label == "Average score" else "rate"
    thr        = st.slider("Threshold", 0.00, 1.00, 0.50, 0.01, disabled=(mode_key=="avg"))
    dims_tox   = group_options[group_key]

    def prep_show_tox(df):
        if df is None or df.empty:
            return df
        agg = reaggregate_tox(df, dims_tox, metric, mode_key, thr)
        if agg is None or agg.empty:
            return agg
        if sort_choice == "Highest (worst)":
            return agg.sort_values("VALUE", ascending=False).head(top_n)
        elif sort_choice == "Lowest (best)":
            return agg.sort_values("VALUE", ascending=True).head(top_n)
        else:
            if "samples" in df.columns:
                s = df.groupby(dims_tox, dropna=False)["samples"].sum().reset_index()
                agg = agg.merge(s, on=dims_tox, how="left")
                return agg.sort_values("samples", ascending=False).head(top_n)
            return agg.head(top_n)

    def three_tox(pre_df, post_df, title_suffix=""):
        pre_show  = prep_show_tox(pre_df)  if pre_df  is not None else None
        post_show = prep_show_tox(post_df) if post_df is not None else None
        delta_show = None
        if (pre_show is not None and not pre_show.empty) and (post_show is not None and not post_show.empty):
            keys = [d for d in dims_tox if (d in pre_show.columns and d in post_show.columns)]
            j = post_show.merge(pre_show, on=keys, how="left", suffixes=("_post","_pre"))
            if {"VALUE_post","VALUE_pre"}.issubset(j.columns):
                j["Δ"] = j["VALUE_post"] - j["VALUE_pre"]
                delta_show = j.sort_values("Δ", ascending=True).head(top_n)
        c1,c2,c3 = st.columns(3)
        with c1:
            st.subheader(f"PRE {title_suffix}")
            if pre_show is None or pre_show.empty:
                st.info("No PRE toxicity.")
            else:
                bar_tox(
                    pre_show,
                    f"PRE {metric} ({'avg' if mode_key=='avg' else f'%≥{thr:.2f}'})",
                    dims_tox[0],
                    mode_key, thr,
                    static=len(pre_show)>80,
                )
        with c2:
            st.subheader(f"POST {title_suffix}")
            if post_show is None or post_show.empty:
                st.info("No POST toxicity.")
            else:
                bar_tox(
                    post_show,
                    f"POST {metric} ({'avg' if mode_key=='avg' else f'%≥{thr:.2f}'})",
                    dims_tox[0],
                    mode_key, thr,
                    static=len(post_show)>80,
                )
        with c3:
            st.subheader(f"Δ (POST−PRE) {title_suffix}")
            if delta_show is None or delta_show.empty:
                st.info("Δ unavailable.")
            else:
                fig = px.bar(
                    delta_show.sort_values("Δ"),
                    x="Δ", y=dims_tox[0], orientation="h",
                    title="Δ (negative = improvement)"
                )
                if mode_key == "rate":
                    fig.update_layout(xaxis_tickformat=".0%")
                    fig.update_xaxes(title_text="Δ (percentage points)")
                else:
                    fig.update_xaxes(title_text="Δ (avg score)")
                show_plot(fig, static=len(delta_show)>80)

    if split_by_g and (("g_type" in (pre_tox.columns if pre_tox is not None else [])) or ("g_type" in (post_tox.columns if post_tox is not None else []))):
        g_vals = []
        if pre_tox is not None: g_vals += list(pre_tox["g_type"].dropna().unique())
        if post_tox is not None: g_vals += list(post_tox["g_type"].dropna().unique())
        g_vals = sorted(sorted(set(g_vals))) or ["G1","G2","G3"]
        tabs = st.tabs(["All G"] + [f"{g}" for g in g_vals])
        with tabs[0]:
            three_tox(filtered(pre_tox), filtered(post_tox), "(All G)")
        for i,g in enumerate(g_vals, start=1):
            with tabs[i]:
                pre_f = filtered(pre_tox)
                post_f = filtered(post_tox)
                if pre_f is not None:  pre_f  = pre_f[pre_f["g_type"]==g]
                if post_f is not None: post_f = post_f[post_f["g_type"]==g]
                three_tox(pre_f, post_f, f"— {g}")
    else:
        three_tox(filtered(pre_tox), filtered(post_tox))

