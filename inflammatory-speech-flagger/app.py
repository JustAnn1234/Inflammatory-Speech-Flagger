import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import os
import altair as alt

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inflammatory Speech Flagger",
    page_icon="🚩",
    layout="wide",
)

MODEL_PATH = "models/inflammatory_flagger_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

EXAMPLES = [
    "This particular tribe don spoil this country, una all no good",
    "Congrats to everyone graduating today, God bless una plenty",
    "@USER @USER I'm in Akwa ibom state, you guys r very stupid",
    "pastor dey shout your children will suck your blood, come and be saved",
    "They citizens, wannabe call patriotic ones",
]


# ── Load model (cached so it only loads once per session) ──────────────────
@st.cache_resource
def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_model()


# ── Same preprocessing/explanation logic as the training notebook ─────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def explain_prediction(raw_text: str, top_k: int = 6):
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0][1]
    flag = bool(prob >= 0.5)

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    present_idx = vec.nonzero()[1]

    if len(present_idx) == 0:
        return flag, prob, []

    word_scores = [(feature_names[i], coefs[i]) for i in present_idx]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    top_pos = [w for w, s in word_scores if s > 0][:top_k]
    top_neg = [w for w, s in word_scores if s < 0][-top_k:]
    return flag, prob, {"toward_flag": top_pos, "away_from_flag": top_neg}


def flag_post(raw_text: str) -> dict:
    flag, prob, words = explain_prediction(raw_text)
    return {
        "text": raw_text,
        "flag": flag,
        "probability": round(float(prob), 3),
        "toward_flag": ", ".join(words.get("toward_flag", [])) or "—",
        "away_from_flag": ", ".join(words.get("away_from_flag", [])) or "—",
    }


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚩 Speech Flagger")
    st.caption("Capstone — AI & ML NextGen Cohort · Grazac Technologies")
    st.markdown("---")
    st.markdown(
        "**What this does**\n\n"
        "Takes a Nigerian social media post (English, Pidgin, Hausa, Igbo, or Yorùbá) "
        "and predicts whether it's likely inflammatory, along with the words that drove "
        "the decision."
    )
    st.markdown("---")
    if model is not None:
        st.success("Model loaded ✅")
    else:
        st.error("Model files not found — see setup note below.")
    st.markdown(
        "**Model:** TF-IDF + Logistic Regression\n\n"
        "**Trained on:** AfriHate (Nigerian Pidgin, Hausa, Igbo, Yorùbá)"
    )
    st.markdown("---")
    st.markdown("[📓 Notebook & repo](#) · [🎥 Demo video](#)")

if model is None:
    st.warning(
        "Couldn't find `models/inflammatory_flagger_model.joblib` and "
        "`models/tfidf_vectorizer.joblib`. Run the training notebook first, "
        "then copy those two files into a `models/` folder next to this app.py "
        "before deploying."
    )
    st.stop()

# ── Main tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Check a Post", "📋 Batch Check (CSV)", "📊 About the Model"])

# ---- TAB 1: single post checker -------------------------------------------------
with tab1:
    st.subheader("Check a single post")

    col_input, col_examples = st.columns([3, 1])
    with col_examples:
        st.markdown("**Try an example:**")
        chosen_example = st.selectbox(
            "Examples", ["— choose one —"] + EXAMPLES, label_visibility="collapsed"
        )

    default_text = "" if chosen_example == "— choose one —" else chosen_example
    with col_input:
        user_text = st.text_area(
            "Enter a post to analyze",
            value=default_text,
            height=120,
            placeholder="Paste or type a post here...",
        )

    threshold = st.slider(
        "Flagging threshold (confidence needed to flag as inflammatory)",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower = flags more posts (higher recall, more false positives). "
             "Higher = flags fewer posts (higher precision, more missed cases).",
    )

    analyze_clicked = st.button("Analyze post", type="primary", use_container_width=False)

    if analyze_clicked and user_text.strip():
        flag, prob, words = explain_prediction(user_text)
        flag = bool(prob >= threshold)

        result_col, gauge_col = st.columns([2, 1])

        with result_col:
            if flag:
                st.error(f"🚩 **Flagged as potentially inflammatory** — confidence {prob:.0%}")
            else:
                st.success(f"✅ **Not flagged** — confidence {(1 - prob):.0%} not inflammatory")

            st.markdown("**Reason — words pushing toward a flag:**")
            st.code(", ".join(words.get("toward_flag", [])) or "none detected", language=None)

            st.markdown("**Words pushing away from a flag:**")
            st.code(", ".join(words.get("away_from_flag", [])) or "none detected", language=None)

            with st.expander("Cleaned text used by the model"):
                st.write(clean_text(user_text))

        with gauge_col:
            st.metric("Inflammatory probability", f"{prob:.0%}")
            st.progress(min(max(prob, 0.0), 1.0))
            chart_df = pd.DataFrame({
                "class": ["not inflammatory", "inflammatory"],
                "probability": [1 - prob, prob],
            })
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("class:N", sort="-x"),
                    color=alt.Color(
                        "class:N",
                        scale=alt.Scale(domain=["not inflammatory", "inflammatory"],
                                         range=["#2ecc71", "#e74c3c"]),
                        legend=None,
                    ),
                )
                .properties(height=140)
            )
            st.altair_chart(chart, use_container_width=True)

    elif analyze_clicked:
        st.info("Type or paste a post first.")

# ---- TAB 2: batch CSV checker -------------------------------------------------
with tab2:
    st.subheader("Check many posts at once")
    st.markdown(
        "Upload a CSV with a column called **text** (one post per row). "
        "The app will flag each one and let you download the results."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            batch_df = None

        if batch_df is not None:
            if "text" not in batch_df.columns:
                st.error("Your CSV needs a column named 'text'. Found columns: "
                          + ", ".join(batch_df.columns))
            else:
                with st.spinner(f"Analyzing {len(batch_df)} posts..."):
                    results = [flag_post(t) for t in batch_df["text"].astype(str)]
                    results_df = pd.DataFrame(results)

                st.success(f"Done — {results_df['flag'].sum()} of {len(results_df)} posts flagged.")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total posts", len(results_df))
                with col_b:
                    st.metric("Flagged as inflammatory", int(results_df["flag"].sum()))

                st.dataframe(
                    results_df[["text", "flag", "probability", "toward_flag"]],
                    use_container_width=True,
                    height=400,
                )

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results as CSV",
                    data=csv_bytes,
                    file_name="flagged_posts_results.csv",
                    mime="text/csv",
                )

# ---- TAB 3: about / model info -------------------------------------------------
with tab3:
    st.subheader("About this model")
    st.markdown(
        """
**Problem:** Platforms need to flag hate/inflammatory posts in a Nigerian context, where
content mixes English, Nigerian Pidgin, and local languages.

**Approach:**
- Dataset: [AfriHate](https://huggingface.co/datasets/afrihate/afrihate) — real annotated
  tweets across Nigerian Pidgin, Hausa, Igbo, and Yorùbá (`Hate`/`Abuse` → flagged,
  `Normal` → not flagged).
- Preprocessing: light cleaning (URLs, mentions, punctuation removed) that preserves slang
  and code-switching, since that carries meaning here.
- Features: TF-IDF (unigrams + bigrams).
- Model: Logistic Regression (`class_weight='balanced'`).
- Explainability: the "reason" for each flag is the set of words in the post with the
  highest learned weight toward/away from the inflammatory class — a simple, interpretable
  proxy for reasoning, not true contextual understanding.

**Evaluation results (held-out test set):**
"""
    )
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (inflammatory)", "Recall (inflammatory)", "Macro F1"],
        "Score": ["0.80", "0.86", "0.77", "0.80"],
    })
    st.table(metrics_df)

    st.markdown(
        """
**Known limitations:**
- Trained on Twitter/X-style short posts — may not generalize to other platforms.
- ~58/42 class split between inflammatory and neutral — some imbalance remains despite
  `class_weight='balanced'`.
- Nearly half the training data is Nigerian Pidgin, so performance likely skews stronger
  there than on pure Hausa, Igbo, or Yorùbá text.
- Word-importance "reasons" can misfire on sarcasm, quotes, or reclaimed language.
- Any hate-speech classifier risks inheriting bias from human annotators — false positives
  can suppress legitimate speech, false negatives can miss real harm. This is a baseline
  academic model, not a production moderation system.
"""
    )
