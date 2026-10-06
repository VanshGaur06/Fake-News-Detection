"""TruthLens Streamlit demo."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.predict import MODEL_DIR, ModelNotTrainedError, load_artifacts, predict_article

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "outputs" / "metrics" / "evaluation.json"
st.set_page_config(page_title="TruthLens · News analysis", page_icon="◉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#e9f1fa; --muted:#91a4ba; --line:rgba(154,184,212,.15); --panel:#101d2b; --cyan:#55d6e8; }
.stApp { background: radial-gradient(ellipse at 72% -15%, #15354a 0, #0a1420 40%, #08111b 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#0b1723; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; color:var(--ink) !important; }
p,li,label { color:#bdcbd9; }
.hero { padding:1rem 0 1.5rem; }
.eyebrow { color:var(--cyan); letter-spacing:.15em; font-size:.72rem; font-weight:700; text-transform:uppercase; }
.hero h1 { font-size:3.6rem; line-height:1.06; margin:.5rem 0 .3rem; }
.hero p { font-size:1.05rem; max-width:740px; color:#aebdcb; }
.panel { background:linear-gradient(145deg,rgba(18,34,49,.94),rgba(13,25,37,.94)); border:1px solid var(--line); border-radius:18px; padding:1.35rem 1.5rem; box-shadow:0 14px 40px rgba(0,0,0,.15); margin:.5rem 0 1rem; }
.chip { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:.35rem .68rem; margin:.2rem .3rem .2rem 0; color:#b9ccdc; font-size:.78rem; }
.metric { background:#102131; border:1px solid var(--line); border-radius:15px; padding:1rem 1.1rem; }
.metric .label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.metric .value { color:var(--ink); font-size:1.75rem; font-weight:700; font-family:'Manrope',sans-serif; margin-top:.3rem; }
.result { border-radius:18px; border:1px solid var(--line); padding:1.4rem; margin:1rem 0; background:#102131; }
.result.fake { border-color:rgba(255,128,106,.5); background:linear-gradient(110deg,rgba(109,37,39,.34),rgba(16,33,49,.9)); }
.result.real { border-color:rgba(72,205,154,.45); background:linear-gradient(110deg,rgba(22,91,73,.3),rgba(16,33,49,.9)); }
.result-label { font-size:2rem; font-weight:800; font-family:'Manrope',sans-serif; }
.fine { font-size:.82rem; color:#91a4ba; }
button[kind="primary"] { background:#33aabd; border:0; color:#03151d; border-radius:10px; font-weight:700; }
button[kind="primary"]:hover { background:#55d6e8; color:#03151d; }
textarea { border-radius:12px !important; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

try:
    artifacts = load_artifacts()
    load_error = None
except ModelNotTrainedError as exc:
    artifacts, load_error = None, str(exc)
except Exception:
    artifacts = None
    load_error = "Saved model files could not be loaded. Check that the model artifacts are intact, then run: python src/train.py"

with st.sidebar:
    st.markdown("## ◉ TruthLens")
    st.caption("AI-POWERED NEWS ANALYSIS")
    page = st.radio("Workspace", ["Analyze", "Explainability", "Model Performance", "Methodology / About"], label_visibility="collapsed")
    st.divider()
    st.markdown("**ENGINE**")
    st.markdown('<span class="chip">NLP · TF-IDF</span><span class="chip">Classical ML</span><span class="chip">Local explanations</span>', unsafe_allow_html=True)
    st.caption("Status: Model ready" if artifacts else "Status: Training required")

st.markdown('<div class="hero"><div class="eyebrow">A transparent research demo</div><h1>TruthLens</h1><div style="font-size:1.25rem;color:#c5d5e3;font-weight:600">AI-Powered Fake News Detection</div><p>Analyze news articles using Natural Language Processing and Machine Learning to identify patterns associated with fake and real news.</p></div>', unsafe_allow_html=True)

if load_error:
    st.info("**Model not trained yet.** Add a labeled dataset as `data/dataset.csv`, then run `python src/train.py`. The expected columns are `label` and either `text`, `title`, or both.")

if page == "Analyze":
    st.markdown('<div class="panel"><div class="eyebrow">ARTICLE ANALYSIS</div><h2>Analyze a News Article</h2><p>Paste a headline, article body, or full article. The model looks for language patterns learned from its training examples.</p></div>', unsafe_allow_html=True)
    upload = st.file_uploader("Optional plain-text article", type=["txt"], help="UTF-8 .txt files only")
    initial = ""
    if upload:
        try:
            initial = upload.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.error("This text file could not be decoded as UTF-8. Save it as UTF-8 and upload again.")
    with st.form("analysis"):
        article = st.text_area("Article text", value=initial, placeholder="Paste a news article here...", height=245)
        submitted = st.form_submit_button("Analyze Article", type="primary", disabled=artifacts is None, use_container_width=True)
    if submitted:
        if not article.strip():
            st.warning("Paste an article before starting the analysis.")
        else:
            with st.spinner("Analyzing article…"):
                try:
                    result = predict_article(article, artifacts)
                    st.session_state["last_result"] = result
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Analysis could not be completed: {exc}")
    result = st.session_state.get("last_result")
    if result:
        label = result["label"]
        display_label = "⚠  FAKE PATTERN" if label == "FAKE" else "✓  REAL PATTERN"
        st.markdown(f'<div class="result {label.lower()}"><div class="eyebrow">ANALYSIS RESULT</div><div class="result-label">{display_label}</div><p>This label describes similarity to the training examples.</p></div>', unsafe_allow_html=True)
        a,b,c = st.columns(3)
        for column, title, value in [(a,"Model confidence",f"{result['confidence']:.1%}"),(b,"Selected model",result["model_name"]),(c,"Article length",f"{result['word_count']:,} words")]:
            column.markdown(f'<div class="metric"><div class="label">{title}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)
        st.progress(result["confidence"], text="Model confidence · based on its predicted class probability")
        if result["confidence"] < .60:
            st.caption("The model is not strongly confident. Consider this result uncertain.")
        st.caption("Confidence is a model probability estimate, not a measure of factual certainty.")

elif page == "Explainability":
    st.header("Why did the model make this prediction?")
    result = st.session_state.get("last_result")
    if result and result["explanation"]["available"]:
        st.caption("Each contribution is the article's TF-IDF feature value multiplied by the fitted linear model coefficient. Positive values point toward FAKE; negative values point toward REAL.")
        features = result["explanation"]["features"]
        if features:
            left, right = st.columns(2)
            for column, direction in [(left,"FAKE"),(right,"REAL")]:
                column.markdown(f"#### Features pushing toward {direction}")
                subset = [item for item in features if item["direction"] == direction]
                if not subset: column.caption("No strong active feature contributions in this direction.")
                for item in subset:
                    column.markdown(f"`{item['feature']}` · {item['contribution']:+.4f}")
                    column.progress(min(abs(item["contribution"]) / max(abs(x["contribution"]) for x in features), 1.0))
        else: st.caption("No influential vocabulary features were found in this article.")
    elif result: st.info("This model does not expose a linear coefficient explanation. Prediction and confidence remain available.")
    else: st.info("Analyze an article first to see its local feature contributions.")
    st.warning("These features influenced the model's classification based on patterns learned from the training data. They do not independently prove that the article is factually false.")

elif page == "Model Performance":
    st.header("Model Performance")
    if METRICS_PATH.exists():
        data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        best = data["metrics"]
        cols = st.columns(4)
        for col, label, key in zip(cols,["Accuracy","Precision","Recall","F1 score"],["accuracy","precision","recall","f1"]):
            col.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{best[key]:.1%}</div></div>',unsafe_allow_html=True)
        st.caption(f"Selected model: {data['model_name']} · Metrics use the held-out 20% test set. Positive class: FAKE.")
        st.subheader("Model comparison")
        comparison = pd.DataFrame([{ "Model": name, **{metric: values[metric] for metric in ("accuracy","precision","recall","f1")} } for name,values in data["all_models"].items()])
        st.dataframe(comparison.set_index("Model").style.format("{:.1%}"), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9,3.8), facecolor="#101d2b")
        comparison.set_index("Model")[["accuracy","precision","recall","f1"]].plot(kind="bar", ax=ax, color=["#55d6e8","#78b7ed","#62d2a3","#c6a5f5"], width=.78)
        ax.set_facecolor("#101d2b"); ax.set_ylim(0,1); ax.set_ylabel("Score", color="#bdcbd9"); ax.tick_params(colors="#bdcbd9", axis="both"); ax.legend(frameon=False, labelcolor="#bdcbd9", ncol=4, loc="upper center", bbox_to_anchor=(.5,1.16)); ax.spines[:].set_color("#33495b"); ax.grid(axis="y",alpha=.15); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.subheader("Selected model confusion matrix")
        matrix = best["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5.2,3.7), facecolor="#101d2b")
        sns.heatmap(matrix, annot=True, fmt="d", cmap="crest", cbar=False, xticklabels=["FAKE","REAL"], yticklabels=["FAKE","REAL"], ax=ax, linewidths=1, linecolor="#101d2b")
        ax.set_facecolor("#101d2b"); ax.set_xlabel("Predicted label", color="#bdcbd9"); ax.set_ylabel("Actual label", color="#bdcbd9"); ax.tick_params(colors="#bdcbd9"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.subheader("Confusion matrices by model")
        matrix_columns = st.columns(2)
        for index, (name, values) in enumerate(data["all_models"].items()):
            with matrix_columns[index % 2]:
                st.markdown(f"**{name}**")
                fig, ax = plt.subplots(figsize=(4, 3), facecolor="#101d2b")
                sns.heatmap(values["confusion_matrix"], annot=True, fmt="d", cmap="crest", cbar=False,
                            xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax,
                            linewidths=1, linecolor="#101d2b")
                ax.set_facecolor("#101d2b"); ax.set_xlabel("Predicted", color="#bdcbd9"); ax.set_ylabel("Actual", color="#bdcbd9"); ax.tick_params(colors="#bdcbd9"); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        cv = data["cross_validation"][data["model_name"]]
        st.subheader("Five-fold cross-validation on training data")
        x,y,z = st.columns(3)
        for col,label,key in [(x,"Mean accuracy","mean_accuracy"),(y,"Mean FAKE-class F1","mean_f1"),(z,"Accuracy standard deviation","std_accuracy")]: col.metric(label,f"{cv[key]:.1%}")
        st.metric("Majority-class baseline accuracy",f"{data['baseline_accuracy']:.1%}")
    else: st.info("Evaluation results will appear after successful training.")

else:
    st.header("Methodology / About")
    st.markdown('<div class="panel"><div class="eyebrow">THE PIPELINE</div><h3>From article to explainable prediction</h3><p>Dataset → Text preprocessing → TF-IDF → Four classifiers → Held-out evaluation → Best model → Prediction → Feature contribution</p></div>',unsafe_allow_html=True)
    cols = st.columns(4)
    for col, title, body in zip(cols,["01 · Dataset","02 · Preprocessing","03 · TF-IDF","04 · Classifiers"],["Labeled examples are loaded from a CSV with text and label fields.","Title and body are joined; URLs, markup, punctuation and common stopwords are removed.","Unigrams and bigrams become sparse term-importance features, fitted only on training folds.","Naive Bayes, Logistic Regression, Linear SVM and Random Forest are compared."]):
        col.markdown(f'<div class="panel"><div class="eyebrow">{title}</div><p>{body}</p></div>',unsafe_allow_html=True)
    st.markdown("### Evaluation and model selection")
    st.write("An 80/20 stratified split holds out the test set. Hyperparameters are selected within the training portion using five-fold cross-validation. Models are ranked by test F1, then precision, recall, and accuracy; report these results as an evaluation of this dataset, not as a guarantee of real-world truth.")
    st.markdown("### Limitations")
    st.write("News language, publishers, topics, and labeling practices can change. Exact duplicate articles are removed, but dataset bias and near duplicates may remain. Model confidence describes estimated class probability, not factual certainty.")

st.markdown('<div class="panel fine"><b>Important disclaimer</b><br>TruthLens predicts whether an article resembles Fake or Real examples based on learned textual patterns. It does not independently verify facts against the real world and should not be treated as a definitive fact-checking system.</div>',unsafe_allow_html=True)
