import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- COLOUR PALETTE ----------------
BENIGN = "#2EC4B6"      # teal
MALIGNANT = "#FF5C8A"   # pink-red
ACCENT = "#7C5CFF"      # violet
GOLD = "#FFC857"
BG_CARD = "rgba(255,255,255,0.05)"
COLOR_MAP = {"Benign": BENIGN, "Malignant": MALIGNANT}

# ---------------- CUSTOM CSS ----------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at 10% 0%, #1b1740 0%, #0f1226 40%, #0b0d1a 100%);
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1a1446 0%, #0e1030 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }}
    .hero {{
        padding: 28px 34px;
        border-radius: 20px;
        background: linear-gradient(120deg, {ACCENT} 0%, #c44fd8 50%, {MALIGNANT} 100%);
        box-shadow: 0 10px 40px rgba(124,92,255,0.35);
        margin-bottom: 22px;
    }}
    .hero h1 {{ color: white; margin: 0; font-size: 2.3rem; }}
    .hero p  {{ color: rgba(255,255,255,0.88); margin: 6px 0 0 0; font-size: 1.05rem; }}
    .kpi {{
        background: {BG_CARD};
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 18px 20px;
        text-align: left;
        transition: transform .2s ease, box-shadow .2s ease;
        backdrop-filter: blur(6px);
    }}
    .kpi:hover {{ transform: translateY(-4px); box-shadow: 0 10px 28px rgba(0,0,0,0.45); }}
    .kpi .label {{ color: #aab0d0; font-size: .85rem; letter-spacing: .04em; text-transform: uppercase; }}
    .kpi .value {{ font-size: 2.1rem; font-weight: 700; margin-top: 4px; }}
    .kpi .sub   {{ color: #8b91b5; font-size: .8rem; }}
    .section-title {{
        font-size: 1.35rem; font-weight: 700; margin: 10px 0 4px 0;
        padding-left: 12px; border-left: 5px solid {ACCENT};
    }}
    .result-box {{
        border-radius: 18px; padding: 22px 26px; margin: 8px 0 16px 0;
        font-size: 1.5rem; font-weight: 700; color: white;
    }}
    .result-benign {{ background: linear-gradient(120deg, #13867c, {BENIGN}); }}
    .result-malignant {{ background: linear-gradient(120deg, #c2275a, {MALIGNANT}); }}
    div[data-testid="stTabs"] button[role="tab"] {{ font-size: 1rem; font-weight: 600; }}
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
        border-radius: 12px; border: none; font-weight: 600;
        background: linear-gradient(120deg, {ACCENT}, #c44fd8); color: white;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{ filter: brightness(1.15); color: white; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- HELPERS ----------------
def show(fig, height=None):
    """Display a plotly figure with a consistent dark theme."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(data, **kwargs):
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def kpi(col, label, value, sub="", color="#ffffff"):
    col.markdown(
        f"""<div class="kpi">
              <div class="label">{label}</div>
              <div class="value" style="color:{color}">{value}</div>
              <div class="sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# FILE LOADING
# =====================================================================
def find_file(filename):
    files = list(Path(".").rglob(filename))
    if not files:
        st.error(f"Required file not found: {filename}")
        st.stop()
    return files[0]


@st.cache_resource
def load_model():
    return joblib.load(find_file("breast_cancer_model.pkl"))


@st.cache_data
def load_features():
    return list(joblib.load(find_file("feature_names.pkl")))


@st.cache_data
def load_data():
    raw = pd.read_csv(find_file("wdbc.data"), header=None)
    feats = load_features()

    if raw.shape[1] != 32 or len(feats) != 30:
        raise ValueError("Unexpected dataset structure (need 32 columns / 30 features).")

    raw.columns = ["id", "diagnosis"] + feats
    raw["diagnosis"] = raw["diagnosis"].astype(str).str.strip()

    for col in feats:
        raw[col] = pd.to_numeric(raw[col], errors="raise")

    raw["target"] = raw["diagnosis"].map({"B": 0, "M": 1})
    if raw["target"].isna().any():
        raise ValueError("Unexpected diagnosis labels in wdbc.data.")

    raw["Diagnosis"] = raw["diagnosis"].map({"B": "Benign", "M": "Malignant"})
    return raw


@st.cache_data
def load_metrics():
    return pd.read_csv(find_file("model_evaluation.csv"))


@st.cache_data
def load_predictions():
    return pd.read_csv(find_file("prediction_results.csv"))


model = load_model()
features = load_features()
df = load_data()
metrics = load_metrics()
results = load_predictions()

mean_feats = [f for f in features if f.startswith("mean")]
error_feats = [f for f in features if f.endswith("error")]
worst_feats = [f for f in features if f.startswith("worst")]


def get_coefficients():
    """Return logistic-regression coefficients (works for Pipeline or plain model)."""
    try:
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        return np.asarray(est.coef_[0])
    except Exception:
        return None


def transform_input(input_df):
    """Apply the preprocessing steps (e.g. scaler) if the model is a Pipeline."""
    try:
        if hasattr(model, "steps") and len(model.steps) > 1:
            return np.asarray(model[:-1].transform(input_df))[0]
    except Exception:
        pass
    return input_df.to_numpy()[0]


# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.markdown("## 🎗️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Data Analytics", "📈 Model Performance", "🔬 Prediction"],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("### Quick Facts")
st.sidebar.markdown(
    f"- **Algorithm:** Logistic Regression\n"
    f"- **Records:** {len(df)}\n"
    f"- **Features:** {len(features)}\n"
    f"- **Dataset:** WDBC (UCI)"
)
st.sidebar.divider()
st.sidebar.caption("⚠️ Educational project – not for clinical use.")


# =====================================================================
# OVERVIEW
# =====================================================================
if page == "🏠 Overview":

    hero("🎗️ Breast Cancer Detection",
         "Machine-learning classification of tumours as Benign or Malignant using Logistic Regression")

    st.write(
        "This project applies **Logistic Regression** to the Wisconsin Diagnostic Breast "
        "Cancer dataset. Each record contains **30 numerical measurements** computed from a "
        "digitised image of a fine-needle aspirate (FNA) of a breast mass."
    )

    n_b = int((df["diagnosis"] == "B").sum())
    n_m = int((df["diagnosis"] == "M").sum())

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Records", len(df), "patient samples", "#ffffff")
    kpi(c2, "Features", len(features), "mean · error · worst", ACCENT)
    kpi(c3, "Benign", n_b, f"{n_b / len(df):.1%} of data", BENIGN)
    kpi(c4, "Malignant", n_m, f"{n_m / len(df):.1%} of data", MALIGNANT)

    st.write("")
    left, right = st.columns([1, 1])

    with left:
        section("Diagnosis Distribution")
        counts = df["Diagnosis"].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index, hole=0.55,
            color=counts.index, color_discrete_map=COLOR_MAP,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14,
                          marker=dict(line=dict(color="#0f1226", width=3)))
        fig.add_annotation(text=f"<b>{len(df)}</b><br>cases", showarrow=False, font_size=20)
        fig.update_layout(showlegend=False)
        show(fig, 380)

    with right:
        section("Class Averages – Key Features")
        key = ["mean radius", "mean texture", "mean perimeter", "mean concavity", "mean concave points"]
        key = [k for k in key if k in features]
        avg = df.groupby("Diagnosis")[key].mean().reset_index().melt(
            id_vars="Diagnosis", var_name="Feature", value_name="Average")
        fig = px.bar(avg, x="Feature", y="Average", color="Diagnosis", barmode="group",
                     color_discrete_map=COLOR_MAP)
        fig.update_layout(legend_title_text="")
        show(fig, 380)

    section("Dataset Preview")
    preview = df.drop(columns=["target", "diagnosis"]).head(10)
    show_df(preview)

    with st.expander("📋 Statistical summary of all features"):
        show_df(df[features].describe().T.round(3))


# =====================================================================
# DATA ANALYTICS
# =====================================================================
elif page == "📊 Data Analytics":

    hero("📊 Data Analytics", "Explore how each tumour measurement differs between Benign and Malignant cases")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📉 Distribution", "🔗 Relationships", "🌡️ Correlation", "🕸️ Radar Profile", "🏆 Top Features"]
    )

    # ---- Distribution ----
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            feat = st.selectbox("Choose a feature", features, key="dist_feat")
        with c2:
            chart_type = st.radio("Chart type", ["Histogram", "Violin", "Box"], horizontal=True)

        if chart_type == "Histogram":
            bins = st.slider("Number of bins", 10, 60, 30)
            fig = px.histogram(df, x=feat, color="Diagnosis", nbins=bins, barmode="overlay",
                               opacity=0.75, marginal="box", color_discrete_map=COLOR_MAP,
                               title=f"Distribution of {feat}")
        elif chart_type == "Violin":
            fig = px.violin(df, x="Diagnosis", y=feat, color="Diagnosis", box=True, points="all",
                            color_discrete_map=COLOR_MAP, title=f"{feat} by Diagnosis")
        else:
            fig = px.box(df, x="Diagnosis", y=feat, color="Diagnosis", points="all",
                         color_discrete_map=COLOR_MAP, title=f"{feat} by Diagnosis")
        show(fig, 520)

        g = df.groupby("Diagnosis")[feat].agg(["mean", "median", "std", "min", "max"]).round(3)
        show_df(g)

    # ---- Relationships ----
    with tab2:
        c1, c2, c3 = st.columns(3)
        x_feat = c1.selectbox("X-axis", features, index=0)
        y_feat = c2.selectbox("Y-axis", features, index=1)
        size_opt = c3.selectbox("Bubble size (optional)", ["None"] + features)

        fig = px.scatter(
            df, x=x_feat, y=y_feat, color="Diagnosis",
            size=None if size_opt == "None" else size_opt,
            color_discrete_map=COLOR_MAP, opacity=0.8,
            trendline="ols" if st.checkbox("Show trend lines") else None,
            hover_data=["id"], title=f"{x_feat} vs {y_feat}",
        )
        show(fig, 520)

        st.markdown("#### 🧊 3D Scatter")
        d1, d2, d3 = st.columns(3)
        fx = d1.selectbox("X", features, index=0, key="3dx")
        fy = d2.selectbox("Y", features, index=1, key="3dy")
        fz = d3.selectbox("Z", features, index=2, key="3dz")
        fig3d = px.scatter_3d(df, x=fx, y=fy, z=fz, color="Diagnosis",
                              color_discrete_map=COLOR_MAP, opacity=0.8)
        fig3d.update_traces(marker=dict(size=4))
        show(fig3d, 600)

    # ---- Correlation ----
    with tab3:
        group = st.radio("Feature group", ["Mean", "Error", "Worst", "Custom"], horizontal=True)
        if group == "Mean":
            sel = mean_feats
        elif group == "Error":
            sel = error_feats
        elif group == "Worst":
            sel = worst_feats
        else:
            sel = st.multiselect("Select features", features, default=features[:8])

        if len(sel) >= 2:
            corr = df[sel].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto", title="Feature Correlation Matrix")
            show(fig, 650)
        else:
            st.info("Select at least two features.")

    # ---- Radar ----
    with tab4:
        st.write("Compare the **average profile** of Benign and Malignant tumours "
                 "(values scaled 0–1 for comparison).")
        radar_feats = st.multiselect("Features on radar", features,
                                     default=mean_feats if mean_feats else features[:10])
        if len(radar_feats) >= 3:
            scaled = (df[radar_feats] - df[radar_feats].min()) / (df[radar_feats].max() - df[radar_feats].min())
            scaled["Diagnosis"] = df["Diagnosis"]
            prof = scaled.groupby("Diagnosis").mean()

            fig = go.Figure()
            for cls, colr in COLOR_MAP.items():
                vals = prof.loc[cls, radar_feats].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=vals + vals[:1], theta=radar_feats + radar_feats[:1],
                    fill="toself", name=cls, line=dict(color=colr),
                    opacity=0.6))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            show(fig, 600)
        else:
            st.info("Select at least three features.")

    # ---- Top features ----
    with tab5:
        st.write("Features most strongly correlated with a **Malignant** diagnosis.")
        corr_t = df[features].corrwith(df["target"]).sort_values()
        top = pd.concat([corr_t.head(8), corr_t.tail(8)])
        cdf = pd.DataFrame({"Feature": top.index, "Correlation": top.values})
        cdf["Direction"] = np.where(cdf["Correlation"] > 0, "Towards Malignant", "Towards Benign")
        fig = px.bar(cdf, x="Correlation", y="Feature", orientation="h", color="Direction",
                     color_discrete_map={"Towards Malignant": MALIGNANT, "Towards Benign": BENIGN})
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        show(fig, 560)


# =====================================================================
# MODEL PERFORMANCE
# =====================================================================
elif page == "📈 Model Performance":

    hero("📈 Model Performance", "How well does the Logistic Regression model classify tumours?")

    # ---- metric cards ----
    section("Evaluation Metrics")
    m = metrics.copy()
    m.columns = ["Metric", "Score"] + list(m.columns[2:])
    palette = [ACCENT, BENIGN, GOLD, MALIGNANT, "#4CC9F0"]
    cols = st.columns(len(m))
    for i, (_, row) in enumerate(m.iterrows()):
        kpi(cols[i], row["Metric"], f"{float(row['Score']):.1%}", "test set", palette[i % len(palette)])

    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(m, x="Metric", y="Score", color="Metric", text=m["Score"].map(lambda v: f"{float(v):.3f}"),
                     color_discrete_sequence=palette, title="Metric Comparison")
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1.05]))
        show(fig, 380)
    with c2:
        acc_row = m[m["Metric"].str.lower() == "accuracy"]
        acc = float(acc_row["Score"].iloc[0]) if not acc_row.empty else float(m["Score"].iloc[0])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=acc * 100,
            number=dict(suffix="%"), title=dict(text="Model Accuracy"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=ACCENT),
                       steps=[dict(range=[0, 70], color="rgba(255,92,138,.35)"),
                              dict(range=[70, 90], color="rgba(255,200,87,.35)"),
                              dict(range=[90, 100], color="rgba(46,196,182,.35)")])))
        show(gauge, 380)

    st.divider()

    # ---- confusion matrix ----
    def normalize_label(value):
        value = str(value).strip().lower()
        if value in ["benign", "b", "0"]:
            return 0
        if value in ["malignant", "m", "1"]:
            return 1
        raise ValueError(f"Unknown prediction label: {value}")

    y_true = results["Actual"].map(normalize_label)
    y_pred = results["Predicted"].map(normalize_label)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    section("Confusion Matrix")
    c1, c2 = st.columns([3, 2])
    with c1:
        view = st.radio("Show as", ["Counts", "Percentage of actual class"], horizontal=True)
        z = cm if view == "Counts" else cm / cm.sum(axis=1, keepdims=True) * 100
        text = [[f"{cm[i][j]}" if view == "Counts" else f"{z[i][j]:.1f}%" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Benign", "Malignant"], y=["Benign", "Malignant"],
            text=text, texttemplate="%{text}", textfont=dict(size=26),
            colorscale=[[0, "#1b1740"], [1, ACCENT]], showscale=False))
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual",
                          yaxis=dict(autorange="reversed"), title="Confusion Matrix")
        show(fig, 430)
    with c2:
        st.write("")
        kpi(st, "True Negatives", tn, "Benign correctly identified", BENIGN)
        st.write("")
        kpi(st, "True Positives", tp, "Malignant correctly identified", BENIGN)
        st.write("")
        kpi(st, "False Positives", fp, "Benign predicted as Malignant", GOLD)
        st.write("")
        kpi(st, "False Negatives", fn, "Malignant missed (most critical)", MALIGNANT)

    st.divider()

    # ---- ROC + coefficients ----
    c1, c2 = st.columns(2)
    with c1:
        section("ROC Curve")
        try:
            proba = model.predict_proba(df[features])[:, 1]
            fpr, tpr, _ = roc_curve(df["target"], proba)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})",
                                     line=dict(color=ACCENT, width=4), fill="tozeroy",
                                     fillcolor="rgba(124,92,255,0.2)"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                     line=dict(color="grey", dash="dash")))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            show(fig, 430)
            st.caption("Computed on the full dataset (train + test) for visualisation.")
        except Exception as e:
            st.info(f"ROC curve unavailable: {e}")

    with c2:
        section("Feature Importance (Model Coefficients)")
        coefs = get_coefficients()
        if coefs is not None and len(coefs) == len(features):
            cdf = pd.DataFrame({"Feature": features, "Coefficient": coefs})
            cdf["Abs"] = cdf["Coefficient"].abs()
            cdf = cdf.sort_values("Abs", ascending=False).head(12)
            cdf["Effect"] = np.where(cdf["Coefficient"] > 0, "Increases Malignant risk", "Increases Benign likelihood")
            fig = px.bar(cdf, x="Coefficient", y="Feature", orientation="h", color="Effect",
                         color_discrete_map={"Increases Malignant risk": MALIGNANT,
                                             "Increases Benign likelihood": BENIGN})
            fig.update_layout(yaxis=dict(categoryorder="total ascending"), legend_title_text="")
            show(fig, 430)
        else:
            st.info("Coefficients could not be extracted from the model.")

    st.divider()
    section("Prediction Results")
    filt = st.selectbox("Filter", ["All", "Correct predictions", "Wrong predictions"])
    res = results.copy()
    correct = y_true.values == y_pred.values
    if filt == "Correct predictions":
        res = res[correct]
    elif filt == "Wrong predictions":
        res = res[~correct]
    show_df(res)


# =====================================================================
# PREDICTION
# =====================================================================
elif page == "🔬 Prediction":

    hero("🔬 Breast Cancer Prediction", "Enter tumour measurements or upload a CSV to get an instant model prediction")
    st.warning("This application is an educational project and is not intended for clinical diagnosis.")

    option = st.radio("Choose Prediction Method", ["✍️ Manual Input", "📁 Upload CSV"], horizontal=True)

    # ------------------------------------------------------------------
    # MANUAL
    # ------------------------------------------------------------------
    if option == "✍️ Manual Input":

        medians = df[features].median()
        benign_avg = df[df["target"] == 0][features].mean()
        malig_avg = df[df["target"] == 1][features].mean()

        # initialise widget state
        for f in features:
            if f"in_{f}" not in st.session_state:
                st.session_state[f"in_{f}"] = float(medians[f])

        def load_values(series):
            for f in features:
                st.session_state[f"in_{f}"] = float(series[f])

        def load_random(target):
            sample = df[df["target"] == target].sample(1).iloc[0]
            load_values(sample)

        section("Quick fill")
        b1, b2, b3, b4 = st.columns(4)
        b1.button("📊 Dataset median", on_click=load_values, args=(medians,))
        b2.button("🟢 Random benign sample", on_click=load_random, args=(0,))
        b3.button("🔴 Random malignant sample", on_click=load_random, args=(1,))
        b4.button("🟣 Average malignant", on_click=load_values, args=(malig_avg,))

        section("Enter Tumour Measurements")

        with st.form("prediction_form"):
            input_values = {}
            groups = [("📏 Mean values", mean_feats),
                      ("📐 Standard-error values", error_feats),
                      ("⚠️ Worst values", worst_feats)]
            tabs = st.tabs([g[0] for g in groups])
            for tab, (_, flist) in zip(tabs, groups):
                with tab:
                    cols = st.columns(3)
                    for i, f in enumerate(flist):
                        with cols[i % 3]:
                            input_values[f] = st.number_input(
                                f, key=f"in_{f}", format="%.6f",
                                min_value=0.0, step=float(max(df[f].std() / 20, 1e-4)),
                                help=f"Dataset range: {df[f].min():.4f} – {df[f].max():.4f}",
                            )
            submitted = st.form_submit_button("🔍 Predict Diagnosis")

        if submitted:
            input_df = pd.DataFrame([input_values], columns=features)
            prediction = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            p_ben, p_mal = float(probs[0]), float(probs[1])

            section("Prediction Result")

            if prediction == 1:
                st.markdown(f'<div class="result-box result-malignant">🔴 Prediction: Malignant '
                            f'({p_mal:.1%} confidence)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box result-benign">🟢 Prediction: Benign '
                            f'({p_ben:.1%} confidence)</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=p_mal * 100, number=dict(suffix="%"),
                    title=dict(text="Malignancy Probability"),
                    gauge=dict(axis=dict(range=[0, 100]),
                               bar=dict(color=MALIGNANT if prediction == 1 else BENIGN),
                               steps=[dict(range=[0, 30], color="rgba(46,196,182,.30)"),
                                      dict(range=[30, 70], color="rgba(255,200,87,.30)"),
                                      dict(range=[70, 100], color="rgba(255,92,138,.30)")],
                               threshold=dict(line=dict(color="white", width=4), value=50))))
                show(gauge, 360)
            with c2:
                prob_df = pd.DataFrame({"Diagnosis": ["Benign", "Malignant"], "Probability": [p_ben, p_mal]})
                fig = px.bar(prob_df, x="Diagnosis", y="Probability", color="Diagnosis",
                             text_auto=".2%", color_discrete_map=COLOR_MAP,
                             title="Prediction Probabilities")
                fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
                show(fig, 360)

            c3, c4 = st.columns(2)

            # radar: input vs class averages (mean features)
            with c3:
                rf = mean_feats
                lo, hi = df[rf].min(), df[rf].max()
                norm = lambda s: ((s[rf] - lo) / (hi - lo)).clip(0, 1)
                fig = go.Figure()
                for name, ser, colr in [("Your input", pd.Series(input_values), GOLD),
                                        ("Avg Benign", benign_avg, BENIGN),
                                        ("Avg Malignant", malig_avg, MALIGNANT)]:
                    v = norm(ser).tolist()
                    fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=rf + rf[:1], name=name,
                                                  line=dict(color=colr, width=3 if name == "Your input" else 2),
                                                  fill="toself" if name == "Your input" else None,
                                                  opacity=0.85))
                fig.update_layout(title="Input vs Class Averages (mean features)",
                                  polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                show(fig, 450)

            # contribution of each feature to the decision
            with c4:
                coefs = get_coefficients()
                if coefs is not None and len(coefs) == len(features):
                    contrib = coefs * transform_input(input_df)
                    cdf = pd.DataFrame({"Feature": features, "Contribution": contrib})
                    cdf["Abs"] = cdf["Contribution"].abs()
                    cdf = cdf.sort_values("Abs", ascending=False).head(10)
                    cdf["Effect"] = np.where(cdf["Contribution"] > 0, "Pushes → Malignant", "Pushes → Benign")
                    fig = px.bar(cdf, x="Contribution", y="Feature", orientation="h", color="Effect",
                                 color_discrete_map={"Pushes → Malignant": MALIGNANT,
                                                     "Pushes → Benign": BENIGN},
                                 title="Top 10 Features Driving This Prediction")
                    fig.update_layout(yaxis=dict(categoryorder="total ascending"), legend_title_text="")
                    show(fig, 450)
                else:
                    st.info("Feature contributions unavailable for this model type.")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    else:
        section("Upload CSV for Batch Prediction")

        template = df[features].head(5).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download sample CSV template", template,
                           file_name="sample_input.csv", mime="text/csv")

        uploaded_file = st.file_uploader("Upload a CSV containing all 30 model features", type=["csv"])

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            missing = [f for f in features if f not in input_df.columns]

            if missing:
                st.error("Missing required feature columns: " + ", ".join(missing))
            else:
                try:
                    input_df = input_df[features].apply(pd.to_numeric, errors="raise")
                    if not np.isfinite(input_df.to_numpy()).all():
                        raise ValueError("CSV contains missing or infinite values.")

                    preds = model.predict(input_df)
                    probs = model.predict_proba(input_df)

                    output = input_df.copy()
                    output.insert(0, "Predicted Diagnosis",
                                  ["Malignant" if int(p) == 1 else "Benign" for p in preds])
                    output["Benign Probability"] = probs[:, 0]
                    output["Malignant Probability"] = probs[:, 1]

                    st.success(f"✅ Predictions generated for {len(output)} records!")

                    n_m = int((preds == 1).sum())
                    n_b = len(preds) - n_m
                    c1, c2, c3 = st.columns(3)
                    kpi(c1, "Records", len(output), "", "#ffffff")
                    kpi(c2, "Predicted Benign", n_b, f"{n_b / len(output):.1%}", BENIGN)
                    kpi(c3, "Predicted Malignant", n_m, f"{n_m / len(output):.1%}", MALIGNANT)
                    st.write("")

                    c1, c2 = st.columns(2)
                    with c1:
                        cnt = output["Predicted Diagnosis"].value_counts()
                        fig = px.pie(values=cnt.values, names=cnt.index, hole=0.5,
                                     color=cnt.index, color_discrete_map=COLOR_MAP,
                                     title="Predicted Class Split")
                        show(fig, 360)
                    with c2:
                        fig = px.histogram(output, x="Malignant Probability", nbins=20,
                                           color="Predicted Diagnosis", color_discrete_map=COLOR_MAP,
                                           title="Distribution of Malignant Probability")
                        show(fig, 360)

                    show_df(output)

                    st.download_button("⬇️ Download Prediction Results",
                                       output.to_csv(index=False).encode("utf-8"),
                                       file_name="breast_cancer_predictions.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"Could not process the uploaded CSV: {e}")


# =====================================================================
# FOOTER
# =====================================================================
st.divider()
st.caption("🎗️ Breast Cancer Detection  |  Logistic Regression  |  Wisconsin Diagnostic Breast Cancer Dataset")
