
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #F7FAFC;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("breast_cancer_dataset.csv")

@st.cache_data
def load_metrics():
    return pd.read_csv("model_evaluation.csv")

@st.cache_data
def load_predictions():
    return pd.read_csv("prediction_results.csv")

model = load_model()
df = load_data()
metrics = load_metrics()
results = load_predictions()

features = joblib.load("feature_names.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎗️ Breast Cancer AI")
st.sidebar.caption("Logistic Regression System")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Data Analytics",
     "Model Performance", "Prediction"]
)

st.sidebar.divider()
st.sidebar.info(
    "Wisconsin Diagnostic Breast Cancer Dataset"
)

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.title("Breast Cancer Detection")
    st.write(
        "Machine learning-based tumour classification "
        "using Logistic Regression."
    )

    accuracy = metrics.loc[
        metrics.Metric == "Accuracy", "Score"
    ].iloc[0]

    auc_score = metrics.loc[
        metrics.Metric == "ROC AUC", "Score"
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", len(df))
    c2.metric("Features", len(features))
    c3.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    c4.metric("ROC AUC", f"{auc_score:.4f}")

    st.divider()

    st.subheader("Diagnosis Distribution")

    counts = df["diagnosis"].value_counts().reset_index()
    counts.columns = ["Diagnosis", "Count"]
    counts["Diagnosis"] = counts["Diagnosis"].map({
        "B": "Benign",
        "M": "Malignant"
    })

    fig = px.pie(
        counts,
        names="Diagnosis",
        values="Count",
        hole=0.45,
        color="Diagnosis",
        color_discrete_map={
            "Benign": "#3498DB",
            "Malignant": "#E74C3C"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ---------------- DATA ANALYTICS ----------------

elif page == "Data Analytics":

    st.title("Data Analytics")
    st.write("Explore the dataset using interactive visualizations.")

    feature1 = st.selectbox(
        "Select X-axis feature", features,
        index=features.index("mean radius")
    )

    feature2 = st.selectbox(
        "Select Y-axis feature", features,
        index=features.index("mean texture")
    )

    plot_df = df.copy()
    plot_df["Diagnosis"] = plot_df["diagnosis"].map({
        "B": "Benign",
        "M": "Malignant"
    })

    fig = px.scatter(
        plot_df,
        x=feature1,
        y=feature2,
        color="Diagnosis",
        hover_data=["target"],
        color_discrete_map={
            "Benign": "#3498DB",
            "Malignant": "#E74C3C"
        },
        title="Interactive Feature Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")

    selected = st.multiselect(
        "Choose features",
        features,
        default=features[:10]
    )

    if len(selected) >= 2:
        corr = df[selected].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least two features.")


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("Model Performance Evaluation")

    metric_names = [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC AUC"
    ]

    values = [
        metrics.loc[
            metrics.Metric == name, "Score"
        ].iloc[0]
        for name in metric_names
    ]

    cols = st.columns(5)

    for col, name, value in zip(cols, metric_names, values):
        col.metric(name, f"{value*100:.2f}%")

    st.divider()

    st.subheader("Confusion Matrix")

    actual = results["Actual"].map({
        "Benign": 0,
        "Malignant": 1
    })

    predicted = results["Predicted"].map({
        "Benign": 0,
        "Malignant": 1
    })

    cm = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Benign", "Malignant"],
        y=["Benign", "Malignant"],
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count"
        },
        title="Confusion Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Comparison")

    metric_df = pd.DataFrame({
        "Metric": metric_names,
        "Score": values
    })

    fig = px.bar(
        metric_df,
        x="Metric",
        y="Score",
        color="Metric",
        text=metric_df["Score"].apply(
            lambda x: f"{x*100:.2f}%"
        ),
        title="Model Evaluation Metrics"
    )

    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True)


# ---------------- PREDICTION ----------------

elif page == "Prediction":

    st.title("Tumour Classification")
    st.write(
        "Enter tumour measurements to obtain a model prediction."
    )

    st.warning(
        "Educational prototype only. This model is not "
        "a clinically validated diagnostic system."
    )

    mode = st.radio(
        "Choose prediction method",
        ["Manual Input", "Upload CSV"]
    )

    if mode == "Manual Input":

        with st.form("prediction_form"):

            st.subheader("Enter 30 Tumour Measurements")

            input_values = {}

            for i in range(0, len(features), 2):

                col1, col2 = st.columns(2)

                f1 = features[i]

                with col1:
                    input_values[f1] = st.number_input(
                        f1.title(),
                        min_value=0.0,
                        value=float(df[f1].median()),
                        format="%.5f",
                        key=f"input_{i}"
                    )

                if i + 1 < len(features):

                    f2 = features[i + 1]

                    with col2:
                        input_values[f2] = st.number_input(
                            f2.title(),
                            min_value=0.0,
                            value=float(df[f2].median()),
                            format="%.5f",
                            key=f"input_{i+1}"
                        )

            submitted = st.form_submit_button(
                "Predict Tumour Class",
                use_container_width=True
            )

        if submitted:

            input_df = pd.DataFrame(
                [input_values],
                columns=features
            )

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            if prediction == 1:
                st.error("Model prediction: MALIGNANT")
            else:
                st.success("Model prediction: BENIGN")

            st.subheader("Class Probabilities")

            prob_df = pd.DataFrame({
                "Diagnosis": ["Benign", "Malignant"],
                "Probability": [
                    probability[0],
                    probability[1]
                ]
            })

            fig = px.bar(
                prob_df,
                x="Diagnosis",
                y="Probability",
                color="Diagnosis",
                text=prob_df["Probability"].apply(
                    lambda x: f"{x*100:.2f}%"
                ),
                color_discrete_map={
                    "Benign": "#3498DB",
                    "Malignant": "#E74C3C"
                },
                range_y=[0, 1]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Probabilities are model estimates, not clinical certainty."
            )

    else:

        uploaded_file = st.file_uploader(
            "Upload a CSV containing all 30 feature columns",
            type=["csv"]
        )

        if uploaded_file:

            batch = pd.read_csv(uploaded_file)

            missing = [
                f for f in features
                if f not in batch.columns
            ]

            if missing:
                st.error(
                    f"Missing required columns: {missing}"
                )

            elif batch.empty:
                st.error("The uploaded CSV is empty.")

            elif not batch[features].apply(
                pd.to_numeric, errors="coerce"
            ).notna().all().all():
                st.error("All 30 features must contain valid numeric values.")

            elif (batch[features] < 0).any().any():
                st.error("Feature measurements cannot be negative.")

            else:

                predictions = model.predict(batch[features])
                probabilities = model.predict_proba(
                    batch[features]
                )

                output = batch.copy()

                output["Predicted Diagnosis"] = [
                    "Malignant" if p == 1 else "Benign"
                    for p in predictions
                ]

                output["Benign Probability"] = probabilities[:, 0]
                output["Malignant Probability"] = probabilities[:, 1]

                st.success("Predictions generated successfully!")

                st.dataframe(
                    output,
                    use_container_width=True
                )

                st.download_button(
                    "Download Prediction Results",
                    data=output.to_csv(index=False),
                    file_name="breast_cancer_predictions.csv",
                    mime="text/csv"
                )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Breast Cancer Detection | Logistic Regression | "
    "Wisconsin Diagnostic Dataset | Academic Project"
)

