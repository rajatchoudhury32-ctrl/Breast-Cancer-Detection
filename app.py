
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import confusion_matrix

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

st.title("🎗️ Breast Cancer Detection")
st.caption("Machine Learning Prediction using Logistic Regression")

# ---------------- FILE LOCATOR ----------------

def find_file(filename):
    files = list(Path(".").rglob(filename))

    if not files:
        st.error(f"Required file not found: {filename}")
        st.stop()

    return files[0]


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load(find_file("breast_cancer_model.pkl"))


@st.cache_data
def load_features():
    return joblib.load(find_file("feature_names.pkl"))


# ---------------- LOAD ORIGINAL DATASET ----------------

@st.cache_data
def load_data():
    dataset_path = find_file("wdbc.data")

    raw = pd.read_csv(dataset_path, header=None)

    features = load_features()

    if raw.shape[1] != 32 or len(features) != 30:
        raise ValueError(
            "Unexpected dataset structure. "
            "Expected 32 columns and 30 model features."
        )

    raw.columns = ["id", "diagnosis"] + list(features)

    raw["diagnosis"] = raw["diagnosis"].astype(str).str.strip()

    for col in features:
        raw[col] = pd.to_numeric(raw[col], errors="raise")

    raw["target"] = raw["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    if raw["target"].isna().any():
        raise ValueError("Unexpected diagnosis labels in wdbc.data.")

    return raw


# ---------------- LOAD EVALUATION FILES ----------------

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

# ---------------- SIDEBAR ----------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Data Analytics",
        "Model Performance",
        "Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Breast Cancer Detection using Logistic Regression"
)

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "This project uses Logistic Regression to classify "
        "breast cancer tumours as Benign or Malignant using "
        "30 numerical features from the Wisconsin Diagnostic "
        "Breast Cancer dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(features))
    col3.metric("Benign Cases", int((df["diagnosis"] == "B").sum()))
    col4.metric("Malignant Cases", int((df["diagnosis"] == "M").sum()))

    st.subheader("Diagnosis Distribution")

    counts = df["diagnosis"].value_counts().rename(
        index={"B": "Benign", "M": "Malignant"}
    )

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.45,
        title="Benign vs Malignant Cases"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")

    display_df = df.copy()
    display_df["diagnosis"] = display_df["diagnosis"].map({
        "B": "Benign",
        "M": "Malignant"
    })

    st.dataframe(display_df.head(10), use_container_width=True)


# ---------------- DATA ANALYTICS ----------------

elif page == "Data Analytics":

    st.header("Data Analytics")

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Choose a feature",
        features
    )

    fig = px.histogram(
        df,
        x=selected_feature,
        color="diagnosis",
        color_discrete_map={
            "B": "#2878B5",
            "M": "#E45756"
        },
        title=f"Distribution of {selected_feature}",
        marginal="box"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Feature Relationship")

    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox(
            "X-axis feature",
            features,
            index=0
        )

    with col2:
        y_feature = st.selectbox(
            "Y-axis feature",
            features,
            index=1
        )

    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="diagnosis",
        color_discrete_map={
            "B": "#2878B5",
            "M": "#E45756"
        },
        title="Feature Relationship"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")

    selected_features = st.multiselect(
        "Select features for correlation analysis",
        features,
        default=features[:8]
    )

    if len(selected_features) >= 2:
        corr = df[selected_features].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation"
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least two features.")


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.header("Model Performance")

    st.subheader("Evaluation Metrics")

    st.dataframe(metrics, use_container_width=True)

    st.divider()

    st.subheader("Confusion Matrix")

    def normalize_label(value):
        value = str(value).strip().lower()

        if value in ["benign", "b", "0"]:
            return 0
        if value in ["malignant", "m", "1"]:
            return 1

        raise ValueError(f"Unknown prediction label: {value}")

    y_true = results["Actual"].map(normalize_label)
    y_pred = results["Predicted"].map(normalize_label)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
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

    st.subheader("Prediction Results")

    st.dataframe(results, use_container_width=True)


# ---------------- PREDICTION ----------------

elif page == "Prediction":

    st.header("Breast Cancer Prediction")

    st.warning(
        "This application is an educational project and "
        "is not intended for clinical diagnosis."
    )

    option = st.radio(
        "Choose Prediction Method",
        ["Manual Input", "Upload CSV"]
    )

    if option == "Manual Input":

        st.subheader("Enter Tumour Measurements")

        st.write(
            "Enter the 30 numerical feature values "
            "to generate a model prediction."
        )

        medians = df[features].median()

        with st.form("prediction_form"):

            input_values = {}

            cols = st.columns(3)

            for i, feature in enumerate(features):
                with cols[i % 3]:
                    input_values[feature] = st.number_input(
                        feature,
                        value=float(medians[feature]),
                        format="%.6f"
                    )

            submitted = st.form_submit_button(
                "Predict Diagnosis",
                use_container_width=True
            )

        if submitted:

            input_df = pd.DataFrame(
                [input_values],
                columns=features
            )

            prediction = model.predict(input_df)[0]

            probabilities = model.predict_proba(input_df)[0]

            result_label = (
                "Malignant" if int(prediction) == 1
                else "Benign"
            )

            st.subheader("Prediction Result")

            if int(prediction) == 1:
                st.error("Prediction: Malignant")
            else:
                st.success("Prediction: Benign")

            prob_df = pd.DataFrame({
                "Diagnosis": ["Benign", "Malignant"],
                "Probability": probabilities
            })

            fig = px.bar(
                prob_df,
                x="Diagnosis",
                y="Probability",
                color="Diagnosis",
                text_auto=".2%",
                title="Prediction Probabilities"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:

        st.subheader("Upload CSV for Batch Prediction")

        uploaded_file = st.file_uploader(
            "Upload a CSV containing all 30 model features",
            type=["csv"]
        )

        if uploaded_file is not None:

            input_df = pd.read_csv(uploaded_file)

            missing = [
                feature for feature in features
                if feature not in input_df.columns
            ]

            if missing:
                st.error(
                    "Missing required feature columns: "
                    + ", ".join(missing)
                )

            else:
                try:
                    input_df = input_df[features].apply(
                        pd.to_numeric,
                        errors="raise"
                    )

                    if not np.isfinite(input_df.to_numpy()).all():
                        raise ValueError(
                            "CSV contains missing or infinite values."
                        )

                    predictions = model.predict(input_df)

                    probabilities = model.predict_proba(input_df)

                    output = input_df.copy()

                    output["Predicted Diagnosis"] = [
                        "Malignant" if int(p) == 1 else "Benign"
                        for p in predictions
                    ]

                    output["Benign Probability"] = probabilities[:, 0]
                    output["Malignant Probability"] = probabilities[:, 1]

                    st.success("Predictions generated successfully!")

                    st.dataframe(
                        output,
                        use_container_width=True
                    )

                    csv = output.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Prediction Results",
                        data=csv,
                        file_name="breast_cancer_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Could not process the uploaded CSV: {e}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Breast Cancer Detection | Logistic Regression | "
    "Wisconsin Diagnostic Breast Cancer Dataset"
)
