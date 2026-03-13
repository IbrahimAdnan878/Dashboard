
# streamlit_app_updated.py
# Streamlit dashboard with:
# - Predictions visualization
# - Model metrics tab
# - Optional fix: if drought_flag_pred == 0 -> drought_severity_pred = "none"

from __future__ import annotations
import io
import os
import pandas as pd
import streamlit as st

try:
    import boto3
except Exception:
    boto3 = None

st.set_page_config(page_title="30-Day Environmental Forecast Dashboard", page_icon="📈", layout="wide")

DEFAULT_PRED_FILE = "unified_next30_predictions.csv"
DEFAULT_METRICS_FILE = "unified_next30_metrics.csv"
DEFAULT_BUCKET = "ibrahim1995-dust-datasets"
DEFAULT_PREFIX = "datasets/predictions/"


def boto_client():
    if boto3 is None:
        return None
    try:
        return boto3.client("s3")
    except Exception:
        return None


def download_s3(bucket, key):
    client = boto_client()
    if client is None:
        return None
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def safe_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def apply_display_fix(df):
    if "drought_flag_pred" in df.columns and "drought_severity_pred" in df.columns:
        mask = df["drought_flag_pred"] == 0
        df.loc[mask, "drought_severity_pred"] = "none"
    return df


st.title("30-Day Environmental Forecast Dashboard")

mode = st.sidebar.radio("Data source", ["S3", "Upload CSV", "Local files"])

pred_df = None
metrics_df = None

if mode == "S3":

    bucket = st.sidebar.text_input("S3 Bucket", DEFAULT_BUCKET)

    pred_key = st.sidebar.text_input(
        "Predictions file",
        DEFAULT_PREFIX + "unified_next30_predictions_LATEST.csv"
    )

    metrics_key = st.sidebar.text_input(
        "Metrics file",
        DEFAULT_PREFIX + "unified_next30_metrics_LATEST.csv"
    )

    if st.sidebar.button("Load from S3"):

        try:
            pred_bytes = download_s3(bucket, pred_key)
            pred_df = pd.read_csv(io.BytesIO(pred_bytes))

            try:
                metrics_bytes = download_s3(bucket, metrics_key)
                metrics_df = pd.read_csv(io.BytesIO(metrics_bytes))
            except Exception:
                st.warning("Metrics file not found on S3.")

        except Exception as e:
            st.error(f"S3 load error: {e}")
            st.stop()

elif mode == "Upload CSV":

    pred_file = st.sidebar.file_uploader("Upload predictions CSV", type=["csv"])
    metrics_file = st.sidebar.file_uploader("Upload metrics CSV", type=["csv"])

    if pred_file:
        pred_df = pd.read_csv(pred_file)

    if metrics_file:
        metrics_df = pd.read_csv(metrics_file)

else:

    pred_path = os.path.join(os.getcwd(), DEFAULT_PRED_FILE)
    metrics_path = os.path.join(os.getcwd(), DEFAULT_METRICS_FILE)

    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)

    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)

if pred_df is None:
    st.info("Load a predictions dataset to start.")
    st.stop()

pred_df = safe_datetime(pred_df, "timestamp")

fix_rule = st.sidebar.checkbox("Fix severity when drought_flag = 0", value=True)

if fix_rule:
    pred_df = apply_display_fix(pred_df)

st.sidebar.header("Filters")

cities = sorted(pred_df["city"].dropna().unique())
city = st.sidebar.selectbox("City", cities)

df_city = pred_df[pred_df["city"] == city].copy()

col1, col2, col3 = st.columns(3)

col1.metric("City", city)
col2.metric("Rows", len(df_city))

if "dust_event_pred" in df_city.columns:
    pct = (df_city["dust_event_pred"] > 0).mean() * 100
    col3.metric("Dust event days", f"{pct:.1f}%")

st.divider()

tab_viz, tab_metrics, tab_data = st.tabs(["📊 Visualization", "📏 Model Metrics", "📄 Data"])

with tab_viz:

    st.subheader("Time Series")

    if "timestamp" in df_city.columns:
        df_city = df_city.sort_values("timestamp")
        df_plot = df_city.set_index("timestamp")
    else:
        df_plot = df_city

    numeric_cols = [
        c for c in df_city.columns
        if pd.api.types.is_numeric_dtype(df_city[c])
    ]

    cols = st.multiselect("Select variables", numeric_cols, default=numeric_cols[:3])

    if cols:
        st.line_chart(df_plot[cols])

with tab_metrics:

    st.subheader("Model Performance Metrics")

    if metrics_df is None:
        st.info("No metrics dataset loaded.")

    else:

        st.dataframe(metrics_df, use_container_width=True)

        numeric_metrics = [
            c for c in metrics_df.columns
            if pd.api.types.is_numeric_dtype(metrics_df[c])
        ]

        if numeric_metrics:

            metric = st.selectbox("Visualize metric", numeric_metrics)

            st.bar_chart(metrics_df.set_index(metrics_df.columns[0])[metric])

with tab_data:

    st.subheader("Filtered dataset")

    st.dataframe(df_city)

    csv = df_city.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        csv,
        f"{city}_predictions.csv",
        "text/csv"
    )
