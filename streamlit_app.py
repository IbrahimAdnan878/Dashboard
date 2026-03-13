
# streamlit_app.py
# Corrected Streamlit dashboard that:
# - reads predictions from S3 / upload / local files
# - reads metrics from S3 / upload / local files
# - supports Streamlit secrets for AWS and S3 paths
# - fixes display-only logic: if drought_flag_pred == 0 -> drought_severity_pred = "none"

from __future__ import annotations

import io
import os
from collections.abc import Mapping

import pandas as pd
import streamlit as st

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
except Exception:
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception
    PartialCredentialsError = Exception


st.set_page_config(
    page_title="30-Day Environmental Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

DEFAULT_PRED_FILE = "unified_next30_predictions.csv"
DEFAULT_METRICS_FILE = "unified_next30_metrics.csv"
DEFAULT_BUCKET = "ibrahim1995-dust-datasets"
DEFAULT_PREFIX = "datasets/predictions/"
DEFAULT_PRED_KEY = "datasets/predictions/unified_next30_predictions_LATEST.csv"
DEFAULT_METRICS_KEY = "datasets/predictions/unified_next30_metrics_LATEST.csv"


def _secrets_container():
    try:
        return st.secrets
    except Exception:
        return {}


def _secret_get(*keys: str, default=None):
    cur = _secrets_container()

    for k in keys:
        if isinstance(cur, Mapping):
            if k in cur:
                cur = cur[k]
                continue
            return default

        try:
            if k in cur:
                cur = cur[k]
                continue
            return default
        except Exception:
            return default

    return cur


def _boto3_client():
    if boto3 is None:
        return None

    aws_access_key_id = _secret_get("aws", "aws_access_key_id", default=None) or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = _secret_get("aws", "aws_secret_access_key", default=None) or os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = _secret_get("aws", "aws_session_token", default=None) or os.getenv("AWS_SESSION_TOKEN")
    region = _secret_get("aws", "region", default=None) or os.getenv("AWS_DEFAULT_REGION") or "eu-north-1"

    kwargs = {"region_name": region}

    if aws_access_key_id and aws_secret_access_key:
        kwargs["aws_access_key_id"] = aws_access_key_id
        kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            kwargs["aws_session_token"] = aws_session_token

    return boto3.client("s3", **kwargs)


def download_s3(bucket: str, key: str) -> bytes:
    client = _boto3_client()
    if client is None:
        raise RuntimeError("boto3 is not installed. Add boto3 to requirements.txt.")
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def safe_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def apply_display_fix(df: pd.DataFrame) -> pd.DataFrame:
    if "drought_flag_pred" in df.columns and "drought_severity_pred" in df.columns:
        mask = pd.to_numeric(df["drought_flag_pred"], errors="coerce").fillna(0) == 0
        df.loc[mask, "drought_severity_pred"] = "none"
    return df


st.title("30-Day Environmental Forecast Dashboard")

mode = st.sidebar.radio("Data source", ["S3", "Upload CSV", "Local files"])

pred_df = None
metrics_df = None

if mode == "S3":
    bucket = st.sidebar.text_input(
        "S3 Bucket",
        _secret_get("s3", "bucket", default=DEFAULT_BUCKET),
    )

    pred_key = st.sidebar.text_input(
        "Predictions file",
        _secret_get("s3", "predictions_latest_key", default=DEFAULT_PRED_KEY),
    )

    metrics_key = st.sidebar.text_input(
        "Metrics file",
        _secret_get("s3", "metrics_latest_key", default=DEFAULT_METRICS_KEY),
    )

    if st.sidebar.button("Load from S3", use_container_width=True):
        try:
            pred_bytes = download_s3(bucket, pred_key)
            pred_df = pd.read_csv(io.BytesIO(pred_bytes))

            try:
                metrics_bytes = download_s3(bucket, metrics_key)
                metrics_df = pd.read_csv(io.BytesIO(metrics_bytes))
            except Exception:
                st.warning("Predictions loaded, but the metrics file was not found or could not be read from S3.")

        except (NoCredentialsError, PartialCredentialsError):
            st.error("AWS credentials were not found. Add them in Streamlit Secrets under [aws].")
            st.stop()
        except ClientError as e:
            st.error(f"S3 access error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"S3 load error: {e}")
            st.stop()

elif mode == "Upload CSV":
    pred_file = st.sidebar.file_uploader("Upload predictions CSV", type=["csv"])
    metrics_file = st.sidebar.file_uploader("Upload metrics CSV", type=["csv"])

    if pred_file is not None:
        pred_df = pd.read_csv(pred_file)

    if metrics_file is not None:
        metrics_df = pd.read_csv(metrics_file)

else:
    pred_path = os.path.join(os.getcwd(), DEFAULT_PRED_FILE)
    metrics_path = os.path.join(os.getcwd(), DEFAULT_METRICS_FILE)

    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)

    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)

if pred_df is None:
    st.info("Choose a data source and load a predictions dataset to start.")
    st.stop()

pred_df = safe_datetime(pred_df, "timestamp")

fix_rule = st.sidebar.checkbox("Fix severity when drought_flag = 0", value=True)
if fix_rule:
    pred_df = apply_display_fix(pred_df)

st.sidebar.header("Filters")

if "city" not in pred_df.columns:
    st.error("The predictions dataset must contain a 'city' column.")
    st.stop()

cities = sorted(pred_df["city"].dropna().astype(str).unique().tolist())
if not cities:
    st.error("No city values were found in the predictions dataset.")
    st.stop()

city = st.sidebar.selectbox("City", cities)
df_city = pred_df[pred_df["city"].astype(str) == city].copy()

col1, col2, col3 = st.columns(3)
col1.metric("City", city)
col2.metric("Rows", len(df_city))

if "dust_event_pred" in df_city.columns:
    dust_series = pd.to_numeric(df_city["dust_event_pred"], errors="coerce").fillna(0)
    pct = (dust_series > 0).mean() * 100 if len(dust_series) else 0.0
    col3.metric("Dust event days", f"{pct:.1f}%")
else:
    col3.metric("Dust event days", "N/A")

st.divider()

tab_viz, tab_metrics, tab_data = st.tabs(["📊 Visualization", "📏 Model Metrics", "📄 Data"])

with tab_viz:
    st.subheader("Time Series")

    if "timestamp" in df_city.columns and pd.api.types.is_datetime64_any_dtype(df_city["timestamp"]):
        df_city = df_city.sort_values("timestamp")
        df_plot = df_city.set_index("timestamp")
    else:
        df_plot = df_city.copy()

    numeric_cols = [c for c in df_city.columns if pd.api.types.is_numeric_dtype(df_city[c])]

    if numeric_cols:
        cols = st.multiselect("Select variables", numeric_cols, default=numeric_cols[: min(3, len(numeric_cols))])
        if cols:
            st.line_chart(df_plot[cols])
        else:
            st.info("Select at least one numeric variable.")
    else:
        st.info("No numeric columns were found for plotting.")

with tab_metrics:
    st.subheader("Model Performance Metrics")

    if metrics_df is None:
        st.info("No metrics dataset loaded.")
    else:
        st.dataframe(metrics_df, use_container_width=True)

        numeric_metrics = [c for c in metrics_df.columns if pd.api.types.is_numeric_dtype(metrics_df[c])]

        if numeric_metrics:
            metric = st.selectbox("Visualize metric", numeric_metrics)
            index_col = metrics_df.columns[0]
            st.bar_chart(metrics_df.set_index(index_col)[metric])
        else:
            st.info("No numeric metrics columns were found.")

with tab_data:
    st.subheader("Filtered dataset")
    st.dataframe(df_city, use_container_width=True)

    csv = df_city.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        f"{city}_predictions.csv",
        "text/csv",
        use_container_width=True,
    )
