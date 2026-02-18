"""
Streamlit dashboard for 30-day environmental predictions (city-separated).

This version supports reading the latest prediction file directly from Amazon S3,
which is the recommended setup when you deploy the dashboard publicly (e.g.,
Streamlit Community Cloud).

Local run:
    pip install -r requirements.txt
    streamlit run streamlit_dashboard_app.py

Streamlit Community Cloud (recommended):
1) Put this file in a GitHub repo.
2) Add requirements (at minimum): streamlit, pandas, boto3
3) Add Streamlit Secrets (Settings → Secrets) like:

[aws]
aws_access_key_id = "AKIA...."
aws_secret_access_key = "...."
region = "eu-north-1"          # optional but recommended
# aws_session_token = "..."    # only if you use temporary creds

[s3]
bucket = "ibrahim1995-dust-datasets"
latest_key = "datasets/predictions/unified_next30_predictions_LATEST.csv"
prefix = "datasets/predictions/"   # optional, used for listing

Security note:
Create an IAM user with read-only permissions limited to:
    s3:GetObject on s3://<bucket>/datasets/predictions/*
and (optional) s3:ListBucket on the bucket with prefix datasets/predictions/*

The app assumes the dataset includes at least:
    - city
    - timestamp
plus prediction columns (numeric and/or categorical).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional dependency (required for S3 mode)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except Exception:  # pragma: no cover
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception


# ----------------------------
# App configuration
# ----------------------------
st.set_page_config(
    page_title="30-Day Environmental Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

DEFAULT_LOCAL_CSV = "unified_next30_predictions.csv"

# Default S3 locations (used if secrets are not provided)
DEFAULT_S3_BUCKET = "ibrahim1995-dust-datasets"
DEFAULT_S3_LATEST_KEY = "datasets/predictions/unified_next30_predictions_LATEST.csv"
DEFAULT_S3_PREFIX = "datasets/predictions/"


def _secret_get(*keys: str, default=None):
    """
    Safe getter for Streamlit secrets.
    Supports both flat secrets (st.secrets["AWS_ACCESS_KEY_ID"]) and TOML sections.

    Examples:
        _secret_get("aws", "aws_access_key_id")
        _secret_get("s3", "bucket")
    """
    cur = st.secrets
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _boto3_client():
    """
    Build an S3 client from Streamlit secrets.
    If no secrets exist, boto3 will still try to use its default credential chain.
    """
    if boto3 is None:
        return None

    # Support [aws] section (recommended)
    aws_access_key_id = (
        _secret_get("aws", "aws_access_key_id", default=None)
        or _secret_get("AWS_ACCESS_KEY_ID", default=None)
    )
    aws_secret_access_key = (
        _secret_get("aws", "aws_secret_access_key", default=None)
        or _secret_get("AWS_SECRET_ACCESS_KEY", default=None)
    )
    aws_session_token = (
        _secret_get("aws", "aws_session_token", default=None)
        or _secret_get("AWS_SESSION_TOKEN", default=None)
    )
    region = (
        _secret_get("aws", "region", default=None)
        or _secret_get("AWS_DEFAULT_REGION", default=None)
        or "eu-north-1"
    )

    kwargs = {"region_name": region}
    if aws_access_key_id and aws_secret_access_key:
        kwargs.update(
            {
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            }
        )
        if aws_session_token:
            kwargs["aws_session_token"] = aws_session_token

    return boto3.client("s3", **kwargs)


@st.cache_data(show_spinner=False)
def _download_s3_object(bucket: str, key: str, cache_buster: str) -> bytes:
    """
    Download an S3 object and cache it.
    cache_buster should change when the S3 object changes (ETag/LastModified),
    so Streamlit cache refreshes automatically.
    """
    client = _boto3_client()
    if client is None:
        raise RuntimeError("boto3 is not available. Add 'boto3' to requirements.txt.")
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def _head_s3_object(bucket: str, key: str) -> Tuple[Optional[str], Optional[datetime], Optional[int]]:
    """
    Return (etag, last_modified, size_bytes) for a key.
    If the object doesn't exist or permissions are missing, returns (None, None, None).
    """
    client = _boto3_client()
    if client is None:
        return None, None, None

    try:
        h = client.head_object(Bucket=bucket, Key=key)
        etag = (h.get("ETag") or "").strip('"') if h.get("ETag") else None
        lm = h.get("LastModified")
        size = h.get("ContentLength")
        return etag, lm, size
    except Exception:
        return None, None, None


def _list_s3_keys(bucket: str, prefix: str, limit: int = 200) -> List[str]:
    client = _boto3_client()
    if client is None:
        return []

    keys: List[str] = []
    try:
        token = None
        while True and len(keys) < limit:
            kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": min(1000, limit - len(keys))}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
    except Exception:
        return []
    return keys


@st.cache_data(show_spinner=False)
def load_predictions_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(csv_bytes))


@st.cache_data(show_spinner=False)
def load_predictions_from_local(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
    return df


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    st.title("30-Day Environmental Forecast Dashboard")
    st.write(
        "This dashboard visualizes the **30-day multi-hazard prediction dataset** produced by the pipeline. "
        "In public deployment, the recommended configuration is to read the latest prediction file from **Amazon S3**."
    )

    # ----------------------------
    # Sidebar: data source
    # ----------------------------
    st.sidebar.header("Data source")

    mode = st.sidebar.radio(
        "Choose input",
        options=["S3 (LATEST predictions)", "Upload CSV", "Local CSV (development)"],
        index=0,
        help="For public deployment, choose S3. For quick tests, upload a CSV.",
    )

    df: Optional[pd.DataFrame] = None

    # ---- S3 mode
    if mode.startswith("S3"):
        if boto3 is None:
            st.error("S3 mode requires boto3. Add `boto3` to requirements.txt and redeploy.")
            st.stop()

        bucket = st.sidebar.text_input(
            "S3 bucket",
            value=_secret_get("s3", "bucket", default=DEFAULT_S3_BUCKET),
        )
        prefix = st.sidebar.text_input(
            "S3 prefix (optional)",
            value=_secret_get("s3", "prefix", default=DEFAULT_S3_PREFIX),
            help="Used only to list files. Requires s3:ListBucket permission.",
        )

        with st.sidebar.expander("Select file (optional)"):
            keys = _list_s3_keys(bucket, prefix) if prefix else []
            default_key = _secret_get("s3", "latest_key", default=DEFAULT_S3_LATEST_KEY)
            if keys:
                key = st.selectbox(
                    "Prediction file key",
                    options=keys,
                    index=keys.index(default_key) if default_key in keys else 0,
                )
            else:
                key = st.text_input("Prediction file key", value=default_key)

        etag, last_modified, size = _head_s3_object(bucket, key)
        if last_modified:
            st.sidebar.caption(f"Last modified: {last_modified}")
        if size is not None:
            st.sidebar.caption(f"Size: {size:,} bytes")

        refresh = st.sidebar.button("Refresh from S3", help="Forces Streamlit to re-check the S3 object metadata.")
        cache_buster = f"{etag}|{last_modified}|{int(refresh)}"

        try:
            csv_bytes = _download_s3_object(bucket, key, cache_buster=cache_buster)
            df = load_predictions_from_bytes(csv_bytes)
        except NoCredentialsError:
            st.error(
                "AWS credentials not found. If you are on Streamlit Community Cloud, set Secrets as described in the file header."
            )
            st.stop()
        except ClientError as e:
            st.error(f"Could not read s3://{bucket}/{key}. Check IAM permissions and the object path.\n\n{e}")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load from S3: {e}")
            st.stop()

        st.sidebar.success("Loaded predictions from S3.")

    # ---- Upload mode
    elif mode.startswith("Upload"):
        uploaded = st.sidebar.file_uploader(
            "Upload predictions CSV",
            type=["csv"],
            help="Upload a timestamped file or unified_next30_predictions_LATEST.csv.",
        )
        if uploaded is None:
            st.info("Upload a CSV to start.")
            st.stop()
        df = load_predictions_from_bytes(uploaded.getvalue())

    # ---- Local mode
    else:
        fallback_path = os.path.join(os.getcwd(), DEFAULT_LOCAL_CSV)
        st.sidebar.caption(f"Local path: {fallback_path}")
        if not os.path.exists(fallback_path):
            st.warning(f"Could not find '{DEFAULT_LOCAL_CSV}' in the current folder.")
            st.stop()
        df = load_predictions_from_local(fallback_path)

    # ----------------------------
    # Prepare data
    # ----------------------------
    df = safe_to_datetime(df, "timestamp")

    if "city" not in df.columns:
        st.error("The dataset must contain a 'city' column.")
        st.stop()

    base_cols = ["city", "timestamp"]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    numeric_candidates = [
        "precipitation_sum_pred",
        "precipitation_sum_normal",
        "precip_deficit_pred",
        "drought_duration_pred",
        "dust_event_prob",
        "pm10",
        "pm25",
        "aod",
        "temp_mean",
        "wind_speed_mean",
        "drought_flag_pred",
        "dust_event_pred",
        "dust_intensity_code",
    ]
    df = coerce_numeric(df, numeric_candidates)

    # ----------------------------
    # Sidebar: filters
    # ----------------------------
    st.sidebar.header("Filters")
    cities = sorted([c for c in df["city"].dropna().unique().tolist()])
    if not cities:
        st.error("No cities found in the 'city' column.")
        st.stop()

    selected_city = st.sidebar.selectbox("City", cities, index=0)
    df_city = df[df["city"] == selected_city].copy()

    if "timestamp" in df_city.columns and pd.api.types.is_datetime64_any_dtype(df_city["timestamp"]):
        df_city = df_city.sort_values("timestamp")
        min_date = df_city["timestamp"].min()
        max_date = df_city["timestamp"].max()

        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = date_range
            end_date = date_range

        mask = (df_city["timestamp"].dt.date >= start_date) & (df_city["timestamp"].dt.date <= end_date)
        df_city = df_city.loc[mask].copy()
    else:
        st.sidebar.info("No usable datetime 'timestamp' column found; date filtering is disabled.")

    # ----------------------------
    # Summary KPIs
    # ----------------------------
    left, mid, right = st.columns(3)
    left.metric("City", selected_city)
    mid.metric("Rows (selected)", f"{len(df_city):,}")

    if "dust_event_pred" in df_city.columns:
        dust_events = df_city["dust_event_pred"].fillna(0).astype(float)
        pct_dust = 100.0 * (dust_events > 0).mean() if len(dust_events) else 0.0
        right.metric("Predicted dust-event days", f"{pct_dust:.1f}%")
    elif "dust_event_prob" in df_city.columns:
        avg_prob = float(df_city["dust_event_prob"].mean()) if len(df_city) else 0.0
        right.metric("Avg dust probability", f"{avg_prob:.2f}")
    else:
        right.metric("Dust signal", "N/A")

    st.divider()

    # ----------------------------
    # Charts
    # ----------------------------
    st.subheader("Time-series depiction")

    if "timestamp" in df_city.columns and pd.api.types.is_datetime64_any_dtype(df_city["timestamp"]):
        df_plot = df_city.set_index("timestamp")
    else:
        df_plot = df_city.copy()
        df_plot.index = range(len(df_plot))

    default_plot_cols = [c for c in ["dust_event_prob", "pm10", "pm25", "aod", "temp_mean", "wind_speed_mean"] if c in df_city.columns]
    numeric_cols = [c for c in df_city.columns if pd.api.types.is_numeric_dtype(df_city[c]) and c not in ["dust_event_pred", "drought_flag_pred"]]

    plot_cols = st.multiselect(
        "Select numeric variables to plot",
        options=numeric_cols,
        default=default_plot_cols if default_plot_cols else numeric_cols[:3],
        help="Choose one or more numeric columns. Each selection will be displayed as a line chart.",
    )

    if plot_cols:
        st.line_chart(df_plot[plot_cols])
    else:
        st.info("Select at least one numeric variable to display a chart.")

    st.subheader("Categorical predictions (overview)")
    cat_cols = [c for c in ["drought_severity_pred", "dust_intensity_level"] if c in df_city.columns]
    if cat_cols:
        st.dataframe(df_city[["timestamp"] + cat_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.write("No categorical severity columns found (e.g., drought_severity_pred, dust_intensity_level).")

    st.subheader("Filtered dataset (table)")
    st.dataframe(df_city.reset_index(drop=True), use_container_width=True)

    # ----------------------------
    # Download filtered data
    # ----------------------------
    st.subheader("Export")
    csv_out = df_city.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered CSV",
        data=csv_out,
        file_name=f"{selected_city}_filtered_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "Tip: In production, keep the dashboard reading from "
        "`datasets/predictions/unified_next30_predictions_LATEST.csv` in S3. "
        "This gives a stable endpoint for the latest results."
    )


if __name__ == "__main__":
    main()
