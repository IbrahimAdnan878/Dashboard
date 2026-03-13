# streamlit_app.py
# Final Streamlit dashboard:
# - auto-load from S3 on every rerun
# - predictions visualization
# - mitigation single-day recommendations
# - mitigation report download for selected city and selected period
# - model metrics tab
# - filtered data export
# - display-only fix: if drought_flag_pred == 0 -> drought_severity_pred = "none"

from __future__ import annotations

import io
import os
from collections.abc import Mapping

import pandas as pd
import streamlit as st

from mitigation_engine import recommend_actions, result_to_action_cards
from mitigation_report_helpers import (
    build_mitigation_report_for_all_days,
    mitigation_report_to_csv_bytes,
)

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


@st.cache_data(show_spinner=False)
def download_s3_cached(bucket: str, key: str) -> bytes:
    client = _boto3_client()
    if client is None:
        raise RuntimeError("boto3 is not installed. Add boto3 to requirements.txt.")
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


def safe_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def apply_display_fix(df: pd.DataFrame) -> pd.DataFrame:
    if "drought_flag_pred" in df.columns and "drought_severity_pred" in df.columns:
        mask = pd.to_numeric(df["drought_flag_pred"], errors="coerce").fillna(0) == 0
        df.loc[mask, "drought_severity_pred"] = "none"
    return df


def _build_mitigation_input(row: dict) -> dict:
    r = dict(row)

    if "timestamp" in r:
        ts = r["timestamp"]
        if isinstance(ts, pd.Timestamp):
            r["date"] = ts.date().isoformat()
        else:
            r["date"] = str(ts).split(" ")[0]

    if "drought_severity" not in r and "drought_severity_pred" in r:
        r["drought_severity"] = r.get("drought_severity_pred")

    if "dust_event" not in r and "dust_event_pred" in r:
        r["dust_event"] = r.get("dust_event_pred")

    return r


def run_mitigation_from_row(row_dict: dict) -> dict:
    """
    Adapter between one dashboard prediction row and mitigation_report_helpers.py.
    """
    row_norm = _build_mitigation_input(row_dict)
    result = recommend_actions(row_norm)
    cards = result_to_action_cards(result)

    action_texts = []
    for card in cards:
        title = str(card.get("title", "")).strip()
        description = str(card.get("description", "")).strip()

        if title and description:
            action_texts.append(f"{title}: {description}")
        elif title:
            action_texts.append(title)
        elif description:
            action_texts.append(description)

    hazards = getattr(result, "hazards_detected", []) or []
    risk_level = getattr(result, "risk_level", None)
    city = getattr(result, "city", row_norm.get("city"))
    date = getattr(result, "date", row_norm.get("date"))

    summary = f"Hazards: {', '.join(hazards) if hazards else 'None'} | Risk: {risk_level}"

    return {
        "status": "ok",
        "city": city,
        "date": date,
        "risk_level": risk_level,
        "actions": action_texts,
        "summary": summary,
        "hazards_detected": hazards,
    }


st.title("30-Day Environmental Forecast Dashboard")

mode = st.sidebar.radio("Data source", ["S3", "Upload CSV", "Local files"], index=0)

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

    refresh = st.sidebar.button("Refresh S3 files", use_container_width=True)

    try:
        if refresh:
            download_s3_cached.clear()
            read_csv_bytes.clear()

        pred_bytes = download_s3_cached(bucket, pred_key)
        pred_df = read_csv_bytes(pred_bytes)

        try:
            metrics_bytes = download_s3_cached(bucket, metrics_key)
            metrics_df = read_csv_bytes(metrics_bytes)
        except Exception:
            metrics_df = None
            st.sidebar.warning("Metrics file was not found or could not be read from S3.")

        st.sidebar.success("Loaded automatically from S3.")

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
    st.info("No predictions dataset is loaded yet.")
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

selected_city = st.sidebar.selectbox("City", cities)
df_city = pred_df[pred_df["city"].astype(str) == selected_city].copy()

selected_start_date = None
selected_end_date = None

if "timestamp" in df_city.columns and pd.api.types.is_datetime64_any_dtype(df_city["timestamp"]):
    df_city = df_city.sort_values("timestamp")
    min_date = df_city["timestamp"].min()
    max_date = df_city["timestamp"].max()

    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            selected_start_date, selected_end_date = date_range
        else:
            selected_start_date = date_range
            selected_end_date = date_range

        mask = (
            (df_city["timestamp"].dt.date >= selected_start_date)
            & (df_city["timestamp"].dt.date <= selected_end_date)
        )
        df_city = df_city.loc[mask].copy()

col1, col2, col3 = st.columns(3)
col1.metric("City", selected_city)
col2.metric("Rows", len(df_city))

if "dust_event_pred" in df_city.columns:
    dust_series = pd.to_numeric(df_city["dust_event_pred"], errors="coerce").fillna(0)
    pct = (dust_series > 0).mean() * 100 if len(dust_series) else 0.0
    col3.metric("Dust event days", f"{pct:.1f}%")
else:
    col3.metric("Dust event days", "N/A")

st.divider()

tab_viz, tab_mitig, tab_metrics, tab_data = st.tabs(
    ["📊 Visualization", "🛡️ Mitigation", "📏 Model Metrics", "📄 Data"]
)

with tab_viz:
    st.subheader("Time Series")

    if "timestamp" in df_city.columns and pd.api.types.is_datetime64_any_dtype(df_city["timestamp"]):
        df_plot = df_city.set_index("timestamp")
    else:
        df_plot = df_city.copy()

    numeric_cols = [c for c in df_city.columns if pd.api.types.is_numeric_dtype(df_city[c])]

    if numeric_cols:
        cols = st.multiselect(
            "Select variables",
            numeric_cols,
            default=numeric_cols[: min(3, len(numeric_cols))],
            key="viz_cols",
        )
        if cols:
            st.line_chart(df_plot[cols])
        else:
            st.info("Select at least one numeric variable.")
    else:
        st.info("No numeric columns were found for plotting.")

    st.subheader("Categorical Predictions")
    cat_cols = [c for c in ["drought_severity_pred", "dust_intensity_level"] if c in df_city.columns]
    if cat_cols:
        display_cols = ["timestamp"] + cat_cols if "timestamp" in df_city.columns else cat_cols
        st.dataframe(df_city[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No categorical severity columns were found.")

with tab_mitig:
    st.subheader("Mitigation Recommendations")

    if df_city.empty:
        st.info("No rows are available after filtering.")
    else:
        chosen_row = None

        if "timestamp" in df_city.columns and pd.api.types.is_datetime64_any_dtype(df_city["timestamp"]):
            available_dates = sorted(df_city["timestamp"].dropna().dt.date.unique().tolist())

            if available_dates:
                chosen_date = st.selectbox(
                    "Choose one day for detailed actions",
                    available_dates,
                    key="mitigation_date",
                )
                row_df = df_city[df_city["timestamp"].dt.date == chosen_date]

                if not row_df.empty:
                    chosen_row = row_df.iloc[0].to_dict()
            else:
                chosen_row = df_city.iloc[0].to_dict()
        else:
            chosen_row = df_city.iloc[0].to_dict()

        if chosen_row is not None:
            chosen_row["city"] = selected_city
            row_norm = _build_mitigation_input(chosen_row)

            result = recommend_actions(row_norm)

            st.markdown(
                f"### Single-day summary — {getattr(result, 'city', selected_city)} — "
                f"{getattr(result, 'date', row_norm.get('date', 'N/A'))}"
            )

            hazards_text = ", ".join(getattr(result, "hazards_detected", []) or []) or "None"
            st.write(f"Detected hazards: {hazards_text}")
            st.write(f"Overall risk level: **{str(getattr(result, 'risk_level', 'unknown')).upper()}**")

            cards = result_to_action_cards(result)
            if not cards:
                st.success("No mitigation actions were triggered for this record.")
            else:
                for c in cards:
                    with st.container(border=True):
                        st.markdown(f"### {c['title']}")
                        st.caption(
                            f"Hazard: {c['hazard']} | Sector: {c['sector']} | Priority: {c['priority']}"
                        )
                        st.write(c["description"])
                        st.write(f"**Lead agency:** {c['lead_agency']}")
                        st.write(f"**Time to act:** {c['time_to_act']} | **Cost:** {c['cost_level']}")
                        triggers = c.get("triggers", [])
                        if triggers:
                            st.caption("Triggers: " + ", ".join(triggers))

        st.divider()
        st.subheader("Mitigation report for selected city and selected period")

        mitigation_report_df = build_mitigation_report_for_all_days(
            prediction_df=df_city,
            mitigation_callback=run_mitigation_from_row,
            city=selected_city,
        )

        if mitigation_report_df.empty:
            st.info("No mitigation report rows were produced for the current filters.")
        else:
            st.dataframe(mitigation_report_df, use_container_width=True)

            if selected_start_date and selected_end_date:
                report_file_name = (
                    f"mitigation_report_{selected_city}_{selected_start_date}_{selected_end_date}.csv"
                )
            else:
                report_file_name = f"mitigation_report_{selected_city}.csv"

            st.download_button(
                label="Download mitigation report CSV for selected city and period",
                data=mitigation_report_to_csv_bytes(mitigation_report_df),
                file_name=report_file_name,
                mime="text/csv",
                use_container_width=True,
            )

with tab_metrics:
    st.subheader("Model Performance Metrics")

    if metrics_df is None:
        st.info("No metrics dataset loaded.")
    else:
        st.dataframe(metrics_df, use_container_width=True)

        numeric_metrics = [c for c in metrics_df.columns if pd.api.types.is_numeric_dtype(metrics_df[c])]

        if numeric_metrics:
            default_metric = "accuracy" if "accuracy" in numeric_metrics else numeric_metrics[0]
            metric = st.selectbox(
                "Visualize metric",
                numeric_metrics,
                index=numeric_metrics.index(default_metric),
                key="metric_select",
            )
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
        f"{selected_city}_predictions.csv",
        "text/csv",
        use_container_width=True,
    )
