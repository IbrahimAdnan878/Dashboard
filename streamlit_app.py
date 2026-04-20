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

MODEL_S3_KEY_DEFAULTS = {
    "Random Forest": {
        "pred": "datasets/predictions/unified_next30_predictions_LATEST.csv",
        "metrics": "datasets/predictions/unified_next30_metrics_LATEST.csv",
        "metrics_label": "Metrics file",
        "metrics_kind": "metrics",
    },
    "XGBoost": {
        "pred": "datasets/predictions/xgboost/unified_next30_predictions_LATEST.csv",
        "metrics": "datasets/predictions/xgboost/unified_next30_metrics_LATEST.csv",
        "metrics_label": "Metrics file",
        "metrics_kind": "metrics",
    },
    "LSTM": {
        "pred": "datasets/predictions/lstm/unified_next30_predictions_LATEST.csv",
        "metrics": "datasets/predictions/lstm/unified_next30_metrics_LATEST.csv",
        "metrics_label": "Metrics file",
        "metrics_kind": "metrics",
    },
    "Combined Best": {
        "pred": "datasets/predictions/combined/combined_best_predictions_LATEST.csv",
        "metrics": "datasets/predictions/combined/combined_best_model_selection_LATEST.csv",
        "metrics_label": "Model selection file",
        "metrics_kind": "model_selection",
    },
}


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



def get_numeric_plot_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = {c for c in df.columns if c.endswith("_source_model")}
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]



def get_source_model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_source_model")]



def show_source_model_summary(df: pd.DataFrame):
    source_cols = get_source_model_columns(df)
    if not source_cols:
        return

    st.subheader("Combined model source summary")
    st.caption("These columns show which model produced each prediction field in the combined-best dataset.")

    summary_frames = []
    for col in source_cols:
        counts = df[col].astype(str).value_counts(dropna=False).rename_axis("source_model").reset_index(name="rows")
        counts.insert(0, "prediction_field", col.replace("_source_model", ""))
        summary_frames.append(counts)

    if summary_frames:
        source_summary_df = pd.concat(summary_frames, ignore_index=True)
        st.dataframe(source_summary_df, use_container_width=True)

        pivot = source_summary_df.pivot(index="prediction_field", columns="source_model", values="rows").fillna(0)
        if not pivot.empty:
            st.bar_chart(pivot)


st.title("30-Day Environmental Forecast Dashboard")

mode = st.sidebar.radio("Data source", ["S3", "Upload CSV", "Local files"], index=0)

pred_df = None
metrics_df = None
selected_model_name = None
metrics_kind = "metrics"
metrics_label = "Metrics file"

if mode == "S3":
    bucket = st.sidebar.text_input(
        "S3 Bucket",
        _secret_get("s3", "bucket", default=DEFAULT_BUCKET),
    )

    selected_model_name = st.sidebar.selectbox(
        "S3 model source",
        list(MODEL_S3_KEY_DEFAULTS.keys()),
        index=0,
    )

    model_cfg = MODEL_S3_KEY_DEFAULTS[selected_model_name]
    default_pred_key = model_cfg["pred"]
    default_metrics_key = model_cfg["metrics"]
    metrics_kind = model_cfg.get("metrics_kind", "metrics")
    metrics_label = model_cfg.get("metrics_label", "Metrics file")

    pred_key = st.sidebar.text_input("Predictions file", default_pred_key)
    metrics_key = st.sidebar.text_input(metrics_label, default_metrics_key)

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
            st.sidebar.warning(f"{metrics_label} was not found or could not be read from S3.")

        st.sidebar.success(f"Loaded automatically from S3 ({selected_model_name}).")

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
    metrics_file = st.sidebar.file_uploader("Upload metrics/model-selection CSV", type=["csv"])

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

col1, col2, col3, col4 = st.columns(4)
col1.metric("City", selected_city)
col2.metric("Rows", len(df_city))
col3.metric("Source", selected_model_name or "Custom upload")

if "dust_event_pred" in df_city.columns:
    dust_series = pd.to_numeric(df_city["dust_event_pred"], errors="coerce").fillna(0)
    pct = (dust_series > 0).mean() * 100 if len(dust_series) else 0.0
    col4.metric("Dust event days", f"{pct:.1f}%")
else:
    col4.metric("Dust event days", "N/A")

st.divider()

extra_tabs = ["🧩 Source Models"] if get_source_model_columns(df_city) else []
tab_names = ["📊 Visualization", "🛡️ Mitigation", "📏 Metrics / Selection", "📄 Data", *extra_tabs]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.subheader("Time Series")

    if "timestamp" in df_city.columns and pd.api.types.is_datetime64_any_dtype(df_city["timestamp"]):
        df_plot = df_city.set_index("timestamp")
    else:
        df_plot = df_city.copy()

    numeric_cols = get_numeric_plot_columns(df_city)

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
    cat_priority = [
        "drought_severity_pred",
        "dust_intensity_level",
        "drought_flag_source_model",
        "drought_severity_code_source_model",
        "precipitation_sum_source_model",
        "dust_event_source_model",
    ]
    cat_cols = [c for c in cat_priority if c in df_city.columns]
    if cat_cols:
        display_cols = ["timestamp"] + cat_cols if "timestamp" in df_city.columns else cat_cols
        st.dataframe(df_city[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No categorical columns were found.")

with tabs[1]:
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

with tabs[2]:
    if metrics_kind == "model_selection":
        st.subheader("Combined model selection")
    else:
        st.subheader("Model performance metrics")

    if metrics_df is None:
        st.info("No metrics or model-selection dataset loaded.")
    else:
        st.dataframe(metrics_df, use_container_width=True)

        numeric_metrics = [c for c in metrics_df.columns if pd.api.types.is_numeric_dtype(metrics_df[c])]

        if numeric_metrics:
            preferred = ["accuracy", "f1_score", "r2", "rmse", "mae"]
            default_metric = next((m for m in preferred if m in numeric_metrics), numeric_metrics[0])
            metric = st.selectbox(
                "Visualize numeric column",
                numeric_metrics,
                index=numeric_metrics.index(default_metric),
                key="metric_select",
            )
            index_col = metrics_df.columns[0]
            chart_df = metrics_df.copy()
            chart_df[index_col] = chart_df[index_col].astype(str)
            st.bar_chart(chart_df.set_index(index_col)[metric])
        else:
            st.info("No numeric columns were found in the metrics/model-selection dataset.")

with tabs[3]:
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

if extra_tabs:
    with tabs[4]:
        show_source_model_summary(df_city)
