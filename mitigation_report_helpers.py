"""
mitigation_report_helpers.py

Helper functions for a Streamlit dashboard that needs to:
1) run mitigation for all future prediction rows,
2) build one mitigation report table for all days,
3) let the user download that report as CSV.

This file is intentionally generic so it can be connected to your existing
streamlit_app.py and mitigation_engine.py with only small edits.

Expected use:
    from mitigation_report_helpers import (
        build_mitigation_report_for_all_days,
        mitigation_report_to_csv_bytes,
    )

    report_df = build_mitigation_report_for_all_days(
        prediction_df=selected_city_df,
        mitigation_callback=run_mitigation_from_row,
    )

    st.download_button(
        "Download mitigation report CSV",
        data=mitigation_report_to_csv_bytes(report_df),
        file_name="mitigation_report_Baghdad.csv",
        mime="text/csv",
    )
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd


def _safe_json(value: Any) -> str:
    """Convert Python object to JSON text safely for CSV storage."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _normalize_actions(actions: Any) -> str:
    """
    Normalize mitigation actions into a readable text field.

    Supported inputs:
    - list[str]
    - list[dict]
    - dict
    - str
    - None
    """
    if actions is None:
        return ""

    if isinstance(actions, str):
        return actions

    if isinstance(actions, dict):
        return _safe_json(actions)

    if isinstance(actions, list):
        cleaned: List[str] = []
        for item in actions:
            if isinstance(item, str):
                cleaned.append(item)
            elif isinstance(item, dict):
                # Try common keys first
                for key in ("action", "title", "message", "recommendation", "description"):
                    if key in item and item[key]:
                        cleaned.append(str(item[key]))
                        break
                else:
                    cleaned.append(_safe_json(item))
            else:
                cleaned.append(str(item))
        return " | ".join(cleaned)

    return str(actions)


def _normalize_mitigation_result(result: Any) -> Dict[str, Any]:
    """
    Convert mitigation engine output into a flat dictionary.

    This function is flexible because mitigation engines often return different
    shapes. It tries to preserve as much information as possible.
    """
    if result is None:
        return {
            "mitigation_status": "no_result",
            "risk_level": None,
            "actions": "",
            "mitigation_summary": "",
            "raw_mitigation_json": "",
        }

    if isinstance(result, str):
        return {
            "mitigation_status": "ok",
            "risk_level": None,
            "actions": result,
            "mitigation_summary": result,
            "raw_mitigation_json": _safe_json(result),
        }

    if isinstance(result, list):
        actions_text = _normalize_actions(result)
        return {
            "mitigation_status": "ok",
            "risk_level": None,
            "actions": actions_text,
            "mitigation_summary": actions_text,
            "raw_mitigation_json": _safe_json(result),
        }

    if isinstance(result, dict):
        actions = (
            result.get("actions")
            or result.get("recommended_actions")
            or result.get("mitigation_actions")
            or result.get("recommendations")
        )
        summary = (
            result.get("summary")
            or result.get("message")
            or result.get("mitigation_summary")
            or _normalize_actions(actions)
        )
        risk_level = (
            result.get("risk_level")
            or result.get("risk")
            or result.get("severity")
            or result.get("alert_level")
        )
        status = result.get("status", "ok")

        return {
            "mitigation_status": status,
            "risk_level": risk_level,
            "actions": _normalize_actions(actions),
            "mitigation_summary": summary,
            "raw_mitigation_json": _safe_json(result),
        }

    # fallback
    text = str(result)
    return {
        "mitigation_status": "ok",
        "risk_level": None,
        "actions": text,
        "mitigation_summary": text,
        "raw_mitigation_json": _safe_json(text),
    }


def build_mitigation_report_for_all_days(
    prediction_df: pd.DataFrame,
    mitigation_callback: Callable[[Dict[str, Any]], Any],
    *,
    city: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build one mitigation report for all available days.

    Parameters
    ----------
    prediction_df:
        Prediction dataframe, usually loaded from:
        unified_next30_predictions_LATEST.csv

    mitigation_callback:
        A function that receives one row as a dictionary and returns the
        mitigation result for that day. Example:
            result = mitigation_callback(row_dict)

    city:
        Optional city filter.

    Returns
    -------
    pd.DataFrame
        A mitigation report dataframe ready for viewing or CSV download.
    """
    if prediction_df is None or prediction_df.empty:
        return pd.DataFrame()

    df = prediction_df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if city and "city" in df.columns:
        df = df[df["city"] == city].copy()

    sort_cols = [c for c in ["city", "timestamp"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    report_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        try:
            mitigation_result = mitigation_callback(row_dict)
            normalized = _normalize_mitigation_result(mitigation_result)
            error_message = ""
        except Exception as exc:
            normalized = {
                "mitigation_status": "error",
                "risk_level": None,
                "actions": "",
                "mitigation_summary": "",
                "raw_mitigation_json": "",
            }
            error_message = str(exc)

        report_row = {
            "city": row_dict.get("city"),
            "timestamp": row_dict.get("timestamp"),
            "drought_flag_pred": row_dict.get("drought_flag_pred"),
            "drought_severity_pred": row_dict.get("drought_severity_pred"),
            "precipitation_sum_pred": row_dict.get("precipitation_sum_pred"),
            "precipitation_sum_normal": row_dict.get("precipitation_sum_normal"),
            "precip_deficit_pred": row_dict.get("precip_deficit_pred"),
            "drought_duration_pred": row_dict.get("drought_duration_pred"),
            "dust_event_pred": row_dict.get("dust_event_pred"),
            "dust_event_prob": row_dict.get("dust_event_prob"),
            "dust_intensity_level": row_dict.get("dust_intensity_level"),
            "dust_intensity_code": row_dict.get("dust_intensity_code"),
            "pm10": row_dict.get("pm10"),
            "pm25": row_dict.get("pm25"),
            "aod": row_dict.get("aod"),
            "temp_mean": row_dict.get("temp_mean"),
            "wind_speed_mean": row_dict.get("wind_speed_mean"),
            "mitigation_status": normalized.get("mitigation_status"),
            "risk_level": normalized.get("risk_level"),
            "actions": normalized.get("actions"),
            "mitigation_summary": normalized.get("mitigation_summary"),
            "mitigation_error": error_message,
            "raw_mitigation_json": normalized.get("raw_mitigation_json"),
        }
        report_rows.append(report_row)

    report_df = pd.DataFrame(report_rows)

    preferred_cols = [
        "city",
        "timestamp",
        "drought_flag_pred",
        "drought_severity_pred",
        "precipitation_sum_pred",
        "precipitation_sum_normal",
        "precip_deficit_pred",
        "drought_duration_pred",
        "dust_event_pred",
        "dust_event_prob",
        "dust_intensity_level",
        "dust_intensity_code",
        "pm10",
        "pm25",
        "aod",
        "temp_mean",
        "wind_speed_mean",
        "risk_level",
        "actions",
        "mitigation_summary",
        "mitigation_status",
        "mitigation_error",
        "raw_mitigation_json",
    ]
    existing_cols = [c for c in preferred_cols if c in report_df.columns]
    return report_df[existing_cols]


def mitigation_report_to_csv_bytes(report_df: pd.DataFrame) -> bytes:
    """Convert mitigation report dataframe into CSV bytes for download."""
    if report_df is None or report_df.empty:
        return b""
    return report_df.to_csv(index=False).encode("utf-8")


STREAMLIT_INTEGRATION_EXAMPLE = r"""
# ---------------------- Streamlit mitigation tab example ----------------------
import streamlit as st
import pandas as pd

from mitigation_report_helpers import (
    build_mitigation_report_for_all_days,
    mitigation_report_to_csv_bytes,
)
from mitigation_engine import run_mitigation_from_row  # adapt to your real function


with mitigation_tab:
    st.subheader("Mitigation actions")

    # prediction_df = dataframe loaded from unified_next30_predictions_LATEST.csv

    selected_city = st.selectbox("City", sorted(prediction_df["city"].dropna().unique()))
    city_df = prediction_df[prediction_df["city"] == selected_city].copy()

    mitigation_report_df = build_mitigation_report_for_all_days(
        prediction_df=city_df,
        mitigation_callback=run_mitigation_from_row,
        city=selected_city,
    )

    st.dataframe(mitigation_report_df, use_container_width=True)

    st.download_button(
        label="Download mitigation report CSV",
        data=mitigation_report_to_csv_bytes(mitigation_report_df),
        file_name=f"mitigation_report_{selected_city}.csv",
        mime="text/csv",
    )

    # Optional: full report for all cities
    all_report_df = build_mitigation_report_for_all_days(
        prediction_df=prediction_df,
        mitigation_callback=run_mitigation_from_row,
        city=None,
    )

    st.download_button(
        label="Download mitigation report for all cities",
        data=mitigation_report_to_csv_bytes(all_report_df),
        file_name="mitigation_report_all_cities.csv",
        mime="text/csv",
    )
# ---------------------------------------------------------------------------
"""
