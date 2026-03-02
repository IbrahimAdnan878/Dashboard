"""
Mitigation Engine (v3) — Dust + Drought + Flood (rule-based)

Purpose
- Reads ONE prediction row (dict-like) produced by your ML/pipeline.
- Detects hazard risks using robust column mappings.
- Returns a structured MitigationResult used by Streamlit to render action cards.

Design notes (important)
- Your current prediction CSV has many "binary/class" outputs always 0 / 'none'.
  Therefore v3 uses both:
    (A) model outputs (e.g., dust_event_pred, drought_severity_pred)
    (B) continuous signals (dust_event_prob, pm10, pm25, aod, wind_speed_mean, precip_deficit_pred)
- Flood detection relies on precipitation_sum_pred (currently all zeros in your file),
  so flood will not trigger until precipitation forecasts are non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# -----------------------------
# Helpers
# -----------------------------
def _get(row: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _to_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _norm_label(x: Any) -> str:
    s = _to_str(x, "").strip().lower()
    return s


# -----------------------------
# Result object
# -----------------------------
@dataclass
class Action:
    title: str
    hazard: str
    sector: str
    priority: str  # LOW / MEDIUM / HIGH
    description: str
    lead_agency: str
    time_to_act: str
    cost_level: str
    triggers: List[str] = field(default_factory=list)


@dataclass
class MitigationResult:
    city: str
    date: str
    risk_level: str  # low/moderate/high/extreme
    hazards_detected: List[str] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)


# -----------------------------
# Core logic
# -----------------------------
def recommend_actions(row: Dict[str, Any]) -> MitigationResult:
    """
    Main entry point used by Streamlit.

    Expected row keys (any subset is okay; v3 is robust):
      - city, timestamp/date
      - Dust: dust_event_pred, dust_event_prob, dust_intensity_level, pm10, pm25, aod, wind_speed_mean
      - Drought: drought_flag_pred, drought_severity_pred, precip_deficit_pred, drought_duration_pred
      - Flood: precipitation_sum_pred, precipitation_sum_normal
    """
    city = _to_str(_get(row, "city", default="Unknown"), "Unknown")

    # date handling (dashboard supplies either date or timestamp)
    date = _to_str(_get(row, "date", "timestamp", default="Unknown"), "Unknown")
    if " " in date:
        date = date.split(" ")[0]

    hazards: List[str] = []
    actions: List[Action] = []

    # -----------------------------
    # Dust detection (hybrid: model output + continuous signals)
    # -----------------------------
    dust_pred = _to_float(_get(row, "dust_event_pred", "dust_event", default=0.0))
    dust_prob = _to_float(_get(row, "dust_event_prob", default=0.0))
    dust_level = _norm_label(_get(row, "dust_intensity_level", default=""))
    pm10 = _to_float(_get(row, "pm10", "pm10_pred", default=0.0))
    pm25 = _to_float(_get(row, "pm25", "pm25_pred", default=0.0))
    aod = _to_float(_get(row, "aod", "aod_pred", default=0.0))
    wind = _to_float(_get(row, "wind_speed_mean", "wind_speed_10m_pred", default=0.0))

    dust_triggered = False
    dust_priority = "LOW"
    dust_triggers: List[str] = []

    # Strong triggers
    if dust_pred >= 1:
        dust_triggered = True
        dust_priority = "HIGH"
        dust_triggers.append("dust_event_pred>=1")

    if dust_level in {"severe", "high"}:
        dust_triggered = True
        dust_priority = "HIGH"
        dust_triggers.append(f"dust_intensity_level={dust_level}")

    # Probabilistic triggers (important for your current file where dust_event_pred is always 0)
    # Calibrated tiers:
    # - >=0.20: high confidence
    # - >=0.10: moderate
    # - >=0.05: low (useful because your current max is ~0.08)
    if dust_prob >= 0.20:
        dust_triggered = True
        dust_priority = "HIGH"
        dust_triggers.append("dust_event_prob>=0.20")
    elif dust_prob >= 0.10:
        dust_triggered = True
        dust_priority = max(dust_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        dust_triggers.append("dust_event_prob>=0.10")
    elif dust_prob >= 0.05:
        dust_triggered = True
        dust_priority = max(dust_priority, "LOW", key=["LOW","MEDIUM","HIGH"].index)
        dust_triggers.append("dust_event_prob>=0.05")

    # Air quality thresholds (WHO-style references are stricter; here we use pragmatic alert thresholds)
    if pm10 >= 150:
        dust_triggered = True
        dust_priority = "HIGH"
        dust_triggers.append("pm10>=150")
    elif pm10 >= 100:
        dust_triggered = True
        dust_priority = max(dust_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        dust_triggers.append("pm10>=100")

    if pm25 >= 75:
        dust_triggered = True
        dust_priority = "HIGH"
        dust_triggers.append("pm25>=75")
    elif pm25 >= 50:
        dust_triggered = True
        dust_priority = max(dust_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        dust_triggers.append("pm25>=50")

    # Remote-sensing proxy
    if aod >= 1.0:
        dust_triggered = True
        dust_priority = max(dust_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        dust_triggers.append("aod>=1.0")

    # Wind mobilization proxy (supports dust uplift)
    if wind >= 12:
        dust_triggered = True
        dust_priority = max(dust_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        dust_triggers.append("wind_speed_mean>=12")

    if dust_triggered:
        hazards.append("dust")
        actions.extend(_dust_actions(priority=dust_priority, triggers=dust_triggers))

    # -----------------------------
    # Drought detection (hybrid)
    # -----------------------------
    drought_flag = _to_float(_get(row, "drought_flag_pred", "drought_flag", default=0.0))
    drought_sev = _norm_label(_get(row, "drought_severity_pred", "drought_severity", default="none"))
    precip_def = _to_float(_get(row, "precip_deficit_pred", default=0.0))
    drought_dur = _to_float(_get(row, "drought_duration_pred", default=0.0))

    drought_triggered = False
    drought_priority = "LOW"
    drought_triggers: List[str] = []

    if drought_flag >= 1:
        drought_triggered = True
        drought_priority = "HIGH"
        drought_triggers.append("drought_flag_pred>=1")

    if drought_sev in {"severe", "extreme"}:
        drought_triggered = True
        drought_priority = "HIGH"
        drought_triggers.append(f"drought_severity_pred={drought_sev}")
    elif drought_sev in {"moderate"}:
        drought_triggered = True
        drought_priority = max(drought_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        drought_triggers.append(f"drought_severity_pred={drought_sev}")
    elif drought_sev in {"mild", "low"}:
        drought_triggered = True
        drought_priority = "LOW"
        drought_triggers.append(f"drought_severity_pred={drought_sev}")

    # Use precipitation deficit as a proxy when severity labels are 'none'
    if precip_def >= 10:
        drought_triggered = True
        drought_priority = "HIGH"
        drought_triggers.append("precip_deficit_pred>=10")
    elif precip_def >= 5:
        drought_triggered = True
        drought_priority = max(drought_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        drought_triggers.append("precip_deficit_pred>=5")
    elif precip_def >= 2:
        drought_triggered = True
        drought_priority = "LOW"
        drought_triggers.append("precip_deficit_pred>=2")

    # Duration (if your model starts producing it later)
    if drought_dur >= 21:
        drought_triggered = True
        drought_priority = "HIGH"
        drought_triggers.append("drought_duration_pred>=21")
    elif drought_dur >= 14:
        drought_triggered = True
        drought_priority = max(drought_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        drought_triggers.append("drought_duration_pred>=14")

    if drought_triggered:
        hazards.append("drought")
        actions.extend(_drought_actions(priority=drought_priority, triggers=drought_triggers))

    # -----------------------------
    # Flood detection (rule-based; currently will not trigger if precip is all zeros)
    # -----------------------------
    precip = _to_float(_get(row, "precipitation_sum_pred", default=0.0))
    precip_norm = _to_float(_get(row, "precipitation_sum_normal", default=0.0))

    flood_triggered = False
    flood_priority = "LOW"
    flood_triggers: List[str] = []

    # Absolute daily rainfall thresholds (mm/day)
    if precip >= 80:
        flood_triggered = True
        flood_priority = "HIGH"
        flood_triggers.append("precipitation_sum_pred>=80mm")
    elif precip >= 50:
        flood_triggered = True
        flood_priority = "MEDIUM"
        flood_triggers.append("precipitation_sum_pred>=50mm")
    elif precip >= 30:
        flood_triggered = True
        flood_priority = "LOW"
        flood_triggers.append("precipitation_sum_pred>=30mm")

    # Relative anomaly threshold (compares to "normal" daily climatology)
    if precip_norm > 0 and precip >= max(20.0, 2.0 * precip_norm):
        flood_triggered = True
        flood_priority = max(flood_priority, "MEDIUM", key=["LOW","MEDIUM","HIGH"].index)
        flood_triggers.append("precipitation_sum_pred>=2x_normal")

    if flood_triggered:
        hazards.append("flood")
        actions.extend(_flood_actions(priority=flood_priority, triggers=flood_triggers))

    # -----------------------------
    # Overall risk
    # -----------------------------
    risk_level = _aggregate_risk(hazards, actions)

    return MitigationResult(
        city=city,
        date=date,
        risk_level=risk_level,
        hazards_detected=hazards,
        actions=actions,
    )


# -----------------------------
# Action libraries
# -----------------------------
def _dust_actions(priority: str, triggers: List[str]) -> List[Action]:
    base = [
        Action(
            title="Public health advisory (respiratory protection)",
            hazard="dust",
            sector="Health",
            priority=priority,
            description="Issue advisories for vulnerable groups; recommend masks, reduce outdoor exposure, and prepare clinics for respiratory cases.",
            lead_agency="Health Directorate / Environmental Authority",
            time_to_act="Within 6–24 hours",
            cost_level="Low",
            triggers=triggers,
        ),
        Action(
            title="Traffic and road safety measures",
            hazard="dust",
            sector="Transport",
            priority=priority,
            description="Warn drivers about reduced visibility; prepare road patrols; consider limiting heavy transport during peak dust periods.",
            lead_agency="Traffic Police / Municipality",
            time_to_act="Within 6–24 hours",
            cost_level="Low",
            triggers=triggers,
        ),
    ]
    if priority in {"MEDIUM", "HIGH"}:
        base.append(
            Action(
                title="Sensitive sites protection (schools, hospitals)",
                hazard="dust",
                sector="Public Services",
                priority=priority,
                description="Improve indoor air filtration where available; limit outdoor school activities; ensure backup power for clinics.",
                lead_agency="Education Directorate / Hospital Admins",
                time_to_act="Within 24 hours",
                cost_level="Medium",
                triggers=triggers,
            )
        )
    return base


def _drought_actions(priority: str, triggers: List[str]) -> List[Action]:
    base = [
        Action(
            title="Water demand management",
            hazard="drought",
            sector="Water",
            priority=priority,
            description="Implement phased demand reduction, leakage control, and public water-saving campaigns; prioritize critical facilities.",
            lead_agency="Water Resources Directorate / Municipality",
            time_to_act="Within 1–7 days",
            cost_level="Medium",
            triggers=triggers,
        ),
        Action(
            title="Agricultural irrigation scheduling",
            hazard="drought",
            sector="Agriculture",
            priority=priority,
            description="Shift to deficit irrigation plans; prioritize high-value crops; promote efficient irrigation and drought-resilient practices.",
            lead_agency="Agriculture Directorate",
            time_to_act="Within 1–14 days",
            cost_level="Medium",
            triggers=triggers,
        ),
    ]
    if priority in {"MEDIUM", "HIGH"}:
        base.append(
            Action(
                title="Reservoir and groundwater monitoring",
                hazard="drought",
                sector="Water",
                priority=priority,
                description="Increase monitoring of reservoirs and wells; prepare emergency allocation rules; coordinate with power sector if needed.",
                lead_agency="Water Resources Directorate",
                time_to_act="Within 7–21 days",
                cost_level="Medium",
                triggers=triggers,
            )
        )
    return base


def _flood_actions(priority: str, triggers: List[str]) -> List[Action]:
    base = [
        Action(
            title="Drainage inspection and debris clearing",
            hazard="flood",
            sector="Municipal",
            priority=priority,
            description="Inspect stormwater drains; clear debris; ensure pumping stations are operational; prioritize known hotspots.",
            lead_agency="Municipality / Civil Defense",
            time_to_act="Within 6–24 hours",
            cost_level="Medium",
            triggers=triggers,
        ),
        Action(
            title="Early warning and emergency readiness",
            hazard="flood",
            sector="Emergency",
            priority=priority,
            description="Prepare response teams; issue warnings for low-lying areas; coordinate shelter readiness and rapid response routes.",
            lead_agency="Civil Defense / Governorate Emergency Cell",
            time_to_act="Within 6–24 hours",
            cost_level="Medium",
            triggers=triggers,
        ),
    ]
    if priority == "HIGH":
        base.append(
            Action(
                title="High-risk zone evacuation planning",
                hazard="flood",
                sector="Emergency",
                priority=priority,
                description="Prepare evacuation advisories for high-risk zones; ensure communication with local councils and critical infrastructure operators.",
                lead_agency="Civil Defense / Police",
                time_to_act="Immediately",
                cost_level="High",
                triggers=triggers,
            )
        )
    return base


def _aggregate_risk(hazards: List[str], actions: List[Action]) -> str:
    if not hazards:
        return "low"

    # Determine max priority among actions
    pr_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    max_pr = 1
    for a in actions:
        max_pr = max(max_pr, pr_map.get(a.priority, 1))

    # Simple rule: multiple hazards increases severity
    if len(hazards) >= 2 and max_pr >= 2:
        return "high"
    if max_pr == 3:
        return "high"
    if max_pr == 2:
        return "moderate"
    return "low"


# -----------------------------
# Streamlit helpers
# -----------------------------
def result_to_action_cards(result: MitigationResult) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for a in result.actions:
        cards.append(
            {
                "title": a.title,
                "hazard": a.hazard,
                "sector": a.sector,
                "priority": a.priority,
                "description": a.description,
                "lead_agency": a.lead_agency,
                "time_to_act": a.time_to_act,
                "cost_level": a.cost_level,
                "triggers": a.triggers,
            }
        )

    # Sort by priority HIGH > MEDIUM > LOW
    pr_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    cards.sort(key=lambda x: pr_order.get(x["priority"], 9))
    return cards
