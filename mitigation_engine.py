#!/usr/bin/env python3
"""
Mitigation Engine for Dust + Drought + Flood (post-prediction decision layer)

What this module does
---------------------
Given ONE prediction row (per city, date), it returns:
- risk summary (hazards detected + overall risk level)
- recommended mitigation actions (ranked)
- action triggers (why the action fired)
- optional logging payload (for S3/DB)

Design goals
------------
1) Transparent & auditable (rule-based, no black-box post-processing).
2) Easy to extend (add new hazard modules and actions without changing the dashboard).
3) Dashboard-friendly outputs (lists of "action cards").

Expected inputs (typical)
-------------------------
Dust (examples):
- dust_event_pred, dust_event_prob, dust_intensity_level, dust_intensity_code
- pm10, pm25, aod, wind_speed_mean (or wind_speed_10m_pred)

Drought (examples):
- drought_flag_pred, drought_severity_pred
- precip_deficit_pred, drought_duration_pred, precipitation_sum_pred

Flood / heavy-rainfall (examples):
- precipitation_sum_pred (daily rainfall, mm)
- precipitation_sum_normal (daily climatological normal, mm) [optional but improves anomaly detection]

Notes
-----
- The mitigation engine DOES NOT update or overwrite your prediction dataset.
- It reads one row and produces a decision/alert layer for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
from datetime import datetime


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        return int(float(x))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _norm_severity_str(s: Any) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip().lower()
    mapping = {
        "0": "none",
        "1": "moderate",
        "2": "severe",
        "3": "extreme",
        "no": "none",
        "yes": "severe",
    }
    return mapping.get(s, s)


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class Action:
    action_id: str
    hazard: str              # "dust" | "drought" | "flood"
    sector: str              # "health", "transport", "water", "agriculture", "environment", "operations", "emergency"
    title: str
    description: str
    lead_agency: str
    time_to_act: str         # "0-6h", "6-24h", "2-7d", "2-4w", "seasonal"
    cost_level: str          # "low", "medium", "high"
    priority: int            # higher = more urgent
    triggers: List[str]      # explanation strings


@dataclass
class MitigationResult:
    city: str
    date: str
    hazards_detected: List[str]
    risk_level: str
    actions: List[Action]
    debug: Dict[str, Any]


# ---------------------------------------------------------------------
# Default action catalog
# ---------------------------------------------------------------------

def default_action_catalog() -> List[Dict[str, Any]]:
    """
    A compact but complete catalog.
    Tailor 'lead_agency' to your local context (e.g., Mosul municipality, civil defense, health directorate).
    """
    return [
        # -------------------- DUST: HEALTH --------------------
        dict(hazard="dust", sector="health", title="Public health alert (masking + indoor stay)",
             description="Issue advisory to reduce outdoor exposure; recommend N95/FFP2 masks for high-risk groups; promote indoor air filtration where possible.",
             lead_agency="Health Directorate", time_to_act="0-6h", cost_level="low"),
        dict(hazard="dust", sector="health", title="Healthcare readiness (respiratory surge planning)",
             description="Notify hospitals/clinics to prepare for increased respiratory cases; ensure inhalers/oxygen availability.",
             lead_agency="Health Directorate", time_to_act="6-24h", cost_level="medium"),

        # -------------------- DUST: TRANSPORT/OPERATIONS --------------------
        dict(hazard="dust", sector="transport", title="Traffic visibility protocol",
             description="Coordinate with traffic police: temporary speed limits, warning signs, consider restricting highway travel if visibility is very low.",
             lead_agency="Traffic Police / Municipality", time_to_act="0-6h", cost_level="low"),
        dict(hazard="dust", sector="transport", title="Aviation/port advisory",
             description="Inform airport/port operations of high dust risk; review contingency for delays and safety checks.",
             lead_agency="Transport Authority", time_to_act="6-24h", cost_level="low"),
        dict(hazard="dust", sector="operations", title="Outdoor work restriction (construction + municipal workers)",
             description="Advise limiting outdoor work during peak dust hours; provide PPE to essential workers.",
             lead_agency="Municipality / Labor Dept", time_to_act="0-6h", cost_level="low"),

        # -------------------- DUST: ENVIRONMENT (short-term) --------------------
        dict(hazard="dust", sector="environment", title="Dust suppression hotspot actions",
             description="Target known dust hotspots: street sweeping/wetting, construction dust control, enforce covering of loose materials.",
             lead_agency="Municipality / Environment Dept", time_to_act="2-7d", cost_level="medium"),

        # -------------------- DROUGHT: WATER --------------------
        dict(hazard="drought", sector="water", title="Water conservation advisory (public + institutions)",
             description="Launch water-saving advisories; reduce non-essential use; coordinate with schools/hospitals for conservation plans.",
             lead_agency="Water Directorate", time_to_act="2-7d", cost_level="low"),
        dict(hazard="drought", sector="water", title="Escalated rationing plan (staged restrictions)",
             description="Prepare staged rationing (Level 1/2/3) based on drought severity; prioritize drinking water and critical services.",
             lead_agency="Water Directorate", time_to_act="2-4w", cost_level="medium"),
        dict(hazard="drought", sector="water", title="Groundwater monitoring + extraction control",
             description="Increase monitoring of groundwater levels; coordinate restrictions to prevent over-extraction during severe drought.",
             lead_agency="Water Resources / Environment Dept", time_to_act="2-4w", cost_level="medium"),

        # -------------------- DROUGHT: AGRICULTURE --------------------
        dict(hazard="drought", sector="agriculture", title="Irrigation scheduling + drip irrigation support",
             description="Promote efficient irrigation schedules; support drip systems where feasible; reduce losses in channels.",
             lead_agency="Agriculture Directorate", time_to_act="2-4w", cost_level="high"),
        dict(hazard="drought", sector="agriculture", title="Crop advisory (shift to drought-tolerant varieties)",
             description="Provide guidance to farmers on crop switching, planting calendar adjustments, and drought-tolerant varieties.",
             lead_agency="Agriculture Directorate", time_to_act="seasonal", cost_level="medium"),

        # -------------------- DROUGHT: ENVIRONMENT (long-term) --------------------
        dict(hazard="drought", sector="environment", title="Anti-desertification and vegetation restoration plan",
             description="Plan shelterbelts/afforestation, soil conservation, and rangeland management to reduce dust sources and drought vulnerability.",
             lead_agency="Environment / Agriculture", time_to_act="seasonal", cost_level="high"),

        # -------------------- FLOOD: EMERGENCY / OPERATIONS --------------------
        dict(hazard="flood", sector="emergency", title="Heavy-rain alert + public safety guidance",
             description="Issue warning for heavy rain / flash flood risk; advise against crossing wadis/low bridges; share emergency contact numbers.",
             lead_agency="Civil Defense / Municipality", time_to_act="0-6h", cost_level="low"),
        dict(hazard="flood", sector="operations", title="Drainage and culvert inspection (hotspots)",
             description="Inspect and clear stormwater drains, culverts, and known blockage points; deploy rapid-response municipal teams.",
             lead_agency="Municipality / Water & Sewage", time_to_act="0-6h", cost_level="medium"),
        dict(hazard="flood", sector="transport", title="Road closures and traffic rerouting plan",
             description="Prepare to close flooded underpasses/low areas; coordinate detours; place warning signage and barriers.",
             lead_agency="Traffic Police / Municipality", time_to_act="0-6h", cost_level="low"),
        dict(hazard="flood", sector="emergency", title="Pre-position pumps/sandbags and response teams",
             description="Stage portable pumps, sandbags, and rescue teams near vulnerable districts; verify shelters and evacuation routes if needed.",
             lead_agency="Civil Defense", time_to_act="6-24h", cost_level="high"),
    ]


# ---------------------------------------------------------------------
# Rule modules
# ---------------------------------------------------------------------

def _dust_severity_from_row(row: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Returns dust severity level 0..3 and trigger notes.
    Uses dust_intensity_level if present; otherwise derives from pollutants/wind proxies.
    """
    notes: List[str] = []
    lvl: Optional[int] = None

    # If your pipeline uses numeric codes 0..3 here
    if "dust_intensity_level" in row:
        lvl = _safe_int(row.get("dust_intensity_level"), 0)
        notes.append(f"dust_intensity_level={lvl}")

    if lvl is None:
        pm10 = _safe_float(row.get("pm10_pred") or row.get("pm10"), 0.0)
        pm25 = _safe_float(row.get("pm25_pred") or row.get("pm25"), 0.0)
        aod  = _safe_float(row.get("aod_pred")  or row.get("aod"),  0.0)
        wind = _safe_float(row.get("wind_speed_10m_pred") or row.get("wind_speed_mean") or row.get("wind_speed_10m"), 0.0)

        # Simple, transparent thresholds (can be tuned to local standards)
        if (pm10 > 450) or (pm25 > 250) or (aod > 1.2):
            lvl = 3
        elif (pm10 > 300) or (pm25 > 150) or (aod > 0.9):
            lvl = 2
        elif (pm10 > 200) or (pm25 > 100) or (aod > 0.7) or ((aod > 0.8) and (wind > 8)):
            lvl = 1
        else:
            lvl = 0
        notes.append("dust:derived_from_pm_aod_wind")

    lvl = max(0, min(3, int(lvl)))
    return lvl, notes


def _drought_severity_from_row(row: Dict[str, Any]) -> Tuple[str, List[str]]:
    notes: List[str] = []
    sev = "unknown"

    # Prefer categorical drought severity
    if "drought_severity_pred" in row:
        sev = _norm_severity_str(row.get("drought_severity_pred"))
        notes.append(f"drought_severity_pred={sev}")
    elif "drought_severity" in row:
        sev = _norm_severity_str(row.get("drought_severity"))
        notes.append(f"drought_severity={sev}")

    # If only drought_flag exists, approximate
    if sev in ("unknown", "", "nan") and ("drought_flag_pred" in row or "drought_flag" in row):
        flag = _safe_int(row.get("drought_flag_pred") or row.get("drought_flag"), 0)
        sev = "none" if flag == 0 else "severe"
        notes.append(f"drought_flag={flag}")

    # If still unknown, try proxies available in your current CSV
    if sev == "unknown":
        # Your file has: precip_deficit_pred and drought_duration_pred
        deficit = _safe_float(row.get("precip_deficit_pred"), 0.0)   # (pred - normal) or similar
        duration = _safe_float(row.get("drought_duration_pred"), 0.0)

        # Transparent heuristic: bigger deficit + longer duration => more severe drought
        if (deficit < -30.0) and (duration >= 30):
            sev = "extreme"
        elif (deficit < -20.0) and (duration >= 14):
            sev = "severe"
        elif (deficit < -10.0) or (duration >= 7):
            sev = "moderate"
        else:
            sev = "none"
        notes.append("drought:derived_from_deficit_duration")

    allowed = {"none", "moderate", "severe", "extreme"}
    if sev not in allowed:
        sev = "unknown"
    return sev, notes


def _flood_severity_from_row(row: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Flood / heavy-rainfall severity level 0..3 from precipitation prediction.

    Works with your current prediction CSV columns:
    - precipitation_sum_pred (daily rainfall, mm)
    - precipitation_sum_normal (daily normal, mm)  [optional]

    Heuristic thresholds (tune later with local hydrology):
    - lvl 1 (moderate): >= 30 mm/day OR (>= 20 and >= 2x normal)
    - lvl 2 (high):     >= 50 mm/day OR (>= 35 and >= 2x normal)
    - lvl 3 (extreme):  >= 80 mm/day OR (>= 60 and >= 2.5x normal)
    """
    notes: List[str] = []
    rain = _safe_float(row.get("precipitation_sum_pred") or row.get("precipitation_sum"), 0.0)
    normal = _safe_float(row.get("precipitation_sum_normal"), 0.0)

    ratio = None
    if normal and normal > 0:
        ratio = rain / normal
        notes.append(f"rain_ratio={ratio:.2f}")

    lvl = 0
    # Extreme
    if (rain >= 80.0) or (ratio is not None and rain >= 60.0 and ratio >= 2.5):
        lvl = 3
    # High
    elif (rain >= 50.0) or (ratio is not None and rain >= 35.0 and ratio >= 2.0):
        lvl = 2
    # Moderate
    elif (rain >= 30.0) or (ratio is not None and rain >= 20.0 and ratio >= 2.0):
        lvl = 1

    notes.append(f"precipitation_sum_pred={rain:.1f}mm")
    if normal:
        notes.append(f"precipitation_sum_normal={normal:.1f}mm")

    return lvl, notes


# ---------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------

def _priority_base(hazard: str, severity: Any) -> int:
    if hazard == "dust":
        return int(severity)  # 0..3
    if hazard == "drought":
        m = {"none": 0, "moderate": 1, "severe": 2, "extreme": 3}
        return m.get(str(severity), 0)
    if hazard == "flood":
        return int(severity)  # 0..3
    return 0


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def recommend_actions(
    row: Dict[str, Any],
    catalog: Optional[List[Dict[str, Any]]] = None,
    min_priority: int = 1,
) -> MitigationResult:
    """
    Main entry point.
    - row: one prediction record (city + date + predicted fields)
    - returns MitigationResult with ranked actions.
    """
    catalog = catalog or default_action_catalog()

    city = str(row.get("city", "Unknown")).strip()
    date = str(row.get("date") or row.get("timestamp") or row.get("day") or "Unknown").split(" ")[0]

    hazards: List[str] = []
    debug: Dict[str, Any] = {"input_keys": sorted(list(row.keys()))}

    # Detect dust
    dust_event = _safe_int(row.get("dust_event_pred") or row.get("dust_event"), 0)
    dust_lvl, dust_notes = _dust_severity_from_row(row)
    if dust_event == 1 or dust_lvl > 0:
        hazards.append("dust")
    debug["dust"] = {"event": dust_event, "level": dust_lvl, "notes": dust_notes}

    # Detect drought
    drought_flag = _safe_int(row.get("drought_flag_pred") or row.get("drought_flag"), 0)
    drought_sev, drought_notes = _drought_severity_from_row(row)
    if drought_flag == 1 or drought_sev in ("moderate", "severe", "extreme"):
        hazards.append("drought")
    debug["drought"] = {"flag": drought_flag, "severity": drought_sev, "notes": drought_notes}

    # Detect flood / heavy rain
    flood_lvl, flood_notes = _flood_severity_from_row(row)
    if flood_lvl > 0:
        hazards.append("flood")
    debug["flood"] = {"level": flood_lvl, "notes": flood_notes}

    # Overall risk label (max severity across hazards)
    risk_score = 0
    risk_score = max(risk_score, dust_lvl)
    risk_score = max(risk_score, _priority_base("drought", drought_sev))
    risk_score = max(risk_score, flood_lvl)

    risk_map = {0: "low", 1: "moderate", 2: "high", 3: "extreme"}
    risk_level = risk_map.get(risk_score, "unknown")

    # Build actions
    actions: List[Action] = []
    for item in catalog:
        hz = item["hazard"]
        if hz not in hazards:
            continue

        if hz == "dust":
            base = _priority_base("dust", dust_lvl)
        elif hz == "drought":
            base = _priority_base("drought", drought_sev)
        else:
            base = _priority_base("flood", flood_lvl)

        if base < min_priority:
            continue

        include = True
        trig: List[str] = []

        # ---------------- hazard-specific gating ----------------
        if hz == "dust":
            if dust_lvl >= 3:
                trig.append("dust_level>=3")
            elif dust_lvl == 2:
                trig.append("dust_level==2")
            elif dust_lvl == 1:
                if item["time_to_act"] not in ("0-6h", "6-24h"):
                    include = False
                trig.append("dust_level==1")

        elif hz == "drought":
            if drought_sev == "extreme":
                trig.append("drought=extreme")
            elif drought_sev == "severe":
                trig.append("drought=severe")
            elif drought_sev == "moderate":
                if item["time_to_act"] in ("seasonal",):
                    include = False
                trig.append("drought=moderate")

        elif hz == "flood":
            if flood_lvl >= 3:
                trig.append("flood_level>=3")
            elif flood_lvl == 2:
                trig.append("flood_level==2")
            elif flood_lvl == 1:
                # keep mainly immediate actions for moderate rainfall
                if item["time_to_act"] not in ("0-6h", "6-24h"):
                    include = False
                trig.append("flood_level==1")

        if not include:
            continue

        # Priority: base severity (0..3) scaled
        priority = 10 * base

        # Sector boosts (keeps urgent actions at top)
        if hz == "dust" and item["sector"] in ("health", "transport", "operations") and item["time_to_act"] in ("0-6h", "6-24h"):
            priority += 5
        if hz == "drought" and item["sector"] in ("water",) and item["time_to_act"] in ("2-7d", "2-4w"):
            priority += 5
        if hz == "flood" and item["sector"] in ("emergency", "operations", "transport") and item["time_to_act"] in ("0-6h", "6-24h"):
            priority += 5

        action_id = _stable_id(city, date, hz, item["sector"], item["title"])

        triggers = []
        triggers.extend(dust_notes)
        triggers.extend(drought_notes)
        triggers.extend(flood_notes)
        triggers.extend(trig)

        actions.append(Action(
            action_id=action_id,
            hazard=hz,
            sector=item["sector"],
            title=item["title"],
            description=item["description"],
            lead_agency=item["lead_agency"],
            time_to_act=item["time_to_act"],
            cost_level=item["cost_level"],
            priority=priority,
            triggers=sorted(list(set([t for t in triggers if t]))),
        ))

    actions.sort(key=lambda a: a.priority, reverse=True)

    return MitigationResult(
        city=city,
        date=date,
        hazards_detected=hazards,
        risk_level=risk_level,
        actions=actions,
        debug=debug,
    )


# ---------------------------------------------------------------------
# Helpers for dashboard integration
# ---------------------------------------------------------------------

def result_to_action_cards(result: MitigationResult) -> List[Dict[str, Any]]:
    """Convert MitigationResult into JSON-serializable dicts for Streamlit."""
    return [asdict(a) for a in result.actions]


def result_to_log_payload(result: MitigationResult, model_version: str = "unknown") -> Dict[str, Any]:
    """A compact log payload suitable for S3/DB storage."""
    return {
        "logged_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_version": model_version,
        "city": result.city,
        "date": result.date,
        "hazards_detected": result.hazards_detected,
        "risk_level": result.risk_level,
        "actions": [asdict(a) for a in result.actions],
        "debug": result.debug,
    }


# ---------------------------------------------------------------------
# CLI quick test (optional)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    sample = {
        "city": "Mosul",
        "date": "2026-03-03",
        "dust_event_pred": 0,
        "dust_intensity_level": 0,
        "pm10": 120,
        "pm25": 60,
        "aod": 0.2,
        "wind_speed_mean": 4.0,
        "drought_flag_pred": 0,
        "drought_severity_pred": "none",
        "precip_deficit_pred": -5.0,
        "drought_duration_pred": 0,
        "precipitation_sum_pred": 85.0,
        "precipitation_sum_normal": 20.0,
    }
    r = recommend_actions(sample)
    print(json.dumps(result_to_log_payload(r, model_version="demo_flood"), indent=2))
