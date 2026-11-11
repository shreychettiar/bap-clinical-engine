"""Headless BAP query layer for Alyf Advisor.

Decoupled from dashboard slider / view state. All functions are read-only
against the full loaded `bap_dashboard_bundle` (and its daily_series).

Public API (names match the product contract):
  resolve_timeframe(bundle, phrase) -> {ok, start, end, ...}
  score_window(...)                 -> via advisor_context.score_window
  get_metric(bundle, variable, start, end, aggregation)
"""

from __future__ import annotations

import re
import statistics
from datetime import date as _date
from datetime import timedelta

import advisor_context as C
import advisor_score_window as SW

# Plain-language aliases → daily_series / engine keys
VARIABLE_ALIASES = {
    "sleep": "sleep_duration_min",
    "sleep duration": "sleep_duration_min",
    "sleep time": "sleep_duration_min",
    "hours of sleep": "sleep_duration_min",
    "sleep_duration": "sleep_duration_min",
    "sleep_duration_min": "sleep_duration_min",
    "resting heart rate": "resting_hr",
    "resting hr": "resting_hr",
    "rhr": "resting_hr",
    "resting_hr": "resting_hr",
    "hrv": "hrv_rmssd",
    "heart rate variability": "hrv_rmssd",
    "hrv_rmssd": "hrv_rmssd",
    "steps": "steps",
    "daily steps": "steps",
    "active calories": "calories_active",
    "calories": "calories_active",
    "calories_active": "calories_active",
    "active_calories": "calories_active",
    "vigorous": "vigorous_intensity_seconds",
    "vigorous minutes": "vigorous_intensity_seconds",
    "vigorous intensity": "vigorous_intensity_seconds",
    "vigorous_intensity_seconds": "vigorous_intensity_seconds",
    "moderate": "moderate_intensity_seconds",
    "moderate intensity": "moderate_intensity_seconds",
    "moderate_intensity_seconds": "moderate_intensity_seconds",
    "low intensity": "low_intensity_seconds",
    "low_intensity_seconds": "low_intensity_seconds",
    "inactivity": "inactivity_seconds",
    "sedentary": "inactivity_seconds",
    "inactivity_seconds": "inactivity_seconds",
    "standing": "low_intensity_seconds",  # best available proxy on Oura trends
    "breathing rate": "breaths_per_min",
    "respiratory rate": "breaths_per_min",
    "breaths_per_min": "breaths_per_min",
    "temperature": "temperature_delta",
    "temperature_delta": "temperature_delta",
    "met": "met_level",
    "met level": "met_level",
    "average met": "met_level",
    "metabolic equivalent": "met_level",
    "met_level": "met_level",
    "distance": "distance",
}

AGGREGATIONS = {
    "avg",
    "average",
    "mean",
    "min",
    "max",
    "sum",
    "median",
    "latest",
    "count",
    "series",
    "trend",
}


def _data_span(bundle):
    dates = list(bundle.get("dates") or [])
    if not dates:
        return None, None, []
    return str(dates[0])[:10], str(dates[-1])[:10], dates


def resolve_timeframe(bundle, phrase, active_start=None, active_end=None):
    """Resolve natural-language range → inclusive start/end dates in the dataset.

    Relative phrases anchor to the DATA's latest timestamp (not wall-clock now)
    and clamp to the loaded span. Never invents days outside the bind.
    """
    data_start, data_end, dates = _data_span(bundle)
    if not dates:
        return {
            "ok": False,
            "missing_data": True,
            "error": "no dates in loaded dataset",
        }
    low = (phrase or "").lower().strip()
    end_d = _date.fromisoformat(data_end)
    start_d = _date.fromisoformat(data_start)

    def _ok(s_iso, e_iso, label=None, clamped=False):
        si, ei, err = C.resolve_date_indices(bundle, s_iso, e_iso)
        if err:
            return {
                "ok": False,
                "missing_data": True,
                "error": err,
                "data_start": data_start,
                "data_end": data_end,
            }
        return {
            "ok": True,
            "missing_data": False,
            "start": dates[si],
            "end": dates[ei],
            "start_idx": si,
            "end_idx": ei,
            "label": label or f"{dates[si]} → {dates[ei]}",
            "clamped": clamped,
            "data_start": data_start,
            "data_end": data_end,
            "window_days": ei - si + 1,
        }

    # Full span / since start
    if any(
        k in low
        for k in (
            "since i started",
            "since the start",
            "full range",
            "full window",
            "entire range",
            "all my data",
            "whole period",
            "everything loaded",
            "all available",
        )
    ):
        return _ok(data_start, data_end, label="full loaded dataset")

    # this month (calendar month of data end, clamped)
    if "this month" in low or "current month" in low:
        month_start = end_d.replace(day=1)
        s = max(month_start, start_d)
        return _ok(s.isoformat(), data_end, label="this month (data-clamped)")

    # last / past N days or weeks
    m = re.search(
        r"\b(?:last|past|previous|prior)\s+(\d+)\s*(day|days|week|weeks)\b",
        low,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        days = n * 7 if unit.startswith("week") else n
        # inclusive window of `days` ending at data_end
        s = end_d - timedelta(days=max(0, days - 1))
        clamped = s < start_d
        if clamped:
            s = start_d
        return _ok(
            s.isoformat(),
            data_end,
            label=f"last {n} {unit} (anchored to data end {data_end})",
            clamped=clamped,
        )

    # past/last two weeks, couple weeks (no digit)
    if re.search(r"\b(past|last|previous)\s+(two|2)\s+weeks?\b", low) or re.search(
        r"\b(past|last)\s+couple\s+(of\s+)?weeks?\b", low
    ):
        s = end_d - timedelta(days=13)
        clamped = s < start_d
        if clamped:
            s = start_d
        return _ok(
            s.isoformat(),
            data_end,
            label=f"past two weeks (anchored to {data_end})",
            clamped=clamped,
        )

    if re.search(r"\b(past|last)\s+(few)\s+weeks?\b", low):
        s = end_d - timedelta(days=20)
        clamped = s < start_d
        if clamped:
            s = start_d
        return _ok(s.isoformat(), data_end, label="past few weeks", clamped=clamped)

    # last 30 days / past month-ish without "this month"
    if re.search(r"\b(last|past)\s+month\b", low) or "past 30" in low or "last 30" in low:
        s = end_d - timedelta(days=29)
        clamped = s < start_d
        if clamped:
            s = start_d
        return _ok(s.isoformat(), data_end, label="last ~30 days", clamped=clamped)

    # Delegate to extract_date_mentions for absolute / weekend / etc.
    p_start, p_end = SW.extract_date_mentions(phrase, dates)
    if p_start:
        return _ok(p_start, p_end or p_start, label=f"{p_start} → {p_end or p_start}")

    # active window fallback if phrase is "baseline" / "current window"
    if any(k in low for k in ("baseline", "current window", "active window", "my window")):
        if active_start is not None and active_end is not None:
            return _ok(
                dates[int(active_start)] if isinstance(active_start, int) else active_start,
                dates[int(active_end)] if isinstance(active_end, int) else active_end,
                label="active/baseline window",
            )

    return {
        "ok": False,
        "missing_data": False,
        "error": "could not resolve a concrete timeframe from the phrase",
        "data_start": data_start,
        "data_end": data_end,
        "phrase": phrase,
    }


def resolve_variable(name: str):
    """Map plain language or key → canonical daily_series key."""
    if not name:
        return None
    key = str(name).strip().lower()
    key = re.sub(r"\s+", " ", key)
    if key in VARIABLE_ALIASES:
        return VARIABLE_ALIASES[key]
    # fuzzy contains
    for alias, canon in VARIABLE_ALIASES.items():
        if alias in key or key in alias:
            return canon
    # raw key present as-is
    return name if name in VARIABLE_ALIASES.values() else None


def _numeric_vals(series_slice):
    out = []
    for v in series_slice:
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def get_metric(bundle, variable, start=None, end=None, aggregation="avg"):
    """Aggregate one underlying variable over any window (read-only).

    aggregation ∈ {avg,min,max,sum,median,latest,count,series,trend}
    """
    data_start, data_end, dates = _data_span(bundle)
    if not dates:
        return {"ok": False, "missing_data": True, "error": "no dates in loaded dataset"}

    canon = resolve_variable(variable)
    if not canon:
        return {
            "ok": False,
            "missing_data": False,
            "error": f"unknown variable '{variable}'",
            "supported": sorted(set(VARIABLE_ALIASES.values())),
        }

    agg = (aggregation or "avg").lower().strip()
    if agg in ("average", "mean"):
        agg = "avg"
    if agg not in AGGREGATIONS and agg not in ("avg",):
        return {"ok": False, "error": f"unknown aggregation '{aggregation}'"}

    si, ei, err = C.resolve_date_indices(bundle, start, end)
    if err:
        return {
            "ok": False,
            "missing_data": True,
            "error": err,
            "data_start": data_start,
            "data_end": data_end,
        }

    ds = bundle.get("daily_series") or {}
    series = ds.get(canon)
    if not isinstance(series, list):
        return {
            "ok": False,
            "missing_data": True,
            "error": f"variable '{canon}' not present in loaded daily series",
            "variable": canon,
        }

    slice_vals = series[si : ei + 1]
    win_dates = dates[si : ei + 1]
    nums = _numeric_vals(slice_vals)

    # MET eligibility note (display OK; scoring may exclude)
    met_note = None
    if canon == "met_level":
        eligible = bool(bundle.get("met_scoring_eligible"))
        form = bundle.get("met_form") or (
            "intraday" if eligible else "interday_daily_average"
        )
        if not eligible:
            met_note = (
                f"MET form={form}: shown for discussion but NOT used in BAP/axis "
                "scoring (daily-average / interday MET is excluded; no imputation)."
            )
        else:
            met_note = f"MET form={form}: eligible for scoring (intraday series present)."

    base = {
        "ok": True,
        "missing_data": False,
        "variable": canon,
        "variable_plain": C.plain_var(canon),
        "aggregation": agg,
        "start": win_dates[0],
        "end": win_dates[-1],
        "start_idx": si,
        "end_idx": ei,
        "window_days": ei - si + 1,
        "n_values": len(nums),
        "met_note": met_note,
    }

    if agg == "count":
        base["value"] = len(nums)
        return base
    if agg == "series":
        base["series"] = [
            {"date": win_dates[i], "value": slice_vals[i]} for i in range(len(win_dates))
        ]
        return base
    if not nums:
        return {
            **base,
            "ok": False,
            "missing_data": True,
            "error": f"no numeric values for {canon} in {win_dates[0]} → {win_dates[-1]}",
        }
    if agg == "avg":
        base["value"] = round(sum(nums) / len(nums), 4)
    elif agg == "min":
        base["value"] = round(min(nums), 4)
    elif agg == "max":
        base["value"] = round(max(nums), 4)
    elif agg == "sum":
        base["value"] = round(sum(nums), 4)
    elif agg == "median":
        base["value"] = round(statistics.median(nums), 4)
    elif agg == "latest":
        base["value"] = round(nums[-1], 4)
    elif agg == "trend":
        # simple first→last delta and slope per day
        first, last = nums[0], nums[-1]
        base["value"] = round(last - first, 4)
        base["first"] = round(first, 4)
        base["last"] = round(last, 4)
        days = max(1, len(nums) - 1)
        base["slope_per_day"] = round((last - first) / days, 4)
    else:
        base["value"] = round(sum(nums) / len(nums), 4)

    # Friendly sleep hours twin
    if canon == "sleep_duration_min" and isinstance(base.get("value"), (int, float)):
        if agg in ("avg", "min", "max", "median", "latest"):
            base["value_hours"] = round(float(base["value"]) / 60.0, 2)
        elif agg == "sum":
            base["value_hours"] = round(float(base["value"]) / 60.0, 2)
    return base


def format_timeframe_block(tf):
    if not tf or not tf.get("ok"):
        return (
            f"resolveTimeframe FAILED: {(tf or {}).get('error') or 'unknown'}. "
            f"Loaded data span: {(tf or {}).get('data_start')} → {(tf or {}).get('data_end')}."
        )
    bits = [
        f"resolveTimeframe → {tf.get('start')} → {tf.get('end')} "
        f"({tf.get('window_days')} days; {tf.get('label')})"
    ]
    if tf.get("clamped"):
        bits.append(
            "(clamped to loaded data span — do not claim coverage outside this)"
        )
    bits.append(
        f"Dataset span: {tf.get('data_start')} → {tf.get('data_end')}. "
        "You computed this headlessly; NEVER ask the patient to move the slider."
    )
    return " ".join(bits)


def format_metric_block(m):
    if not m or not m.get("ok"):
        err = (m or {}).get("error") or "unknown"
        return (
            f"getMetric FAILED: {err}. "
            "If the range is outside the loaded dataset, say exactly what is missing. "
            "Do NOT ask the user to move the dashboard slider."
        )
    plain = m.get("variable_plain") or m.get("variable")
    agg = m.get("aggregation")
    lines = [
        f"getMetric({plain}, {m.get('start')} → {m.get('end')}, {agg}) "
        f"— real values from the full daily series (headless; no UI change):",
    ]
    if "value_hours" in m and m.get("variable") == "sleep_duration_min":
        lines.append(
            f"  Result: {m['value_hours']} hours "
            f"({m.get('value')} minutes), n={m.get('n_values')} days"
        )
    elif agg == "series":
        # compact preview
        series = m.get("series") or []
        lines.append(f"  Series length: {len(series)} points (not dumping all)")
        head = series[:3]
        tail = series[-2:] if len(series) > 5 else []
        for pt in head + tail:
            lines.append(f"    {pt.get('date')}: {pt.get('value')}")
    elif agg == "trend":
        lines.append(
            f"  Delta end−start: {m.get('value')}; first {m.get('first')} → last {m.get('last')}; "
            f"slope/day {m.get('slope_per_day')}"
        )
    else:
        lines.append(f"  Result: {m.get('value')} (n={m.get('n_values')} days with values)")
    if m.get("met_note"):
        lines.append(f"  NOTE: {m['met_note']}")
    return "\n".join(lines)


def detect_metric_request(text: str):
    """If the user asks for a variable aggregate, return (variable, aggregation) else None."""
    low = (text or "").lower()
    if not low:
        return None
    # aggregation cue
    agg = "avg"
    if re.search(r"\b(median)\b", low):
        agg = "median"
    elif re.search(r"\b(sum|total)\b", low):
        agg = "sum"
    elif re.search(r"\b(maximum|max|highest|peak)\b", low):
        agg = "max"
    elif re.search(r"\b(minimum|min|lowest)\b", low):
        agg = "min"
    elif re.search(r"\b(latest|most recent|last night|last value)\b", low):
        agg = "latest"
    elif re.search(r"\b(trend|changed|change in|going up|going down)\b", low):
        agg = "trend"
    elif re.search(r"\b(average|avg|mean|typically|usually)\b", low):
        agg = "avg"
    elif re.search(r"\b(how much sleep|how many steps|how much)\b", low):
        agg = "avg"
    else:
        # still allow bare "what was my sleep over X" as avg
        if not re.search(
            r"\b(sleep|steps|hrv|rhr|resting|calories|met|distance|breath|inactiv|vigorous|moderate)\b",
            low,
        ):
            return None

    # variable detection (order matters — longer phrases first)
    candidates = sorted(VARIABLE_ALIASES.keys(), key=len, reverse=True)
    for alias in candidates:
        if alias in low:
            return resolve_variable(alias), agg
    return None


def detect_wants_any_query(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    keys = (
        "average",
        "avg",
        "over the",
        "past ",
        "last ",
        "score for",
        "score from",
        "how did",
        "compare",
        "till now",
        "until now",
        "this month",
        "two weeks",
        "30 days",
        "my sleep",
        "my steps",
        "my hrv",
        "resting",
        "what was my",
        "what's my",
        "how much",
        "how many",
        "isolated",
        "window",
        "between ",
        "from july",
        "from june",
        "from may",
    )
    return any(k in low for k in keys)