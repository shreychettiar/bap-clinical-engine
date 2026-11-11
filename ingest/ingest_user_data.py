"""Ingest user-uploaded wearable CSV/JSON, validate schema, score via live BAP engine,
and rebuild bap_dashboard_bundle. Conditions NEVER passed into score_bap.
"""
from executor import aio
import types, sys, json, csv, io, re
from datetime import datetime

# Canonical Oura-style headers + common aliases → internal field used by build_day
OURA_ALIASES = {
    "date": ["date", "day", "Date", "DAY", "timestamp", "Timestamp"],
    "High Activity Time": [
        "High Activity Time", "high_activity_time", "high activity time",
        "Vigorous Activity", "vigorous_intensity_seconds",
    ],
    "Medium Activity Time": [
        "Medium Activity Time", "medium_activity_time", "medium activity time",
        "Moderate Activity", "moderate_intensity_seconds",
    ],
    "Low Activity Time": [
        "Low Activity Time", "low_activity_time", "low activity time",
        "Light Activity", "low_intensity_seconds",
    ],
    "Steps": ["Steps", "steps", "step_count", "Step Count"],
    "Activity Burn": [
        "Activity Burn", "activity_burn", "Active Calories", "active_calories",
        "calories_active", "Calories Active",
    ],
    "Equivalent Walking Distance": [
        "Equivalent Walking Distance", "equivalent_walking_distance",
        "Distance", "distance", "Walking Distance",
    ],
    "Inactive Time": [
        "Inactive Time", "inactive_time", "Inactivity", "inactivity_seconds",
        "Sedentary Time",
    ],
    "Non-wear Time": ["Non-wear Time", "non_wear_time", "Nonwear Time", "nonwear_time"],
    "Total Sleep Duration": [
        "Total Sleep Duration", "total_sleep_duration", "Sleep Duration",
        "sleep_duration", "sleep_duration_min",
    ],
    "Average MET": ["Average MET", "average_met", "met_level", "Average Met", "MET"],
    "Average HRV": ["Average HRV", "average_hrv", "HRV", "hrv_rmssd", "HRV RMSSD"],
    "Average Resting Heart Rate": [
        "Average Resting Heart Rate", "average_resting_heart_rate",
        "Resting Heart Rate", "resting_hr", "RHR",
    ],
    "Respiratory Rate": [
        "Respiratory Rate", "respiratory_rate", "breaths_per_min", "Breathing Rate",
    ],
    "Temperature Deviation (°C)": [
        "Temperature Deviation (°C)", "Temperature Deviation (\u00b0C)",
        "temperature_delta", "Temperature Deviation", "Temp Deviation",
    ],
}

# At least one activity/recovery signal beyond date is required
REQUIRED_SIGNAL_GROUPS = [
    ["Steps", "Activity Burn", "High Activity Time", "Medium Activity Time", "Average MET"],
    ["Average HRV", "Average Resting Heart Rate", "Total Sleep Duration"],
]


def _norm(h):
    return re.sub(r"\s+", " ", str(h or "").strip().lower())


def build_header_map(headers):
    """Map canonical Oura column → actual header present in file."""
    norm_to_actual = {_norm(h): h for h in headers}
    mapping = {}
    for canon, aliases in OURA_ALIASES.items():
        for a in aliases:
            key = _norm(a)
            if key in norm_to_actual:
                mapping[canon] = norm_to_actual[key]
                break
    return mapping


def parse_date(v):
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip()
    # ISO date or datetime
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19] if "T" in s or " " in s else s[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # epoch seconds / ms
    try:
        n = float(s)
        if n > 1e12:
            n = n / 1000.0
        if n > 1e9:
            return datetime.utcfromtimestamp(n).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        pass
    return None


def parse_csv_text(text):
    # strip BOM
    if text.startswith("\ufeff"):
        text = text[1:]
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("CSV has no header row. Expected a first row of column names (e.g. date, Steps, Average HRV).")
    headers = list(reader.fieldnames)
    rows = []
    bad = []
    for i, row in enumerate(reader, start=2):
        if row is None:
            continue
        # skip fully empty rows
        if all(v is None or str(v).strip() == "" for v in row.values()):
            continue
        rows.append(dict(row))
        if len(rows) > 5000:
            raise ValueError("File has more than 5,000 data rows. Please upload a smaller export.")
    if not rows:
        raise ValueError("CSV has a header but no data rows.")
    return headers, rows


def parse_json_text(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}.")
    if isinstance(data, dict):
        # common wrappers
        for k in ("rows", "data", "days", "records", "trends"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            raise ValueError(
                "JSON object must contain an array under one of: rows, data, days, records, trends — "
                "or be a top-level array of day objects."
            )
    if not isinstance(data, list):
        raise ValueError("JSON must be an array of day objects, or an object wrapping such an array.")
    if not data:
        raise ValueError("JSON array is empty.")
    if not all(isinstance(r, dict) for r in data[:50]):
        raise ValueError("JSON array items must be objects with column keys.")
    # union of keys as headers
    headers = []
    seen = set()
    for r in data:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)
    return headers, [dict(r) for r in data]


def remap_rows(headers, rows, header_map):
    """Produce Oura-canonical rows with a 'date' field."""
    if "date" not in header_map:
        raise ValueError(
            "Missing required column: date. "
            f"Found columns: {', '.join(headers[:20])}{'…' if len(headers) > 20 else ''}. "
            "Expected a date/day column (e.g. date, Date, timestamp)."
        )
    # need at least one signal from either group
    has_signal = False
    for group in REQUIRED_SIGNAL_GROUPS:
        if any(c in header_map for c in group):
            has_signal = True
            break
    if not has_signal:
        raise ValueError(
            "Could not find any activity or recovery columns. "
            "Need at least one of: Steps, Activity Burn, High/Medium Activity Time, Average MET, "
            "Average HRV, Average Resting Heart Rate, or Total Sleep Duration. "
            f"Found: {', '.join(headers[:25])}{'…' if len(headers) > 25 else ''}."
        )

    out = []
    bad_dates = []
    for i, row in enumerate(rows):
        src_date_key = header_map["date"]
        raw_date = row.get(src_date_key)
        d = parse_date(raw_date)
        if d is None:
            bad_dates.append(i + 1)  # 1-based data row approx
            continue
        canon = {"date": d}
        for c, src in header_map.items():
            if c == "date":
                continue
            val = row.get(src)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                continue
            canon[c] = val
        out.append(canon)

    if bad_dates and not out:
        raise ValueError(
            f"Could not parse any dates. First bad row index ~{bad_dates[0]}. "
            "Expected formats like YYYY-MM-DD or MM/DD/YYYY."
        )
    if not out:
        raise ValueError("No usable rows after date parsing.")
    if bad_dates and len(bad_dates) > len(out) * 0.5:
        raise ValueError(
            f"{len(bad_dates)} of {len(rows)} rows had unparseable dates "
            f"(examples at data rows {bad_dates[:5]}). Check the date column format."
        )

    # sort + de-dupe by date (keep last)
    by_date = {}
    for r in out:
        by_date[r["date"]] = r
    sorted_rows = [by_date[k] for k in sorted(by_date.keys())]
    if len(sorted_rows) < 1:
        raise ValueError("Need at least 1 day of data to score.")
    return sorted_rows, bad_dates


async def load_engine():
    src = await aio.get_artifact("bap_engine_source")
    if isinstance(src, (bytes, bytearray)):
        src = src.decode("utf-8")
    mod = types.ModuleType("bap")
    mod.__dict__["__name__"] = "bap"
    sys.modules["bap"] = mod
    exec(compile(src, "bap.py", "exec"), mod.__dict__)
    return mod


def f(v):
    try:
        if v in (None, "", " "):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_day(row):
    HAT = f(row.get("High Activity Time"))
    MAT = f(row.get("Medium Activity Time"))
    LAT = f(row.get("Low Activity Time"))
    steps = f(row.get("Steps"))
    burn = f(row.get("Activity Burn"))
    dist = f(row.get("Equivalent Walking Distance"))
    inact = f(row.get("Inactive Time"))
    nonwear = f(row.get("Non-wear Time"))
    tsd = f(row.get("Total Sleep Duration"))
    d = {}
    d["valid_wear_hours"] = (24.0 - (nonwear or 0) / 3600.0) if nonwear is not None else 24.0
    if tsd is not None:
        # if looks like minutes already (< 24*60+), keep; if seconds-like, convert
        if tsd > 24 * 60:
            d["sleep_duration_min"] = tsd / 60.0
            d["sleep_duration"] = tsd
        else:
            d["sleep_duration_min"] = tsd
            d["sleep_duration"] = tsd * 60.0
    if steps is not None:
        d["steps"] = steps
    if HAT is not None:
        d["vigorous_intensity_seconds"] = HAT
        d["activity_high"] = HAT
    if MAT is not None:
        d["moderate_intensity_seconds"] = MAT
        d["activity_medium"] = MAT
    if burn is not None:
        d["calories_active"] = burn
    if dist is not None:
        d["distance"] = dist
        d["daily_movement"] = dist
    if None not in (HAT, MAT, LAT):
        d["activity_seconds"] = HAT + MAT + LAT
    met = f(row.get("Average MET"))
    if met is not None:
        d["met_level"] = met
    if inact is not None:
        d["inactivity_seconds"] = inact
    if LAT is not None:
        d["low_intensity_seconds"] = LAT
    hrv = f(row.get("Average HRV"))
    rhr = f(row.get("Average Resting Heart Rate"))
    rr = f(row.get("Respiratory Rate"))
    temp = f(row.get("Temperature Deviation (°C)"))
    if temp is None:
        temp = f(row.get("Temperature Deviation (\u00b0C)"))
    if hrv is not None:
        d["hrv_rmssd"] = hrv
    if rhr is not None:
        d["resting_hr"] = rhr
    if rr is not None:
        d["breaths_per_min"] = rr
    if temp is not None:
        d["temperature_delta"] = temp
    return d


def mean_sd(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        sd = max(1e-6, abs(m) * 0.05)
    else:
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        if sd <= 0:
            sd = max(1e-6, abs(m) * 0.05)
    return m, sd


def cv(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    if m == 0:
        return None
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return sd / abs(m)


def build_prior_baseline(bap, prior_days):
    baseline = {}
    specs = [s for ax in bap.AXES.values() for s in ax]
    n = len(prior_days)
    for s in specs:
        name, field, stat = s.name, s.field, s.stat
        daily = [d.get(field) for d in prior_days if isinstance(d.get(field), (int, float))]
        if stat in ("cv", "isd", "weekly_freq_sd") or name.endswith("_cv") or name.endswith("_isd"):
            cvs = []
            W = 7
            if n < W:
                continue
            for end in range(W, n + 1):
                chunk = prior_days[end - W : end]
                vals = [d.get(field) for d in chunk if isinstance(d.get(field), (int, float))]
                c = cv(vals)
                if c is not None:
                    cvs.append(c)
            ms = mean_sd(cvs)
            if ms:
                baseline[name] = {"mean": ms[0], "sd": ms[1]}
            continue
        if "freq" in (stat or "") or name.endswith("_freq"):
            freqs = []
            W = 7
            for end in range(W, n + 1):
                chunk = prior_days[end - W : end]
                cnt = sum(1 for d in chunk if isinstance(d.get(field), (int, float)) and d.get(field) > 0)
                freqs.append(float(cnt))
            ms = mean_sd(freqs)
            if ms:
                baseline[name] = {"mean": ms[0], "sd": ms[1]}
            continue
        ms = mean_sd(daily)
        if ms:
            baseline[name] = {"mean": ms[0], "sd": ms[1]}
    return baseline


def result_payload(r):
    def ax(v):
        return float(v) if isinstance(v, (int, float)) else None
    subs = {}
    for a, rows in (r.subscores or {}).items():
        out = []
        for row in rows:
            spec = row.get("spec")
            name = getattr(spec, "name", None) or row.get("name")
            raw = row.get("raw")
            if isinstance(raw, (int, float)):
                raw = float(raw)
            out.append({
                "name": name,
                "present": bool(row.get("present")),
                "s": ax(row.get("s")),
                "w": ax(row.get("w")) if row.get("w") is not None else 0.0,
                "raw": raw,
                "n": int(row.get("n") or 0),
                "tier": int(row.get("tier") or 0),
                "s_pop": ax(row.get("s_pop")),
                "s_per": ax(row.get("s_per")),
                "alpha": ax(row.get("alpha")),
                "suppressed": bool(row.get("suppressed") or False),
            })
        subs[a] = out
    d = r.detail or {}
    detail = {
        "S_I_pop": ax(d.get("S_I_pop")) or 0.0,
        "m_v": ax(d.get("m_v")),
        "P_couch": ax(d.get("P_couch")) or 0.0,
        "base": ax(d.get("base")) or 0.0,
        "scenario": d.get("scenario") or "A",
        "stream_coverage": ax(d.get("stream_coverage")) or 0.0,
        "L_recent": ax(d.get("L_recent")) or 0.0,
        "frac_low_recovery": ax(d.get("frac_low_recovery")) or 0.0,
        "a_persist": ax(d.get("a_persist")) or 1.0,
        "W_hours": ax(d.get("W_hours")) or 0.0,
        "inter_weights": {k: float(v) for k, v in (d.get("inter_weights") or {}).items()
                          if isinstance(v, (int, float))},
        "axis_available": d.get("axis_available") or {},
        "rho_axis": {k: float(v) for k, v in (d.get("rho_axis") or {}).items()
                     if isinstance(v, (int, float))},
        "reliability_factor": ax(d.get("reliability_factor")) or 1.0,
        "fallback_tiers": d.get("fallback_tiers") or {},
    }
    return {
        "bap": ax(r.bap),
        "axis_scores": {k: ax(v) for k, v in (r.axis_scores or {}).items()},
        "subscores": subs,
        "confidence": ax(r.confidence) or 0.0,
        "flags": list(r.flags or []),
        "detail": detail,
    }


def pack_suggestions(bap, r):
    try:
        sug = bap.suggest_improvements(r)
        d = bap.suggestion_to_dict(sug)
    except Exception as e:
        return {
            "mode": "unavailable",
            "why_line": "Suggestions could not be computed for this window.",
            "suggestions": [],
            "standing_frame": "",
            "gates_fired": [],
            "error": str(e),
            "packaged": "No packaged recommendation available for this window.",
            "predictions": [],
        }
    preds = []
    actions = []
    for s in d.get("suggestions") or []:
        pts = s.get("recoverable_points")
        text = (s.get("text") or "").strip()
        lever = s.get("lever_class") or "lifestyle"
        if pts is not None and text:
            preds.append({
                "action": text,
                "predicted_bap_gain_points": round(float(pts), 2),
                "lever": lever,
            })
            short = {
                "spread": "break up long sitting with light movement across the day",
                "consistency": "keep a more regular activity and sleep rhythm day to day",
                "variety": "mix different types of movement you enjoy",
                "recovery": "protect sleep and easier days after harder ones",
                "intensity": "add a bit more moderate activity when recovery allows",
                "p_couch": "avoid long inactive stretches right after active periods",
            }.get(lever, text[:80])
            actions.append((short, round(float(pts), 2)))
    if actions:
        parts = []
        total = 0.0
        for short, pts in actions:
            parts.append(f"{short} (about +{pts:g} points)")
            total += pts
        if len(parts) == 1:
            packaged = f"You can raise your BAP by about {parts[0]}."
        elif len(parts) == 2:
            packaged = f"You can raise your BAP by about {total:.1f} points by doing two things: {parts[0]}; and {parts[-1]}."
        else:
            packaged = (
                f"You can raise your BAP by about {total:.1f} points by combining these: "
                + "; ".join(parts[:-1])
                + f"; and {parts[-1]}."
            )
    else:
        mode = d.get("mode") or "maintain"
        if mode in ("maintain", "maintain_protect"):
            packaged = "Your profile looks balanced for this window — focus on maintaining current habits rather than chasing more points."
        elif mode == "decline_clinician":
            packaged = "Suggestions are paused for lifestyle coaching while decline monitoring is active — clinician review takes priority."
        else:
            packaged = d.get("why_line") or "No high-leverage lifestyle actions cleared the suggestion gates for this window."
    return {
        "mode": d.get("mode"),
        "why_line": d.get("why_line"),
        "suggestions": d.get("suggestions") or [],
        "standing_frame": d.get("standing_frame") or "",
        "confidence_caveat": d.get("confidence_caveat"),
        "gates_fired": d.get("gates_fired") or [],
        "predictions": preds,
        "packaged": packaged,
    }


def score_range(bap, days, age, sex, start, end, path):
    if start < 0 or end >= len(days) or start > end:
        return None
    sub = days[start : end + 1]
    w_days = end - start + 1
    H = end + 1
    baseline = {}
    if path == "blend":
        prior = days[:start]
        if len(prior) >= 7:
            baseline = build_prior_baseline(bap, prior)
        elif w_days >= 14 and start == 0:
            half = max(7, w_days // 2)
            baseline = build_prior_baseline(bap, sub[:half])
    ml = [d.get("met_level") for d in sub[-7:] if d.get("met_level") is not None]
    window = dict(
        H=H,
        W_days=w_days,
        age=age,
        sex=sex,
        days=sub,
        baseline=baseline,
        event_counts={"default": 0},
        recent_intensity_load=ml,
    )
    return bap.score_bap(window)


async def build_bundle_from_rows(canon_rows, meta):
    AGE = int(meta.get("age") or 19)
    SEX = meta.get("sex") or "M"
    conditions_flag = list(meta.get("conditions_flag") or [])
    subject = meta.get("subject") or "Uploaded subject"
    profile_label = meta.get("profile_label") or "Uploaded wearable data"
    source = meta.get("source") or "user_upload"
    drop_met = bool(meta.get("drop_met_from_intensity", True))

    bap = await load_engine()
    if drop_met:
        bap.AXES["I"] = [s for s in bap.AXES["I"] if s.name != "met_level"]

    dates = [r["date"] for r in canon_rows]
    days = [build_day(r) for r in canon_rows]
    n = len(days)

    daily_series = {
        "dates": dates,
        "steps": [d.get("steps") for d in days],
        "calories_active": [d.get("calories_active") for d in days],
        "vigorous_intensity_seconds": [d.get("vigorous_intensity_seconds") for d in days],
        "moderate_intensity_seconds": [d.get("moderate_intensity_seconds") for d in days],
        "inactivity_seconds": [d.get("inactivity_seconds") for d in days],
        "low_intensity_seconds": [d.get("low_intensity_seconds") for d in days],
        "hrv_rmssd": [d.get("hrv_rmssd") for d in days],
        "resting_hr": [d.get("resting_hr") for d in days],
        "breaths_per_min": [d.get("breaths_per_min") for d in days],
        "temperature_delta": [d.get("temperature_delta") for d in days],
        "met_level": [d.get("met_level") for d in days],
        "distance": [d.get("distance") for d in days],
        "sleep_duration_min": [d.get("sleep_duration_min") for d in days],
    }

    # Cap dense O(n^2) precompute for large uploads: still full for n<=90, else sample strategy
    # For large n, compute all windows but this can be slow — keep full for correctness.
    windows = {}
    count = 0
    for start in range(n):
        for end in range(start, n):
            w_days = end - start + 1
            r_pure = score_range(bap, days, AGE, SEX, start, end, "pure")
            r_blend = score_range(bap, days, AGE, SEX, start, end, "blend")
            key = f"{start}_{end}"
            windows[key] = {
                "start_idx": start,
                "end_idx": end,
                "window_days": w_days,
                "window_hours": w_days * 24,
                "date_start": dates[start],
                "date_end": dates[end],
                "pure": result_payload(r_pure),
                "blend": result_payload(r_blend),
                "suggestions_pure": pack_suggestions(bap, r_pure),
                "suggestions_blend": pack_suggestions(bap, r_blend),
            }
            count += 1

    full_series = []
    for e in range(n):
        rp = score_range(bap, days, AGE, SEX, e, e, "pure")
        rb = score_range(bap, days, AGE, SEX, e, e, "blend")
        t14p = t14b = None
        if e >= 13:
            t14p = score_range(bap, days, AGE, SEX, e - 13, e, "pure")
            t14b = score_range(bap, days, AGE, SEX, e - 13, e, "blend")
        full_series.append({
            "date": dates[e],
            "idx": e,
            "BAP_1d_pure": None if rp is None or rp.bap is None else round(float(rp.bap), 2),
            "BAP_1d_blend": None if rb is None or rb.bap is None else round(float(rb.bap), 2),
            "BAP_14d_pure": None if t14p is None or t14p.bap is None else round(float(t14p.bap), 2),
            "BAP_14d_blend": None if t14b is None or t14b.bap is None else round(float(t14b.bap), 2),
            "I_1d_pure": None if rp is None or rp.axis_scores.get("I") is None else round(float(rp.axis_scores["I"]), 3),
            "R_1d_pure": None if rp is None or rp.axis_scores.get("R") is None else round(float(rp.axis_scores["R"]), 3),
            "Sp_1d_pure": None if rp is None or rp.axis_scores.get("Sp") is None else round(float(rp.axis_scores["Sp"]), 3),
            "C_1d_pure": None if rp is None or rp.axis_scores.get("C") is None else round(float(rp.axis_scores["C"]), 3),
            "V_1d_pure": None if rp is None or rp.axis_scores.get("V") is None else round(float(rp.axis_scores["V"]), 3),
            "I_1d_blend": None if rb is None or rb.axis_scores.get("I") is None else round(float(rb.axis_scores["I"]), 3),
            "R_1d_blend": None if rb is None or rb.axis_scores.get("R") is None else round(float(rb.axis_scores["R"]), 3),
            "Sp_1d_blend": None if rb is None or rb.axis_scores.get("Sp") is None else round(float(rb.axis_scores["Sp"]), 3),
            "C_1d_blend": None if rb is None or rb.axis_scores.get("C") is None else round(float(rb.axis_scores["C"]), 3),
            "V_1d_blend": None if rb is None or rb.axis_scores.get("V") is None else round(float(rb.axis_scores["V"]), 3),
        })

    trends = {}
    # Only materialize common trailing widths + all for small n to keep artifact size sane
    widths = list(range(1, n + 1)) if n <= 120 else sorted(set(
        list(range(1, min(31, n + 1))) + [n] + ([14] if n >= 14 else [])
    ))
    for w in widths:
        series = []
        for e in range(w - 1, n):
            s = e - w + 1
            key = f"{s}_{e}"
            ent = windows[key]
            series.append({
                "date": dates[e],
                "start_idx": s,
                "end_idx": e,
                "BAP_pure": None if ent["pure"]["bap"] is None else round(float(ent["pure"]["bap"]), 2),
                "BAP_blend": None if ent["blend"]["bap"] is None else round(float(ent["blend"]["bap"]), 2),
                "Confidence_pure": round(float(ent["pure"]["confidence"] or 0), 3),
                "Confidence_blend": round(float(ent["blend"]["confidence"] or 0), 3),
                "I_pure": None if ent["pure"]["axis_scores"].get("I") is None else round(float(ent["pure"]["axis_scores"]["I"]), 3),
                "R_pure": None if ent["pure"]["axis_scores"].get("R") is None else round(float(ent["pure"]["axis_scores"]["R"]), 3),
                "Sp_pure": None if ent["pure"]["axis_scores"].get("Sp") is None else round(float(ent["pure"]["axis_scores"]["Sp"]), 3),
                "C_pure": None if ent["pure"]["axis_scores"].get("C") is None else round(float(ent["pure"]["axis_scores"]["C"]), 3),
                "V_pure": None if ent["pure"]["axis_scores"].get("V") is None else round(float(ent["pure"]["axis_scores"]["V"]), 3),
                "I_blend": None if ent["blend"]["axis_scores"].get("I") is None else round(float(ent["blend"]["axis_scores"]["I"]), 3),
                "R_blend": None if ent["blend"]["axis_scores"].get("R") is None else round(float(ent["blend"]["axis_scores"]["R"]), 3),
                "Sp_blend": None if ent["blend"]["axis_scores"].get("Sp") is None else round(float(ent["blend"]["axis_scores"]["Sp"]), 3),
                "C_blend": None if ent["blend"]["axis_scores"].get("C") is None else round(float(ent["blend"]["axis_scores"]["C"]), 3),
                "V_blend": None if ent["blend"]["axis_scores"].get("V") is None else round(float(ent["blend"]["axis_scores"]["V"]), 3),
            })
        trends[str(w)] = series

    half_ticks = []
    for di, date in enumerate(dates):
        half_ticks.append({"tick": di * 2, "date": date, "time": "00:00", "label": f"{date} 00:00", "day_idx": di, "half": 0})
        half_ticks.append({"tick": di * 2 + 1, "date": date, "time": "12:00", "label": f"{date} 12:00", "day_idx": di, "half": 1})
    half_ticks.append({
        "tick": n * 2,
        "date": dates[-1],
        "time": "24:00",
        "label": f"{dates[-1]} 24:00",
        "day_idx": n - 1,
        "half": 2,
    })

    bundle = {
        "version": 4,
        "subject": subject,
        "profile_label": profile_label,
        "source": source,
        "age": AGE,
        "sex": SEX,
        "conditions_flag": conditions_flag,
        "condition_personalization": False,
        "drop_met_from_intensity": drop_met,
        "met_scoring_eligible": False if drop_met else True,
        "met_form": (
            "interday_daily_average" if drop_met else "intraday_or_scoring_eligible"
        ),
        "met_excluded_reason": (
            "Whole-day / interday daily-average MET (≤1 value per day, no intraday "
            "resolution) is not meaningful for BAP Intensity scoring and is treated "
            "as not present for the engine — no imputation. Raw values still display "
            "in the UI and are queryable via getMetric for discussion only."
            if drop_met
            else None
        ),
        "dates": dates,
        "n_days": n,
        "min_hours": 12,
        "max_hours": n * 24,
        "hours_step": 12,
        "half_ticks": half_ticks,
        "windows": windows,
        "trends": trends,
        "full_series": full_series,
        "daily_series": daily_series,
        "upload": {
            "file_name": meta.get("file_name"),
            "row_count": n,
            "date_start": dates[0],
            "date_end": dates[-1],
            "skipped_bad_dates": int(meta.get("skipped_bad_dates") or 0),
        },
        "notes": [
            "Free-range dual-handle windows keyed start_end (inclusive day indices).",
            "12h UI steps map onto day boundaries for scoring (daily source).",
            "conditions are flags only and are not passed into score_bap.",
            "Bundle built from user upload via ingest_user_data.py.",
        ],
    }

    await aio.store_artifact(
        identifier="bap_dashboard_bundle",
        title="BAP interactive dashboard data bundle (v4 upload)",
        artifact_type="text",
        data=json.dumps(bundle),
        metadata={"text": {"content_type": "application/json"}},
        hidden=False,
    )

    # keep a compact bind result
    full = windows[f"0_{n-1}"]
    await aio.store_artifact(
        identifier="bap_result_oura",
        title="BAP result — current bind pure-pop",
        artifact_type="text",
        data=json.dumps(full["pure"]),
        metadata={"text": {"content_type": "application/json"}},
        hidden=False,
    )

    # upload receipt for UI
    receipt = {
        "ok": True,
        "file_name": meta.get("file_name"),
        "row_count": n,
        "date_start": dates[0],
        "date_end": dates[-1],
        "full_pure_bap": full["pure"]["bap"],
        "windows": count,
        "skipped_bad_dates": int(meta.get("skipped_bad_dates") or 0),
    }
    await aio.store_artifact(
        identifier="bap_upload_receipt",
        title="BAP upload ingest receipt",
        artifact_type="text",
        data=json.dumps(receipt),
        metadata={"text": {"content_type": "application/json"}},
        hidden=True,
    )
    executor.print(
        f"ingest ok n={n} windows={count} full_pure={full['pure']['bap']} "
        f"range={dates[0]}→{dates[-1]} file={meta.get('file_name')}"
    )
    return receipt


async def main():
    # Load staged upload payload
    try:
        raw = await aio.get_artifact("bap_upload_pending")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"No upload payload found (bap_upload_pending). {e}")

    if not raw or not str(raw).strip() or str(raw).strip() == "{}":
        raise RuntimeError("Upload payload is empty.")

    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Upload payload is not valid JSON: {e.msg}")

    if not isinstance(payload, dict):
        raise RuntimeError("Upload payload must be a JSON object.")

    file_name = payload.get("file_name") or "upload.csv"
    content = payload.get("content")
    if content is None:
        raise RuntimeError("Upload payload missing 'content' (file text).")
    if not isinstance(content, str):
        content = str(content)

    # config defaults
    AGE, SEX = 19, "M"
    conditions_flag = []
    try:
        cfg_raw = await aio.get_artifact("bap_config")
        if isinstance(cfg_raw, (bytes, bytearray)):
            cfg_raw = cfg_raw.decode("utf-8")
        cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else (cfg_raw or {})
        AGE = int(cfg.get("age", AGE))
        SEX = cfg.get("sex", SEX)
        raw_c = cfg.get("conditions_display") or cfg.get("conditions") or []
        if isinstance(raw_c, list):
            conditions_flag = [str(c) for c in raw_c]
    except Exception:
        pass

    # override from payload
    if payload.get("age") is not None:
        try:
            AGE = int(payload["age"])
        except Exception:
            pass
    if payload.get("sex"):
        SEX = str(payload["sex"])
    if isinstance(payload.get("conditions_flag"), list):
        conditions_flag = [str(c) for c in payload["conditions_flag"]]

    lower = file_name.lower()
    try:
        if lower.endswith(".json") or content.lstrip().startswith(("[", "{")):
            headers, rows = parse_json_text(content)
        else:
            headers, rows = parse_csv_text(content)
    except ValueError as e:
        raise RuntimeError(str(e))

    header_map = build_header_map(headers)
    try:
        canon_rows, bad_dates = remap_rows(headers, rows, header_map)
    except ValueError as e:
        raise RuntimeError(str(e))

    # store normalized table for reuse / refresh
    await aio.store_artifact(
        identifier="bap_upload_rows",
        title=f"BAP upload rows — {file_name}",
        artifact_type="table",
        data=canon_rows,
        hidden=True,
    )

    meta = {
        "age": AGE,
        "sex": SEX,
        "conditions_flag": conditions_flag,
        "subject": payload.get("subject") or "Uploaded subject",
        "profile_label": payload.get("profile_label") or "Uploaded wearable data",
        "source": file_name,
        "file_name": file_name,
        "drop_met_from_intensity": payload.get("drop_met_from_intensity", True),
        "skipped_bad_dates": len(bad_dates),
    }
    await build_bundle_from_rows(canon_rows, meta)

    # clear pending (leave empty object)
    try:
        await aio.store_artifact(
            identifier="bap_upload_pending",
            title="BAP upload pending (cleared)",
            artifact_type="text",
            data="{}",
            metadata={"text": {"content_type": "application/json"}},
            hidden=True,
        )
    except Exception:
        pass