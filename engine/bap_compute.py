from executor import aio
import types, sys, json


async def load_engine():
    """Load the BAP reference engine from the bap_engine_source text artifact."""
    src = await aio.get_artifact("bap_engine_source")
    if isinstance(src, (bytes, bytearray)):
        src = src.decode("utf-8")
    mod = types.ModuleType("bap")
    mod.__dict__["__name__"] = "bap"
    sys.modules["bap"] = mod
    exec(compile(src, "bap.py", "exec"), mod.__dict__)
    return mod


async def load_config():
    default = {"age": 30, "sex": "M", "drop_met_from_intensity": True, "input_artifact": "oura_trends",
               "form_personal_baseline": False}
    try:
        raw = await aio.get_artifact("bap_config")
    except Exception:
        return default
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    try:
        cfg = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except Exception:
        return default
    default.update({k: v for k, v in cfg.items() if v is not None})
    return default


def f(v):
    try:
        if v in (None, "", " "):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_day(row):
    HAT = f(row.get("High Activity Time")); MAT = f(row.get("Medium Activity Time"))
    LAT = f(row.get("Low Activity Time")); steps = f(row.get("Steps"))
    burn = f(row.get("Activity Burn")); dist = f(row.get("Equivalent Walking Distance"))
    inact = f(row.get("Inactive Time")); nonwear = f(row.get("Non-wear Time"))
    tsd = f(row.get("Total Sleep Duration"))
    d = {}
    d["valid_wear_hours"] = (24.0 - (nonwear or 0) / 3600.0) if nonwear is not None else 24.0
    if tsd is not None: d["sleep_duration_min"] = tsd / 60.0
    if steps is not None: d["steps"] = steps
    if HAT is not None: d["vigorous_intensity_seconds"] = HAT; d["activity_high"] = HAT
    if MAT is not None: d["moderate_intensity_seconds"] = MAT; d["activity_medium"] = MAT
    if burn is not None: d["calories_active"] = burn
    if dist is not None: d["distance"] = dist; d["daily_movement"] = dist
    if None not in (HAT, MAT, LAT): d["activity_seconds"] = HAT + MAT + LAT
    met = f(row.get("Average MET"))
    if met is not None: d["met_level"] = met  # retained for Consistency's avg_met_level_cv
    if inact is not None: d["inactivity_seconds"] = inact
    if LAT is not None: d["low_intensity_seconds"] = LAT
    hrv = f(row.get("Average HRV")); rhr = f(row.get("Average Resting Heart Rate"))
    rr = f(row.get("Respiratory Rate")); temp = f(row.get("Temperature Deviation (\u00b0C)"))
    if hrv is not None: d["hrv_rmssd"] = hrv
    if rhr is not None: d["resting_hr"] = rhr
    if rr is not None: d["breaths_per_min"] = rr
    if temp is not None: d["temperature_delta"] = temp
    return d


def auto_baseline(bap, days, names):
    out = {}; field_of = {}
    for a in bap.AXES.values():
        for s in a:
            field_of[s.name] = (s.field, s.stat)
    for nm in names:
        fld, _stat = field_of.get(nm, (nm, "mean"))
        vals = [d[fld] for d in days if fld in d and d[fld] is not None]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else max(1e-6, abs(m) * 0.05)
        out[nm] = {"mean": m, "sd": sd if sd > 0 else max(1e-6, abs(m) * 0.05)}
    return out


def result_to_dict(r):
    return {"bap": r.bap, "axis_scores": r.axis_scores, "subscores": r.subscores,
            "confidence": r.confidence, "flags": r.flags, "detail": r.detail}


async def main():
    cfg = await load_config()
    AGE = int(cfg.get("age", 30)); SEX = str(cfg.get("sex", "M"))
    DROP_MET = bool(cfg.get("drop_met_from_intensity", True))
    FORM_BASELINE = bool(cfg.get("form_personal_baseline", False))
    input_artifact = cfg.get("input_artifact", "oura_trends")

    bap = await load_engine()
    if DROP_MET:
        # Oura Average MET is a whole-day average (incl. sleep/rest) → understates Intensity.
        # Drop met_level from Intensity; MET stays in Consistency's avg_met_level_cv.
        bap.AXES["I"] = [s for s in bap.AXES["I"] if s.name != "met_level"]

    rows = sorted(await aio.get_artifact(input_artifact), key=lambda r: r.get("date"))
    dates = [r.get("date") for r in rows]
    days = [build_day(r) for r in rows]
    n = len(days)
    names = [s.name for a in bap.AXES.values() for s in a]
    # Spec-correct default: no self-referential baseline. Only form a personal baseline
    # when explicitly enabled (and even then it should come from PRIOR history, not this window).
    base = auto_baseline(bap, days, names) if FORM_BASELINE else {}
    executor.print(f"FORM_BASELINE={FORM_BASELINE} -> baseline vars={len(base)} "
                   f"({'pure-population path, alpha=0' if not base else 'personal blend active'})")

    met_load = [d.get("met_level") for d in days[-7:] if d.get("met_level") is not None]
    window = dict(H=n, W_days=n, age=AGE, sex=SEX, days=days, baseline=base,
                  event_counts={"default": 0}, recent_intensity_load=met_load)
    r = bap.score_bap(window)
    executor.print(f"FULL WINDOW  BAP={round(r.bap, 2) if r.bap is not None else None} "
                   f"conf={round(r.confidence, 3)} flags={r.flags} age={AGE} sex={SEX} drop_met={DROP_MET}")

    await aio.store_artifact(identifier="bap_result_oura",
        title=f"BAP result — {input_artifact} ({n}-day window, age {AGE} {SEX}, MET {'excluded' if DROP_MET else 'included'} in Intensity)",
        artifact_type="text", data=json.dumps(result_to_dict(r)),
        metadata={"text": {"content_type": "application/json"}}, hidden=True)

    WIN = min(14, n)
    trend = []
    for end in range(WIN, n + 1):
        sub = days[end - WIN:end]
        ml = [d.get("met_level") for d in sub[-7:] if d.get("met_level") is not None]
        w = dict(H=end, W_days=WIN, age=AGE, sex=SEX, days=sub, baseline=base,
                 event_counts={"default": 0}, recent_intensity_load=ml)
        rr = bap.score_bap(w)
        def ax(k):
            v = rr.axis_scores.get(k)
            return round(v, 3) if v is not None else None
        trend.append({
            "date": dates[end - 1],
            "BAP": round(rr.bap, 2) if rr.bap is not None else None,
            "Confidence": round(rr.confidence, 3),
            "Intensity": ax("I"), "Recovery": ax("R"), "Variety": ax("V"),
            "Spread": ax("Sp"), "Consistency": ax("C"),
            "Flags": ",".join(rr.flags),
        })
    baps = [t["BAP"] for t in trend if t["BAP"] is not None]
    if baps:
        executor.print(f"TREND points={len(trend)} range {min(baps)}..{max(baps)}")
    await aio.store_artifact(identifier="bap_dashboard_trend",
        title=f"BAP rolling {WIN}-day trend ({input_artifact}, age {AGE} {SEX})",
        artifact_type="table", data=trend)
    executor.print("Stored bap_result_oura (json, hidden) + bap_dashboard_trend (table).")