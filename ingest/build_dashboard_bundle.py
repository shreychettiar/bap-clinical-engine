"""Precompute pure-pop + blend scores for every [start_idx, end_idx] day window.
Conditions NEVER passed into score_bap. Engine math untouched.
Windows keyed as "start_end" (inclusive day indices).
Half-day (12h) UI steps map onto day indices (daily Oura source).
"""
from executor import aio
import types, sys, json

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
        d["sleep_duration_min"] = tsd / 60.0
        d["sleep_duration"] = tsd
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
            packaged = f"You can raise your BAP by about {total:.1f} points by doing two things: {parts[0]}; and {parts[1]}."
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
    """Inclusive day indices [start, end]."""
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

async def main():
    # Defaults match bap_compute / saved-program contract. Override via bap_config.
    AGE, SEX = 30, "M"
    conditions_flag = []
    DROP_MET = True
    input_artifact = "oura_trends"
    subject = "Uploaded subject"
    profile_label = "Oura Ring trends"
    try:
        cfg_raw = await aio.get_artifact("bap_config")
        if isinstance(cfg_raw, (bytes, bytearray)):
            cfg_raw = cfg_raw.decode("utf-8")
        cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else (cfg_raw or {})
        AGE = int(cfg.get("age", AGE))
        SEX = cfg.get("sex", SEX)
        if "drop_met_from_intensity" in cfg:
            DROP_MET = bool(cfg.get("drop_met_from_intensity"))
        if cfg.get("input_artifact"):
            input_artifact = str(cfg.get("input_artifact"))
        if cfg.get("subject"):
            subject = str(cfg.get("subject"))
        if cfg.get("profile_label"):
            profile_label = str(cfg.get("profile_label"))
        raw_c = cfg.get("conditions_display") or cfg.get("conditions") or []
        if isinstance(raw_c, list):
            conditions_flag = [str(c) for c in raw_c]
    except Exception:
        pass

    bap = await load_engine()
    if DROP_MET:
        bap.AXES["I"] = [s for s in bap.AXES["I"] if s.name != "met_level"]

    rows = sorted(
        await aio.get_artifact(input_artifact),
        key=lambda r: r.get("date") or "",
    )
    dates = [r.get("date") for r in rows]
    days = [build_day(r) for r in rows]
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

    # All inclusive [start, end] windows (O(n^2) day pairs — n=62 → ~1953)
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
        if start % 10 == 0:
            executor.print(f"start={start}/{n-1} windows_so_far={count}")

    # Full-span rolling trend series (for chart with free range highlight)
    # Also trailing-window trends keyed by width for backward compat
    full_series = []
    for e in range(n):
        # score each single day pure/blend for dense chart markers
        rp = score_range(bap, days, AGE, SEX, e, e, "pure")
        rb = score_range(bap, days, AGE, SEX, e, e, "blend")
        # also trailing-14 ending at e when possible
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

    # Trailing trends by width (for rolling history chart under current end)
    trends = {}
    for w in range(1, n + 1):
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

    full_key = f"0_{n-1}"
    full = windows[full_key]
    await aio.store_artifact(
        identifier="bap_result_oura",
        title=f"BAP result — {subject} pure-pop (dashboard bind)",
        artifact_type="text",
        data=json.dumps(full["pure"]),
        metadata={"text": {"content_type": "application/json"}},
        hidden=False,
    )
    trend14 = [
        {
            "date": p["date"],
            "BAP": p["BAP_pure"],
            "Confidence": p["Confidence_pure"],
            "Intensity": p["I_pure"],
            "Recovery": p["R_pure"],
            "Variety": p["V_pure"],
            "Spread": p["Sp_pure"],
            "Consistency": p["C_pure"],
            "Flags": "",
        }
        for p in trends.get("14", [])
    ]
    await aio.store_artifact(
        identifier="bap_dashboard_trend",
        title=f"BAP rolling 14-day trend — {subject} pure-pop",
        artifact_type="table",
        data=trend14,
        hidden=False,
    )

    # Half-day timeline ticks for dual slider (12h steps)
    # tick i = start of half-day slot; date_idx = i // 2, half = i % 2 (0=00:00, 1=12:00)
    half_ticks = []
    for di, date in enumerate(dates):
        half_ticks.append({"tick": di * 2, "date": date, "time": "00:00", "label": f"{date} 00:00", "day_idx": di, "half": 0})
        half_ticks.append({"tick": di * 2 + 1, "date": date, "time": "12:00", "label": f"{date} 12:00", "day_idx": di, "half": 1})
    # end boundary after last day
    half_ticks.append({
        "tick": n * 2,
        "date": dates[-1],
        "time": "24:00",
        "label": f"{dates[-1]} 24:00",
        "day_idx": n - 1,
        "half": 2,
    })

    bundle = {
        "version": 3,
        "subject": subject,
        "profile_label": profile_label,
        "source": input_artifact,
        "age": AGE,
        "sex": SEX,
        "conditions_flag": conditions_flag,
        "condition_personalization": False,
        "drop_met_from_intensity": DROP_MET,
        # MET eligibility: daily-average (≤1 value/day, no intraday timestamps) is
        # NOT used in BAP/axis scoring. Intraday MET series would set eligible=True.
        # Assumption (confirm with clinical): Oura trends Average MET = interday form.
        "met_scoring_eligible": False if DROP_MET else True,
        "met_form": (
            "interday_daily_average" if DROP_MET else "intraday_or_scoring_eligible"
        ),
        "met_excluded_reason": (
            "Whole-day / interday daily-average MET (≤1 value per day, no intraday "
            "resolution) is not meaningful for BAP Intensity scoring and is treated "
            "as not present for the engine — no imputation. Raw values still display "
            "in the UI and are queryable via getMetric for discussion only."
            if DROP_MET
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
        "notes": [
            "Free-range dual-handle windows keyed start_end (inclusive day indices).",
            "12h UI steps map onto day boundaries for scoring (daily Oura source).",
            "conditions are flags only and are not passed into score_bap.",
        ],
    }
    await aio.store_artifact(
        identifier="bap_dashboard_bundle",
        title="BAP interactive dashboard data bundle (free-range v3)",
        artifact_type="text",
        data=json.dumps(bundle),
        metadata={"text": {"content_type": "application/json"}},
        hidden=False,
    )

    # Migrate annotations to v2 schema if needed
    try:
        raw = await aio.get_artifact("bap_dashboard_annotations")
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        doc = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        doc = {"annotations": []}
    anns = list(doc.get("annotations") or [])
    migrated = []
    for a in anns:
        aa = dict(a)
        tt = aa.get("target_type")
        if tt == "graph_region":
            aa["target_type"] = "score"
            aa["target_key"] = "bap"
            aa["variable"] = None
        elif tt == "variable":
            aa["target_type"] = "axis"
            # keep target_key as variable name temporarily; UI will resolve
            aa["variable"] = aa.get("target_key")
            # try to leave axis unknown — set to first axis if missing
            aa["axis_key"] = aa.get("axis_key") or "C"
            aa["target_key"] = aa["axis_key"]
        else:
            if tt == "axis":
                aa["axis_key"] = aa.get("axis_key") or aa.get("target_key")
                aa["variable"] = aa.get("variable")
            if tt == "score":
                aa["target_key"] = "bap"
                aa["variable"] = None
        if not aa.get("start_date"):
            # map old window_days to a trailing range ending at last day
            wd = int(aa.get("window_days") or n)
            s = max(0, n - wd)
            aa["start_date"] = dates[s]
            aa["end_date"] = dates[n - 1]
            aa["start_time"] = "00:00"
            aa["end_time"] = "24:00"
            aa["start_idx"] = s
            aa["end_idx"] = n - 1
        aa.setdefault("updated_at", aa.get("created_at"))
        migrated.append(aa)
    await aio.store_artifact(
        identifier="bap_dashboard_annotations",
        title="BAP dashboard administrator annotations",
        artifact_type="text",
        data=json.dumps({"version": 2, "annotations": migrated}),
        metadata={"text": {"content_type": "application/json"}},
        hidden=False,
    )

    sample = windows[f"{n-14}_{n-1}"]
    executor.print(
        f"bundle v3 ready n={n} windows={count} "
        f"full_pure={full['pure']['bap']} last14_pure={sample['pure']['bap']} "
        f"last14_blend={sample['blend']['bap']} anns={len(migrated)}"
    )