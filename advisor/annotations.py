"""Conversational annotation proposals for the Alyf Advisor.

Produces a structured annotation draft the UI previews and confirms. The actual
write always goes through the dashboard's existing saveAnnotations path — this
module never touches `bap_dashboard_annotations` itself.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from executor import aio, Chat, UserPrompt

AXIS_KEYS = ("I", "R", "V", "Sp", "C")
AXIS_NAMES = {
    "I": "Intensity",
    "R": "Recovery",
    "V": "Variety",
    "Sp": "Spread",
    "C": "Consistency",
}

# Common Oura/BAP variables the assistant may narrow to (axis-scoped).
AXIS_VARS = {
    "I": [
        "vigorous_intensity_seconds",
        "moderate_intensity_seconds",
        "activity_high",
        "activity_medium",
        "steps",
        "calories_active",
        "distance",
        "activity_seconds",
        "met_level",
    ],
    "R": [
        "hrv_rmssd",
        "resting_hr",
        "breaths_per_min",
        "temperature_delta",
        "sleep_score",
        "total_sleep",
        "deep_sleep",
        "rem_sleep",
        "sleep_efficiency",
    ],
    "V": [],
    "Sp": [
        "inactivity_fraction",
        "inactivity_seconds",
        "sedentary_seconds",
        "low_intensity_seconds",
    ],
    "C": [
        "steps_cv",
        "activity_cv",
        "sleep_midpoint_cv",
        "sleep_duration_cv",
    ],
}

EXPLICIT_PATTERNS = [
    r"\b(add|create|make|save|log|record|put|leave)\b.{0,40}\b(annotation|note|tag|marker)\b",
    r"\b(annotation|note|tag)\b.{0,40}\b(from|for|between|on)\b",
    r"\bmark (this|that|it|the)\b",
    r"\bannotate\b",
    r"\badd a note\b",
    r"\bnote that\b.{0,20}\b(from|between|on|during)\b",
]

# Soft life-context cues that may warrant a proactive draft (never silent write).
# Bias toward INCLUSION: better to surface a draft the patient can ignore than miss
# context that explains their data. Match meaning via broad lexicon + paraphrase
# patterns + heuristic (timeframe and/or score-impact + life circumstance).
# Situational only → annotation path; never durable profile. No clinical diagnosis.
# Acute/emergency still suppressed upstream (guardrails) — not annotation material.
#
# Categories (illustrative, not exhaustive): travel/disruption, sedentary/immobility,
# injury/setbacks, illness, sleep disruption, stress/life load, routine breaks,
# volunteered substance/behavior context, environmental limits.

_LIFE_EVENT = (
    # Travel & disruption
    r"travel|traveling|travelling|trip|trips|vacation|holiday|holidays|"
    r"flight|flights|flying|plane|airport|red-?eye|layover|"
    r"jet ?lag|time[- ]?zone|long drive|road trip|hotel|away from home|"
    r"relocation|relocated|out of town|"
    # Sedentary / immobility
    r"sitting|sedentary|desk(?:bound)?|desk[- ]bound|commute|commuting|"
    r"glued to (?:my )?desk|stuck indoors|bed rest|immobile|on the couch|"
    r"long (?:haul|day|days|hours|drive|drives)|"
    # Injury & physical setbacks
    r"injur(?:y|ed|ies)|sprain|sprained|strain(?:ed)?|tweaked|pulled (?:a )?muscle|"
    r"soreness|sore|flare[- ]?ups?|surgery|procedure|physical therap(?:y|ist)|"
    r"pt sessions?|fell|fall|broke|broken|fracture|"
    # Illness (patient-stated; draft describes their words — do not diagnose)
    r"sick|ill|illness|flu|cold|covid|fever|infection|under the weather|"
    r"run down|run-down|unwell|stomach bug|migraine|"
    # Sleep disruption
    r"part(?:y|ies|ying)|hangover|out late|stayed up|all[- ]nighter|"
    r"newborn|baby|childcare|kids? (?:up|awake)|noisy|insomnia|"
    r"shift work|night shift|on call|sleep(?:ing)? (?:well|poorly|bad|worse)|"
    r"didn'?t (?:really )?sleep|did not (?:really )?sleep|couldn'?t sleep|could not sleep|"
    r"poor sleep|bad sleep|rough night|rough nights|"
    # Stress & life load
    r"finals?|exam|exams|deadline|deadlines|crunch|stress(?:ed|ful)?|"
    r"work (?:week|crunch|travel)|overtime|grief|funeral|bereavement|"
    r"major (?:life )?event|high[- ]stress|"
    # Routine breaks / events
    r"wedding|celebration|celebrating|conference|detrain(?:ing|ed)?|"
    r"broke? (?:from )?training|off (?:my )?routine|schedule change|"
    r"moved|moving|house move|"
    # Substance / behavioral (patient-volunteered only)
    r"drinking|drank|alcohol|hungover|dehydrated|hydration|"
    r"diet change|fasting|"
    # Environmental
    r"heat ?wave|heatwave|altitude|extreme weather|wildfire smoke|smoke day|"
    r"too hot|too cold to (?:run|train|walk)"
)

# Casual paraphrases → same meaning as a life circumstance (not strict keywords).
_PARAPHRASE_PATTERNS = [
    # Sedentary / desk
    r"\bglued to (?:my )?(?:desk|chair|laptop|computer)\b",
    r"\bon my feet all (?:day|week)\b",
    r"\bstuck (?:at|in) (?:my )?desk\b",
    r"\bdidn'?t (?:really )?move\b",
    r"\bhardly (?:moved|walked)\b",
    r"\bcouch[- ]potato\b",
    # Travel paraphrases
    r"\baway from home\b",
    r"\bout of town\b",
    r"\bcrossing time zones?\b",
    r"\blong haul\b",
    r"\bback[- ]to[- ]back flights?\b",
    # Illness paraphrases
    r"\bunder the weather\b",
    r"\bcoming down with\b",
    r"\bfighting (?:a |something )?(?:cold|bug|something)\b",
    r"\bnot feeling (?:great|well|myself)\b",
    # Injury paraphrases
    r"\btweaked my\b",
    r"\bdid something to my\b",
    r"\bmessed up my (?:knee|back|ankle|shoulder|hip|wrist)\b",
    r"\bout with (?:a |an )?(?:injury|bad (?:knee|back|ankle))\b",
    # Sleep paraphrases
    r"\bup all night\b",
    r"\bbarely slept\b",
    r"\btossed and turned\b",
    r"\bwired at night\b",
    r"\bbaby (?:kept me up|was up)\b",
    # Stress / load
    r"\bslammed at work\b",
    r"\bcrushing it at work\b",  # busy load, still situational if timed
    r"\bwork(?:ed)? late every\b",
    r"\bnon[- ]stop (?:week|month)\b",
    r"\bburning the candle\b",
]

IMPLICIT_PATTERNS = [
    r"\b(that'?s why|which is why|explains why|because i|because of)\b",
    rf"\b(i (was|had|have been|ve been|didn'?t|did not|couldn'?t|could not)|we were)\b.{{0,100}}\b({_LIFE_EVENT})\b",
    rf"\b(on|after|from|during|while|since)\b.{{0,50}}\b({_LIFE_EVENT})\b",
    r"\b(didn'?t|did not|couldn'?t|could not)\b.{0,40}\b(sleep|rest|train|work out|workout|exercise|move)\b",
    r"\b(poor|bad|rough|terrible|awful)\b.{0,20}\b(sleep|night|nights)\b",
    r"\b(sitting|sedentary|desk)\b.{0,40}\b(long|hours|periods?|all day|all week|stretch(?:es)?|week|forever|non[- ]stop)\b",
    r"\b(long periods?|long hours|all day|all week|forever)\b.{0,30}\b(sitting|sedentary|desk|indoors)\b",
    r"\blong drives?\b",
    r"\b(flight|flights|flying|plane|airport|red-?eye)\b",
    r"\bpart(?:y|ies|ying)\b",
    r"\b(injured|injury|surgery|sprain|tweaked)\b",
    r"\b(sick|flu|covid|fever|under the weather)\b",
    r"\b(finals?|exams?|deadline|deadlines|crunch)\b",
    r"\b(jet ?lag|time[- ]?zone)\b",
    r"\b(wedding|funeral|grief|moved house|moving house)\b",
    *_PARAPHRASE_PATTERNS,
]

# Relative / bounded time anchors that make life context annotation-worthy.
_TIME_ANCHOR = re.compile(
    r"\b("
    r"past couple (?:of )?(?:days?|nights?|weeks?|months?)|"
    r"last couple (?:of )?(?:days?|nights?|weeks?|months?)|"
    r"past few (?:days?|nights?|weeks?|months?)|"
    r"last few (?:days?|nights?|weeks?|months?)|"
    r"past \d+ (?:days?|nights?|weeks?|months?)|"
    r"last \d+ (?:days?|nights?|weeks?|months?)|"
    r"(?:a |one |the )?(?:day|night|week|weekend|month) (?:or so )?ago|"
    r"this week|this weekend|this month|yesterday|last night|last week|last weekend|"
    r"the other day|recently|lately|these days|"
    r"all week|all month|over the weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun|"
    r"from .+ to .+|between .+ and .+"
    r")\b",
    re.I,
)

_LIFE_EVENT_RE = re.compile(rf"\b({_LIFE_EVENT})\b", re.I)
_PARAPHRASE_RE = re.compile("|".join(f"(?:{p})" for p in _PARAPHRASE_PATTERNS), re.I)

# Score-impact / explain questions that often carry situational context.
_SCORE_IMPACT_RE = re.compile(
    r"\b(how does that affect|affect(?:ed|s)? my (?:score|bap|axes?|sleep|recovery|spread|consistency|intensity)|"
    r"impact(?:ed|s)? my (?:score|bap)|why (?:is|was) my (?:score|bap)|"
    r"what happened to my (?:score|bap)|does that (?:show|explain|show up)|"
    r"how (?:did|does) (?:that|this) (?:show|look|read) on)\b",
    re.I,
)


def _has_situational_cue(low: str) -> bool:
    """True if message mentions a life circumstance (lexicon or paraphrase)."""
    if _LIFE_EVENT_RE.search(low):
        return True
    if _PARAPHRASE_RE.search(low):
        return True
    return False


def _implicit_situational_heuristic(low: str) -> bool:
    """True when situational cue pairs with timeframe and/or score-impact.

    Bias toward inclusion when both a circumstance and a data-link signal exist.
    Bare score questions without life context stay False.
    """
    has_life = _has_situational_cue(low)
    has_time = bool(_TIME_ANCHOR.search(low))
    has_impact = bool(_SCORE_IMPACT_RE.search(low))
    if has_life and has_time:
        return True
    if has_life and has_impact:
        return True
    # Causal framing often implies situational annotation-worthiness with impact.
    if has_impact and re.search(
        r"\b(because|after|from|since|while|during)\b", low
    ) and len(low) > 40:
        # Soft: impact + causal connector + enough content (not "how does that affect")
        if has_life:
            return True
    return False


def detect_annotation_intent(user_message: str) -> str | None:
    """Return 'explicit', 'implicit', or None.

    Implicit = situational, time-bound (or score-impact-linked) life context that
    may warrant a ready annotation draft. Accept is still the write gate.
    Does not diagnose; does not write the store; emergency turns suppressed upstream.
    """
    low = user_message.lower().strip()
    if not low:
        return None
    for pat in EXPLICIT_PATTERNS:
        if re.search(pat, low, re.I):
            return "explicit"
    for pat in IMPLICIT_PATTERNS:
        if re.search(pat, low, re.I):
            return "implicit"
    if _implicit_situational_heuristic(low):
        return "implicit"
    return None


def _parse_iso(d: str) -> date | None:
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def clamp_range(
    start_s: str | None,
    end_s: str | None,
    data_start: str,
    data_end: str,
) -> tuple[str, str] | None:
    ds = _parse_iso(data_start)
    de = _parse_iso(data_end)
    if not ds or not de:
        return None
    s = _parse_iso(start_s or "") or ds
    e = _parse_iso(end_s or "") or s
    if e < s:
        s, e = e, s
    if s < ds:
        s = ds
    if e > de:
        e = de
    if s > de or e < ds:
        return None
    return s.isoformat(), e.isoformat()


def nearest_data_date(target: str, dates: list[str]) -> str | None:
    t = _parse_iso(target)
    if not t or not dates:
        return None
    best = None
    best_d = None
    for d in dates:
        dd = _parse_iso(d)
        if not dd:
            continue
        dist = abs((dd - t).days)
        if best is None or dist < best:
            best = dist
            best_d = d
    return best_d


def resolve_indices(dates: list[str], start: str, end: str) -> tuple[int, int]:
    """Map YYYY-MM-DD to inclusive indices, snapping to nearest available dates."""
    if not dates:
        return 0, 0
    s_date = nearest_data_date(start, dates) or dates[0]
    e_date = nearest_data_date(end, dates) or dates[-1]
    try:
        s = dates.index(s_date)
    except ValueError:
        s = 0
    try:
        e = dates.index(e_date)
    except ValueError:
        e = len(dates) - 1
    if e < s:
        s, e = e, s
    return s, e


EXTRACT_PROMPT = """You extract a timeline annotation draft for the BAP dashboard.

The patient may be explicitly asking to add a note/annotation, or may have shared
situational life context (travel, sitting, illness they mentioned, injury they
mentioned, stress, sleep disruption, routine breaks, environment, etc.) that
could become an annotation. Bias toward drafting when timeframe + data link are
plausible — the patient still Accepts before anything is saved.

Describe ONLY what the patient said. Do NOT diagnose, assess severity, or add
clinical interpretation. Do NOT invent issues or dates they did not imply.

Rules:
- target_type is ONLY "score" or "axis" (never anything else).
- If axis: axis_key is one of I, R, V, Sp, C.
  - I Intensity = effort/activity volume
  - R Recovery = sleep, HRV, resting HR, recovery signals
  - V Variety = activity-type diversity (often unavailable on ring exports)
  - Sp Spread = movement distributed vs long sitting
  - C Consistency = day-to-day steadiness of activity/sleep timing
- variable is optional and only when clearly one metric within that axis.
- If the issue is general life context or spans multiple axes, use target_type=score.
- Prefer target_type=score for multi-signal life events (travel, illness, stress). Do NOT
  set needs_clarification just to pick between two plausible axes in offer mode — use score.
- start_date / end_date must be YYYY-MM-DD, resolved from explicit or relative
  language against anchor_today, and MUST fall within data_start..data_end when possible.
- title: short label (e.g. "Travel — reduced sleep").
- description: fuller context in the patient's own words (paraphrase lightly OK).
- TIMEFRAMES:
  - Explicit or clear relative ranges ("June 1–10", "last week", "this past weekend") → resolve.
  - Vague relatives ("recently", "lately", "the other day", "this trip") in mode=offer:
    DO NOT clarify. Default start_date/end_date to the active analysis window
    (window_start..window_end), or data_start..data_end if no window. Patient can edit dates
    on the Accept card.
  - Only set needs_clarification=true for timeframe when mode=explicit AND dates are missing
    or contradictory with no usable window default.
- mode is provided by the system; keep it.

Axes and example variables:
{axis_catalog}

Data date range available: {data_start} to {data_end}
Anchor "today" for relative phrases: {today}
Active analysis window (hint only): {window_start} to {window_end}
Intent mode: {mode}
"""


async def extract_proposal(
    user_message: str,
    *,
    mode: str,
    dates: list[str],
    today: str,
    window_start: str | None = None,
    window_end: str | None = None,
    history_text: str = "",
) -> dict:
    """Return a proposal dict ready for the UI (or a clarification-only payload)."""
    if not dates:
        return {
            "status": "error",
            "mode": mode,
            "needs_clarification": True,
            "clarifier": "I don't have loaded score dates to attach an annotation to yet.",
        }

    data_start, data_end = dates[0], dates[-1]
    axis_catalog = "\n".join(
        f"- {k} {AXIS_NAMES[k]}: " + (", ".join(AXIS_VARS.get(k) or ["(axis-level only)"]))
        for k in AXIS_KEYS
    )
    schema = {
        "type": "object",
        "properties": {
            "needs_clarification": {"type": "boolean"},
            "clarifier": {"type": "string"},
            "target_type": {"type": "string", "enum": ["score", "axis"]},
            "axis_key": {"type": "string", "enum": list(AXIS_KEYS) + [""]},
            "variable": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "timeframe_raw": {"type": "string"},
        },
        "required": [
            "needs_clarification",
            "target_type",
            "title",
            "description",
            "start_date",
            "end_date",
        ],
    }
    system = EXTRACT_PROMPT.format(
        axis_catalog=axis_catalog,
        data_start=data_start,
        data_end=data_end,
        today=today,
        window_start=window_start or data_start,
        window_end=window_end or data_end,
        mode=mode,
    )
    user = (
        (f"Recent conversation:\n{history_text}\n\n" if history_text else "")
        + f"Patient message:\n{user_message}"
    )
    try:
        out = await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=system,
                user_prompt=UserPrompt.from_text(user),
            ),
            output_schema=schema,
        )
    except Exception as e:
        return {
            "status": "error",
            "mode": mode,
            "needs_clarification": True,
            "clarifier": (
                "I couldn't draft that annotation cleanly. "
                f"Try naming the dates and what to note. ({type(e).__name__})"
            ),
        }

    if out.get("needs_clarification"):
        # Offer mode: prefer a ready draft on the active window over a dead-end clarifier.
        # Patient still Accepts/edits dates on the card. Explicit mode may still clarify.
        if mode == "offer":
            ws = (window_start or data_start or "").strip()
            we = (window_end or data_end or "").strip()
            if ws and we:
                out = dict(out)
                out["needs_clarification"] = False
                if not (out.get("start_date") or "").strip():
                    out["start_date"] = ws
                if not (out.get("end_date") or "").strip():
                    out["end_date"] = we
                if out.get("target_type") not in ("score", "axis"):
                    out["target_type"] = "score"
                if not (out.get("title") or "").strip():
                    out["title"] = "Note"
                if not (out.get("description") or "").strip():
                    out["description"] = user_message.strip()
            else:
                return {
                    "status": "clarify",
                    "mode": mode,
                    "needs_clarification": True,
                    "clarifier": (out.get("clarifier") or "").strip()
                    or "Which dates should I use for this timeline note?",
                }
        else:
            return {
                "status": "clarify",
                "mode": mode,
                "needs_clarification": True,
                "clarifier": (out.get("clarifier") or "").strip()
                or "Which dates should I use, and should this tag overall BAP or a specific axis?",
            }

    target_type = out.get("target_type") if out.get("target_type") in ("score", "axis") else "score"
    axis_key = (out.get("axis_key") or "").strip()
    if target_type == "axis" and axis_key not in AXIS_KEYS:
        # Fall back to score rather than inventing an axis.
        target_type = "score"
        axis_key = ""
    variable = (out.get("variable") or "").strip() or None
    if target_type != "axis":
        variable = None
        axis_key = None
    elif variable:
        allowed = set(AXIS_VARS.get(axis_key or "", []) or [])
        # Allow unknown variable strings only if they look like snake_case metrics;
        # otherwise drop narrowing.
        if allowed and variable not in allowed:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", variable):
                variable = None

    clamped = clamp_range(out.get("start_date"), out.get("end_date"), data_start, data_end)
    if not clamped:
        if mode == "offer":
            # Last resort: active window / whole loaded span so the draft still appears.
            fallback_s = window_start or data_start
            fallback_e = window_end or data_end
            clamped = clamp_range(fallback_s, fallback_e, data_start, data_end)
        if not clamped:
            return {
                "status": "clarify",
                "mode": mode,
                "needs_clarification": True,
                "clarifier": (
                    f"I couldn't place those dates inside your loaded data "
                    f"({data_start} → {data_end}). What range should I use?"
                ),
            }
    start_date, end_date = clamped
    # Snap to actual series dates for stable markers.
    s_idx, e_idx = resolve_indices(dates, start_date, end_date)
    start_date = dates[s_idx]
    end_date = dates[e_idx]

    title = (out.get("title") or "").strip() or "Note"
    description = (out.get("description") or "").strip() or user_message.strip()
    title = title[:80]
    description = description[:2000]

    return {
        "status": "proposal",
        "mode": mode,  # explicit | offer
        "needs_clarification": False,
        "target_type": target_type,
        "axis_key": axis_key if target_type == "axis" else None,
        "variable": variable,
        "title": title,
        "description": description,
        "start_date": start_date,
        "end_date": end_date,
        "start_time": "00:00",
        "end_time": "24:00",
        "start_idx": s_idx,
        "end_idx": e_idx,
        "timeframe_raw": (out.get("timeframe_raw") or "").strip(),
        "source": "patient_via_advisor",
        "data_range": {"start": data_start, "end": data_end},
    }


def proposal_reply_text(proposal: dict) -> str:
    """Statement lead-in for the inline annotation preview card (never a permission ask)."""
    if proposal.get("status") == "clarify" or proposal.get("needs_clarification"):
        return proposal.get("clarifier") or (
            "I need a bit more detail to finish the annotation draft. Which dates and target should I use?"
        )
    if proposal.get("status") == "error":
        return proposal.get("clarifier") or "I could not draft that annotation from what was shared."

    tgt = "overall BAP score"
    if proposal.get("target_type") == "axis":
        ak = proposal.get("axis_key") or ""
        name = AXIS_NAMES.get(ak, ak)
        var = proposal.get("variable")
        tgt = f"{name} axis" + (f" · {var}" if var else "")

    rng = f"{proposal.get('start_date')} → {proposal.get('end_date')}"
    if proposal.get("mode") == "offer":
        # Proactive path: answer already complete; present a ready draft, do not ask permission.
        return (
            "Here is an annotation you can add to your timeline if you'd like:"
        )
    return (
        f"Here is a draft annotation for **{rng}** on **{tgt}**. "
        f"Accept to save it, edit any field first, or ignore to skip."
    )