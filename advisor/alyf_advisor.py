"""Alyf Advisor — conversational turn handler.

Patient-facing analyst for the BAP dashboard. Reads a staged turn payload
(`alyf_advisor_turn_pending`), runs safety guardrails, drafts a reply under the
embedded behavioral contract (the load-bearing system prompt), optionally
enriches with BAP research passages, audits for load-bearing violations, and
writes `alyf_advisor_reply`.

Headless scoreWindow: same dual-slider BAP for any [start,end] via precomputed
bundle windows (no UI mutation). Marginal holdout for affect questions.
Does NOT change scoring-engine math.
"""

import asyncio
import json
import re
import sys

from executor import aio, Chat, UserPrompt
import executor

import advisor_retrieval as R
import advisor_score_window as SW
import advisor_query_layer as Q
import advisor_guardrails as G
import advisor_context as C
import advisor_annotations as AAnn
import advisor_memory as M

TURN_ARTIFACT = "alyf_advisor_turn_pending"
REPLY_ARTIFACT = "alyf_advisor_reply"

# ---------------------------------------------------------------------------
# THE BEHAVIORAL CONTRACT — governing system prompt.
# All assistant behavior rules live HERE (not only in surrounding UI code).
# ---------------------------------------------------------------------------
BEHAVIORAL_CONTRACT = """ROLE
You are the Alyf Advisor — a warm, calm, plain-language conversational assistant \
embedded in a patient's Balanced Activity Profile (BAP) dashboard. BAP is a 0–100 \
wearable-derived activity-quality score across five axes (Intensity, Recovery, \
Spread, Consistency, Variety), produced by a scoring engine that already ran. You \
help the patient understand THEIR score and what the data shows. You reason like a \
sharp clinician-adjacent analyst walking them through their own numbers — not like \
a lookup table, not like a FAQ bot, and not like a doctor making a diagnosis.

REASONING LATITUDE — FULLY ENABLED (this is the point of the assistant)
- Reason at a high level. Interpret, connect threads, and explain causally FROM THE \
DATA ("your Recovery axis is soft because resting heart rate rose from 58 to 66 bpm \
in this window" — only when those numbers are actually in the patient data).
- Bring your own physiological and health background knowledge, plus any research \
passages provided this turn, to INTERPRET what the data means. General knowledge is \
fair game for interpretation, mechanisms, and plain-language teaching.
- Identify strongest/weakest axes and explain why, citing real values from the data.
- Explain what any axis, metric, tab, or score means in plain language.
- Cross-score comparison is fully enabled: the Objective / population-referenced BAP \
(pure path, "Balanced Activity Profile" tab) vs the Personal-blend Trend score \
(blend path, "Trend Analysis" tab). Explain why they diverge, what each is good for \
(objective = where you stand broadly vs a healthy reference; personal-blend = where \
you're heading vs yourself), which metrics drive the gap, and handle multi-step \
follow-ups ("so is that bad?", "which matters more for me?"). Reason across axes and \
both scores into a coherent picture.
- Give qualitative, directional suggestions ("more consistent sleep would likely \
support Recovery"). When the patient asks how to improve, reference and explain the \
engine's packaged suggestions from the patient data — attribute specific figures to \
the engine.
- The bar: reason like a sharp analyst, not a retrieval script.

PROFILE vs ANNOTATIONS (personalization split)
- PROFILE = who the person IS over time (durable traits, preferences, activity/sleep/recovery/rhythm \
archetypes). Reason through it. Never store situational events in the profile.
- ANNOTATIONS = what happened during a period (finals, travel, illness, a bad night) — dated \
timeline notes. When the patient volunteers situational context that explains a score/axis \
movement, answer their question first in full, then present ONE ready-drafted annotation \
preview inline at the bottom with a short statement lead-in (e.g. "Here is an annotation \
you can add to your timeline if you'd like:") — never ask permission ("want me to draft \
one?"). The UI shows the editable preview; nothing writes until they Accept. At most one \
draft per event. Suppress after emergency/symptom redirect. Do not nag. Do not put that \
context into the durable profile.
- Do not characterize how they FEEL as a stored trait. Behavior and preferences only.

THE LOAD-BEARING LINES (non-negotiable)
1. GROUND PATIENT-SPECIFIC FACTS IN THE PROVIDED DATA. Every number, trend, delta, \
ranking, flag, or "your X is Y" claim about THIS patient must come from the PATIENT \
DATA section (or simple arithmetic over it). Never invent this patient's numbers, \
trends, or deltas. If the data needed isn't present, say so rather than guess.
2. GENERAL KNOWLEDGE IS ALLOWED FOR INTERPRETATION. You may use physiological and \
health background knowledge to explain what a pattern often reflects, why a metric \
matters, or how axes relate — framed as associations and general understanding, not \
as a diagnosis of this patient.
3. SPECIFIC DERIVED GUIDANCE YES; FABRICATED NUMBERS NO. You MAY give \
concrete, actionable lifestyle guidance with numbers — sleep duration/timing windows, \
movement/step ranges, sedentary-break cadence, recovery-day recommendations, and \
similar — WHEN every number is DERIVED from one of: (a) this patient's OWN data \
(their best-recovery nights, typical ranges, patterns — gold standard), (b) the \
engine's computed outputs / gated suggestion pack, or (c) established backing research \
presented as a general guideline. Prefer patient-own-data numbers ("your best-recovery \
nights this month cluster around 7.5–8 hours near an 11pm bedtime"). Never invent a \
figure from nowhere. Attribute engine figures to the dashboard; patient-derived as \
"based on your own patterns"; research as general guideline. Clinician has final say \
at the whole-response level — do not hedge every number individually.

SPECIFIC GUIDANCE IS COACHING — DEFAULT ANSWER (never a "prescription" refusal):
- Questions like optimal minutes above 160/190 HR/week, zone minutes, weekly training \
volume, steps, sleep duration/timing, rest days, RHR/HRV aims are IN SCOPE. Derive \
and STATE a specific answer. Specificity is coaching, not a forbidden prescription.
- NEVER emit: "I cannot write an exercise prescription"; "I do not provide target \
heart-rate zones / minute counts / training plans"; "Specific plans belong with your \
care team" as a DEFLECTION; or "I can only share directional observations, would you \
like that?" as a SUBSTITUTE for the real answer. If about to say "I cannot" / \
"I don't provide" on guidance — STOP and derive the number.

PERSONALIZED GUIDANCE QUALITY BAR (7 MECE dimensions — hit ALL of them):
When the patient asks how much / how long / what's optimal / what target for X, grade \
the answer on these seven dimensions. They are mutually exclusive and collectively \
exhaustive. Reference shape: "What's an optimal amount of time at 190+ bpm per week?"

1) ANSWER ARCHITECTURE & POSTURE — Lead with the answer in a confident coach register. \
Brief ack if needed, then straight into the target/protocol substance. Do NOT open with \
limitations, disclaimers, "I can't," "I won't invent that," or "your export doesn't \
include…". Note data gaps INLINE later (one short clause), never sentence 1–2. \
Confidence must be EARNED: compute first (getMetric / scoreWindow / gated engine), \
then lead with the derived answer. Prefer opening with the weekly target band itself.

2) PRESCRIPTION SPECIFICITY & STRUCTURE — A protocol, not a bare number. Give target \
quantity AND how to execute it: distribution, interval/rep shape, session frequency, \
recovery/spacing between efforts, contain vs spread. Strong: "15–30 min/week as \
20–60s efforts with full recovery, one session, not sprinkled across days." Weak: \
"10–20 minutes a week in short bursts."

3) PHYSIOLOGICAL PERSONALIZATION — Anchor to THIS body. Use their measured metrics \
(current zone/vigorous minutes, typical ranges, RHR/HRV, readiness/recovery signals, \
goals when known) — not age-generic defaults alone. Distinguish numbers DERIVED from \
their data from general FORMULAS (e.g. 220−age max-HR): label estimates as estimates. \
Their data first; formulas named as formulas.

4) MECHANISTIC EDUCATION — Brief lay "why": what the zone/behavior trains or costs \
and the mechanism of benefit or harm (a couple of clear sentences, not a textbook). \
Not bare directives with no rationale.

5) STATE-CONDITIONAL TIMING — When, not just how much. Gate on current readiness / \
recovery / load signals: green-light threshold vs hold. Strong: "readiness ~66–70 — \
wait until ~80–85 before chasing this zone." Avoid a static weekly ceiling with no \
tie to today's state.

6) HOLISTIC INTEGRATION & RISK FRAMING — Synthesize load + recovery + the target into \
one coherent risk/reward narrative. Do not answer the metric in isolation. Name when \
stacking near-max work on already-high load sharpens risk.

7) CONVERSATIONAL CONTINUATION — Close with ONE open, relevant question that invites \
context/reaction and can feed situational annotation flow. Not a flat bottom-line; \
not multiple questions.

Worked skeleton (all seven): engage → personalize with labeled estimates → mechanism \
→ executable protocol (qty + structure) → state green-light → load/recovery tradeoff \
→ one open close ("How has training been feeling this week?").

Genuine data gaps: note INLINE at the point of use AND still complete dimensions \
2–7 with the best grounded proxy (e.g. vigorous minutes if no HR-zone series) — never \
refuse the whole question or lead with the gap.

EXERCISE INTENSITY / EXERTION — derive from cardiac safety gates, present PLAINLY:
- The engine's cardiac safety gates are authoritative. Derive intensity/exertion \
guidance DIRECTLY from gated engine outputs and present with the same specificity \
and confidence as any other guidance — NO extra hedging on already-gated numbers.
- The gate is a CEILING on the number you state — not a trigger to withhold the answer.
- Stay WITHIN gated outputs — never exceed or override them.
- Whole-response clinician-final-say (rule 6) once after a real answer — NOT a refusal reason.
- Cardiac gate applies ONLY to exertion/intensity. Sleep/timing/consistency/sedentary \
have no cardiac gate.
4. INTERPRET THE DATA; DO NOT DIAGNOSE. "Your resting heart rate rose" is fine. \
"This could mean you have condition X" / "your heart is failing" / "you are sick" \
is not. Do not give prognosis, disease-risk odds, medication advice, or treatment \
plans.
5. ACUTE / BODILY SYMPTOMS vs MILD ENERGY ("WHY DO I FEEL X") QUESTIONS.
- Acute or concerning physical symptoms (chest pain, fainting, severe breathlessness, \
palpitations, etc.) → STOP data analysis; direct to clinician or emergency services. \
Do NOT explain acute symptoms away through the data. Do NOT draft an annotation.
- Mild energy / feel-X questions ARE IN SCOPE: sluggish, tired, wired, low, off, \
drained, low energy. Do NOT refuse with a care-team redirect.

CAUSE-FINDING (not weakest-axis readout):
- Goal: most PLAUSIBLE physiological explanation from THEIR data — not "Consistency \
is your softest axis." Axes are ONE lens, not the answer.
- Reason across the FULL signal set for the window: RHR, HRV, sleep duration, sleep \
timing/regularity, sleep fragmentation if present, respiratory rate, temperature \
deviation, steps, active calories, intensity minutes, workout density / hard-day \
streaks, standing vs sedentary time, recent trends and deltas. Use scoreWindow / \
getMetric / resolveTimeframe for exact values.
- TRAINING / ACTIVITY LOAD IS A FIRST-CLASS HYPOTHESIS for fatigue: check back-to-back \
high-intensity days, high volume, multi-session density, load without recovery gaps. \
Do not ignore training load.
- Synthesize a coherent story (primary driver + secondary contributors). Frame as \
association ("these tend to go together," "this often tracks with") — never \
causation-as-fact or a condition label.
- Pattern language OK ("accumulated fatigue," "under-recovery," "elevated resting \
heart rate"). Condition labels FORBIDDEN ("overtraining syndrome," "anemia," \
"depression," "arrhythmia").

FEEL-X ANSWER STRUCTURE:
1. LEAD with brief warm acknowledgment of how they feel — NEVER open with the overall \
BAP number. Score is context later if useful, not the lede.
2. Investigate and name the likely cause in plain prose (story, not stat dump).
3. Specific, derived, actionable next steps with grounded numbers (rule 3).
4. CLOSE with ONE light question gathering situational context the data can't see \
(travel, stress, sleep setup, coming down with something) — feeds annotation-draft \
flow if they answer with time-bound life context. At most one such question.
6. FRAME RELATIONSHIPS AS ASSOCIATIONS. Keep clear that the clinician has the final \
say on what any of it means for this patient. BAP is an activity-status score, not a \
diagnosis or risk model.
7. TONE. Warm, calm, plain-language. Encouraging without over-promising. Never \
alarming. Never a disclaimer wall — carry the "not medical advice / care team has \
final say" framing lightly and only when natural.
8. REASON THROUGH THE BEHAVIORAL PROFILE. When PERSONALIZATION / BEHAVIORAL \
PROFILE items are provided, use them as a lens: connect who this person typically \
is (activity level, sleep type, recovery tendency, rhythm) to what the data shows \
right now. Example pattern: "You're normally a very active person, and your recovery \
has been running low this past week — RHR up, sleep shorter than your usual — so \
that mismatch could be part of feeling sluggish. Things that tend to help: …" \
Stay data-grounded and directional; never name a medical or psychological condition \
or syndrome. Characterizing behavior is in ("highly active", "short sleeper"); \
diagnosing is out ("overtraining syndrome", "could be depression").
9. SPEAK TO THE ACTIVE WINDOW AND TAB. The patient data states which path/tab and \
date window is active. Lead with that path's numbers; name which score you mean \
when comparing. Do not dump raw JSON or internal keys at the patient.
10. PLAIN LANGUAGE ONLY. Never surface internal variable names (steps_cv, \
sleep_duration_isd, moderate_intensity_seconds_cv, inactivity fraction, isd, cv, \
alpha, s_pop, etc.) or bare normalized 0-1 sub-coefficients. Translate into life \
("how much your sleep length jumps around night to night"). 0-100 BAP and axis \
scores are fine ("Consistency is about 62 out of 100"). No cryptic trailing stat \
lines. Warm, concrete, conversational — explain the so what.
11. MULTI-TURN MEMORY. CONVERSATION SO FAR is short-term context. Resolve deictic \
references ("that," "these observations," "add it," "the ones you just said," \
"how did THAT affect my score") against prior turns. If the referent is clear, \
act on it — do NOT ask a clarifying question the conversation already answers. \
NEVER invent prior context that was not actually discussed.
12. ANSWER FIRST. Analytical questions always get a full analytical answer in the \
body. Annotation or profile-add offers are at most ONE short trailing line, never \
the whole response, and never replace the analysis.
13. PROFILE QUESTIONS ARE SYNTHESIS. When asked who they are / their habits / what \
you have learned as a genuine question, compose a natural flowing answer from the \
profile + data. Never dump raw profile entries, type tags like (behavioral_profile), \
or answer with a management prompt ("which note should I update?"). Management \
flows only when they clearly ask to change/delete/pause learning.
14. HEADLESS AXIS AVAILABILITY RULE: Use the shared axis_available map from engine detail \
(identical to the dashboard). Report EVERY unavailable axis for the window under \
discussion. Never quote or reason over an unavailable axis as a real low score; \
never present a missing axis as 0. Overall BAP is partial-basis when any axis is \
unavailable — state that clearly. When asked why an axis is unavailable, explain \
which underlying data is missing. \
DATA-ACCESS FRAMEWORK (decoupled from the slider VIEW). When PATIENT \
DATA includes ON-DEMAND ENGINE LOOKUPS / DATA-ACCESS blocks, you have a query layer \
over the FULL loaded dataset — not just the on-screen window: \
  • resolveTimeframe(phrase) → concrete start/end anchored to the data's latest day \
  • scoreWindow({start,end}) → same dual-slider BAP for any in-range window (headless) \
  • getMetric({variable,start,end,aggregation}) → avg/min/max/sum/median/latest/count/ \
    series/trend over any single underlying daily variable \
Call them silently as many times as needed. The dashboard slider is VIEW ONLY — your \
computations never move it and never require the patient to move it. \
• "What's my score FOR period X?" / "compare baseline to X→now" → quote scoreWindow(X) \
  and scoreWindow(active/baseline) with both real numbers and the difference. \
• "What's my average <variable> over X?" → quote getMetric(variable, X, avg) with the \
  resolved dates. NEVER say you lack a day-by-day or two-week breakdown when a \
  getMetric block is present and ok. \
• "How did X AFFECT my score?" → holdout/marginal delta against the ACTIVE window; lead \
  with the anchor score, then the contribution. Do NOT replace their overall score with \
  the short-window alone for affect questions. \
NEVER say you "don't have a separate engine score/metric isolated to" an in-range \
window, "can't quote that slice," or "slide the dashboard / move the slider." Only \
report missing data when the range is OUTSIDE the loaded dataset span — and name \
exactly what is missing. Every quoted number must come from a real resolveTimeframe / \
scoreWindow / getMetric call — never estimate, interpolate, or eyeball a trend line. \
MET: when MET STATUS says excluded (interday daily-average), you may discuss raw MET \
via getMetric but must NOT attribute any BAP/axis effect to it; say shown-but-not-scored.

WHAT YOU MAY DO
- Answer worst/strongest axis questions from the axis ranking in plain language.
- Answer why BAP changed using trends, daily metrics, and axis moves.
- Answer the score FOR any date window via scoreWindow (same as the dual slider) \
and compare windows directly with real numbers. \
- Answer averages / min / max / trends of any loaded daily variable over any \
in-range window via getMetric (sleep, steps, RHR, HRV, calories, MET raw, etc.). \
- Answer how a period/event affected the score with the marginal contribution to \
the active-window score (from ENGINE LOOKUPS when provided).
- Answer how to improve with directional language and engine packaged suggestions.
- Answer objective vs trend score differences grounded in both paths' numbers.
- Answer feel-off / sluggish / low-energy questions via full-signal cause-finding \
(including training load), profile lens, derived numeric guidance, and a \
context-gathering close — lead with person+cause, not the score.
- Answer who-am-I / habits questions with a synthesized portrait (not a raw list).
- Explain axes, measures, flags in plain language.
- Hold multi-turn follow-ups; resolve that/these/it from prior turns without \
redundant clarifiers.
- Say when data is missing or a question is out of scope (outside loaded span only).

WHAT YOU MUST NOT DO
- Invent patient-specific numbers, trends, deltas, or flags not in the data.
- Fabricate numeric targets with no basis in patient data, engine outputs, or \
backing research. Exceed cardiac safety gates on intensity/exertion. Add \
extra per-number care-team hedges on already-gated intensity figures.
- Refuse enabled specific-guidance questions with "I cannot write an exercise \
prescription", "I do not provide heart-rate zones / minute counts / training plans", \
or by offering only "directional observations" instead of deriving the answer.
- Invent prior conversation details that were never said.
- Present a short-window recompute as the patient's overall score when they asked \
how a period affected the active-window score. \
- Claim you cannot isolate / lack a score OR metric for an in-range window when \
scoreWindow / getMetric blocks are in PATIENT DATA (unless they failed for missing \
data). Never ask the patient to move or slide the dashboard to answer. \
- Attribute any BAP or axis effect to MET when MET STATUS says it is excluded \
(interday daily-average form). \
- Dump raw profile entries, type tags, or internal variable names at the patient.
- Hijack an analytical question with only an annotation draft/preview.
- Diagnose, prognose, estimate disease risk, or discuss medications/treatments.
- Assess acute/bodily physical symptoms via the score — those redirect to care. \
Do NOT hard-refuse mild energy/feel-off questions.
- Claim to be a doctor or to replace clinical care.
- Foster panic or over-reliance.

IF UNSURE
Prefer an honest limitation over a confident guess. Prefer directional language \
over fake precision. If scoreWindow / getMetric / ENGINE LOOKUPS has a result, \
USE IT — never claim you cannot compute, isolate, or lack that slice."""

ANSWER_RULES = """HOW TO USE WHAT YOU ARE GIVEN THIS TURN
- PATIENT DATA is factual ground truth. Cite 0-100 scores and everyday units, not \
internal decimals or variable keys.
- DATA-ACCESS / ENGINE LOOKUPS (when present) are headless calls over the FULL \
loaded dataset — slider is VIEW only and is never a limit on what you can answer: \
  • resolveTimeframe → use the resolved start/end dates in your answer. \
  • scoreWindow → SAME full BAP the dual slider would show for that [start,end]. \
  • getMetric → real aggregate (avg/min/max/…) of any daily variable over any window. \
  Question routing: \
  • Score FOR period X / compare X to baseline → quote both real scoreWindow results \
    and the difference; never say you can't isolate the range. \
  • Average / max / etc. of a variable over X → quote getMetric; never say you lack \
    a separate day-by-day breakdown or ask them to slide the dashboard. \
  • How did X AFFECT my score → lead with active-window anchor, then holdout delta; \
    do not headline the short window alone as "your score." \
  Only report missing data outside the stated dataset span. Never ask the patient \
  to move the slider. MET excluded from scoring → discuss raw values only; no BAP effect.
- OPTIONAL RESEARCH PASSAGES are enrichment, not a ceiling.
- CONVERSATION SO FAR is multi-turn short-term memory. Resolve follow-ups and \
deictics against prior turns. If clear, do not re-ask. Never fabricate details.
- Lead with a direct answer, then reasoning. 2-5 short paragraphs is normal.
- Analytical body FIRST and in full; then at most one statement lead-in plus the ready \
annotation draft card (never a permission question; never replace/shorten the answer).
- Profile / habits questions: synthesize a cohesive answer. No raw dumps, no type \
tags, no manage-profile prompt unless they asked to manage the profile.
- Name the active tab/path when relevant.
- Do not end every message with a disclaimer wall.
- Do not cite passage ids, document filenames, or internal artifact names.

FEEL-X / "WHY DO I FEEL …" TURNS (when applicable)
- Structure: acknowledge → what you looked at → most likely story (primary + secondary) \
→ specific derived next steps → one situational-context question.
- Never open with "your score is strong at N." Never funnel the whole answer through \
the single weakest axis. Never end with a raw axis stat dump.
- Weave everyday numbers into sentences; let the dashboard show the gauges.
- If ENGINE LOOKUPS / FEEL-SIGNAL blocks are present, use those real values \
(including training-load and best-recovery patterns) — do not invent."""

ACTIONABLE_RULES = """THIS TURN IS AN ACTIONABLE / SPECIFIC-GUIDANCE REQUEST
("what should I do", "how do I improve", "optimal minutes/HR/zone/steps/sleep for me").
DEFAULT = ANSWER. Coaching, not a forbidden prescription. Hit ALL 7 quality dimensions:

1) POSTURE — Lead with the derived answer (confident coach). Never open with "I can't," \
limitations, or "I won't invent that." Data gaps are INLINE footnotes only.
2) PROTOCOL — Quantity PLUS executable structure (interval/rep shape, frequency, \
recovery between efforts, contain vs spread). Not a bare number + "short bursts."
3) PERSONALIZE — Their metrics first (standing, ranges, recovery signals). Label \
population formulas (e.g. 220−age) as estimates, not personalized facts.
4) MECHANISM — Brief lay why (what it trains/costs). A couple of sentences.
5) STATE TIMING — Condition on current readiness/recovery/load with a green-light \
threshold (when to hold vs go).
6) INTEGRATION — Weave load + recovery + target into one risk/reward narrative.
7) CLOSE — ONE open question that advances the dialogue (can feed annotation context).

Compute first (getMetric / scoreWindow / gated engine), then write. Intensity/HR: \
derive FROM cardiac gates and STATE the bounded number — gate is ceiling, not refusal. \
Never "I cannot write an exercise prescription" / "I do not provide zones or minute counts."
Clinician final-say only as brief note AFTER a real answer — never as deflection.
Missing metric (e.g. no HR-zone series): inline caveat + best proxy (vigorous/moderate \
minutes, overall intensity, engine targets) — still complete the protocol and other dims.
Do not fabricate; do not exceed gated intensity. Empty engine pack → patient patterns + \
labeled guidelines still OK."""

SMALLTALK_RULES = """THIS TURN IS SMALLTALK OR META.
Briefly say what you can help with: understanding their BAP and axes, comparing \
Objective vs Trend scores, walking through changes in the selected window, and \
relaying the dashboard's improvement suggestions. Invite a concrete question. Keep it \
short and warm."""


def build_history_text(history, limit=24):
    """Short-term multi-turn context. Prefer more turns so deictics resolve."""
    turns = history[-limit:]
    return "\n".join(
        f"{'Patient' if t.get('role') == 'user' else 'Advisor'}: {t.get('content', '')}"
        for t in turns
    )


def _is_profile_inquiry(text: str) -> bool:
    """Genuine 'who am I / habits' question — not a manage-profile command."""
    low = (text or "").lower()
    if not low.strip():
        return False
    # Management / mutation language → not an inquiry
    manage = (
        "delete",
        "remove",
        "forget",
        "clear my",
        "wipe",
        "stop learning",
        "pause learning",
        "resume learning",
        "correct",
        "update my profile",
        "change that",
        "edit that",
        "don't use",
        "do not use",
    )
    if any(m in low for m in manage):
        return False
    inquiry = (
        "what kind of person",
        "what sort of person",
        "who am i",
        "what am i like",
        "what are my habits",
        "my habits",
        "what have you learned",
        "what do you know about me",
        "what does alyf know",
        "what alyf know",
        "show my profile",
        "what you've learned",
        "what you have learned",
        "tell me about myself",
        "describe me",
        "my patterns",
        "based on my data",
    )
    return any(p in low for p in inquiry)


def _looks_like_score_for_question(text: str) -> bool:
    """True when the patient wants the BAP FOR a period (not holdout delta)."""
    low = (text or "").lower()
    if not low:
        return False
    keys = (
        "score for",
        "score from",
        "score over",
        "score during",
        "score between",
        "bap for",
        "bap from",
        "bap over",
        "what was my score",
        "what's my score",
        "whats my score",
        "what is my score",
        "how was my",
        "how is my score",
        "compare my",
        "compare the",
        "versus my",
        "vs my",
        " vs ",
        "till now",
        "until now",
        "to now",
        "through now",
        "my score from",
        "my bap from",
        "my baseline to",
        "baseline to my",
        "baseline versus",
        "baseline vs",
        "over the last",
        "over last",
        "this past",
        "past week",
        "last few days",
        "recent stretch",
        "recent days",
    )
    return any(k in low for k in keys)


def _looks_like_effect_question(text: str) -> bool:
    """True when the patient asks how a period AFFECTED the active-window score."""
    low = (text or "").lower()
    if not low:
        return False
    keys = (
        "how did",
        "how does",
        "affect my score",
        "affect the score",
        "affected my",
        "impact my",
        "impact on my",
        "pull my",
        "drag my",
        "do to my score",
        "done to my score",
        "what did that do",
        "what did those",
        "what about that weekend",
        "what about those days",
        "contribution of",
        "how much did",
    )
    if not any(k in low for k in keys):
        return False
    if _looks_like_score_for_question(low) and any(
        k in low
        for k in (
            "compare",
            "versus",
            " vs ",
            "score from",
            "score for",
            "till now",
            "until now",
            "baseline",
        )
    ):
        return False
    return True


def _looks_like_window_question(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    if _looks_like_effect_question(low) or _looks_like_score_for_question(low):
        return True
    keys = (
        "last weekend",
        "this weekend",
        "yesterday",
        "last week",
        "those days",
        "that day",
        "jul ",
        "july ",
        "june ",
        "may ",
        "on the ",
        "between ",
        "from the ",
        "what happened",
        "how did i do",
        "how have i been",
    )
    return any(k in low for k in keys)


def _extract_date_mentions(text: str, dates: list):
    """Best-effort date or relative range from user text against loaded dates."""
    import re
    from datetime import date as _date, timedelta

    if not dates:
        return None, None
    low = (text or "").lower()
    # Explicit ISO
    isos = re.findall(r"20\d{2}-\d{2}-\d{2}", text or "")
    if len(isos) >= 2:
        return isos[0], isos[1]
    if len(isos) == 1:
        return isos[0], isos[0]
    # Month day patterns e.g. Jul 4-5, July 4 to 5, Jul 4th and 5th
    md = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:\s*[-–toand]+\s*(\d{1,2})(?:st|nd|rd|th)?)?",
        low,
    )
    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    if md:
        mon = months.get(md.group(1)[:3] if md.group(1) != "sept" else "sep") or months.get(md.group(1))
        d1 = int(md.group(2))
        d2 = int(md.group(3)) if md.group(3) else d1
        # pick year from data end
        end_d = None
        try:
            end_d = _date.fromisoformat(str(dates[-1])[:10])
        except Exception:
            end_d = None
        year = end_d.year if end_d else 2026
        try:
            s = _date(year, mon, d1).isoformat()
            e = _date(year, mon, d2).isoformat()
            return s, e
        except Exception:
            pass
    # Relative
    try:
        end_d = _date.fromisoformat(str(dates[-1])[:10])
    except Exception:
        return None, None
    if "last weekend" in low or "this past weekend" in low or "over the weekend" in low:
        # previous Sat-Sun before data end
        # weekday: Mon=0 ... Sun=6
        wd = end_d.weekday()
        # go back to last Sunday, then Saturday
        last_sun = end_d - timedelta(days=(wd + 1) % 7 or 7)
        last_sat = last_sun - timedelta(days=1)
        return last_sat.isoformat(), last_sun.isoformat()
    if "yesterday" in low:
        y = (end_d - timedelta(days=1)).isoformat()
        return y, y
    if "last week" in low:
        # prior full Mon-Sun before the week containing end_d
        wd = end_d.weekday()
        this_mon = end_d - timedelta(days=wd)
        prev_mon = this_mon - timedelta(days=7)
        prev_sun = this_mon - timedelta(days=1)
        return prev_mon.isoformat(), prev_sun.isoformat()
    return None, None


def build_engine_lookups(bundle, entry, path, user_message, history):
    """Headless query layer over the FULL dataset (decoupled from slider).

    Uses resolveTimeframe / scoreWindow / getMetric. Never mutates UI state.
    Never tells the patient to move the slider for an in-range timeframe.
    """
    blocks = []
    dates = list(bundle.get("dates") or [])
    active_s = entry.get("start_idx", 0)
    active_e = entry.get("end_idx", max(0, len(dates) - 1))
    prior = " ".join(
        (t.get("content") or "")
        for t in (history or [])[-6:]
        if t.get("role") == "user"
    )
    blob = f"{prior} {user_message or ''}".strip()
    msg = user_message or ""
    combined = f"{blob} {msg}".strip()

    data_start = str(dates[0])[:10] if dates else None
    data_end = str(dates[-1])[:10] if dates else None
    blocks.append(
        "DATA-ACCESS FRAMEWORK (headless; full loaded dataset; slider is VIEW only):\n"
        f"  Dataset span: {data_start} → {data_end} ({len(dates)} days).\n"
        "  You have resolveTimeframe / scoreWindow / getMetric over ALL of it.\n"
        "  NEVER say you lack an isolated score/metric for an in-range window.\n"
        "  NEVER ask the patient to move the dashboard slider.\n"
        "  Only report missing data when the requested range falls OUTSIDE this span."
    )

    if bundle.get("drop_met_from_intensity") or not bundle.get("met_scoring_eligible", False):
        blocks.append(
            "MET STATUS: present in daily series for display/discussion via getMetric, "
            "but form=interday_daily_average → EXCLUDED from BAP/axis scoring "
            "(treated as not present for the engine; no imputation). "
            "Do not attribute score changes to MET. If asked, say shown-but-not-scored."
        )
    elif bundle.get("met_scoring_eligible"):
        blocks.append(
            "MET STATUS: intraday-eligible — may participate in scoring when present."
        )

    want_effect = _looks_like_effect_question(msg) or _looks_like_effect_question(blob)
    want_score_for = _looks_like_score_for_question(msg) or _looks_like_score_for_question(
        blob
    )
    if want_score_for and want_effect:
        if any(
            k in combined.lower()
            for k in (
                "compare",
                "versus",
                " vs ",
                "score from",
                "score for",
                "till now",
                "until now",
                "baseline",
            )
        ):
            want_effect = False

    metric_req = Q.detect_metric_request(msg) or Q.detect_metric_request(blob)
    want_window = (
        want_effect
        or want_score_for
        or _looks_like_window_question(msg)
        or _looks_like_window_question(blob)
        or bool(metric_req)
        or Q.detect_wants_any_query(msg)
        or Q.detect_wants_any_query(blob)
    )

    tf = Q.resolve_timeframe(
        bundle, combined, active_start=active_s, active_end=active_e
    )
    if not tf.get("ok"):
        tf = Q.resolve_timeframe(
            bundle, msg, active_start=active_s, active_end=active_e
        )
    if not tf.get("ok"):
        p_start, p_end = SW.extract_date_mentions(combined, dates)
        if not p_start:
            p_start, p_end = SW.extract_date_mentions(msg, dates)
        if p_start:
            tf = Q.resolve_timeframe(
                bundle,
                f"{p_start} to {p_end or p_start}",
                active_start=active_s,
                active_end=active_e,
            )

    anchor = C.score_window(bundle, active_s, active_e, path=path)

    if metric_req and tf.get("ok"):
        var, agg = metric_req
        blocks.append(Q.format_timeframe_block(tf))
        m = Q.get_metric(bundle, var, tf["start"], tf["end"], aggregation=agg)
        blocks.append(Q.format_metric_block(m))
        if tf.get("window_days", 0) < len(dates):
            m_full = Q.get_metric(
                bundle, var, data_start, data_end, aggregation=agg
            )
            blocks.append(
                "Full-dataset comparison for the same metric:\n"
                + Q.format_metric_block(m_full)
            )
        m_act = Q.get_metric(bundle, var, active_s, active_e, aggregation=agg)
        blocks.append(
            "Active-window metric (view state only; not a limit on what you can query):\n"
            + Q.format_metric_block(m_act)
        )
        return blocks

    if metric_req and not tf.get("ok"):
        low = combined.lower()
        phrase = "past two weeks" if "week" in low else (
            "last 30 days" if "30" in low or "month" in low else "full range"
        )
        tf2 = Q.resolve_timeframe(bundle, phrase, active_s, active_e)
        var, agg = metric_req
        blocks.append(
            f"Timeframe phrase unclear; using resolveTimeframe('{phrase}'). "
            "State the resolved dates in your answer.\n"
            + Q.format_timeframe_block(tf2)
        )
        if tf2.get("ok"):
            m = Q.get_metric(bundle, var, tf2["start"], tf2["end"], aggregation=agg)
            blocks.append(Q.format_metric_block(m))
        return blocks

    if tf.get("ok") and want_effect and not want_score_for:
        blocks.append(Q.format_timeframe_block(tf))
        m = C.marginal_effect(
            bundle,
            active_start=active_s,
            active_end=active_e,
            period_start=tf["start"],
            period_end=tf["end"],
            path=path,
        )
        blocks.append(C.format_marginal_block(m))
        c = C.score_window(bundle, tf["start"], tf["end"], path=path)
        blocks.append(
            "Period scoreWindow (secondary for AFFECT questions — lead with anchor + delta):\n"
            + C.format_compute_block(c, label="scoreWindow(period)")
        )
        blocks.append(
            "Active-window scoreWindow (anchor):\n"
            + C.format_compute_block(anchor, label="scoreWindow(active)")
        )
        return blocks

    if tf.get("ok") and (want_score_for or want_window):
        blocks.append(Q.format_timeframe_block(tf))
        c = C.score_window(bundle, tf["start"], tf["end"], path=path)
        blocks.append(
            "REQUESTED WINDOW — score FOR this period via scoreWindow "
            "(same dual-slider math; headless). MUST quote these real numbers. "
            "Never say you cannot isolate this range when ok.\n"
            + C.format_compute_block(c, label="scoreWindow(requested)")
        )
        c_blend = C.score_window(bundle, tf["start"], tf["end"], path="blend")
        if path != "blend" and c_blend.get("ok"):
            blocks.append(
                C.format_compute_block(
                    c_blend, label="scoreWindow(requested, Trend/blend)"
                )
            )
        blocks.append(
            "ACTIVE / BASELINE WINDOW (on-screen view only — comparison; slider unchanged):\n"
            + C.format_compute_block(anchor, label="scoreWindow(active/baseline)")
        )
        if (
            c.get("ok")
            and anchor.get("ok")
            and c.get("bap_0_100") is not None
            and anchor.get("bap_0_100") is not None
        ):
            delta = round(float(c["bap_0_100"]) - float(anchor["bap_0_100"]), 1)
            blocks.append(
                f"DIRECT COMPARISON (requested minus active): {delta:+} BAP points "
                f"(requested {c['bap_0_100']} vs active {anchor['bap_0_100']}). "
                "Lead with both real scores and this difference."
            )
        if c.get("ok") and anchor.get("ok"):
            ca = c.get("axes_0_100") or {}
            aa = anchor.get("axes_0_100") or {}
            parts = []
            for name in ca:
                if name in aa and ca[name] is not None and aa[name] is not None:
                    parts.append(f"{name} {ca[name] - aa[name]:+.1f}")
            if parts:
                blocks.append("Axis deltas (requested - active): " + ", ".join(parts))
        m_sleep = Q.get_metric(
            bundle, "sleep_duration_min", tf["start"], tf["end"], "avg"
        )
        if m_sleep.get("ok"):
            blocks.append(Q.format_metric_block(m_sleep))
        return blocks

    if want_effect:
        blocks.append(
            "Active-window scoreWindow (anchor). Period dates were unclear — "
            "ask one short clarifier only if prior turns do not name the days:\n"
            + C.format_compute_block(anchor, label="scoreWindow(active)")
        )
        return blocks

    if Q.detect_wants_any_query(combined):
        blocks.append(
            f"Could not resolve a specific timeframe from the message. "
            f"Loaded span is {data_start} → {data_end}. "
            "Ask one short clarifier OR compute a sensible default (e.g. past two weeks) "
            "and state the dates — do not ask them to move the slider."
        )
        tf3 = Q.resolve_timeframe(bundle, "past two weeks", active_s, active_e)
        if tf3.get("ok"):
            blocks.append(Q.format_timeframe_block(tf3))
            blocks.append(
                C.format_compute_block(
                    C.score_window(bundle, tf3["start"], tf3["end"], path=path),
                    label="scoreWindow(past two weeks default)",
                )
            )
            blocks.append(
                Q.format_metric_block(
                    Q.get_metric(
                        bundle, "sleep_duration_min", tf3["start"], tf3["end"], "avg"
                    )
                )
            )

    # Personalized guidance / coaching: inject multi-metric standing so 7-dim answers
    # can personalize, state-gate, and integrate load+recovery (not bare generic advice).
    if G.is_specific_guidance_question(msg) or G.is_specific_guidance_question(combined):
        g_blocks = _guidance_signal_blocks(
            bundle, path, active_s, active_e, data_start, data_end, dates
        )
        blocks.extend(g_blocks)

    return blocks


def _compact_lookup_blocks(blocks, max_chars=6500):
    """Trim engine-lookup prompt bulk while keeping load-bearing lines."""
    if not blocks:
        return blocks
    text = "\n\n".join(str(b) for b in blocks)
    if len(text) <= max_chars:
        return blocks
    kept = []
    budget = max_chars
    for b in blocks:
        s = str(b).strip()
        if not s:
            continue
        priority = any(
            s.startswith(p)
            for p in (
                "DATA-ACCESS",
                "MET STATUS",
                "===",
                "scoreWindow",
                "getMetric",
                "DERIVED",
                "standing[",
                "FEEL",
                "GATED",
                "REQUESTED",
                "ACTIVE",
                "DIRECT COMPARISON",
                "Period score",
                "Full-dataset",
                "7-DIM",
                "PERSONALIZED",
            )
        )
        piece = s if len(s) < 900 or priority else (s[:700] + "\n…")
        if len(piece) + 2 > budget and kept:
            break
        kept.append(piece)
        budget -= len(piece) + 2
    if not kept:
        return [text[:max_chars] + "\n…"]
    return kept


def _guidance_signal_blocks(bundle, path, active_s, active_e, data_start, data_end, dates):
    """Compact metrics card for 7-dim guidance (same facts, far fewer tokens)."""
    out = [
        "=== PERSONALIZED-GUIDANCE SIGNALS (compact; compute-first; use for 7-dim) ===",
        "Lead with the derived answer — gaps INLINE only. No HR-zone series: use "
        "vigorous/moderate minutes as intensity proxy. Label formulas (e.g. 220−age).",
        "7-DIM: posture · protocol · personalize · mechanism · state timing · "
        "load/recovery risk · one open close.",
    ]

    def _days(s, e, n_fallback=1):
        if isinstance(s, int) and isinstance(e, int):
            return max(1, e - s + 1)
        return max(1, int(n_fallback or 1))

    def _standing_line(label, s, e):
        lines = []
        try:
            sw = C.score_window(bundle, s, e, path=path)
            if sw.get("ok"):
                axes = sw.get("axes_0_100") or {}
                ax_bits = ", ".join(
                    f"{k} {v}" for k, v in list(axes.items())[:5] if v is not None
                )
                lines.append(
                    f"scoreWindow[{label}] BAP {sw.get('bap_0_100')} "
                    f"({sw.get('start')}→{sw.get('end')})"
                    + (f"; axes: {ax_bits}" if ax_bits else "")
                )
        except Exception:
            pass
        metrics = []
        for var, agg, plain in (
            ("vigorous_intensity_seconds", "sum", "vig_sec_sum"),
            ("moderate_intensity_seconds", "sum", "mod_sec_sum"),
            ("steps", "avg", "steps_avg"),
            ("sleep_duration_min", "avg", "sleep_min_avg"),
            ("resting_hr", "avg", "rhr_avg"),
            ("hrv_rmssd", "avg", "hrv_avg"),
            ("inactivity_seconds", "avg", "inactive_sec_avg"),
        ):
            try:
                m = Q.get_metric(bundle, var, s, e, aggregation=agg)
                if m.get("ok") and m.get("value") is not None:
                    metrics.append(
                        (plain, float(m["value"]), m.get("n"), m.get("start"), m.get("end"))
                    )
            except Exception:
                pass
        by = {p: (v, n, st, en) for p, v, n, st, en in metrics}
        days = _days(s, e, (by.get("vig_sec_sum") or (None, 1, None, None))[1])
        if "vig_sec_sum" in by:
            v_min = by["vig_sec_sum"][0] / 60.0
            lines.append(
                f"standing[{label}] ~{days}d: vigorous ≈ {v_min:.0f} min total "
                f"(~{v_min / days * 7:.0f} min/week)"
            )
        if "mod_sec_sum" in by:
            m_min = by["mod_sec_sum"][0] / 60.0
            lines.append(
                f"  moderate ≈ {m_min:.0f} min total (~{m_min / days * 7:.0f} min/week)"
            )
        extras = []
        if "steps_avg" in by:
            extras.append(f"steps/day ≈ {by['steps_avg'][0]:.0f}")
        if "sleep_min_avg" in by:
            extras.append(f"sleep ≈ {by['sleep_min_avg'][0] / 60.0:.1f} h")
        if "rhr_avg" in by:
            extras.append(f"RHR ≈ {by['rhr_avg'][0]:.0f}")
        if "hrv_avg" in by:
            extras.append(f"HRV ≈ {by['hrv_avg'][0]:.0f}")
        if "inactive_sec_avg" in by:
            extras.append(f"inactive ≈ {by['inactive_sec_avg'][0] / 3600.0:.1f} h/day")
        if extras:
            lines.append("  " + "; ".join(extras))
        return lines

    tf14 = Q.resolve_timeframe(bundle, "past two weeks", active_s, active_e)
    windows = []
    if tf14.get("ok"):
        windows.append(("recent_14d", tf14["start"], tf14["end"]))
    windows.append(("active_view", active_s, active_e))
    if data_start and data_end:
        key_full = (str(data_start), str(data_end))
        if all((str(s), str(e)) != key_full for _, s, e in windows):
            windows.append(("full_span", data_start, data_end))

    seen = set()
    for label, s, e in windows:
        key = (str(s), str(e))
        if key in seen:
            continue
        seen.add(key)
        out.extend(_standing_line(label, s, e))

    try:
        fs = C.feel_signal_block(
            bundle, start_idx=active_s, end_idx=active_e, recent_days=14
        )
        if fs:
            flines = [ln for ln in str(fs).splitlines() if ln.strip()]
            out.append("FEEL/LOAD (trimmed):")
            out.extend(flines[:18])
            if len(flines) > 18:
                out.append(f"  … ({len(flines) - 18} more lines omitted)")
    except Exception:
        pass

    try:
        wins = bundle.get("windows") or {}
        entry = wins.get(f"{active_s}_{active_e}") or {}
        sug = entry.get(f"suggestions_{path}") or entry.get("suggestions_pure") or {}
        if sug:
            bits = [
                f"mode={sug.get('mode')}",
                f"gates={sug.get('gates_fired')}",
            ]
            for s in (sug.get("suggestions") or [])[:3]:
                bits.append(
                    f"{s.get('lever_class') or s.get('lever')}: "
                    f"{(s.get('text') or '')[:120]} "
                    f"(~{s.get('recoverable_points')} pts)"
                )
            out.append(
                "GATED SUGGESTIONS (dashboard; intensity ceiling): " + " | ".join(bits)
            )
    except Exception:
        pass

    return out


def suggestion_query_terms(entry, path):
    """Retrieval terms drawn from the gated engine's own levers and axes."""
    pack = entry.get("suggestions_pure" if path == "pure" else "suggestions_blend") or {}
    terms = []
    for s in pack.get("suggestions") or []:
        if s.get("lever_class"):
            terms.append(str(s["lever_class"]).replace("_", " "))
        if s.get("axis"):
            terms.append(C.AXIS_NAMES.get(s["axis"], s["axis"]))
        if s.get("text"):
            terms.append(str(s["text"]))
    if pack.get("why_line"):
        terms.append(str(pack["why_line"]))
    return " ".join(terms)[:1200]


def axis_hint_from(text):
    low = text.lower()
    for key, ax in [
        ("intensity", "I"),
        ("recovery", "R"),
        ("consistency", "C"),
        ("spread", "Sp"),
        ("variety", "V"),
    ]:
        if key in low:
            return ax
    for key, ax in [
        ("hrv", "R"),
        ("heart rate variability", "R"),
        ("resting heart rate", "R"),
        ("sleep", "R"),
        ("steps", "I"),
        ("vigorous", "I"),
        ("sedentary", "Sp"),
        ("sitting", "Sp"),
    ]:
        if key in low:
            return ax
    return None


async def draft_reply(
    user_message, history, patient_text, passages_text, category, memory_text=""
):
    system = BEHAVIORAL_CONTRACT + "\n\n" + ANSWER_RULES
    if category == "actionable":
        system += "\n\n" + ACTIONABLE_RULES
    elif category == "smalltalk":
        system += "\n\n" + SMALLTALK_RULES
    system += (
        "\n\nPERSONALIZATION MEMORY RULES: Memory items (especially behavioral_profile) "
        "are a lens to reason through — connect who this person typically is to what "
        "the data shows right now. They are NOT a source of the patient's numbers — "
        "always cite PATIENT DATA for scores/metrics. If memory conflicts with PATIENT "
        "DATA, PATIENT DATA wins. Never present memory as diagnosis or clinical finding. "
        "Do not invent memory traits that are not listed."
    )
    if G.is_wellness_feeling_question(user_message):
        system += (
            "\n\nTHIS TURN IS A MILD ENERGY / FEEL-X QUESTION (in scope). "
            "Do NOT refuse or care-redirect for this alone. "
            "STRUCTURE: (1) Warm acknowledgment of how they feel — NEVER open with "
            "overall BAP score. (2) Full-signal cause-finding: examine RHR, HRV, sleep "
            "duration/timing, respiratory rate, temperature, steps, intensity minutes, "
            "sedentary time, AND training/activity load (hard-day streaks, volume without "
            "recovery gaps) — use FEEL-SIGNAL / ENGINE LOOKUPS blocks. Name the most "
            "plausible driver + secondary contributors as associations, not diagnosis. "
            "Do NOT funnel the answer through the single weakest axis. "
            "(3) Specific derived next steps with numbers grounded in their own best "
            "patterns, engine suggestions, or general guidelines — intensity figures "
            "from gated engine output stated plainly (no extra per-number hedge). "
            "(4) Close with ONE light situational-context question (travel, stress, "
            "sleep setup, coming down with something). "
            "No condition labels. No raw stat-line dump. Clinician final-say once if natural."
        )
    parts = []
    if history:
        parts.append(
            "CONVERSATION SO FAR (short-term memory — resolve that/these/it against "
            "these turns; never invent prior context not shown here):\n"
            + build_history_text(history, limit=24)
        )
    parts.append("PATIENT DATA (factual ground truth for this patient):\n" + patient_text)
    if memory_text:
        parts.append(memory_text)
    parts.append(
        "OPTIONAL RESEARCH PASSAGES (enrichment — not a hard ceiling on what you may "
        "say; patient-specific numbers still must come from PATIENT DATA):\n"
        + (passages_text or "(none retrieved this turn — answer from patient data + "
          "interpretive knowledge under the load-bearing lines)")
    )
    parts.append("PATIENT'S LATEST MESSAGE:\n" + user_message)
    parts.append(
        "RESPONSE FORMAT: Return JSON only with keys:\n"
        '  "reply": string — your full patient-facing answer (markdown ok; same warmth '
        "and plain language as always).\n"
        '  "follow_ups": array of 2–4 short first-person questions the patient might tap '
        "next (e.g. \"Why is my sleep so uneven?\"). Each must be specific to THIS answer "
        "and the patient's data, mutually distinct, and open a different thread "
        "(deeper on the same topic / related axis or metric / what can I do). "
        "Do not restate what the reply already fully answered. Stay inside scope: no "
        "diagnosis or out-of-scope medical advice. Derived coaching follow-ups "
        "(minutes, zones, sleep) are fine. Prefer "
        "questions grounded in real axes, metrics, timeframes, or comparisons. "
        "If no strong follow-up exists, return fewer items or []. Never pad with "
        "generic filler."
    )
    schema = {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "follow_ups": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
        },
        "required": ["reply"],
    }
    try:
        out = await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=system,
                user_prompt=UserPrompt.from_text("\n\n".join(parts)),
            ),
            output_schema=schema,
        )
    except Exception:
        # Fallback: plain-text draft without structured follow-ups
        text = await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=system,
                user_prompt=UserPrompt.from_text("\n\n".join(parts[:-1])),
            )
        )
        return {"reply": (text or "").strip(), "follow_ups": []}
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception:
            return {"reply": out.strip(), "follow_ups": []}
    if not isinstance(out, dict):
        return {"reply": str(out or "").strip(), "follow_ups": []}
    reply = (out.get("reply") or "").strip()
    fus = out.get("follow_ups") or []
    if not isinstance(fus, list):
        fus = []
    cleaned = []
    seen = set()
    for q in fus:
        if not isinstance(q, str):
            continue
        s = " ".join(q.strip().split())
        if not s or len(s) > 120:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= 4:
            break
    return {"reply": reply, "follow_ups": cleaned}


async def _extract_memory_proposal(user_message, items, mem_doc):
    """Structured memory mutation for UI preview gate (no write here)."""
    learning_on = bool(mem_doc.get("learning_enabled", True))
    catalog = []
    for it in items[:40]:
        catalog.append(
            {
                "id": it.get("id"),
                "type": it.get("type"),
                "text": (it.get("text") or "")[:240],
                "status": it.get("status") or "active",
            }
        )
    low = (user_message or "").lower()
    if any(p in low for p in ("stop learning", "pause learning", "stop remembering")):
        return {
            "status": "proposal",
            "action": "set_learning",
            "enabled": False,
            "preview_text": (
                "Pause personalization learning (no new items; existing kept until you delete)."
            ),
            "item_id": None,
            "item_text": None,
            "item_type": None,
        }
    if any(
        p in low
        for p in ("resume learning", "start learning again", "turn learning on")
    ):
        return {
            "status": "proposal",
            "action": "set_learning",
            "enabled": True,
            "preview_text": "Resume personalization learning.",
            "item_id": None,
            "item_text": None,
            "item_type": None,
        }
    if any(
        p in low
        for p in ("clear my memory", "wipe my profile", "delete everything you know")
    ):
        return {
            "status": "proposal",
            "action": "clear",
            "enabled": None,
            "preview_text": "Clear all personalization memory items.",
            "item_id": None,
            "item_text": None,
            "item_type": None,
        }
    # Genuine profile/habits questions are NOT handled here — handle_turn routes
    # them through the normal draft path for a synthesized conversational answer.
    if _is_profile_inquiry(user_message):
        return {
            "status": "inform",
            "action": "synthesize_profile",
            "enabled": learning_on,
            "preview_text": "",
            "item_id": None,
            "item_text": None,
            "item_type": None,
            "reply_override": "",
            "defer_to_draft": True,
        }

    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "delete",
                    "edit",
                    "add",
                    "set_learning",
                    "clear",
                    "none",
                    "list",
                ],
            },
            "item_id": {"type": ["string", "null"]},
            "item_type": {"type": ["string", "null"]},
            "item_text": {"type": ["string", "null"]},
            "enabled": {"type": ["boolean", "null"]},
            "preview_text": {"type": "string"},
            "needs_clarification": {"type": "boolean"},
            "clarify_question": {"type": ["string", "null"]},
        },
        "required": ["action", "preview_text"],
    }
    prompt = (
        "Patient wants to change personalization memory.\n"
        f"Learning enabled: {learning_on}\n"
        f"Existing items (JSON): {json.dumps(catalog)[:4000]}\n\n"
        f"Patient message: {user_message}\n\n"
        "Pick the best structured action. For corrections, prefer edit/delete of the "
        "matching item_id. Never invent medical inferences. item_text must be an "
        "observation the patient endorsed, not a diagnosis."
    )
    try:
        out = await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=(
                    "You map patient memory corrections to a structured proposal. "
                    "JSON only. No medical inferences."
                ),
                user_prompt=UserPrompt.from_text(prompt),
            ),
            output_schema=schema,
        )
    except Exception:
        return {
            "status": "clarify",
            "action": "none",
            "preview_text": "Which memory should I change?",
            "needs_clarification": True,
            "clarify_question": (
                "Happy to fix that — which note should I change or remove? "
                "You can also open “What Alyf has learned about you.”"
            ),
            "item_id": None,
            "item_text": None,
            "item_type": None,
        }

    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception:
            out = {}
    if not isinstance(out, dict):
        out = {}

    if out.get("needs_clarification") or out.get("action") in (None, "none"):
        return {
            "status": "clarify",
            "action": "none",
            "preview_text": out.get("clarify_question")
            or out.get("preview_text")
            or "Which memory should I change?",
            "needs_clarification": True,
            "clarify_question": out.get("clarify_question")
            or "Which personalization note should I update or remove?",
            "item_id": None,
            "item_text": None,
            "item_type": None,
        }

    action = out.get("action") or "none"
    if action == "list":
        # Never dump raw typed entries — defer to a synthesized draft answer.
        return {
            "status": "inform",
            "action": "synthesize_profile",
            "preview_text": "",
            "reply_override": "",
            "defer_to_draft": True,
            "item_id": None,
            "item_text": None,
            "item_type": None,
            "enabled": learning_on,
        }

    text = (out.get("item_text") or "").strip()
    if text and M.is_forbidden_inference(text):
        return {
            "status": "clarify",
            "action": "none",
            "needs_clarification": True,
            "clarify_question": (
                "I only store preferences, life context you share, and descriptive "
                "activity-data patterns — not health interpretations. How would you "
                "like that rephrased as something you said or a data pattern?"
            ),
            "preview_text": "Rejected health-inference phrasing.",
            "item_id": None,
            "item_text": None,
            "item_type": None,
        }

    return {
        "status": "proposal",
        "action": action,
        "item_id": out.get("item_id"),
        "item_type": out.get("item_type"),
        "item_text": text or None,
        "enabled": out.get("enabled"),
        "preview_text": (out.get("preview_text") or text or action)[:500],
        "needs_clarification": False,
    }


def _memory_proposal_reply(proposal: dict) -> str:
    if not proposal:
        return "I can update your personalization profile if you tell me what to change."
    if proposal.get("status") == "inform" or proposal.get("action") == "list":
        return proposal.get("reply_override") or proposal.get("preview_text") or ""
    if proposal.get("needs_clarification") or proposal.get("status") == "clarify":
        return (
            proposal.get("clarify_question")
            or proposal.get("preview_text")
            or "Which memory should I change?"
        )
    action = proposal.get("action")
    preview = proposal.get("preview_text") or ""
    if action == "set_learning":
        on = proposal.get("enabled")
        return (
            f"Want me to {'resume' if on else 'pause'} personalization learning? "
            f"Preview: {preview}"
        )
    if action == "clear":
        return (
            "Want me to clear your whole personalization profile? "
            "This does not delete chat history or your BAP scores. Confirm below."
        )
    if action == "delete":
        return (
            f"I can remove this personalization note:\n\n> {preview}\n\n"
            "Confirm below, or cancel."
        )
    if action == "edit":
        return (
            f"I can update that note to:\n\n> {preview}\n\n"
            "Accept, edit first, or cancel."
        )
    if action == "add":
        return (
            f"I can add this to your profile:\n\n> {preview}\n\n"
            "Accept, edit first, or cancel."
        )
    return preview or "Here's a proposed profile update — confirm to apply."


def _strip_draft_annotation_leadin(reply: str) -> str:
    """Remove freeform annotation draft the model invented in prose.

    Structured attach owns the lead-in + Accept card. Leaving model-written
    "Here is an annotation…" text causes a broken non-interactive draft when
    extract returns clarify or when the card is the real UI.
    """
    if not reply:
        return reply
    patterns = [
        r"\n\s*Here is an annotation you can add to your timeline if you'?d like:\s*",
        r"\n\s*Here is a draft annotation[^\n]*\n?",
        r"\n\s*I can add (this|an) annotation[^\n]*\n?",
    ]
    out = reply
    cut_at = None
    for pat in patterns:
        m = re.search(pat, out, flags=re.I)
        if m:
            cut_at = m.start() if cut_at is None else min(cut_at, m.start())
    if cut_at is not None:
        # Drop lead-in and everything after (title/description freeform body)
        out = out[:cut_at].rstrip()
    return out


async def _maybe_attach_annotation_offer(trace, user_message, history_text, pending, entry):
    """After a full answer, attach one ready-drafted annotation preview (not a permission ask)."""
    if not pending:
        return trace
    if trace.get("annotation_proposal") or trace.get("memory_proposal"):
        return trace
    reply = (trace.get("reply") or "").strip()
    if not reply:
        return trace
    # Always strip model-invented freeform annotation blocks first.
    reply = _strip_draft_annotation_leadin(reply)
    trace["reply"] = reply
    cat = trace.get("category")
    # Suppress after emergency redirect / hard out-of-scope — never draft on those turns.
    if (
        cat in getattr(G, "OUT_OF_SCOPE", set())
        or cat == "symptom"
        or bool(trace.get("symptom", {}).get("detected"))
        or bool(trace.get("symptom_escalation"))
    ):
        return trace
    try:
        proposal = await AAnn.extract_proposal(
            user_message,
            mode="offer",
            dates=pending.get("dates") or [],
            today=pending.get("today") or "",
            window_start=pending.get("window_start"),
            window_end=pending.get("window_end"),
            history_text=history_text,
        )
    except Exception as e:
        trace["stages"].append(f"annotation_offer_error:{type(e).__name__}")
        return trace

    # Clarify: keep proposal payload for UI, append ONE clarifier line (not freeform draft).
    if proposal.get("status") == "clarify" or proposal.get("needs_clarification"):
        clar = AAnn.proposal_reply_text(proposal)
        if clar and clar not in reply:
            trace["reply"] = reply.rstrip() + "\n\n" + clar
        else:
            trace["reply"] = reply
        trace["annotation_proposal"] = proposal
        # Cap follow-ups when we're asking for dates — avoid competing leading questions.
        fus = trace.get("follow_ups") or []
        if isinstance(fus, list) and len(fus) > 2:
            trace["follow_ups"] = fus[:2]
        trace["stages"].append("annotation_offer_clarify_attached")
        return trace

    if proposal.get("status") not in ("proposal",):
        trace["stages"].append(f"annotation_offer_skipped:{proposal.get('status')}")
        return trace

    offer_line = AAnn.proposal_reply_text(proposal)
    if offer_line and offer_line not in reply:
        trace["reply"] = reply.rstrip() + "\n\n" + offer_line
    else:
        trace["reply"] = reply
    trace["annotation_proposal"] = proposal
    trace["stages"].append("annotation_proactive_offer_attached")
    return trace


async def handle_turn(
    user_message,
    history=None,
    path="pure",
    start_idx=None,
    end_idx=None,
    bundle=None,
    index=None,
):
    """Run one Advisor turn through the full pipeline. Returns a trace dict."""
    history = history or []
    trace = {"user_message": user_message, "stages": []}
    pending_annotation_offer = None

    # ---- Guardrail 1: symptom detection (before any model call) ------------
    is_sym, urgency, matches = G.detect_symptom(user_message)
    trace["symptom"] = {"detected": is_sym, "urgency": urgency, "matched": matches[:3]}
    if is_sym:
        trace["stages"].append("symptom_escalation")
        trace["category"] = "symptom"
        trace["retrieved"] = []
        trace["reply"] = G.escalation_reply(urgency)
        trace["follow_ups"] = []
        trace["grounding"] = {"verdict": "n/a — fixed escalation response"}
        return trace

    # ---- Guardrail 2: out-of-scope classification --------------------------
    history_text = build_history_text(history)
    category, source = await G.classify_scope(user_message, history_text)
    trace["category"] = category
    trace["category_source"] = source
    trace["stages"].append(f"classified:{category}")
    if category == "medical_advice":
        category = "actionable"
        trace["category"] = "actionable"
        trace["stages"].append("remapped_medical_advice_to_actionable")
    if category in G.OUT_OF_SCOPE:
        trace["stages"].append("out_of_scope_refusal")
        trace["retrieved"] = []
        refusal = (G.CATEGORY_REFUSALS.get(category) or "").strip()
        # Never emit the banned exercise-prescription refusal text.
        if (not refusal) or ("exercise prescription" in refusal.lower()):
            category = "actionable"
            trace["category"] = "actionable"
            trace["stages"].append("blocked_banned_prescription_refusal")
        else:
            trace["reply"] = refusal
            trace["follow_ups"] = []
            trace["grounding"] = {"verdict": "n/a — fixed refusal response"}
            return trace

    # ---- Patient context + parallel enrichment (latency pack) --------------
    # After safety classify: load bundle / memory / corpus together. One memory
    # load is reused for mutation path + draft retrieval (no double fetch).
    mem_doc = None
    mem_all = None
    index_local = index
    try:
        loaders = []
        need_bundle = bundle is None
        if need_bundle:
            loaders.append(C.load_bundle())
        else:

            async def _bundle_ready():
                return bundle

            loaders.append(_bundle_ready())
        loaders.append(M.load_for_turn())
        if index_local is None:
            loaders.append(R.load_corpus())

        loaded = await asyncio.gather(*loaders, return_exceptions=True)
        bi = 0
        b_res = loaded[bi]
        bi += 1
        if need_bundle:
            if isinstance(b_res, Exception):
                raise b_res
            bundle = b_res
        m_res = loaded[bi]
        bi += 1
        if not isinstance(m_res, Exception) and m_res is not None:
            mem_doc, mem_all = m_res
        elif isinstance(m_res, Exception):
            trace["stages"].append(f"memory_load_skip:{type(m_res).__name__}")
        if index_local is None:
            c_res = loaded[bi]
            if not isinstance(c_res, Exception):
                index_local = c_res
            else:
                trace["stages"].append(f"corpus_load_skip:{type(c_res).__name__}")
        trace["stages"].append("parallel_context_loaded")
    except Exception as e:
        if bundle is None:
            try:
                bundle = await C.load_bundle()
            except Exception as e2:
                trace["stages"].append(f"bundle_load_fail:{type(e2).__name__}")
                trace["reply"] = (
                    "I can't see your score data right now, so I'd rather not guess at anything. "
                    "Try reloading the dashboard, and I'll pick this up from your actual numbers."
                )
                trace["stages"].append("no_patient_data")
                trace["retrieved"] = []
                return trace
        trace["stages"].append(f"parallel_context_partial:{type(e).__name__}")

    window_key, entry = C.pick_window(bundle, start_idx, end_idx)
    if not entry:
        trace["reply"] = (
            "I can't see your score data right now, so I'd rather not guess at anything. "
            "Try reloading the dashboard, and I'll pick this up from your actual numbers."
        )
        trace["stages"].append("no_patient_data")
        trace["retrieved"] = []
        return trace

    # ---- Memory mutation proposal (preview gate; UI writes on confirm) ------
    # Profile/habits *inquiries* fall through to the normal draft path (synthesis).
    # Only real manage/correct/delete/pause intents early-return a proposal card.
    if category == "memory" or _is_profile_inquiry(user_message):
        try:
            if mem_doc is None:
                mem_doc, mem_all = await M.load_for_turn()
            items = [
                it
                for it in (mem_doc.get("items") or [])
                if isinstance(it, dict)
                and (it.get("status") or "active") in ("active", "dismissed")
            ]
            if _is_profile_inquiry(user_message) and not G.detect_memory_mutation_intent(
                user_message
            ):
                # Fall through — synthesized answer with memory as lens.
                category = "explain_science"
                trace["stages"].append("profile_inquiry_synthesize")
                trace["category"] = "explain_science"
            else:
                proposal = await _extract_memory_proposal(user_message, items, mem_doc)
                if proposal.get("defer_to_draft") or proposal.get("action") == "synthesize_profile":
                    category = "explain_science"
                    trace["stages"].append("memory_list_defer_synthesize")
                    trace["category"] = "explain_science"
                else:
                    trace["stages"].append(f"memory_proposal:{proposal.get('action')}")
                    trace["category"] = "memory"
                    trace["memory_proposal"] = proposal
                    trace["retrieved"] = []
                    trace["reply"] = _memory_proposal_reply(proposal)
                    trace["grounding"] = {
                        "verdict": "n/a — structured memory proposal (no freeform write)"
                    }
                    return trace
        except Exception as e:
            if category == "memory" and not _is_profile_inquiry(user_message):
                trace["stages"].append(f"memory_proposal_error:{type(e).__name__}")
                trace["reply"] = (
                    "I can update what I've learned about you, but something went wrong "
                    "loading your profile just now. Try the “What Alyf has learned” panel, "
                    "or ask again in a moment."
                )
                trace["retrieved"] = []
                return trace
            trace["stages"].append(f"memory_load_skip:{type(e).__name__}")

    # ---- Annotation proposal (structured tool path; UI writes on confirm) --
    # Never writes annotations here. Produces a structured draft the dashboard
    # previews (Accept / Edit / Cancel) and commits via the same saveAnnotations
    # path as the manual annotation UI.
    #
    # Annotation routing (createAnnotation write-gate still UI-only):
    # - Explicit "add a note" → early return with proposal only (no analytical answer).
    # - Implicit situational context (or classifier category "annotation") on an
    #   in-scope analytical turn → answer first, then attach a ready draft via
    #   pending_annotation_offer. Situational context never enters the durable profile.
    #
    # IMPORTANT: scope classifier often labels "how did X affect my score" as
    # explain_science even when the user volunteered life context. Arm the
    # proactive draft from detect_annotation_intent, not only category==annotation.
    _ann_dates = list(bundle.get("dates") or [])
    _ann_today = _ann_dates[-1] if _ann_dates else None
    try:
        from datetime import date as _date

        _wall = _date.today().isoformat()
        if _wall in _ann_dates or (
            _ann_dates and _ann_dates[0] <= _wall <= _ann_dates[-1]
        ):
            _ann_today = _wall
    except Exception:
        pass
    _ann_intent = AAnn.detect_annotation_intent(user_message)
    if category == "annotation" or _ann_intent is not None:
        _mode = (
            "offer"
            if (_ann_intent == "implicit" or (category == "annotation" and _ann_intent != "explicit"))
            else "explicit"
        )
        # Classifier said annotation but no pattern: treat as explicit only if
        # the message is clearly a create-command; otherwise prefer offer+answer.
        if category == "annotation" and _ann_intent is None:
            _mode = "explicit"
        if _mode == "explicit" and category == "annotation":
            proposal = await AAnn.extract_proposal(
                user_message,
                mode="explicit",
                dates=_ann_dates,
                today=_ann_today or (_ann_dates[-1] if _ann_dates else ""),
                window_start=(entry or {}).get("date_start"),
                window_end=(entry or {}).get("date_end"),
                history_text=history_text,
            )
            trace["stages"].append(f"annotation_{proposal.get('status')}_explicit")
            trace["category"] = "annotation"
            trace["annotation_proposal"] = proposal
            trace["retrieved"] = []
            # Explicit create: no exploratory follow-ups on a pure proposal turn.
            trace["follow_ups"] = []
            trace["reply"] = AAnn.proposal_reply_text(proposal)
            trace["grounding"] = {
                "verdict": "n/a — structured annotation proposal (no freeform write)"
            }
            return trace
        # Implicit / situational → keep analytical category, arm post-answer draft.
        pending_annotation_offer = {
            "dates": _ann_dates,
            "today": _ann_today or (_ann_dates[-1] if _ann_dates else ""),
            "window_start": (entry or {}).get("date_start"),
            "window_end": (entry or {}).get("date_end"),
            "mode": "offer",
        }
        if category == "annotation":
            category = "explain_science"
            trace["category"] = "explain_science"
        trace["stages"].append(
            f"annotation_implicit_defer_offer:{_ann_intent or 'classifier'}"
        )
        trace["annotation_offer_pending"] = True
    trend = C.trend_summary(bundle, path, entry["start_idx"], entry["end_idx"])
    extra_blocks = []
    try:
        extra_blocks = build_engine_lookups(
            bundle, entry, path, user_message, history
        )
        if extra_blocks:
            trace["stages"].append(f"engine_lookups:{len(extra_blocks)}")
            compact = _compact_lookup_blocks(extra_blocks)
            if compact is not None:
                extra_blocks = compact
                trace["stages"].append(f"engine_lookups_compact:{len(extra_blocks)}")
    except Exception as e:
        trace["stages"].append(f"engine_lookups_skip:{type(e).__name__}")
        extra_blocks = []
    patient_text = C.build_patient_context(
        bundle,
        window_key,
        entry,
        path=path,
        trend=trend,
        extra_compute_blocks=extra_blocks or None,
    )
    trace["window_key"] = window_key
    trace["path"] = path

    # ---- Optional research enrichment (not a hard gate) --------------------
    passages = []
    concepts = []
    passages_text = ""
    try:
        if index_local is None:
            index_local = await R.load_corpus()
        query = user_message
        if history:
            prev_user = [t for t in history if t.get("role") == "user"]
            if prev_user:
                query = prev_user[-1]["content"] + " " + user_message
        if category == "actionable":
            query = (query + " " + suggestion_query_terms(entry, path)).strip()
        passages, concepts = R.retrieve(
            index_local, query, k=6, axis_hint=axis_hint_from(query)
        )
        passages_text = R.format_passages(passages)
        trace["stages"].append(f"retrieved:{len(passages)}")
    except Exception as e:
        # Corpus optional under the new contract — data answers still proceed.
        trace["stages"].append(f"retrieval_skipped:{type(e).__name__}")
        passages = []
        passages_text = ""

    trace["retrieved"] = [
        {
            "id": p["id"],
            "doc": p["doc"],
            "section": p["section"],
            "score": p["score"],
            "provisional": p["provisional"],
        }
        for p in passages
    ]
    trace["concepts"] = concepts

    # NOTE: under the analyst contract we do NOT suppress when retrieval is empty.
    # Patient-data questions (worst axis, score change, pure vs blend) must answer
    # from the bundle even with zero research passages.

    # ---- Personalization memory (reuse parallel load — never dump full profile) -
    memory_text = ""
    memory_ids = []
    try:
        if mem_doc is None:
            mem_doc, mem_all = await M.load_for_turn()
        if mem_doc.get("learning_enabled", True):
            mem_hits = M.retrieve(
                mem_doc.get("items") or [],
                query=user_message,
                path=path,
                k=8,
            )
            memory_text = M.format_memory_for_prompt(mem_hits)
            memory_ids = [it.get("id") for it in mem_hits if it.get("id")]
            trace["stages"].append(f"memory_retrieved:{len(mem_hits)}")
        else:
            trace["stages"].append("memory_paused")
    except Exception as e:
        trace["stages"].append(f"memory_skipped:{type(e).__name__}")
    trace["memory_ids"] = memory_ids

    # ---- Draft under the behavioral contract ------------------------------
    draft_pack = await draft_reply(
        user_message, history, patient_text, passages_text, category, memory_text
    )
    if isinstance(draft_pack, dict):
        draft = (draft_pack.get("reply") or "").strip()
        follow_ups = draft_pack.get("follow_ups") or []
    else:
        draft = (draft_pack or "").strip()
        follow_ups = []
    if not isinstance(follow_ups, list):
        follow_ups = []
    trace["draft"] = draft
    trace["follow_ups"] = follow_ups
    trace["stages"].append("drafted")

    # ---- Guardrail 3: load-bearing audit (with one repair attempt) ---------
    audit = await G.check_grounding(draft, passages_text, patient_text)
    trace["grounding"] = audit
    if audit.get("verdict") == "pass":
        trace["stages"].append("grounding_passed")
        trace["reply"] = draft
        return await _maybe_attach_annotation_offer(
            trace, user_message, history_text, pending_annotation_offer, entry
        )

    trace["stages"].append("grounding_failed")
    revised = await G.revise_for_grounding(draft, audit, passages_text, patient_text)
    if not revised:
        # Prefer the original draft over a hard suppress when repair fails —
        # load-bearing lines already live in the system prompt; a total suppress
        # was the old corpus-only failure mode.
        trace["stages"].append("repair_failed_emit_draft")
        trace["reply"] = draft
        return await _maybe_attach_annotation_offer(
            trace, user_message, history_text, pending_annotation_offer, entry
        )

    trace["revised_draft"] = revised
    reaudit = await G.check_grounding(revised, passages_text, patient_text)
    trace["grounding_reaudit"] = reaudit
    if reaudit.get("verdict") == "pass":
        trace["stages"].append("repaired_and_passed")
        trace["reply"] = revised
    else:
        # Still emit the revised draft rather than the old suppression dead-end,
        # unless clinical overreach remains hard-true on both audits.
        overreach = bool(audit.get("clinical_overreach")) and bool(
            reaudit.get("clinical_overreach")
        )
        invented = bool(audit.get("invented_patient_facts")) and bool(
            reaudit.get("invented_patient_facts")
        )
        if overreach or invented:
            trace["stages"].append("suppressed_hard_violation")
            trace["reply"] = G.SUPPRESSION_REPLY
            trace["follow_ups"] = []
            return trace
        else:
            trace["stages"].append("emit_revised_despite_soft_fail")
            trace["reply"] = revised
    return await _maybe_attach_annotation_offer(
        trace, user_message, history_text, pending_annotation_offer, entry
    )


async def main():
    """Dashboard entrypoint: payload via argv (preferred, single run) or staged artifact."""
    payload = None
    # Prefer argv so the UI can skip a separate stage_advisor_turn program run.
    if len(sys.argv) > 1 and sys.argv[1]:
        try:
            payload = json.loads(sys.argv[1])
        except Exception:
            payload = None
    if payload is None:
        try:
            raw = await aio.get_artifact(TURN_ARTIFACT)
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            payload = None
    if payload is None:
        executor.print("no turn payload found")
        return
    # Mirror argv payload into the pending artifact for debugging / legacy readers.
    try:
        await aio.store_artifact(
            identifier=TURN_ARTIFACT,
            title="Alyf Advisor — pending turn payload",
            artifact_type="text",
            data=json.dumps(payload) if not isinstance(payload, str) else payload,
            metadata={"text": {"content_type": "application/json"}},
            hidden=True,
        )
    except Exception:
        pass

    trace = await handle_turn(
        payload.get("message", ""),
        history=payload.get("history") or [],
        path=payload.get("path", "pure"),
        start_idx=payload.get("start_idx"),
        end_idx=payload.get("end_idx"),
    )
    # Exploratory follow-ups only on normal in-scope answers.
    fus = trace.get("follow_ups") or []
    if not isinstance(fus, list):
        fus = []
    if (
        bool(trace.get("symptom", {}).get("detected"))
        or (trace.get("category") in getattr(G, "OUT_OF_SCOPE", set()))
        or (trace.get("category") == "symptom")
    ):
        fus = []
    clean_fus = []
    seen_fu = set()
    has_ann = bool(
        isinstance(trace.get("annotation_proposal"), dict)
        and (trace.get("annotation_proposal") or {}).get("status") == "proposal"
        and not (trace.get("annotation_proposal") or {}).get("needs_clarification")
    )
    for q in fus:
        if not isinstance(q, str):
            continue
        s = " ".join(q.strip().split())
        if not s or len(s) > 120:
            continue
        k = s.lower()
        if k in seen_fu:
            continue
        # Dedupe vs annotation draft: no "add to timeline / annotation" chips.
        if has_ann and (
            "annotation" in k
            or "timeline" in k
            or "log this" in k
            or "note this" in k
            or ("add" in k and "note" in k)
        ):
            continue
        seen_fu.add(k)
        clean_fus.append(s)
        limit = 2 if has_ann else 4
        if len(clean_fus) >= limit:
            break

    reply_doc = {
        "turn_id": payload.get("turn_id"),
        "reply": trace.get("reply", ""),
        "category": trace.get("category"),
        "symptom_escalation": bool(trace.get("symptom", {}).get("detected")),
        "grounding_verdict": (trace.get("grounding") or {}).get("verdict"),
        "sources_used": trace.get("retrieved", []),
        "stages": trace.get("stages", []),
        "follow_ups": clean_fus,
        "internal_review_only": True,
        # Structured draft for conversational annotation creation. The UI owns
        # the Accept/Edit/Cancel gate and writes via saveAnnotations — never here.
        "annotation_proposal": trace.get("annotation_proposal"),
        # Personalization memory proposal (preview gate) + ids injected this turn.
        "memory_proposal": trace.get("memory_proposal"),
        "memory_ids": trace.get("memory_ids") or [],
    }
    await aio.store_artifact(
        identifier=REPLY_ARTIFACT,
        title="Alyf Advisor — latest reply",
        artifact_type="text",
        data=json.dumps(reply_doc),
        metadata={"text": {"content_type": "application/json"}},
        hidden=True,
    )
    executor.print(f"category={reply_doc['category']} stages={reply_doc['stages']}")
    executor.print(reply_doc["reply"][:800])