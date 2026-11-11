"""Alyf Advisor — guardrail layer.

Independent of the system prompt so a jailbreak does not remove protections:

1. `detect_symptom`  — deterministic symptom detection on user input; forces
                       fixed escalation text (no model prose).
2. `classify_scope`  — out-of-scope classifier (patterns + LLM) for diagnosis /
                       disease-risk / medication / pure medical-advice asks.
3. `check_grounding` — audits the draft against the LOAD-BEARING lines:
                       no invented patient facts, no diagnosis/prognosis,
                       no self-invented numeric targets, no symptom assessment.
                       General-knowledge interpretation is ALLOWED.
"""

import re

from executor import aio, Chat, UserPrompt

# ---------------------------------------------------------------- 1. symptoms

URGENT_SYMPTOMS = [
    r"chest (pain|pressure|tightness|discomfort)",
    r"can'?t breathe",
    r"cannot breathe",
    r"trouble breathing",
    r"short(ness)? of breath",
    r"breathless",
    r"faint(ed|ing)?",
    r"pass(ed)? out",
    r"black(ed)? out",
    r"collaps(e|ed)",
    r"palpitations?",
    r"heart (is )?(racing|pounding|fluttering|skipping)",
    r"irregular heart ?beat",
    r"numb(ness)? in my (arm|face)",
    r"slurred speech",
    r"cold sweat",
    r"vomit(ing)?",
]

# True bodily / clinical-feeling symptoms → routine care redirect.
# Intentionally EXCLUDES mild energy/wellness language ("feel off", tired,
# sluggish, wiped out, exhausted). Those are answered with profile + data
# associations (see BEHAVIORAL_CONTRACT), not hard-refused.
OTHER_SYMPTOMS = [
    r"dizz(y|iness)",
    r"light ?headed",
    r"my (chest|heart|legs?|arms?|back|knee|hip|shoulder|head) (hurts?|aches?)",
    r"\bpain\b",
    r"ach(e|ing|y)\b",
    r"sore(ness)?\b",
    r"swollen",
    r"swelling",
    r"edema",
    r"nause(a|ous)",
    r"headache",
    r"migraine",
    r"cramp(s|ing)?",
    r"feel(ing)? (unwell|awful|terrible|sick)",
    r"can'?t sleep at all",
    r"night sweats",
]

# Mild subjective energy / "how I feel" language that is IN-SCOPE for a
# data-grounded, non-diagnostic answer (profile + current trends). Do not
# hard-escalate these — the model reasons through behavioral profile + BAP.
WELLNESS_FEELING_OK = [
    r"feel(ing)? off",
    r"feel(ing)? sluggish",
    r"feel(ing)? (low|run[- ]?down|drained)",
    r"\bsluggish\b",
    r"\btired\b",
    r"\bfatigued\b",
    r"\bexhaust(ed|ion)\b",
    r"wiped out",
    r"why do i feel",
    r"why am i (so )?(tired|exhausted|sluggish)",
    r"low energy",
    r"no energy",
]


def detect_symptom(text):
    """Return (is_symptom, urgency, matches).

    Urgent bodily red flags always escalate.
    Routine bodily symptoms escalate.
    Mild energy/wellness phrasing alone does NOT escalate — the Advisor should
    answer with behavioral profile + data associations (non-diagnostic).
    If both a wellness phrase and a true bodily symptom appear, bodily wins.
    """
    low = f" {text.lower()} "
    urgent = [p for p in URGENT_SYMPTOMS if re.search(p, low)]
    other = [p for p in OTHER_SYMPTOMS if re.search(p, low)]
    if urgent:
        return True, "urgent", urgent
    if other:
        return True, "routine", other
    return False, None, []


def is_wellness_feeling_question(text: str) -> bool:
    """True when the user is asking a mild energy/feel-off style question."""
    if not text:
        return False
    low = f" {text.lower()} "
    return any(re.search(p, low) for p in WELLNESS_FEELING_OK)


ESCALATION_URGENT = (
    "I need to stop and flag that right away. What you are describing is something "
    "I am not able to assess or reassure you about. Because it could be urgent, "
    "please contact your care team now, or seek emergency care by calling your local "
    "emergency number if it is severe, new, or getting worse.\n\n"
    "I am only here to help you understand your activity score. Anything about how "
    "you are physically feeling belongs with a clinician who can examine you. "
    "I am glad to continue the score conversation whenever you are ready."
)

ESCALATION_ROUTINE = (
    "Thank you for sharing that. I cannot assess or interpret how you are physically "
    "feeling, even in passing. Please bring that to your care team, and if it is "
    "severe, sudden, or getting worse, treat it as urgent and seek care right away.\n\n"
    "What I can do is help you understand your activity score and the patterns in "
    "your data. We can return to that whenever you are ready."
)


def escalation_reply(urgency):
    return ESCALATION_URGENT if urgency == "urgent" else ESCALATION_ROUTINE


# ------------------------------------------------------------------- 2. scope

DISEASE_RISK_PATTERNS = [
    r"odds of (a )?(heart attack|stroke|dying|death)",
    r"(chance|risk|likelihood|probability) of (a |an )?(heart attack|stroke|cardiac|"
    r"event|dying|death|diabetes|cancer|disease)",
    r"am i (going to|gonna) (have|get) (a )?(heart attack|stroke)",
    r"how long (do i|have i got) to live",
    r"predict my (risk|health|disease)",
    r"will i (develop|get) ",
    r"life expectancy",
]

MEDICATION_PATTERNS = [
    r"\b(beta ?blocker|statin|metoprolol|lisinopril|warfarin|eliquis|apixaban|"
    r"aspirin|insulin|metformin|amiodarone|diuretic)\b",
    r"(should|can) i (stop|start|skip|change|adjust|take|halve|double) (my |the )?"
    r"(med|meds|medication|dose|pill|tablet)",
    r"my (doctor|cardiologist) (put me on|prescribed)",
]

DIAGNOSIS_PATTERNS = [
    r"do i have ",
    r"is (this|that|it) (a sign of|because of|caused by) ",
    r"what'?s wrong with (me|my)",
    r"is my (heart|condition|disease|health) (getting|going) (worse|better)",
    r"is my (heart|health) (ok|okay|fine|failing|damaged)",
    r"(getting|going) worse\b.*\b(heart|health|condition)",
    r"does (this|that|my).*(mean|indicate).*(heart disease|prognosis|dying|disease|damage)",
    r"am i (sick|ill|dying)",
]

# Narrowed: only clearly non-BAP clinical topics. Sleep/activity lifestyle
# language is in-scope for the analyst Advisor.
NON_BAP_HEALTH_PATTERNS = [
    r"\b(keto|paleo|vegan diet|intermittent fasting)\b",
    r"\b(supplement|vitamin|creatine|protein powder)\b",
    r"\b(cholesterol|a1c|blood sugar|thyroid|bmi)\b",
    r"should i (get|have) (a |an )?(scan|test|mri|ekg|ecg|blood test)",
    r"is .* (good|bad) for my (blood pressure|cholesterol)",
]

CATEGORY_REFUSALS = {
    "disease_risk": (
        "That is not something I can answer. I do not estimate disease risk or predict "
        "outcomes, and this score is not a risk model. Your Balanced Activity Profile "
        "describes your recent activity patterns; it says nothing about your odds of "
        "any medical event.\n\n"
        "Questions about cardiac or disease risk belong with your care team. If it is "
        "useful, I am glad to walk through what your score and its axes actually show."
    ),
    "medication": (
        "I cannot speak to medications: not what they do, and not whether to take, "
        "change, or stop anything. That is strictly for your prescriber.\n\n"
        "If you would like, I can walk through what your score is showing so you can "
        "take that conversation to your care team."
    ),
    "diagnosis": (
        "I am not able to answer that. Interpreting what your data means for a medical "
        "condition, organ health, or prognosis is clinical judgment, and it belongs "
        "with your care team.\n\n"
        "What I can do is walk through the activity-score side: which axes are strong "
        "or soft, how your metrics moved in this window, and what the dashboard "
        "suggestions say. Would you like me to start there?"
    ),
    "non_bap_health": (
        "That is outside what I cover here. I am focused on your Balanced Activity "
        "Profile: the activity-quality score and the wearable patterns behind it.\n\n"
        "Your care team is the right place for that question. I am glad to continue "
        "on your score, axes, or the dashboard improvement suggestions."
    ),
    # Deprecated — specific HR/minute/training guidance is answered as coaching.
    # Empty so banned "exercise prescription" phrasing can never surface.
    "medical_advice": "",
}


# Conversational personalization-memory corrections (preview gate in UI).
# True manage/correct intents only — NOT profile/habits inquiry questions.
_MEMORY_MUTATION_RE = re.compile(
    r"\b("
    r"that'?s not right|that is not right|don'?t usually|do not usually|"
    r"remove that (from )?memory|delete that memory|forget that|"
    r"stop (learning|remembering)|pause learning|resume learning|"
    r"clear (my )?memory|wipe (my )?profile|"
    r"i don'?t travel|i do not travel|wrong about me|"
    r"update my (profile|memory|preferences)|correct (your|the) memory|"
    r"delete (that|this) (note|item|trait)|remove (that|this) (note|item|trait)|"
    r"don'?t use that|do not use that"
    r")\b",
    re.I,
)

# Profile *inquiry* — must NOT route to memory-management proposal cards.
_PROFILE_INQUIRY_RE = re.compile(
    r"\b("
    r"what kind of person|what sort of person|who am i|what am i like|"
    r"what are my habits|my habits|tell me about myself|describe me|"
    r"what (have )?you learned|what do you know about me|"
    r"what (does )?alyf know|show my profile|my patterns|"
    r"based on my data"
    r")\b",
    re.I,
)


def detect_memory_mutation_intent(text: str) -> str | None:
    """Return 'mutate' if the patient is correcting/deleting/pausing memory.

    Profile/habits questions are NOT mutations — they get a synthesized answer.
    """
    if not text:
        return None
    # Inquiry wins over loose "what have you learned" if no manage verbs.
    if _PROFILE_INQUIRY_RE.search(text) and not _MEMORY_MUTATION_RE.search(text):
        return None
    if _MEMORY_MUTATION_RE.search(text):
        return "mutate"
    return None


def is_profile_inquiry(text: str) -> bool:
    if not text:
        return False
    if detect_memory_mutation_intent(text):
        return False
    return bool(_PROFILE_INQUIRY_RE.search(text))


def is_specific_guidance_question(text: str) -> bool:
    """True for in-scope coaching asks: how much / optimal / target for HR, minutes,
    zones, volume, steps, sleep, rest days, RHR/HRV aims, etc.

    These must NEVER route to medical_advice refusal. Specificity is coaching.
    """
    low = (text or "").lower().strip()
    if not low:
        return False
    want = bool(
        re.search(
            r"\b("
            r"optimal|optimum|target|aim for|should i (get|have|do|aim|spend)|"
            r"how (much|many|long|often)|what(?:'s| is) a good|"
            r"how much .* (should|per week|a week|weekly)|"
            r"weekly (target|volume|goal)|per week|a week|each week"
            r")\b",
            low,
        )
    )
    if not want:
        return False
    domain = bool(
        re.search(
            r"\b("
            r"heart[- ]?rate|\bhr\b|bpm|zone\s*[0-9]|zones?|"
            r"160\+|above 160|minutes? (at|above|in)|vigorous|moderate|"
            r"intensity|exertion|training (volume|load|plan)?|workout|"
            r"steps?|sleep|bedtime|rest days?|recovery days?|"
            r"hrv|resting hr|resting heart|active minutes|cardio"
            r")\b",
            low,
        )
    )
    return domain


def _deterministic_scope(text):
    low = f" {text.lower()} "
    for pat in DISEASE_RISK_PATTERNS:
        if re.search(pat, low):
            return "disease_risk", pat
    for pat in MEDICATION_PATTERNS:
        if re.search(pat, low):
            return "medication", pat
    for pat in DIAGNOSIS_PATTERNS:
        if re.search(pat, low):
            return "diagnosis", pat
    # Mild energy / feel-off questions are IN-SCOPE score personalization —
    # never route to diagnosis/refusal. Must beat the LLM classifier.
    if is_wellness_feeling_question(text):
        return "explain_science", "wellness_feeling_profile"
    # Memory profile corrections before annotation (so "that's not right" isn't
    # treated as a chart annotation).
    if detect_memory_mutation_intent(text):
        return "memory", "memory_mutation"
    # Explicit annotation asks win before non-BAP health patterns.
    try:
        import advisor_annotations as AA

        intent = AA.detect_annotation_intent(text)
        if intent == "explicit":
            return "annotation", f"annotation_explicit"
        if intent == "implicit":
            return "annotation", f"annotation_implicit"
    except Exception:
        pass
    for pat in NON_BAP_HEALTH_PATTERNS:
        if re.search(pat, low):
            return "non_bap_health", pat
    # Specific coaching targets (HR zones, weekly minutes, steps, sleep, rest days)
    # are IN SCOPE — never medical_advice. Beat the LLM classifier.
    if is_specific_guidance_question(text):
        return "actionable", "specific_guidance_coaching"
    # Common in-scope score questions — skip classifier LLM (latency).
    if re.search(
        r"\b(what('?s| is) my (bap|score|overall)|worst|weakest|strongest|best axis|"
        r"why did my score|score change|how did .+ affect|"
        r"objective vs|pure.?pop|blend|trend score|compare .{0,40}score|"
        r"what does (recovery|intensity|consistency|spread|variety) mean|"
        r"explain (my )?(score|bap|axis)|how am i doing)\b",
        low,
    ):
        return "explain_science", "score_understanding_pattern"
    if re.search(
        r"\b(how (can|do) i improve|what should i (do|change)|how to improve|"
        r"suggestions? for me|what would help my)\b",
        low,
    ):
        return "actionable", "improve_pattern"
    if re.search(r"\b(hi|hello|hey|thanks|thank you|good (morning|afternoon))\b", low) and len(
        text.strip()
    ) < 48:
        return "smalltalk", "smalltalk_pattern"
    return None, None


SCOPE_CLASSIFIER_PROMPT = """You are the routing classifier for the Alyf Advisor, a \
patient-facing analyst for the Balanced Activity Profile (BAP) activity score. \
Classify the patient's latest message into exactly one category.

Categories:
- "explain_science": asking what a variable/axis/score is, why it matters, what a \
trend means, why their score changed, which axis is worst/strongest and why, how \
Objective vs personal-blend scores compare, how their numbers look, follow-ups about \
those things. THIS IS THE DEFAULT for score-understanding questions.
- "actionable": asking what they should DO, what to change, how to improve their \
score or a specific axis.
- "annotation": asking to add/create/save a note, tag, marker, or annotation on their \
score timeline, OR sharing life context that clearly explains a score dip and could \
be recorded as an annotation (travel, finals, illness, poor sleep stretch). Prefer \
annotation when they want something written onto the chart/notes.
- "memory": correcting, deleting, pausing, or clearing personalization \
("that's not right", "I don't usually travel", "stop learning", "remove that note"). \
NOT "what kind of person am I" / "what are my habits" / "what have you learned" as \
genuine questions — those are explain_science (synthesize a portrait answer).
- "diagnosis": asking whether they have a disease/condition, whether an organ is \
failing, prognosis, or "am I sick/dying" — clinical judgment about their HEALTH, not \
about their activity score.
- "disease_risk": asking for risk, odds, probability, or prediction of a disease or \
medical event, or life expectancy.
- "medication": anything about medicines, doses, prescriptions, or treatment changes.
- "non_bap_health": clearly outside activity-score scope (specific diets like keto, \
supplements, lab values like A1c/cholesterol as clinical topics, ordering medical tests).
- "medical_advice": DEPRECATED — do NOT use. Specific exercise/HR/minute/training \
targets are "actionable" (coaching). Only diagnosis/disease_risk/medication/true \
non-BAP clinical topics are out of scope.
- "smalltalk": greetings, thanks, or meta questions about what the Advisor can do.

Important boundary:
- "Which axis is worst and why?" / "Why did my BAP fall?" / "Why do my two scores \
differ?" → explain_science (ALWAYS).
- "Why do I feel off / sluggish / tired / low energy lately?" → explain_science \
(ALWAYS). This is a core personalization path: connect behavioral profile + current \
Recovery/sleep/RHR/activity data as associations. It is NOT diagnosis and NOT \
medical_advice. Only true disease/organ/prognosis asks are diagnosis.
- "What kind of person am I?" / "What are my habits?" / "What have you learned about \
me?" as genuine questions → explain_science (synthesize). Only explicit correct/delete/ \
pause/clear intents are "memory".
- "How did that weekend / those days affect my score?" → explain_science (ALWAYS).
- "Add an annotation from June 3–7, I was traveling" → annotation.
- "I had finals this week, that's why I haven't slept" → annotation (offer path).
- "What does this mean for my heart disease / am I dying?" → diagnosis.
- "What should I do to improve?" → actionable.
- "Optimal time per week above 160 HR?" / zone minutes / weekly training volume / \
steps / sleep duration / rest days / RHR-HRV aims → actionable (ALWAYS). Specific \
derived coaching — NOT medical_advice. Answer with data + gates; never refuse.
- Sleep consistency, breaking up sitting, activity patterns as lifestyle observations \
about the SCORE are explain_science or actionable — NOT non_bap_health.
When a message mixes in-scope and out-of-scope, choose the out-of-scope category \
— EXCEPT mild feel-off/energy language alone stays explain_science, AND specific \
activity/HR/sleep/recovery target questions stay actionable."""


async def classify_scope(user_message, history_text=""):
    """Return (category, source). Deterministic patterns win over the classifier."""
    cat, pat = _deterministic_scope(user_message)
    if cat:
        return cat, f"pattern:{pat}"

    schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "explain_science",
                    "actionable",
                    "annotation",
                    "memory",
                    "diagnosis",
                    "disease_risk",
                    "medication",
                    "non_bap_health",
                    "medical_advice",
                    "smalltalk",
                ],
            },
            "reason": {"type": "string"},
        },
        "required": ["category", "reason"],
    }
    prompt = (
        (
            f"Recent conversation (for context only):\n{history_text}\n\n"
            if history_text
            else ""
        )
        + f"Patient's latest message:\n{user_message}"
    )
    try:
        out = await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=SCOPE_CLASSIFIER_PROMPT,
                user_prompt=UserPrompt.from_text(prompt),
            ),
            output_schema=schema,
        )
        cat = out["category"]
        # Never hard-refuse specific coaching as medical_advice.
        if cat == "medical_advice":
            return (
                "actionable",
                f"classifier_remapped_medical_advice:{out.get('reason', '')[:100]}",
            )
        return cat, f"classifier:{out.get('reason', '')[:120]}"
    except Exception as e:
        # Fail open toward answering score questions; hard safety still sits in
        # symptom detection + grounding overreach flags.
        return "explain_science", f"classifier_error:{type(e).__name__}"


OUT_OF_SCOPE = {
    "diagnosis",
    "disease_risk",
    "medication",
    "non_bap_health",
    # medical_advice removed: HR-zone / minute / training targets are in-scope coaching
}


# --------------------------------------------------------------- 3. grounding

GROUNDING_PROMPT = """You are the safety auditor for the Alyf Advisor. The Advisor is \
a data-grounded analyst: it MAY use general physiological knowledge to interpret, and \
it MUST ground every patient-specific fact in the provided patient data.

Audit the DRAFT against the load-bearing lines only:

1. invented_patient_facts (list of quotes): patient-specific numbers, trends, deltas, \
rankings, flags, or "your X is Y" claims that are NOT present in and NOT simple \
arithmetic over the PATIENT DATA section. General statements without a patient-specific \
number are fine. Axis rankings and comparisons derived from supplied axis scores are \
fine.

2. clinical_overreach (boolean): true only if the draft diagnoses a disease/condition, \
gives prognosis, estimates disease risk/odds, medically assesses or reassures about an \
acute/bodily physical symptom, discusses medications/treatment changes, or claims the \
patient's organ/health is failing/fine as a medical judgment. \
FALSE (allowed) examples: explaining that higher resting heart rate often associates \
with lower recovery capacity; multi-signal fatigue stories (training load + RHR up + \
HRV down + short nights) framed as associations / "accumulated fatigue" / \
"under-recovery" WITHOUT condition labels; saying Consistency is soft because \
day-to-day rhythm varies; comparing Objective vs personal-blend scores; specific \
derived lifestyle guidance from the patient's own best nights or engine outputs \
("aim back toward your ~7.5–8 hour nights"); relaying gated intensity targets plainly; \
answering feel-off via profile + full data as associations without naming a condition \
or syndrome. TRUE examples: "overtraining syndrome", "you have anemia", "this is \
depression", "possible arrhythmia", diagnosing from feel language.

3. invented_numeric_targets (boolean): true ONLY if the draft asserts a specific \
quantitative training/sleep/HR/step/duration target or recoverable-point claim that \
cannot be derived from ANY of: (a) numbers or simple arithmetic over PATIENT DATA / \
FEEL-SIGNAL / getMetric / scoreWindow blocks, (b) gated suggestion-engine output, or \
(c) clearly marked general research guidelines (not personalized). \
FALSE (allowed): patient-own-pattern targets ("your best nights cluster around 7.5–8h"), \
engine suggestion minutes/points attributed to the dashboard, weekly vigorous/moderate \
minute targets derived from intensity data + gates, HR-zone / above-threshold coaching \
with an executable protocol (e.g. "15–30 min/week as 20–60s efforts, one session") when \
framed as gate-bounded / data-informed / labeled general high-intensity guidance \
(including proxies when exact zone series is missing), population formulas clearly \
labeled as estimates (e.g. 220−age), general guideline ranges labeled as general. \
Directional language is fine. \
Do NOT fail intensity guidance that stays within gated engine suggestion text merely \
for lacking an extra care-team hedge. Do NOT fail a draft for answering a specific \
guidance question instead of refusing it. Do NOT fail a coherent near-max / zone \
protocol quantity solely because the export lacks HR-zone minutes — fail only if the \
number is free-standing with no patient standing, no engine text, and no labeled \
guideline/formula basis at all.

4. provisional_misrepresentation (boolean): true only if optional research passages \
mark something provisional and the draft states it as settled medical fact with no hedge.

5. verdict: "pass" if invented_patient_facts is empty AND clinical_overreach is false \
AND invented_numeric_targets is false AND provisional_misrepresentation is false; else \
"fail".

Do NOT fail a draft merely for using general knowledge, paraphrase, or interpretive \
language. Do NOT require research passages to be present."""

REVISION_PROMPT = """You are the repair step for the Alyf Advisor. A draft failed the \
load-bearing safety audit. Rewrite it so it keeps the useful analysis and removes only \
the violations.

Rules:
- Remove or correct every invented patient-specific fact. Replace with values from the \
PATIENT DATA section, or say the number isn't in this window.
- If clinical_overreach: remove diagnosis/prognosis/risk/symptom-assessment/medication \
language. You may still interpret activity-score patterns and point health questions to \
the care team.
- If invented_numeric_targets: remove only quantities with NO basis. KEEP patient-derived \
standing, engine/dashboard figures (attributed), labeled formulas (220−age etc.), and \
labeled general high-intensity protocol ranges (e.g. brief weekly near-max minutes with \
interval structure). Prefer tightening attribution over deleting the whole protocol.
- If provisional_misrepresentation: hedge as still-developing science.
- Keep the warm, plain-language analyst voice. Keep grounded analysis intact. Do not \
add a disclaimer wall."""


async def check_grounding(draft, passages_text, patient_text):
    """Audit a draft against load-bearing lines. Returns dict with verdict."""
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "invented_patient_facts": {
                "type": "array",
                "items": {"type": "string"},
            },
            # keep unsupported_claims alias for older trace readers
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
            },
            "clinical_overreach": {"type": "boolean"},
            "invented_numeric_targets": {"type": "boolean"},
            "provisional_misrepresentation": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "verdict",
            "invented_patient_facts",
            "clinical_overreach",
            "invented_numeric_targets",
            "provisional_misrepresentation",
            "notes",
        ],
    }
    prompt = (
        f"PATIENT DATA SUMMARY:\n{patient_text}\n\n"
        f"OPTIONAL RESEARCH PASSAGES:\n{passages_text or '(none)'}\n\n"
        f"DRAFT REPLY TO AUDIT:\n{draft}"
    )
    try:
        out = await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=GROUNDING_PROMPT,
                user_prompt=UserPrompt.from_text(prompt),
            ),
            output_schema=schema,
        )
        # Mirror invented facts into unsupported_claims for trace compatibility.
        if not out.get("unsupported_claims"):
            out["unsupported_claims"] = list(out.get("invented_patient_facts") or [])
        return out
    except Exception as e:
        # Fail open on auditor errors — system prompt still carries the contract.
        return {
            "verdict": "pass",
            "invented_patient_facts": [],
            "unsupported_claims": [],
            "clinical_overreach": False,
            "invented_numeric_targets": False,
            "provisional_misrepresentation": False,
            "notes": f"auditor_error_fail_open:{type(e).__name__}",
        }


async def revise_for_grounding(draft, audit, passages_text, patient_text):
    """One repair attempt: strip load-bearing violations, keep the analysis."""
    facts = audit.get("invented_patient_facts") or audit.get("unsupported_claims") or []
    findings = [
        "Invented patient-specific facts to remove/correct:\n"
        + ("\n".join(f"- {c}" for c in facts) or "- (none)"),
        f"clinical_overreach: {audit.get('clinical_overreach')}",
        f"invented_numeric_targets: {audit.get('invented_numeric_targets')}",
        f"provisional_misrepresentation: {audit.get('provisional_misrepresentation')}",
    ]
    prompt = (
        f"PATIENT DATA SUMMARY:\n{patient_text}\n\n"
        f"OPTIONAL RESEARCH PASSAGES:\n{passages_text or '(none)'}\n\n"
        f"AUDIT FINDINGS:\n" + "\n".join(findings) + "\n\n"
        f"DRAFT TO REPAIR:\n{draft}"
    )
    try:
        return await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=REVISION_PROMPT,
                user_prompt=UserPrompt.from_text(prompt),
            )
        )
    except Exception:
        return None


SUPPRESSION_REPLY = (
    "I need to stay inside what I can responsibly say. I should not diagnose, estimate "
    "medical risk, or invent numbers that are not in your score data.\n\n"
    "If you have a health concern, please take it to your care team. If you would like, "
    "I can walk through your actual BAP: which axes are strong or soft in this window, "
    "how Objective and Trend scores compare, or what the dashboard suggestions say."
)