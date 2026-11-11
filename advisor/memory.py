"""Alyf Advisor — continuous personalization memory (long-term profile).

Separate from chat transcripts. Pipeline: EXTRACT → CONSOLIDATE → DECAY → RETRIEVE.

PROFILE = who the person IS (durable traits/preferences/archetypes).
ANNOTATIONS = what happened during a period (situational, dated) — never here.

Primary ingestion: durable characterizations the ASSISTANT asserted in its own
answers, plus standing archetypes from the data pass. Situational patient
context (finals, travel, one bad night) is rejected by the durability filter.

Store: encrypted `alyf_advisor_memory`
View:  decrypted `alyf_advisor_memory_view` (for UI bind) — includes portrait.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from executor import aio, Chat, UserPrompt
import executor

import advisor_memory_crypto as crypto
import advisor_context as C

STORE_ID = "alyf_advisor_memory"
VIEW_ID = "alyf_advisor_memory_view"

SCHEMA_VERSION = 2
RETENTION_NOTE = (
    "Chat transcripts are deleted after 7 days. A personalization profile that "
    "captures durable activity, sleep, recovery patterns, and preferences is "
    "retained longer and can be viewed, edited, deleted, or paused at any time. "
    "Time-bound events such as travel or exams belong on the timeline as "
    "annotations, not in this profile. The profile is not a medical record."
)

OUTER_BOUND_MS = 180 * 24 * 60 * 60 * 1000
DATA_PATTERN_TTL_MS = 90 * 24 * 60 * 60 * 1000

ITEM_TYPES = ("preference", "stated", "data_pattern", "behavioral_profile")
ACTIVE_STATUSES = ("active",)

PORTRAIT_SECTIONS = (
    ("activity", "Activity"),
    ("sleep", "Sleep"),
    ("recovery", "Recovery"),
    ("rhythm", "Rhythm & Consistency"),
    ("communication", "How you like to talk with me"),
)

_SITUATIONAL_RE = re.compile(
    r"\b("
    r"finals?|exams?|midterms?|test\s+week|exam\s+week|"
    r"travel(?:ing|ling)?|trip|vacation|jet\s*lag|flight|"
    r"sick|illness|flu|covid|cold\b|fever|"
    r"last\s+night|yesterday|this\s+morning|tonight|"
    r"this\s+week|last\s+week|this\s+weekend|last\s+weekend|"
    r"wedding|funeral|moved|moving|relocation|"
    r"night\s+shift|on\s+call|deadline|crunch\s+time|"
    r"hangover|one[- ]off|for\s+now|temporarily|"
    r"didn'?t\s+sleep\s+(well\s+)?last|couldn'?t\s+sleep\s+last|"
    r"fighting\s+a\s+cold|been\s+ill"
    r")\b",
    re.I,
)

_FEELING_CLINICAL_RE = re.compile(
    r"\b("
    r"you\s+feel|feeling\s+(?:off|anxious|depressed|sad|stressed)|"
    r"anxious|depression|depressed|mental\s+health|"
    r"diagnos|prognos|syndrome|disorder|disease|"
    r"non[- ]?compliant|compliance"
    r")\b",
    re.I,
)


def is_situational_text(text: str) -> bool:
    """True if text describes a time-bound event / passing circumstance."""
    if not text:
        return True
    return bool(_SITUATIONAL_RE.search(text))


def is_durable_characterization(text: str, typ: str) -> bool:
    """Durability test: will this still be true of the person next month?"""
    if not text or len(text.strip()) < 8:
        return False
    if is_situational_text(text):
        return False
    if _FEELING_CLINICAL_RE.search(text) and typ != "preference":
        if typ == "preference" and re.search(
            r"\b(prefer|like|want|tone|detail|short|direct)\b", text, re.I
        ):
            return not bool(
                re.search(r"\b(diagnos|syndrome|disorder|disease|depress|anxi)\b", text, re.I)
            )
        return False
    if typ == "preference":
        return True
    durable_cues = re.search(
        r"\b("
        r"tend(?:s|ed|ing)?|typically|usually|often|generally|"
        r"highly\s+active|moderately\s+active|lower\s+activity|"
        r"short\s+sleep|long\s+sleep|irregular\s+sleep|"
        r"recovery|consistent|consistency|rhythm|mid[- ]week|"
        r"prefer|preference|profile|archetype|"
        r"who\s+you\s+are|standing|over\s+time"
        r")\b",
        text,
        re.I,
    )
    if typ in ("behavioral_profile", "data_pattern"):
        return True if durable_cues or typ == "behavioral_profile" else bool(durable_cues)
    if typ == "stated":
        return bool(durable_cues) and not is_situational_text(text)
    return bool(durable_cues)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _uid() -> str:
    return "mem_" + uuid.uuid4().hex[:12]


def _empty_doc(now: int | None = None) -> dict:
    now = now or _now_ms()
    return {
        "version": SCHEMA_VERSION,
        "learning_enabled": True,
        "updated_at_ms": now,
        "retention_note": RETENTION_NOTE,
        "items": [],
    }


def _loads_json_lenient(raw: str):
    """Parse JSON, tolerating accidental trailing comments from older writers."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Strip // or /* trailers appended after a top-level JSON value
        if "\n//" in s:
            s = s.split("\n//", 1)[0].strip()
        if "\n/*" in s:
            s = s.split("\n/*", 1)[0].strip()
        # Last resort: cut to last closing brace/bracket
        end_obj = s.rfind("}")
        end_arr = s.rfind("]")
        end = max(end_obj, end_arr)
        if end > 0:
            s = s[: end + 1]
        return json.loads(s)


async def read_store() -> dict:
    try:
        raw = await aio.get_artifact(STORE_ID)
    except Exception:
        return _empty_doc()
    if raw is None:
        return _empty_doc()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str) or not raw.strip():
        return _empty_doc()
    try:
        env = _loads_json_lenient(raw)
    except Exception:
        return _empty_doc()
    if isinstance(env, dict) and env.get("ciphertext"):
        try:
            plain = crypto.decrypt_text(env["ciphertext"])
            doc = json.loads(plain)
            if isinstance(doc, dict):
                doc.setdefault("items", [])
                doc.setdefault("learning_enabled", True)
                doc.setdefault("retention_note", RETENTION_NOTE)
                return doc
        except Exception:
            return _empty_doc()
    if isinstance(env, dict) and "items" in env:
        env.setdefault("learning_enabled", True)
        env.setdefault("retention_note", RETENTION_NOTE)
        return env
    return _empty_doc()


async def write_store(doc: dict) -> None:
    now = _now_ms()
    doc = dict(doc)
    doc["version"] = SCHEMA_VERSION
    doc["updated_at_ms"] = now
    doc["retention_note"] = RETENTION_NOTE
    items = doc.get("items") or []
    plain = json.dumps(
        {
            "version": SCHEMA_VERSION,
            "learning_enabled": bool(doc.get("learning_enabled", True)),
            "updated_at_ms": now,
            "retention_note": RETENTION_NOTE,
            "items": items,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    token = crypto.encrypt_text(plain)
    envelope = {
        "version": SCHEMA_VERSION,
        "alg": "hmac-sha256-xor-v1",
        "updated_at_ms": now,
        "n_items": len(items),
        "ciphertext": token,
    }
    # Nonce keeps ciphertext unique across rewrites (content-addressed blob store).
    envelope["write_nonce"] = now
    await aio.store_artifact(
        identifier=STORE_ID,
        title="Alyf Advisor personalization memory (encrypted)",
        artifact_type="text",
        data=json.dumps(envelope, ensure_ascii=False),
        metadata={"text": {"content_type": "application/json"}},
        hidden=True,
    )
    try:
        await write_view(doc)
    except Exception as e:
        # View refresh is best-effort — never fail a store write on view blob quirks.
        import executor as _ex

        _ex.print(f"memory view refresh skipped: {type(e).__name__}")


def _section_for_item(it: dict) -> str:
    typ = (it.get("type") or "").lower()
    text = (it.get("text") or "").lower()
    if typ == "preference":
        return "communication"
    if re.search(r"\bsleep|night|bedtime\b", text):
        return "sleep"
    if re.search(r"\brecover|rhr|resting\s+heart|hrv\b", text):
        return "recovery"
    if re.search(r"\bconsistenc|rhythm|mid[- ]week|weekend|day[- ]to[- ]day|regular\b", text):
        return "rhythm"
    if re.search(r"\bactiv|intensity|steps|movement|sedentary|exercise\b", text):
        return "activity"
    if typ == "behavioral_profile":
        return "activity"
    if typ == "data_pattern":
        if re.search(r"sleep", text):
            return "sleep"
        if re.search(r"recover|rhr|heart", text):
            return "recovery"
        if re.search(r"consist|weekend|weekday|rhythm", text):
            return "rhythm"
        return "activity"
    return "activity"


def _article_for(word: str) -> str:
    w = (word or "").strip().lower()
    if not w:
        return "a"
    return "an" if w[0] in "aeiou" else "a"


def _split_lead_rest(t: str):
    """Split 'Lead label — rest' / 'Lead: rest' into (lead, rest)."""
    for sep in ("—", "–", " - ", ": "):
        if sep in t:
            a, b = t.split(sep, 1)
            return a.strip(), b.strip()
    return t.strip(), ""


def _to_second_person(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(r"\s*\([^)]*≈[^)]*\)", "", t)
    t = re.sub(r"\s*in this window\.?", ".", t, flags=re.I)
    t = re.sub(r"\s*in [0-9]{4}-[0-9]{2}-[0-9]{2}→[^.]+\.", ".", t)
    t = re.sub(r"Data-derived activity profile:\s*", "", t, flags=re.I)
    t = re.sub(r"Recovery profile:\s*", "", t, flags=re.I)
    t = re.sub(r"Rhythm profile:\s*", "", t, flags=re.I)
    t = re.sub(r"Sleep profile:\s*", "", t, flags=re.I)
    t = re.sub(r"\b[Tt]he patient\b", "You", t)
    t = re.sub(r"\b[Pp]atient\b", "You", t)
    t = re.sub(r"\bObserved:\s*", "", t)
    t = re.sub(r"\bdescribes self as\b", "come across as", t, flags=re.I)
    t = re.sub(
        r"\bIntensity tends to sit near the top of the healthy range\b",
        "activity level tends to sit near the top of the healthy range",
        t,
        flags=re.I,
    )

    if not re.match(r"^(You|Your|You're|You are)\b", t, re.I):
        lead, rest = _split_lead_rest(t)
        low_lead = lead.lower().strip()

        def join(core: str, r: str) -> str:
            r = (r or "").strip()
            if not r:
                return core if core.endswith(".") else core + "."
            if r[0].isupper() and not r.startswith("I "):
                r = r[0].lower() + r[1:]
            out = f"{core} — {r}"
            return out if out.endswith((".", "!", "?")) else out + "."

        if re.match(r"^(highly|moderately)\s+active(\s+person)?$", low_lead):
            kind = "highly" if low_lead.startswith("highly") else "moderately"
            # Avoid "person — person"
            if rest.lower().startswith("person"):
                rest = rest[6:].lstrip(" —-–:")
            t = join(f"You're a {kind} active person", rest)
        elif low_lead in ("lower activity", "lower active", "lower activity volume"):
            t = join("You show lower activity volume than a balanced reference profile", rest)
        elif re.match(r"^irregular sleeper$", low_lead) or re.match(
            r"^irregular\s+sleep", low_lead
        ):
            t = join("You're an irregular sleeper", rest)
        elif re.match(r"^short sleeper$", low_lead) or re.match(r"^short\s+sleep", low_lead):
            t = join("You're a short sleeper", rest)
        elif re.match(r"^(long|longer)\s+sleep", low_lead):
            t = join("You're a longer sleeper", rest)
        elif low_lead.startswith("typically well-recovered"):
            t = join("You're typically well-recovered", rest or "recovery usually holds up well")
        elif low_lead.startswith("mixed recovery"):
            t = join("Your recovery is mixed", rest or "some stretches look solid, others softer")
        elif low_lead.startswith("recovery often"):
            t = "Your " + t[0].lower() + t[1:]
        elif low_lead.startswith("day-to-day consistency"):
            t = "Your " + t[0].lower() + t[1:]
        elif low_lead.startswith("rhythm is"):
            t = "Your rhythm is" + lead[len("Rhythm is") :] + ((" — " + rest) if rest else "")
        elif low_lead.startswith("prefers "):
            t = "You prefer " + lead[len("Prefers ") :] + ((" — " + rest) if rest else "")
        elif re.match(r"^step counts", low_lead):
            t = re.sub(r"^Step counts tend", "Your steps tend", t, flags=re.I)
        elif re.match(r"^sleep duration", low_lead):
            t = re.sub(r"^Sleep duration is often", "Your sleep is often", t, flags=re.I)
            t = re.sub(r"^Sleep duration", "Your sleep duration", t, flags=re.I)
        elif re.match(r"^resting heart", low_lead):
            t = re.sub(r"^Resting heart rate tends", "Your resting heart rate tends", t, flags=re.I)
        elif re.match(r"^mid-range sleep", low_lead):
            t = join("Your sleep duration is typically mid-range", rest)
        elif re.match(r"^tends toward", low_lead):
            t = "You " + t[0].lower() + t[1:]
        elif re.match(r"^typically ", low_lead):
            t = "You're " + t[len("Typically ") :]
        else:
            # Generic: if it looks like a noun phrase archetype, prefix You're a/an
            if len(lead.split()) <= 5 and not rest:
                words = lead.split()
                if words:
                    art = _article_for(words[0])
                    t = f"You're {art} {lead[0].lower() + lead[1:]}"
                    if not t.endswith("."):
                        t += "."
            elif rest and len(lead.split()) <= 6:
                # "Something — detail" → keep detail with light you-prefix if needed
                t = t  # leave; grammar pass below

    # Grammar / article fixes (idempotent)
    t = re.sub(r"\bYou're highly active person\b", "You're a highly active person", t, flags=re.I)
    t = re.sub(
        r"\bYou're moderately active person\b",
        "You're a moderately active person",
        t,
        flags=re.I,
    )
    t = re.sub(r"\bYou're a\s+Irregular\b", "You're an irregular", t, flags=re.I)
    t = re.sub(r"\bYou're a irregular\b", "You're an irregular", t, flags=re.I)
    t = re.sub(r"\bYou're an Irregular\b", "You're an irregular", t, flags=re.I)
    t = re.sub(r"\bYou're a active\b", "You're an active", t, flags=re.I)
    t = re.sub(r"person — person\b", "person", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\.\.+", ".", t)
    t = re.sub(r"\s+—\s*\.", ".", t)
    if t and t[-1] not in ".!?":
        t += "."
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def _dedupe_sentences(sentences: list[str]) -> list[str]:
    out = []
    for s in sentences:
        s = (s or "").strip()
        if not s:
            continue
        if any(_sim(s, prev) >= 0.62 for prev in out):
            continue
        out.append(s)
    return out


def _synthesize_summary(section_leads: list[tuple[str, str]]) -> str:
    """2–4 flowing sentences — whole picture, not a restatement list."""
    if not section_leads:
        return ""
    # Prefer activity, sleep, rhythm, recovery, communication order for narrative
    order = ["activity", "sleep", "rhythm", "recovery", "communication"]
    by = {k: v for k, v in section_leads}
    bits = []
    for k in order:
        if k in by:
            bits.append(by[k])
    bits = _dedupe_sentences(bits)[:4]
    if not bits:
        return ""
    if len(bits) == 1:
        return bits[0]
    # Light glue so it reads as one portrait, not four labels
    return " ".join(bits)


def _section_detail_prose(key: str, kept: list, summary_used: set[str]) -> str:
    """Section adds supporting detail — not a verbatim repeat of the portrait line."""
    parts = []
    for it in kept:
        p = _to_second_person(it.get("text") or "")
        if not p:
            continue
        # Prefer the clause after an em-dash as section detail (new info)
        if "—" in p:
            tail = p.split("—", 1)[1].strip()
            if tail:
                if tail[0].islower():
                    tail = tail[0].upper() + tail[1:]
                if not tail.endswith((".", "!", "?")):
                    tail += "."
                if not any(_sim(tail, s) >= 0.7 for s in summary_used):
                    parts.append(tail)
                    if len(parts) >= 2:
                        break
                    continue
        if any(_sim(p, s) >= 0.7 for s in summary_used):
            # Never surface internal provenance ("Assistant characterized…") as patient prose.
            continue
        parts.append(p)
        if len(parts) >= 2:
            break
    if not parts:
        # Key-specific soft fallback that doesn't clone the portrait lead
        fallbacks = {
            "activity": "Your day-to-day movement volume usually runs high compared with a balanced reference.",
            "sleep": "Night-to-night sleep timing and length are a defining pattern for you.",
            "recovery": "Recovery tends to track with how steady your sleep has been.",
            "rhythm": "Day-to-day rhythm is the piece that varies most.",
            "communication": "You like answers kept short and practical.",
        }
        fb = fallbacks.get(key)
        if fb and not any(_sim(fb, s) >= 0.7 for s in summary_used):
            parts.append(fb)
        else:
            for it in kept:
                p = _to_second_person(it.get("text") or "")
                if p and "—" in p:
                    tail = p.split("—", 1)[1].strip()
                    if tail:
                        if tail[0].islower():
                            tail = tail[0].upper() + tail[1:]
                        parts.append(tail if tail.endswith(".") else tail + ".")
                        break
    return " ".join(_dedupe_sentences(parts)[:2])


def build_portrait(items: list) -> dict:
    """Cohesive second-person portrait from active items. Superseded/expired omitted.

    Portrait summary = whole-picture synthesis (2–4 sentences).
    Sections = supporting detail that must NOT repeat the summary verbatim.
    """
    active = active_items(items)
    by_sec: dict[str, list[dict]] = {k: [] for k, _ in PORTRAIT_SECTIONS}
    for it in active:
        if is_forbidden_inference(it.get("text") or ""):
            continue
        if is_situational_text(it.get("text") or ""):
            continue
        sec = it.get("section") or _section_for_item(it)
        by_sec.setdefault(sec, []).append(it)

    # First pass: pick lead item per section for summary synthesis
    section_leads: list[tuple[str, str]] = []
    kept_by_sec: dict[str, list] = {}
    for key, title in PORTRAIT_SECTIONS:
        bucket = by_sec.get(key) or []
        if not bucket:
            continue
        bucket.sort(
            key=lambda it: (
                0
                if it.get("type") == "behavioral_profile"
                else 1
                if it.get("type") == "preference"
                else 2
                if it.get("type") == "stated"
                else 3,
                -int(it.get("last_reinforced_at") or it.get("learned_at") or 0),
            )
        )
        kept = []
        for it in bucket:
            if any(_sim(it.get("text") or "", k.get("text") or "") >= 0.72 for k in kept):
                continue
            kept.append(it)
            if len(kept) >= 3:
                break
        if not kept:
            continue
        kept_by_sec[key] = kept
        lead = _to_second_person(kept[0].get("text") or "")
        if lead:
            section_leads.append((key, lead))

    summary = _synthesize_summary(section_leads)
    summary_used = set(re.split(r"(?<=[.!?])\s+", summary)) if summary else set()
    summary_used = {s for s in summary_used if s}

    sections = []
    for key, title in PORTRAIT_SECTIONS:
        kept = kept_by_sec.get(key) or []
        if not kept:
            continue
        prose = _section_detail_prose(key, kept, summary_used)
        if not prose:
            # Last resort: still show something distinct from empty
            prose = _to_second_person(kept[0].get("text") or "")
        evidence = []
        for it in kept:
            prov = it.get("provenance") or {}
            evidence.append(
                {
                    "item_id": it.get("id"),
                    "text": it.get("text"),
                    "type": it.get("type"),
                    "provenance_summary": prov.get("summary"),
                    "data_window": prov.get("data_window"),
                    "learned_at": it.get("learned_at"),
                    "last_reinforced_at": it.get("last_reinforced_at"),
                }
            )
        sections.append(
            {
                "key": key,
                "title": title,
                "prose": prose,
                "item_ids": [it.get("id") for it in kept],
                "evidence": evidence,
            }
        )
        # After adding section, treat its prose as used so later sections don't clone it
        for s in re.split(r"(?<=[.!?])\s+", prose):
            if s:
                summary_used.add(s)

    return {
        "summary": summary,
        "sections": sections,
        "generated_at_ms": _now_ms(),
        "voice": "second_person_observation",
        "disclaimer": (
            "These are observations about activity patterns and preferences. "
            "They are not conclusions about how you feel and are not a clinical record."
        ),
    }


async def write_view(doc: dict) -> None:
    """Decrypted session view for UI — active + dismissed, plus portrait.

    Always also writes:
      - alyf_advisor_memory_portrait (portrait + learning flag)
      - alyf_advisor_memory_items (flat items list)

    The full VIEW_ID blob can stick in content-addressed storage; the UI prefers
    the items + portrait sidecars when they are fresher.
    """
    now = _now_ms()
    items = []
    for it in doc.get("items") or []:
        if not isinstance(it, dict):
            continue
        st = it.get("status") or "active"
        if st in ("deleted",):
            continue
        items.append(it)
    portrait = build_portrait(items)
    stamp = _now_ms()
    nonce = uuid.uuid4().hex
    view = {
        "version": SCHEMA_VERSION,
        "learning_enabled": bool(doc.get("learning_enabled", True)),
        "updated_at_ms": stamp,
        "retention_note": RETENTION_NOTE,
        "items": items,
        "portrait": portrait,
        "disclaimer": (
            "Personalization context only — not a clinical record, diagnosis, "
            "or medical finding. Built from durable characterizations in Alyf's "
            "answers and descriptive patterns in your activity data. "
            "Situational events belong on your annotation timeline."
        ),
        "view_stamp_ms": stamp,
        "portrait_rev": 4,
        "write_nonce": nonce,
    }
    payload = json.dumps(view, ensure_ascii=False, separators=(",", ":"))
    try:
        await aio.store_artifact(
            identifier=VIEW_ID,
            title="Alyf Advisor personalization memory (session view)",
            artifact_type="text",
            data=payload,
            metadata={"text": {"content_type": "application/json"}},
            hidden=True,
        )
    except Exception:
        # Full view can stick in content-addressed storage; sidecars below.
        pass
    # Portrait sidecar so the UI can pick up updates even if VIEW_ID is sticky.
    try:
        side = {
            "version": SCHEMA_VERSION,
            "updated_at_ms": stamp,
            "portrait": view.get("portrait"),
            "learning_enabled": view.get("learning_enabled", True),
            "disclaimer": view.get("disclaimer"),
            "portrait_rev": 4,
            "stamp": stamp,
            "write_nonce": nonce,
        }
        await aio.store_artifact(
            identifier="alyf_advisor_memory_portrait",
            title="Alyf Advisor portrait sidecar",
            artifact_type="text",
            data=json.dumps(side, ensure_ascii=False, separators=(",", ":")),
            metadata={"text": {"content_type": "application/json"}},
            hidden=True,
        )
    except Exception:
        pass
    # Items sidecar — authoritative item status when full view blob is sticky.
    try:
        items_side = {
            "version": SCHEMA_VERSION,
            "updated_at_ms": stamp,
            "learning_enabled": view.get("learning_enabled", True),
            "items": items,
            "write_nonce": nonce,
        }
        await aio.store_artifact(
            identifier="alyf_advisor_memory_items",
            title="Alyf Advisor memory items sidecar",
            artifact_type="text",
            data=json.dumps(items_side, ensure_ascii=False, separators=(",", ":")),
            metadata={"text": {"content_type": "application/json"}},
            hidden=True,
        )
    except Exception:
        pass


def decay_items(items: list, now: int | None = None) -> tuple[list, int]:
    now = now if now is not None else _now_ms()
    n = 0
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        it = dict(it)
        st = it.get("status") or "active"
        if st != "active":
            out.append(it)
            continue
        if is_situational_text(it.get("text") or "") and it.get("type") in (
            "stated",
            "data_pattern",
        ):
            it["status"] = "expired"
            it["expired_at"] = now
            it["expire_reason"] = "situational_not_profile"
            n += 1
            out.append(it)
            continue
        exp = it.get("expires_at")
        if exp and int(exp) <= now:
            it["status"] = "expired"
            it["expired_at"] = now
            n += 1
            out.append(it)
            continue
        last = int(it.get("last_reinforced_at") or it.get("learned_at") or 0)
        if last and (now - last) > OUTER_BOUND_MS:
            it["status"] = "expired"
            it["expired_at"] = now
            it["expire_reason"] = "outer_bound_unreinforced"
            n += 1
        out.append(it)
    return out, n


def active_items(items: list) -> list:
    return [
        it
        for it in items
        if isinstance(it, dict)
        and (it.get("status") or "active") == "active"
        and not it.get("dismissed")
    ]


_INFERENCE_RE = re.compile(
    r"\b("
    r"sleep\s*apnea|overtraining\s+syndrome|anemia|arrhythmia|atrial\s+fib|"
    r"heart\s+failure|coronary|myocardial|angina|"
    r"anxiety\s+disorder|major\s+depress(?:ion|ive)|clinical\s+depression|"
    r"bipolar|ptsd|ocd|panic\s+disorder|"
    r"cardiac[- ]?decondition(?:ed|ing)?|"
    r"non[- ]?compliant|noncompliance|"
    r"diagnos(?:is|e|ed|ing)|prognos(?:is|e|ed)|"
    r"disease\s*risk|disease\s+odds|"
    r"may\s+have\s+(?:a\s+)?(?:condition|disease|disorder|syndrome)|"
    r"appears\s+to\s+have\s+(?:a\s+)?(?:condition|disease|disorder|syndrome)|"
    r"seems\s+to\s+have\s+(?:a\s+)?(?:condition|disease|disorder|syndrome)|"
    r"could\s+be\s+(?:depression|anemia|a\s+heart\s+problem|sleep\s+apnea)|"
    r"looks\s+like\s+(?:overtraining|depression|apnea)|"
    r"patholog(?:y|ical)|medication\s+advice|prescri(?:be|ption)"
    r")\b",
    re.I,
)


def is_forbidden_inference(text: str) -> bool:
    if not text or not isinstance(text, str):
        return True
    return bool(_INFERENCE_RE.search(text))


def normalize_candidate(c: dict, now: int) -> dict | None:
    if not isinstance(c, dict):
        return None
    typ = (c.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if typ in ("data-derived", "data_derived", "pattern", "behavioral_pattern"):
        typ = "data_pattern"
    if typ in ("behavioral", "profile", "archetype", "persona", "behavioral_profile"):
        typ = "behavioral_profile"
    if typ in ("life_context", "context", "stated_context"):
        typ = "stated"
    if typ in ("pref", "style"):
        typ = "preference"
    if typ not in ITEM_TYPES:
        return None
    text = (c.get("text") or c.get("content") or "").strip()
    if not text or len(text) < 8:
        return None
    if is_forbidden_inference(text):
        return None
    if not is_durable_characterization(text, typ):
        return None
    if is_situational_text(text):
        return None
    text = re.sub(
        r"^(the patient|patient)\s+(is|has|shows)\s+", "Observed: ", text, flags=re.I
    )
    text = text[:400]
    section = c.get("section") or _section_for_item({"type": typ, "text": text})
    prov = c.get("provenance") or c.get("source") or {}
    if isinstance(prov, str):
        prov = {"summary": prov}
    if not isinstance(prov, dict):
        prov = {}
    source_kind = prov.get("kind") or (
        "patient_stated" if typ in ("preference", "stated") else "data_derived"
    )
    if prov.get("kind") == "assistant_derived":
        source_kind = "assistant_derived"
    if source_kind not in (
        "patient_stated",
        "data_derived",
        "patient_edit",
        "clinician_edit",
        "assistant_derived",
    ):
        source_kind = (
            "data_derived"
            if typ in ("data_pattern", "behavioral_profile")
            else "patient_stated"
        )
    expires = c.get("expires_at")
    if expires is None and typ == "data_pattern":
        expires = now + DATA_PATTERN_TTL_MS
    if expires is None and typ in ("preference", "behavioral_profile", "stated"):
        expires = now + OUTER_BOUND_MS
    return {
        "id": c.get("id") or _uid(),
        "type": typ,
        "section": section,
        "text": text,
        "provenance": {
            "kind": source_kind,
            "summary": str(prov.get("summary") or c.get("provenance_summary") or "")[:300],
            "data_window": prov.get("data_window"),
            "quote": (prov.get("quote") or "")[:240] or None,
        },
        "learned_at": int(c.get("learned_at") or now),
        "last_reinforced_at": int(c.get("last_reinforced_at") or c.get("learned_at") or now),
        "expires_at": int(expires) if expires else None,
        "status": "active",
        "dismissed": False,
        "clinician_notes": [],
    }


EXTRACT_SYSTEM = """You extract LONG-TERM personalization PROFILE candidates from a patient chat \
with an activity-score assistant (Alyf Advisor / BAP).

THE PROFILE = who the person IS over time. Not what happened this week.

PRIMARY SOURCE — learn from the ASSISTANT'S OWN ANSWERS:
Review the assistant messages and the reasoning in them. Pull out durable \
characterizations of the person that the assistant asserted or relied on.
Example: assistant says "you're a highly active person who tends to sleep short \
mid-week and recovers slowly after poor sleep" → three profile candidates.

Also capture durable preferences the patient stated about how they like to talk \
(short answers, more charts, etc.).

ALLOWED item types:
1) behavioral_profile — standing who-they-are characterizations (activity level, \
sleep style, recovery tendency, rhythm). Prefer these.
2) preference — how they like info (tone, detail, charts vs prose)
3) data_pattern — ONLY if the assistant asserted a standing descriptive pattern \
as part of who they are (not a one-week blip)
4) stated — ONLY durable self-descriptions the patient claims as lasting traits \
("I'm a short sleeper"). NEVER exams/travel/illness/one bad night.

DURABILITY TEST (must pass):
Ask: "Will this still be true of the person next month?"
✓ "You're a highly active person."
✓ "You're a consistently short sleeper."
✓ "You prefer short, direct answers."
✗ "You had finals this week."
✗ "You traveled last week and slept badly."
✗ "You didn't sleep well last night."

HARD REJECT:
- Situational / time-bound events (finals, travel, illness, one bad night, this week)
- Medical or psychological CONDITION / syndrome / clinical state / compliance judgment
- How the person FEELS (anxious, depressed, "feel off" as a stored trait)
- Disease-risk, medication, treatment
- Fabricated traits not supported by assistant answers or clear patient claims
- Raw score numbers as the memory itself

ALLOWED behavioral language (NOT diagnoses):
- highly active / moderately active / lower activity volume
- short sleeper / longer sleeper / irregular sleeper
- recovery often lags after hard days / typically well-recovered
- most consistent mid-week / softer day-to-day consistency

Each item = an OBSERVATION about behavior/preference, never a clinical conclusion.

Also return situational_mentions: any time-bound events the PATIENT volunteered \
that should NOT go in the profile (for the annotation system). Do not put these \
in candidates.

Return JSON only:
{
  "candidates": [
    {
      "type": "behavioral_profile"|"preference"|"stated"|"data_pattern",
      "text": "observation sentence (who they are)",
      "section": "activity"|"sleep"|"recovery"|"rhythm"|"communication",
      "provenance_summary": "why grounded (cite assistant characterization or patient claim)",
      "quote": "short supporting quote from assistant or patient",
      "source": "assistant_answer"|"patient_claim"
    }
  ],
  "situational_mentions": [
    {"text": "finals this week", "reason": "time-bound event — annotation, not profile"}
  ],
  "rejected_inferences": ["brief notes"]
}
If nothing durable, return empty candidates."""


async def extract_from_conversation(
    messages: list,
    *,
    path: str = "pure",
    window_label: str = "",
) -> list[dict]:
    """LLM extract focusing on ASSISTANT answers. Returns normalized candidates."""
    if not messages:
        return []
    lines = []
    assistant_n = 0
    for m in messages[-24:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if m.get("failed"):
            continue
        if role == "assistant":
            assistant_n += 1
        lines.append(f"{role.upper()}: {content[:1200]}")
    if not lines:
        return []
    if assistant_n == 0:
        user_blob = " ".join(
            (m.get("content") or "") for m in messages if m.get("role") == "user"
        )
        if len(user_blob.strip()) < 20:
            return []

    prompt = (
        f"Active tab/path: {path}. Window: {window_label or 'n/a'}.\n\n"
        "Focus on ASSISTANT characterizations of who the person is. "
        "Apply the durability test strictly.\n\n"
        "CONVERSATION:\n" + "\n".join(lines)
    )
    try:
        raw = await aio.call_llm(
            chat=Chat.from_prompt(
                system_prompt=EXTRACT_SYSTEM,
                user_prompt=UserPrompt.from_text(prompt),
            ),
            output_schema={
                "type": "object",
                "properties": {
                    "candidates": {"type": "array"},
                    "situational_mentions": {"type": "array"},
                    "rejected_inferences": {"type": "array"},
                },
            },
        )
    except Exception as e:
        executor.print(f"memory extract llm error: {type(e).__name__}")
        return []

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, dict):
        return []
    now = _now_ms()
    out = []
    for c in raw.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        src = (c.get("source") or "").strip().lower()
        kind = (
            "assistant_derived"
            if src in ("assistant_answer", "assistant", "answer")
            else "patient_stated"
        )
        cand = {
            "type": c.get("type"),
            "text": c.get("text"),
            "section": c.get("section"),
            "provenance": {
                "kind": kind
                if c.get("type") != "preference"
                else (
                    "patient_stated"
                    if kind == "patient_stated"
                    else "assistant_derived"
                ),
                "summary": c.get("provenance_summary")
                or (
                    "From assistant characterization in chat"
                    if kind == "assistant_derived"
                    else "From patient conversation"
                ),
                "quote": c.get("quote"),
                "data_window": window_label or None,
            },
        }
        if is_situational_text(str(c.get("text") or "")):
            continue
        norm = normalize_candidate(cand, now)
        if norm:
            out.append(norm)
    n_sit = len(raw.get("situational_mentions") or [])
    if n_sit:
        executor.print(
            f"memory extract: {n_sit} situational mention(s) deferred (not profile)"
        )
    return out


def extract_data_patterns(
    bundle: dict, path: str = "pure", start_idx=None, end_idx=None
) -> list[dict]:
    """Standing behavioral archetypes + durable descriptive patterns."""
    if not bundle:
        return []
    now = _now_ms()
    window_key, entry = C.pick_window(bundle, start_idx, end_idx)
    if not entry:
        return []
    dates = list(bundle.get("dates") or [])
    si = int(entry.get("start_idx") or 0)
    ei = int(entry.get("end_idx") or max(0, len(dates) - 1))
    window_label = f"{dates[si] if dates else '?'}→{dates[ei] if dates else '?'} ({path})"

    candidates = []
    daily = bundle.get("daily_series") or bundle.get("daily") or {}
    if not daily and isinstance(bundle.get("series"), dict):
        daily = bundle["series"]

    def series_vals(key):
        arr = daily.get(key)
        if not isinstance(arr, list):
            return []
        out = []
        for i in range(si, min(ei + 1, len(arr))):
            v = arr[i]
            if v is None:
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out

    def add_pattern(text: str, summary: str, section: str = "activity"):
        norm = normalize_candidate(
            {
                "type": "data_pattern",
                "text": text,
                "section": section,
                "provenance": {
                    "kind": "data_derived",
                    "summary": summary,
                    "data_window": window_label,
                },
                "expires_at": now + DATA_PATTERN_TTL_MS,
            },
            now,
        )
        if norm:
            candidates.append(norm)

    def add_profile(text: str, summary: str, section: str):
        norm = normalize_candidate(
            {
                "type": "behavioral_profile",
                "text": text,
                "section": section,
                "provenance": {
                    "kind": "data_derived",
                    "summary": summary,
                    "data_window": window_label,
                },
            },
            now,
        )
        if norm:
            candidates.append(norm)

    rhr = series_vals("resting_hr")
    sleep = series_vals("sleep_duration_min")
    steps = series_vals("steps")

    if len(rhr) >= 10 and len(sleep) >= 10 and len(rhr) == len(sleep):
        pairs = list(zip(sleep, rhr))
        med_sleep = sorted(sleep)[len(sleep) // 2]
        low_sleep_rhr = [hr for sl, hr in pairs if sl < med_sleep]
        high_sleep_rhr = [hr for sl, hr in pairs if sl >= med_sleep]
        if len(low_sleep_rhr) >= 4 and len(high_sleep_rhr) >= 4:
            m_low = sum(low_sleep_rhr) / len(low_sleep_rhr)
            m_high = sum(high_sleep_rhr) / len(high_sleep_rhr)
            if m_low - m_high >= 1.5:
                add_pattern(
                    "Resting heart rate tends to run higher on shorter-sleep days "
                    f"(about {m_low:.0f} bpm when sleep is below the window median, vs "
                    f"{m_high:.0f} bpm on longer-sleep days).",
                    f"Compared mean RHR on days below vs above median sleep in {window_label}.",
                    section="recovery",
                )

    if sleep and dates and len(sleep) >= 14:
        sleep_by_i = []
        arr = daily.get("sleep_duration_min") or []
        for i in range(si, min(ei + 1, len(arr))):
            try:
                if arr[i] is not None:
                    sleep_by_i.append((dates[i], float(arr[i])))
            except (TypeError, ValueError, IndexError):
                continue
        wd, we = [], []
        for d, v in sleep_by_i:
            try:
                from datetime import date as _date

                dow = _date.fromisoformat(d[:10]).weekday()
            except Exception:
                continue
            (we if dow >= 5 else wd).append(v)
        if len(wd) >= 5 and len(we) >= 2:
            m_wd = sum(wd) / len(wd)
            m_we = sum(we) / len(we)
            if m_wd - m_we >= 25:
                add_pattern(
                    f"Sleep duration is often shorter on weekends than weekdays "
                    f"(weekend mean ~{m_we:.0f} min vs weekday ~{m_wd:.0f} min).",
                    "Weekend vs weekday means of sleep_duration_min.",
                    section="sleep",
                )
            elif m_we - m_wd >= 25:
                add_pattern(
                    f"Sleep duration is often longer on weekends than weekdays "
                    f"(weekend mean ~{m_we:.0f} min vs weekday ~{m_wd:.0f} min).",
                    "Weekend vs weekday means of sleep_duration_min.",
                    section="sleep",
                )

    if steps and dates and len(steps) >= 14:
        by_dow = {i: [] for i in range(7)}
        arr = daily.get("steps") or []
        for i in range(si, min(ei + 1, len(arr))):
            try:
                if arr[i] is None:
                    continue
                from datetime import date as _date

                dow = _date.fromisoformat(dates[i][:10]).weekday()
                by_dow[dow].append(float(arr[i]))
            except Exception:
                continue
        means = {d: (sum(v) / len(v) if v else None) for d, v in by_dow.items()}
        present = [
            (d, m) for d, m in means.items() if m is not None and len(by_dow[d]) >= 2
        ]
        if len(present) >= 5:
            present.sort(key=lambda x: -x[1])
            best_d, best_m = present[0]
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            if best_d in (1, 2, 3):
                add_pattern(
                    f"Step counts tend to be highest mid-week "
                    f"({names[best_d]} mean ~{best_m:.0f} steps).",
                    "Highest weekday mean of steps in window.",
                    section="activity",
                )

    try:
        path_block = (entry.get("paths") or {}).get(path) or entry.get(path) or {}
        axes = path_block.get("axes") or entry.get("axes") or {}

        def axis_score(key, name):
            for k in (key, name, name.lower()):
                if k in axes:
                    v = axes[k]
                    if isinstance(v, dict):
                        return v.get("s") or v.get("score")
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
            return None

        c_sc = axis_score("C", "Consistency")
        r_sc = axis_score("R", "Recovery")
        i_sc = axis_score("I", "Intensity")

        if i_sc is not None:
            iv = float(i_sc)
            if iv >= 0.85:
                add_profile(
                    "Highly active person — activity level tends to sit near the top of the healthy range.",
                    f"Intensity axis ≈ {iv:.2f} in {window_label}",
                    "activity",
                )
            elif iv >= 0.55:
                add_profile(
                    "Moderately active person — movement volume is solid without sitting at the ceiling.",
                    f"Intensity axis ≈ {iv:.2f} in {window_label}",
                    "activity",
                )
            else:
                add_profile(
                    "Lower activity volume than a balanced reference profile in recent data.",
                    f"Intensity axis ≈ {iv:.2f} in {window_label}",
                    "activity",
                )
        if r_sc is not None:
            rv = float(r_sc)
            if rv >= 0.85:
                add_profile(
                    "Typically well-recovered — Recovery usually holds up well.",
                    f"Recovery axis ≈ {rv:.2f} in {window_label}",
                    "recovery",
                )
            elif rv >= 0.6:
                add_profile(
                    "Mixed recovery — some stretches look solid, others softer.",
                    f"Recovery axis ≈ {rv:.2f} in {window_label}",
                    "recovery",
                )
            else:
                add_profile(
                    "Recovery often runs softer than activity volume would suggest.",
                    f"Recovery axis ≈ {rv:.2f} in {window_label}",
                    "recovery",
                )
        if c_sc is not None:
            cv = float(c_sc)
            if cv < 0.7:
                add_profile(
                    "Day-to-day consistency is a softer spot — rhythm varies more than ideal.",
                    f"Consistency axis ≈ {cv:.2f} in {window_label}",
                    "rhythm",
                )
            elif cv >= 0.85:
                add_profile(
                    "Rhythm is steady — day-to-day consistency usually holds.",
                    f"Consistency axis ≈ {cv:.2f} in {window_label}",
                    "rhythm",
                )

        if sleep and len(sleep) >= 7:
            mean_sl = sum(sleep) / len(sleep)
            if len(sleep) >= 10:
                mu = mean_sl
                var = sum((x - mu) ** 2 for x in sleep) / len(sleep)
                sd = var ** 0.5
            else:
                sd = 0.0
            if sd >= 70:
                add_profile(
                    "Irregular sleeper — night length varies a lot from day to day.",
                    f"Sleep mean ≈ {mean_sl:.0f} min, SD ≈ {sd:.0f} in {window_label}",
                    "sleep",
                )
            elif mean_sl < 380:
                add_profile(
                    "Tends toward shorter nights — sleep duration usually runs on the short side.",
                    f"Mean sleep ≈ {mean_sl:.0f} min in {window_label}",
                    "sleep",
                )
            elif mean_sl > 480:
                add_profile(
                    "Tends toward longer nights — sleep duration usually runs on the long side.",
                    f"Mean sleep ≈ {mean_sl:.0f} min in {window_label}",
                    "sleep",
                )
            else:
                add_profile(
                    "Mid-range sleep duration is typical.",
                    f"Mean sleep ≈ {mean_sl:.0f} min in {window_label}",
                    "sleep",
                )
    except Exception:
        pass

    return candidates


def _sim(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def consolidate(existing: list, candidates: list, now: int | None = None) -> tuple[list, list]:
    now = now if now is not None else _now_ms()
    items = [dict(it) for it in existing if isinstance(it, dict)]
    ops = []

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        c = normalize_candidate(cand, now) if cand.get("status") != "active" else cand
        if c is None:
            c = normalize_candidate(cand, now)
        if c is None:
            ops.append(
                {
                    "op": "REJECT",
                    "reason": "normalize_durability_or_inference",
                    "text": str(cand.get("text", ""))[:80],
                }
            )
            continue

        best_i, best_s = -1, 0.0
        for i, it in enumerate(items):
            if (it.get("status") or "active") != "active":
                continue
            if it.get("dismissed"):
                continue
            same_type = it.get("type") == c["type"]
            same_section = (it.get("section") or _section_for_item(it)) == (
                c.get("section") or _section_for_item(c)
            )
            if not same_type and not (
                same_section
                and it.get("type") in ("behavioral_profile", "data_pattern")
                and c.get("type") in ("behavioral_profile", "data_pattern")
            ):
                continue
            s = _sim(it.get("text") or "", c.get("text") or "")
            if (
                same_section
                and it.get("type") == "behavioral_profile"
                and c.get("type") == "behavioral_profile"
            ):
                s = max(s, 0.6)
            if s > best_s:
                best_s, best_i = s, i

        if best_i >= 0 and best_s >= 0.72:
            old = items[best_i]
            new_text = (
                c["text"]
                if len(c["text"]) > len(old.get("text") or "")
                else old.get("text")
            )
            if (old.get("text") or "") != new_text and best_s < 0.92:
                old["status"] = "superseded"
                old["superseded_at"] = now
                old["superseded_by"] = c["id"]
                c["replaces"] = old.get("id")
                c["learned_at"] = old.get("learned_at") or now
                c["last_reinforced_at"] = now
                items.append(c)
                ops.append({"op": "SUPERSEDE", "old": old.get("id"), "new": c["id"]})
            else:
                old["last_reinforced_at"] = now
                if c.get("expires_at"):
                    old_exp = old.get("expires_at") or 0
                    old["expires_at"] = max(int(old_exp), int(c["expires_at"]))
                if (c.get("provenance") or {}).get("summary") and len(
                    (c.get("provenance") or {}).get("summary") or ""
                ) > len((old.get("provenance") or {}).get("summary") or ""):
                    old["provenance"] = dict(old.get("provenance") or {})
                    old["provenance"]["summary"] = c["provenance"]["summary"]
                ops.append({"op": "UPDATE", "id": old.get("id"), "sim": round(best_s, 2)})
            continue

        if best_i >= 0 and best_s >= 0.55 and c["type"] in ("stated", "behavioral_profile"):
            old = items[best_i]
            old["status"] = "superseded"
            old["superseded_at"] = now
            old["superseded_by"] = c["id"]
            c["replaces"] = old.get("id")
            c["last_reinforced_at"] = now
            items.append(c)
            ops.append({"op": "SUPERSEDE", "old": old.get("id"), "new": c["id"]})
            continue

        items.append(c)
        ops.append({"op": "ADD", "id": c["id"], "type": c["type"]})

    return items, ops


def retrieve(
    items: list,
    *,
    query: str = "",
    path: str = "pure",
    k: int = 8,
    now: int | None = None,
) -> list[dict]:
    now = now if now is not None else _now_ms()
    q_tokens = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    axis_words = {
        "intensity",
        "recovery",
        "spread",
        "consistency",
        "variety",
        "sleep",
        "steps",
        "heart",
        "rhr",
        "hrv",
        "weekend",
        "active",
        "rhythm",
    }
    scored = []
    for it in active_items(items):
        if is_situational_text(it.get("text") or ""):
            continue
        text = it.get("text") or ""
        t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens)) if q_tokens else 0.15
        typ = it.get("type")
        prior = (
            0.28
            if typ == "behavioral_profile"
            else 0.22
            if typ == "preference"
            else 0.12
            if typ == "data_pattern"
            else 0.1
        )
        if q_tokens & axis_words and t_tokens & axis_words:
            prior += 0.15
        last = int(it.get("last_reinforced_at") or it.get("learned_at") or now)
        age_days = max(0, (now - last) / 86400_000)
        recency = max(0.0, 1.0 - age_days / 180.0)
        score = overlap * 0.55 + prior + recency * 0.25
        scored.append((score, it))
    scored.sort(key=lambda x: -x[0])
    out = []
    seen = set()
    for s, it in scored:
        if it["id"] in seen:
            continue
        if s < 0.18 and len(out) >= 3:
            continue
        out.append(it)
        seen.add(it["id"])
        if len(out) >= k:
            break
    return out


def format_memory_for_prompt(items: list) -> str:
    if not items:
        return (
            "(no personalization memory items retrieved this turn — answer from "
            "patient data + conversation only)"
        )
    portrait = build_portrait(items)
    lines = [
        "PERSONALIZATION / BEHAVIORAL PROFILE (context only — NOT a source of patient numbers; "
        "if memory conflicts with PATIENT DATA above, PATIENT DATA wins). "
        "REASON THROUGH this profile: connect who they are (activity/sleep/recovery/rhythm "
        "archetypes and preferences) to what the data is doing right now. "
        "Never name a medical/psychological condition. Never store or treat situational "
        "events (finals, travel, one bad night) as who they are — those are annotations. "
        "Profile items are durable observations, not diagnoses."
    ]
    if portrait.get("summary"):
        lines.append(f"PORTRAIT: {portrait['summary']}")
    for sec in portrait.get("sections") or []:
        lines.append(f"- [{sec.get('title')}] {sec.get('prose')}")
    lines.append("Structured items:")
    for it in items:
        typ = it.get("type") or "?"
        prov = it.get("provenance") or {}
        kind = prov.get("kind") or "?"
        learned = it.get("learned_at")
        try:
            from datetime import datetime, timezone

            ds = datetime.fromtimestamp(learned / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            ds = "?"
        lines.append(
            f"- [{typ} | {kind} | learned {ds}] {it.get('text')}"
            f" (provenance: {prov.get('summary') or 'n/a'})"
        )
    return "\n".join(lines)


def apply_patient_mutation(doc: dict, payload: dict) -> dict:
    now = _now_ms()
    doc = dict(doc)
    items = [dict(it) for it in (doc.get("items") or [])]
    op = (payload.get("op") or "").strip()

    if op == "set_learning":
        doc["learning_enabled"] = bool(payload.get("enabled"))
        doc["items"] = items
        return doc

    if op == "clear":
        for it in items:
            if (it.get("status") or "active") == "active":
                it["status"] = "deleted"
                it["deleted_at"] = now
                it["deleted_by"] = "patient"
        doc["items"] = items
        return doc

    if op == "delete":
        iid = str(payload.get("id") or "")
        for it in items:
            if it.get("id") == iid:
                it["status"] = "deleted"
                it["deleted_at"] = now
                it["deleted_by"] = "patient"
        doc["items"] = items
        return doc

    if op == "mute":
        iid = str(payload.get("id") or "")
        for it in items:
            if it.get("id") == iid:
                it["dismissed"] = True
                it["status"] = "dismissed"
                it["muted_by"] = "patient"
                it["muted_at"] = now
        doc["items"] = items
        return doc

    if op == "edit":
        # Patient corrections are trusted: only block empty text and forbidden
        # clinical inference. Do NOT re-run durability/situational extract gates —
        # those caused silent no-ops when the patient rewrote an existing item.
        iid = str(payload.get("id") or "")
        new_text = (payload.get("text") or "").strip()
        if not iid:
            doc["items"] = items
            return doc
        if new_text and not is_forbidden_inference(new_text) and len(new_text) >= 3:
            for it in items:
                if it.get("id") == iid and (it.get("status") or "active") == "active":
                    it["text"] = new_text[:400]
                    it["last_reinforced_at"] = now
                    it["section"] = _section_for_item(it)
                    it["provenance"] = dict(it.get("provenance") or {})
                    it["provenance"]["kind"] = "patient_edit"
                    it["provenance"]["summary"] = "Corrected by patient"
                    break
        doc["items"] = items
        return doc

    if op == "add":
        c = normalize_candidate(
            {
                "type": payload.get("type") or "preference",
                "text": payload.get("text"),
                "provenance": {
                    "kind": "patient_edit",
                    "summary": payload.get("provenance_summary") or "Added by patient",
                    "quote": payload.get("quote"),
                },
            },
            now,
        )
        if c:
            items, _ = consolidate(items, [c], now)
        doc["items"] = items
        return doc

    if op == "clinician_dismiss":
        iid = str(payload.get("id") or "")
        who = str(payload.get("who") or "clinician")
        note = (payload.get("note") or "").strip()[:300]
        for it in items:
            if it.get("id") == iid:
                it["dismissed"] = True
                it["status"] = "dismissed"
                notes = list(it.get("clinician_notes") or [])
                notes.append({"at": now, "who": who, "action": "dismiss", "note": note})
                it["clinician_notes"] = notes
        doc["items"] = items
        return doc

    if op == "clinician_correct":
        iid = str(payload.get("id") or "")
        who = str(payload.get("who") or "clinician")
        new_text = (payload.get("text") or "").strip()
        note = (payload.get("note") or "").strip()[:300]
        if (
            new_text
            and not is_forbidden_inference(new_text)
            and not is_situational_text(new_text)
        ):
            for it in items:
                if it.get("id") == iid:
                    it["text"] = new_text[:400]
                    it["dismissed"] = False
                    if it.get("status") == "dismissed":
                        it["status"] = "active"
                    it["last_reinforced_at"] = now
                    it["section"] = _section_for_item(it)
                    it["provenance"] = dict(it.get("provenance") or {})
                    it["provenance"]["kind"] = "clinician_edit"
                    notes = list(it.get("clinician_notes") or [])
                    notes.append({"at": now, "who": who, "action": "correct", "note": note})
                    it["clinician_notes"] = notes
        doc["items"] = items
        return doc

    if op == "clinician_restore":
        iid = str(payload.get("id") or "")
        who = str(payload.get("who") or "clinician")
        for it in items:
            if it.get("id") == iid and it.get("status") == "dismissed":
                it["dismissed"] = False
                it["status"] = "active"
                notes = list(it.get("clinician_notes") or [])
                notes.append({"at": now, "who": who, "action": "restore", "note": ""})
                it["clinician_notes"] = notes
        doc["items"] = items
        return doc

    doc["items"] = items
    return doc


async def run_pipeline(
    *,
    messages: list | None = None,
    path: str = "pure",
    start_idx=None,
    end_idx=None,
    run_data_pass: bool = True,
    bundle=None,
) -> dict:
    doc = await read_store()
    if not doc.get("learning_enabled", True):
        items, n_decay = decay_items(doc.get("items") or [])
        doc["items"] = items
        await write_store(doc)
        return {"skipped": True, "reason": "learning_disabled", "decayed": n_decay}

    now = _now_ms()
    items, n_decay = decay_items(doc.get("items") or [], now)
    candidates: list[dict] = []

    window_label = ""
    if bundle is None:
        try:
            bundle = await C.load_bundle()
        except Exception:
            bundle = None
    if bundle:
        _, entry = C.pick_window(bundle, start_idx, end_idx)
        dates = list(bundle.get("dates") or [])
        if entry and dates:
            si = int(entry.get("start_idx") or 0)
            ei = int(entry.get("end_idx") or 0)
            window_label = f"{dates[si]}→{dates[ei]}"

    if messages:
        conv_cands = await extract_from_conversation(
            messages, path=path, window_label=window_label
        )
        candidates.extend(conv_cands)

    if run_data_pass and bundle is not None:
        candidates.extend(extract_data_patterns(bundle, path, start_idx, end_idx))

    items, ops = consolidate(items, candidates, now)
    items, n_decay2 = decay_items(items, now)
    doc["items"] = items
    await write_store(doc)

    n_add = sum(1 for o in ops if o.get("op") == "ADD")
    n_upd = sum(1 for o in ops if o.get("op") == "UPDATE")
    n_sup = sum(1 for o in ops if o.get("op") == "SUPERSEDE")
    n_rej = sum(1 for o in ops if o.get("op") == "REJECT")
    portrait = build_portrait(items)
    return {
        "skipped": False,
        "candidates": len(candidates),
        "ops_add": n_add,
        "ops_update": n_upd,
        "ops_supersede": n_sup,
        "ops_reject": n_rej,
        "decayed": n_decay + n_decay2,
        "active": len(active_items(items)),
        "portrait_sections": len(portrait.get("sections") or []),
        "has_summary": bool(portrait.get("summary")),
    }


async def load_for_turn() -> tuple[dict, list]:
    doc = await read_store()
    items, n = decay_items(doc.get("items") or [])
    if n:
        doc["items"] = items
        await write_store(doc)
    return doc, active_items(items)