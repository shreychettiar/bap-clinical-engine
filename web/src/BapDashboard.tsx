import './App.css'

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
  type DragEvent,
  type ChangeEvent,
} from 'react'
import { createPortal } from 'react-dom'
import SaveAsImage from './SaveAsImage'
import { ResponsiveRadar } from '@nivo/radar'
import { ResponsiveLine } from '@nivo/line'
import {
  usePromptQlArtifact,
  usePromptQlExecuteProgram,
} from './promptQlHostHooks'

// ---------- Types ----------
type Sub = {
  name: string
  present: boolean
  s: number | null
  w: number
  raw: number | string | null
  n: number
  tier: number
  s_pop: number | null
  s_per: number | null
  alpha: number | null
  suppressed: boolean
}

type Detail = {
  S_I_pop: number
  m_v: number | null
  P_couch: number
  base: number
  scenario: string
  stream_coverage: number
  L_recent: number
  frac_low_recovery: number
  a_persist: number
  W_hours: number
  inter_weights: Record<string, number>
  axis_available: Record<string, boolean>
  rho_axis: Record<string, number>
  reliability_factor: number
  fallback_tiers: Record<string, unknown[]>
}

type Result = {
  bap: number | null
  axis_scores: Record<string, number | null>
  subscores: Record<string, Sub[]>
  confidence: number
  flags: string[]
  detail: Detail
}

type Suggestion = {
  lever_class: string
  text: string
  recoverable_points: number
  leverage?: number
  load_increasing?: boolean
  axis?: string
}

type SuggestionPack = {
  mode: string
  why_line: string
  suggestions: Suggestion[]
  standing_frame: string
  confidence_caveat?: string | null
  gates_fired: string[]
  predictions: { action: string; predicted_bap_gain_points: number; lever: string }[]
  packaged: string
}

type WindowEntry = {
  start_idx: number
  end_idx: number
  window_days: number
  window_hours: number
  date_start: string
  date_end: string
  pure: Result
  blend: Result
  suggestions_pure: SuggestionPack
  suggestions_blend: SuggestionPack
}

type TrendPoint = {
  date: string
  start_idx: number
  end_idx: number
  BAP_pure: number | null
  BAP_blend: number | null
  Confidence_pure: number | null
  Confidence_blend: number | null
  I_pure: number | null
  R_pure: number | null
  Sp_pure: number | null
  C_pure: number | null
  V_pure: number | null
  I_blend: number | null
  R_blend: number | null
  Sp_blend: number | null
  C_blend: number | null
  V_blend: number | null
}

type FullSeriesPoint = {
  date: string
  idx: number
  BAP_1d_pure: number | null
  BAP_1d_blend: number | null
  BAP_14d_pure: number | null
  BAP_14d_blend: number | null
  I_1d_pure: number | null
  R_1d_pure: number | null
  Sp_1d_pure: number | null
  C_1d_pure: number | null
  V_1d_pure: number | null
  I_1d_blend: number | null
  R_1d_blend: number | null
  Sp_1d_blend: number | null
  C_1d_blend: number | null
  V_1d_blend: number | null
}

type HalfTick = {
  tick: number
  date: string
  time: string
  label: string
  day_idx: number
  half: number
}

type BundleUploadMeta = {
  file_name?: string
  row_count?: number
  date_start?: string
  date_end?: string
  skipped_bad_dates?: number
}

type Bundle = {
  version: number
  subject: string
  profile_label: string
  source: string
  age: number
  sex: string
  conditions_flag: string[]
  condition_personalization: boolean
  drop_met_from_intensity: boolean
  /** False when only interday daily-average MET is available (excluded from scoring). */
  met_scoring_eligible?: boolean
  met_form?: string
  met_excluded_reason?: string | null
  dates: string[]
  n_days: number
  min_hours: number
  max_hours: number
  hours_step: number
  half_ticks: HalfTick[]
  windows: Record<string, WindowEntry>
  trends: Record<string, TrendPoint[]>
  full_series: FullSeriesPoint[]
  daily_series: Record<string, (number | null)[] | string[]>
  notes: string[]
  upload?: BundleUploadMeta
}

type AnnTargetType = 'score' | 'axis'

type Annotation = {
  id: string
  created_at: string
  updated_at?: string
  target_type: AnnTargetType
  target_key: string
  axis_key?: string | null
  variable?: string | null
  label: string
  text: string
  start_date: string
  end_date: string
  start_time: string
  end_time: string
  start_idx: number
  end_idx: number
  tab?: string
  window_days?: number
  /**
   * Provenance. Manual UI defaults to 'manual'. Conversational Advisor creates
   * use 'patient_via_advisor'. Same schema/store either way.
   */
  source?: 'manual' | 'patient_via_advisor'
}

/** Structured draft from the Advisor — nothing is written until Accept. */
type AnnotationProposal = {
  status?: 'proposal' | 'clarify' | 'error'
  mode?: 'explicit' | 'offer'
  needs_clarification?: boolean
  clarifier?: string
  target_type?: AnnTargetType
  axis_key?: string | null
  variable?: string | null
  title?: string
  description?: string
  start_date?: string
  end_date?: string
  start_time?: string
  end_time?: string
  start_idx?: number
  end_idx?: number
  source?: 'patient_via_advisor' | string
  timeframe_raw?: string
  data_range?: { start?: string; end?: string }
}

type AnnotationsDoc = { version?: number; annotations: Annotation[] }

type MemoryItem = {
  id: string
  type: 'preference' | 'stated' | 'data_pattern' | 'behavioral_profile' | string
  section?: string
  text: string
  provenance?: {
    kind?: string
    summary?: string
    data_window?: string | null
    quote?: string | null
  }
  learned_at?: number
  last_reinforced_at?: number
  expires_at?: number | null
  status?: string
  dismissed?: boolean
  clinician_notes?: { at: number; who: string; action: string; note?: string }[]
}

type MemoryPortraitEvidence = {
  item_id?: string
  text?: string
  type?: string
  provenance_summary?: string
  data_window?: string | null
  learned_at?: number
  last_reinforced_at?: number
}

type MemoryPortraitSection = {
  key: string
  title: string
  prose: string
  item_ids?: string[]
  evidence?: MemoryPortraitEvidence[]
}

type MemoryPortrait = {
  summary?: string
  sections?: MemoryPortraitSection[]
  generated_at_ms?: number
  voice?: string
  disclaimer?: string
}

type MemoryViewDoc = {
  version?: number
  learning_enabled?: boolean
  updated_at_ms?: number
  retention_note?: string
  disclaimer?: string
  items?: MemoryItem[]
  portrait?: MemoryPortrait
}

type MemoryProposal = {
  status?: string
  action?: string
  item_id?: string | null
  item_type?: string | null
  item_text?: string | null
  enabled?: boolean | null
  preview_text?: string
  needs_clarification?: boolean
  clarify_question?: string | null
  reply_override?: string
}

type TabId = 'bap' | 'trend'

type UploadInfo = {
  fileName: string
  rowCount: number
  dateStart: string
  dateEnd: string
}

// ---------- Alyf Advisor (conversational explainer) ----------
type AdvisorSource = {
  id: string
  doc: string
  section: string
  score: number
  provisional: boolean
}

type AdvisorReplyDoc = {
  turn_id?: string
  reply: string
  category?: string
  symptom_escalation?: boolean
  grounding_verdict?: string
  sources_used?: AdvisorSource[]
  stages?: string[]
  /** Context-derived first-person follow-up questions (2–4); empty after escalation. */
  follow_ups?: string[]
  internal_review_only?: boolean
  annotation_proposal?: AnnotationProposal | null
  memory_proposal?: MemoryProposal | null
  memory_ids?: string[]
}

type AdvisorMsg = {
  id: string
  role: 'user' | 'assistant'
  content: string
  category?: string
  grounding?: string
  stages?: string[]
  sources?: AdvisorSource[]
  escalation?: boolean
  /** A turn that failed: rendered as a recoverable bubble with a retry, not a crash. */
  failed?: boolean
  retryText?: string
  /** Editable annotation draft attached to this assistant turn (preview gate). */
  annotationProposal?: AnnotationProposal | null
  /** Local UI state for the proposal card on this message. */
  annotationUi?: 'pending' | 'editing' | 'saved' | 'cancelled'
  annotationSavedId?: string
  memoryProposal?: MemoryProposal | null
  memoryUi?: 'pending' | 'editing' | 'saved' | 'cancelled'
  /** Context-derived follow-up chips for this assistant turn only. */
  followUps?: string[]
}

/**
 * A patient conversation with the Alyf Advisor.
 *
 * The artifact iframe has an opaque origin — no localStorage/sessionStorage —
 * so durable history lives in a dedicated encrypted transcript store
 * (`alyf_advisor_chat_history`), fully separate from BAP clinical artifacts.
 * Each conversation carries a rolling 7-day expiry from createdAt; purge is
 * permanent and removes ONLY transcripts.
 */
type AdvisorConvo = {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  /** Absolute expiry (ms). Policy: createdAt + 7 days. Rolling per conversation. */
  expiresAt: number
  msgs: AdvisorMsg[]
  /** Opened from history as a past transcript — composer disabled. */
  readOnly?: boolean
}

/** Session-view document bound from the purged transcript store. */
type AdvisorHistoryView = {
  version?: number
  retention_days?: number
  retention_note?: string
  updated_at_ms?: number
  conversations?: AdvisorConvo[]
}

/** Stable empty array so effects depending on the message list don't thrash. */
const EMPTY_ADV_MSGS: AdvisorMsg[] = []

/** Rolling retention: each conversation lives 7 days from when it started. */
const ADV_RETENTION_MS = 7 * 24 * 60 * 60 * 1000
const ADV_RETENTION_DAYS = 7

/**
 * Starter chips for the data-grounded analyst Advisor. These are answered from
 * the patient's bundle (axis rankings, both score paths, trends, engine
 * suggestions) plus interpretive knowledge — they no longer depend on corpus
 * retrieval clearing a relevance floor.
 */
const ADVISOR_STARTERS = [
  "What's my worst axis and why?",
  'Why did my BAP change?',
  'How could I improve?',
  'Why is my objective score different from my trend?',
  'What does Recovery mean?',
  'Why do I feel off lately?',
]

// Type-out cadence for the streamed reply, in characters per second.
const STREAM_CPS = 460
// Conversation history sent per turn. The full thread stays on screen; only the
// tail is forwarded, because the payload travels as a single argv string.
// More turns so deictics ("that/these/it") resolve; still argv-safe with char cap.
const ADV_HISTORY_TURNS = 16
const ADV_HISTORY_CHARS = 900
// How long to keep resolving the reply artifact before offering a retry.
const ADV_REPLY_TIMEOUT_MS = 90000

/** Preview line for the history list (first user message, trimmed). */
function advConvoPreview(c: AdvisorConvo): string {
  const firstUser = c.msgs.find((m) => m.role === 'user' && m.content.trim())
  const raw = (firstUser?.content || c.title || 'Conversation').replace(/\s+/g, ' ').trim()
  return raw.length > 72 ? `${raw.slice(0, 72)}…` : raw
}

/** Plain-language countdown until a conversation auto-deletes. */
function advDeletesInLabel(expiresAt: number, now = Date.now()): string {
  const ms = expiresAt - now
  if (ms <= 0) return 'deletes soon'
  const days = Math.floor(ms / (24 * 60 * 60 * 1000))
  const hours = Math.floor(ms / (60 * 60 * 1000))
  if (days >= 2) return `deletes in ${days} days`
  if (days === 1) return 'deletes in 1 day'
  if (hours >= 2) return `deletes in ${hours} hours`
  if (hours === 1) return 'deletes in 1 hour'
  const mins = Math.max(1, Math.floor(ms / (60 * 1000)))
  return `deletes in ${mins} min`
}

function advFormatWhen(ts: number): string {
  try {
    const d = new Date(ts)
    const now = new Date()
    const sameDay =
      d.getFullYear() === now.getFullYear() &&
      d.getMonth() === now.getMonth() &&
      d.getDate() === now.getDate()
    if (sameDay) {
      return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    }
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' })
  } catch {
    return ''
  }
}

/** Drop expired conversations permanently from an in-memory list. */
function advPurgeExpired(list: AdvisorConvo[], now = Date.now()): AdvisorConvo[] {
  // Only drop expired threads. Do NOT drop empty drafts here — the history panel
  // already filters to msgs.length > 0 for display, and purging empty drafts was
  // clearing the active "new chat", re-seeding, and force-closing the history UI.
  return list.filter((c) => {
    const exp = c.expiresAt || c.createdAt + ADV_RETENTION_MS || 0
    return exp > now
  })
}

function advWithExpiry(c: Omit<AdvisorConvo, 'expiresAt'> & { expiresAt?: number }): AdvisorConvo {
  const createdAt = c.createdAt || Date.now()
  return {
    ...c,
    createdAt,
    updatedAt: c.updatedAt || createdAt,
    expiresAt: createdAt + ADV_RETENTION_MS,
    msgs: c.msgs || [],
  }
}

// Axis words the Advisor may name in prose → axis key, for the inline data chips.
const AXIS_WORDS: [string, string][] = [
  ['Intensity', 'I'],
  ['Recovery', 'R'],
  ['Spread', 'Sp'],
  ['Consistency', 'C'],
  ['Variety', 'V'],
]
const TREND_RE =
  /trend|trending|drift|rose|fell|climb|declin|improv|dropp|over the window|week over week|this week/i

// How a routed turn is labelled — surfaced only in the hidden review trace.
const ADVISOR_ROUTE_LABELS: Record<string, string> = {
  explain_science: 'Science · from the research corpus',
  actionable: 'Routed to the gated suggestion engine',
  annotation: 'Annotation draft · confirm to save',
  smalltalk: 'Conversation',
  symptom: 'Symptom → escalated to care team',
  diagnosis: 'Out of scope · clinical interpretation',
  disease_risk: 'Out of scope · disease risk',
  medication: 'Out of scope · medication',
  non_bap_health: 'Out of scope · outside the corpus',
  medical_advice: 'Out of scope · medical/exercise advice',
}

// Chart margins — must match ResponsiveLine margin props for drag/markers
const CHART_ML = 42
const CHART_MR = 18
const CHART_MT = 24
const CHART_MB = 48

// ---------- Display labels ----------
const AXIS_META: Record<string, { name: string; color: string; tip: string }> = {
  I: {
    name: 'Intensity',
    color: '#ef4444',
    tip: 'How much physical effort and active energy you put in over the window, compared with a healthy reference group.',
  },
  R: {
    name: 'Recovery',
    color: '#10b981',
    tip: 'How well your body is recovering — sleep-related and resting signals such as heart-rate variability and resting heart rate.',
  },
  V: {
    name: 'Variety',
    color: '#8b5cf6',
    tip: 'How diverse your activity types are. Needs workout-type or heart-rate zone data; often unavailable from daily ring exports.',
  },
  Sp: {
    name: 'Spread',
    color: '#3b82f6',
    tip: 'How movement is distributed across the day versus long stretches of sitting. Balanced days score higher than one workout plus inactivity.',
  },
  C: {
    name: 'Consistency',
    color: '#f59e0b',
    tip: 'How steady your day-to-day activity and sleep timing are. Large swings between days lower this part of the score.',
  },
}

const VAR_LABELS: Record<string, string> = {
  vigorous_intensity_seconds: 'Vigorous activity time',
  moderate_intensity_seconds: 'Moderate activity time',
  workouts_average_hr: 'Average workout heart rate',
  activity_high: 'High activity minutes',
  floors_climbed: 'Floors climbed',
  steps: 'Steps',
  workouts_max_hr: 'Peak workout heart rate',
  calories_active: 'Active calories',
  activity_medium: 'Medium activity minutes',
  distance: 'Walking distance',
  activity_seconds: 'Total active time',
  met_level: 'Average metabolic level',
  met_level_excluded: 'Average metabolic level (not used in score)',
  hrv_rmssd: 'Heart-rate variability',
  resting_hr: 'Resting heart rate',
  hrr_1min: 'One-minute heart-rate recovery',
  breaths_per_min: 'Breathing rate',
  temperature_delta: 'Skin temperature change',
  sleep_efficiency: 'Sleep efficiency',
  modality_diversity: 'Activity type mix',
  zone_diversity: 'Heart-rate zone mix',
  inactivity_fraction: 'Sitting and inactive share',
  standing_hours_ratio: 'Standing hours share',
  low_intensity_fraction: 'Light activity share',
  standing_seconds: 'Standing time',
  activity_level_bout: 'Activity bout structure',
  met_level_bout: 'Effort bout structure',
  steps_cv: 'Day-to-day step steadiness',
  calories_active_cv: 'Day-to-day calorie steadiness',
  avg_met_level_cv: 'Day-to-day effort steadiness',
  moderate_intensity_seconds_cv: 'Day-to-day moderate activity steadiness',
  daily_movement_cv: 'Day-to-day movement steadiness',
  vigorous_engagement_freq: 'Vigorous activity frequency',
  sleep_duration_isd: 'Sleep timing regularity',
  sleep_duration: 'Sleep duration',
}

const VAR_TIPS: Record<string, string> = {
  vigorous_intensity_seconds: 'Time spent in higher-effort movement during the window.',
  moderate_intensity_seconds: 'Time spent in medium-effort movement such as brisk walking.',
  activity_high: 'Daily high-activity minutes reported by the wearable.',
  activity_medium: 'Daily medium-activity minutes reported by the wearable.',
  steps: 'Average daily step count versus a healthy reference range.',
  calories_active: 'Calories burned through movement, not including resting burn.',
  distance: 'Estimated walking-equivalent distance covered.',
  activity_seconds: 'Combined active minutes across intensity bands.',
  met_level:
    'Whole-day average metabolic equivalent. On some wearables this includes sleep and can understate true workout effort.',
  met_level_excluded:
    'Shown for context only — not used in the BAP score. Interday daily-average MET (≤1 value per day, no intraday resolution) is excluded from Intensity and all axis computation; the engine treats it as not present (no imputation).',
  hrv_rmssd: 'Beat-to-beat heart-rate variability — higher often reflects better recovery capacity.',
  resting_hr: 'Heart rate while at rest. Lower is generally better within healthy bounds.',
  breaths_per_min: 'Breaths per minute during rest or sleep.',
  temperature_delta: 'Deviation of skin temperature from your usual baseline.',
  inactivity_fraction: 'Share of the day spent inactive or sitting.',
  low_intensity_fraction: 'Share of light everyday movement such as slow walking around.',
  standing_hours_ratio: 'How much of waking time includes standing rather than prolonged sitting.',
  steps_cv: 'How much your steps bounce around day to day. Lower variation scores better here.',
  calories_active_cv: 'How steady your active calorie burn is across days.',
  avg_met_level_cv:
    'How steady average daily effort is across days. Uses the MET series for Consistency only when present; whole-day daily-average MET is still excluded from Intensity scoring.',
  moderate_intensity_seconds_cv: 'How steady moderate activity minutes are across days.',
  daily_movement_cv: 'How steady overall daily movement is across days.',
  vigorous_engagement_freq: 'How often you engage in vigorous activity across weeks.',
  sleep_duration_isd: 'How irregular sleep duration is from night to night.',
  modality_diversity: 'Mix of different activity types (running, strength, cycling, and so on).',
  zone_diversity: 'Mix of heart-rate training zones used.',
}

const FLAG_INFO: Record<string, { label: string; why: string }> = {
  SHORT_WINDOW: {
    label: 'Short window',
    why: 'The selected window is shorter than preferred for a fully stable score.',
  },
  COLD_START: {
    label: 'Cold start',
    why: 'Not enough enrolled history yet for a fully mature personal baseline.',
  },
  MODALITY_BLIND: {
    label: 'Activity types unavailable',
    why: 'No workout-type or heart-rate zone data — variety inputs are limited.',
  },
  RECOVERY_INSUFFICIENT: {
    label: 'Limited recovery signal',
    why: 'Too little recovery data in this window for a full recovery read.',
  },
  INSUFFICIENT_DATA: {
    label: 'Insufficient data',
    why: 'Not enough valid wearable data to score reliably.',
  },
}

const HEADER_TIPS: Record<string, string> = {
  advisor:
    'A conversational explainer for this score. It answers only from the BAP research corpus and this patient’s own engine output — it does not diagnose, prescribe, or interpret health. Internal review prototype.',
  bap: 'Balanced Activity Profile — a 0 to 100 clinical status score of how balanced and healthy your activity looks versus a healthy reference population.',
  confidence:
    'How much of the score’s inputs were available and reliable in this window. Higher means more of the model could see usable data.',
  reliability: 'An internal reliability factor applied when some signals are thin or fallback tiers are used.',
  scenario:
    'Spread scoring mode. Daily aggregates means the wearable only provided day-level totals, not minute-by-minute bout structure.',
  couch: 'Active-then-sit penalty: how much the score is pulled down when active periods are followed by long inactive stretches.',
  window: 'Length of wearable data currently included in the score. Both ends are freely adjustable.',
  pure_tab:
    'Population-normalized clinical view. Compares you to a healthy reference group without personalizing to your own history.',
  trend_tab: 'Personal trend view. Blends the population score with how you are doing versus your own prior baseline.',
  slider:
    'Drag either handle to set the start and end of the analysis window freely. Steps are 12 hours. Minimum width is 12 hours. The score updates live.',
  suggestions:
    'Lifestyle directions the suggestion engine thinks could recover open points. Not medical advice and not a training prescription.',
  annotations:
    'Notes attached to the overall score or an axis (optionally a variable). Add from the panel, drag a range on the chart, or describe one in the Alyf Advisor chat — same store either way.',
  conditions:
    'Reported conditions shown as flags only. They do not change the numeric BAP score in this dashboard.',
  rolling:
    'Rolling 14-day BAP history for each day that falls inside the selected analysis window. Independent of the dual-handle score window length.',
  dropbox:
    'Drop a wearable export (CSV or JSON). The file is validated, scored with the live BAP engine, and the full dashboard rebinds to it.',
}

// Client-side schema helpers (mirrors server aliases for fast feedback)
const DATE_ALIASES = ['date', 'day', 'timestamp']
const SIGNAL_ALIASES = [
  'steps',
  'step_count',
  'activity burn',
  'active calories',
  'calories_active',
  'high activity time',
  'medium activity time',
  'vigorous',
  'moderate',
  'average met',
  'met_level',
  'average hrv',
  'hrv',
  'hrv_rmssd',
  'resting heart rate',
  'resting_hr',
  'rhr',
  'total sleep duration',
  'sleep duration',
  'sleep_duration',
]

// ---------- Helpers ----------
function humanVar(name: string): string {
  if (VAR_LABELS[name]) return VAR_LABELS[name]
  return name
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^\w/, (c) => c.toUpperCase())
}

/** True when MET is display-only (interday daily-average excluded from scoring). */
function metExcludedFromScore(b: Bundle | null | undefined): boolean {
  if (!b) return false
  if (typeof b.met_scoring_eligible === 'boolean') return !b.met_scoring_eligible
  return !!b.drop_met_from_intensity
}

function metVarLabel(b: Bundle | null | undefined): string {
  return metExcludedFromScore(b)
    ? VAR_LABELS.met_level_excluded || 'Average metabolic level (not used in score)'
    : VAR_LABELS.met_level || 'Average metabolic level'
}

function metVarTip(b: Bundle | null | undefined): string {
  if (metExcludedFromScore(b)) {
    return (
      b?.met_excluded_reason ||
      VAR_TIPS.met_level_excluded ||
      'Not used in score — daily-average MET lacks intraday resolution.'
    )
  }
  return VAR_TIPS.met_level || 'Metabolic equivalent level.'
}

function pct(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return Number(v).toFixed(digits)
}

function scoreBand(bap: number): { label: string; color: string } {
  if (bap >= 85) return { label: 'Excellent', color: '#059669' }
  if (bap >= 65) return { label: 'Strong', color: '#22c55e' }
  if (bap >= 50) return { label: 'Moderate', color: '#d97706' }
  if (bap >= 35) return { label: 'Limited', color: '#f97316' }
  return { label: 'Low', color: '#dc2626' }
}

function uid(): string {
  return `ann_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

/** Map half-tick pair → inclusive day indices for scoring (min 1 day / 12h). */
function ticksToDayRange(
  startTick: number,
  endTick: number,
  nDays: number,
): { start: number; end: number } {
  const maxTick = nDays * 2
  let a = clamp(Math.min(startTick, endTick), 0, maxTick)
  let b = clamp(Math.max(startTick, endTick), 0, maxTick)
  if (b - a < 1) b = Math.min(maxTick, a + 1)
  let start = Math.floor(a / 2)
  let end = Math.ceil(b / 2) - 1
  if (end < start) end = start
  start = clamp(start, 0, nDays - 1)
  end = clamp(end, 0, nDays - 1)
  return { start, end }
}

function dayRangeToTicks(start: number, end: number, nDays: number): { lo: number; hi: number } {
  const lo = clamp(start, 0, nDays - 1) * 2
  const hi = (clamp(end, 0, nDays - 1) + 1) * 2
  return { lo, hi: Math.max(lo + 1, hi) }
}

function formatTickLabel(tick: number, dates: string[]): string {
  const n = dates.length
  if (tick <= 0) return `${dates[0]} 00:00`
  if (tick >= n * 2) return `${dates[n - 1]} 24:00`
  const day = Math.floor(tick / 2)
  const half = tick % 2
  return `${dates[day]} ${half === 0 ? '00:00' : '12:00'}`
}

function resolveAnnTarget(a: Annotation): string {
  if (a.target_type === 'score') return 'Overall BAP score'
  const axisName = AXIS_META[a.axis_key || a.target_key || '']?.name || a.target_key
  if (a.variable) return `${axisName} · ${humanVar(a.variable)}`
  return `${axisName} axis`
}

function resolveProposalTarget(p: AnnotationProposal): string {
  if (p.target_type !== 'axis') return 'Overall BAP score'
  const axisName = AXIS_META[p.axis_key || '']?.name || p.axis_key || 'Axis'
  if (p.variable) return `${axisName} · ${humanVar(p.variable)}`
  return `${axisName} axis`
}

/** Inline editable preview for a conversational annotation draft. */
function AnnotationProposalCard({
  msg,
  dates,
  varOptionsFor,
  onChange,
  onAccept,
  onCancel,
  busy,
  compact,
}: {
  msg: AdvisorMsg
  dates: string[]
  varOptionsFor: (axisKey: string) => { value: string; label: string }[]
  onChange: (next: AnnotationProposal) => void
  onAccept: () => void
  onCancel: () => void
  busy?: boolean
  /** Tighter summary when follow-up chips share the footer. */
  compact?: boolean
}) {
  const p = msg.annotationProposal
  if (!p) return null
  const ui = msg.annotationUi || 'pending'
  if (ui === 'cancelled') {
    return (
      <div className="adv-ann-card cancelled">
        <p className="adv-ann-status">Annotation discarded — nothing was saved.</p>
      </div>
    )
  }
  if (ui === 'saved') {
    return (
      <div className="adv-ann-card saved">
        <p className="adv-ann-status">
          Saved to your annotations (panel + graph marker)
          {msg.annotationSavedId ? '.' : '.'}
        </p>
      </div>
    )
  }
  if (p.needs_clarification || p.status === 'clarify' || p.status === 'error') {
    return null // clarifier text is already in the bubble
  }
  const editing = ui === 'editing'
  const targetType: AnnTargetType = p.target_type === 'axis' ? 'axis' : 'score'
  const axisKey = p.axis_key || 'R'
  const minDate = dates[0] || ''
  const maxDate = dates[dates.length - 1] || ''
  const vars = varOptionsFor(axisKey)

  const set = (patch: Partial<AnnotationProposal>) => {
    onChange({ ...p, ...patch, status: 'proposal', needs_clarification: false })
  }

  return (
    <div
      className={`adv-ann-card${compact && !editing ? ' compact' : ''}`}
      role="group"
      aria-label="Annotation preview"
    >
      <div className="adv-ann-card-head">
        <span className="adv-ann-kicker">
          {p.mode === 'offer' ? 'Timeline annotation draft' : 'Annotation preview'}
        </span>
        <span className="adv-ann-source">via chat · not saved until you accept</span>
      </div>

      {editing ? (
        <div className="adv-ann-fields">
          <label>
            Target
            <select
              value={targetType}
              onChange={(e) => {
                const t = e.target.value === 'axis' ? 'axis' : 'score'
                set({
                  target_type: t,
                  axis_key: t === 'axis' ? axisKey : null,
                  variable: t === 'axis' ? p.variable : null,
                })
              }}
            >
              <option value="score">Overall BAP score</option>
              <option value="axis">Axis</option>
            </select>
          </label>
          {targetType === 'axis' ? (
            <>
              <label>
                Axis
                <select
                  value={axisKey}
                  onChange={(e) => set({ axis_key: e.target.value, variable: null })}
                >
                  {Object.entries(AXIS_META).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Variable (optional)
                <select
                  value={p.variable || ''}
                  onChange={(e) => set({ variable: e.target.value || null })}
                >
                  <option value="">Whole axis</option>
                  {vars.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : null}
          <label>
            Title
            <input
              type="text"
              value={p.title || ''}
              maxLength={80}
              onChange={(e) => set({ title: e.target.value })}
            />
          </label>
          <label>
            Description
            <textarea
              value={p.description || ''}
              rows={3}
              onChange={(e) => set({ description: e.target.value })}
            />
          </label>
          <div className="adv-ann-dates">
            <label>
              Start
              <input
                type="date"
                value={p.start_date || minDate}
                min={minDate}
                max={maxDate}
                onChange={(e) => set({ start_date: e.target.value })}
              />
            </label>
            <label>
              End
              <input
                type="date"
                value={p.end_date || maxDate}
                min={minDate}
                max={maxDate}
                onChange={(e) => set({ end_date: e.target.value })}
              />
            </label>
          </div>
          {minDate && maxDate ? (
            <p className="adv-ann-hint">
              Data range: {minDate} → {maxDate}
            </p>
          ) : null}
        </div>
      ) : compact ? (
        <div className="adv-ann-summary compact-line">
          <span className="adv-ann-compact-title">{p.title || 'Timeline note'}</span>
          <span className="adv-ann-compact-meta">
            {resolveProposalTarget(p)} · {p.start_date} → {p.end_date}
          </span>
        </div>
      ) : (
        <div className="adv-ann-summary">
          <div>
            <span className="adv-ann-label">Target</span>
            <span>{resolveProposalTarget(p)}</span>
          </div>
          <div>
            <span className="adv-ann-label">Dates</span>
            <span>
              {p.start_date} → {p.end_date}
            </span>
          </div>
          <div>
            <span className="adv-ann-label">Title</span>
            <span>{p.title}</span>
          </div>
          <div>
            <span className="adv-ann-label">Note</span>
            <span className="adv-ann-desc">{p.description}</span>
          </div>
        </div>
      )}

      <div className="adv-ann-actions">
        <button
          type="button"
          className={`btn sm ${editing ? '' : 'primary'}`}
          onClick={() => onAccept()}
          disabled={busy}
        >
          {editing ? 'Save annotation' : 'Accept'}
        </button>
        {!editing ? (
          <button
            type="button"
            className="btn sm ghost"
            onClick={() => onChange({ ...p })}
            disabled={busy}
          >
            Edit
          </button>
        ) : null}
        <button type="button" className="btn sm ghost" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
    </div>
  )
}

function PortraitSectionCard({
  section,
  expanded,
  onToggle,
  itemsById,
  editId,
  editText,
  setEditId,
  setEditText,
  onCorrect,
  onRemove,
  onMute,
  clinician,
  onClinicianCorrect,
  onClinicianDismiss,
  onClinicianRestore,
  busy,
}: {
  section: MemoryPortraitSection
  expanded: boolean
  onToggle: () => void
  itemsById: Record<string, MemoryItem>
  editId: string | null
  editText: string
  setEditId: (id: string | null) => void
  setEditText: (t: string) => void
  onCorrect?: (id: string, text: string) => void
  onRemove?: (id: string) => void
  onMute?: (id: string) => void
  clinician?: boolean
  onClinicianCorrect?: (id: string, text: string) => void
  onClinicianDismiss?: (id: string) => void
  onClinicianRestore?: (id: string) => void
  busy?: boolean
}) {
  const ids = section.item_ids || []
  const evidence = section.evidence || []
  return (
    <div className={'mem-section' + (expanded ? ' open' : '')}>
      <button type="button" className="mem-section-head" onClick={onToggle}>
        <div>
          <div className="mem-section-title">{section.title}</div>
          <p className="mem-section-prose">{section.prose}</p>
        </div>
        <span className="mem-section-chevron" aria-hidden>
          {expanded ? '▾' : '▸'}
        </span>
      </button>
      {expanded ? (
        <div className="mem-section-body">
          <div className="mem-evidence-label">Why Alyf believes this</div>
          {evidence.length === 0 ? (
            <p className="mem-prov">No detailed evidence stored for this section.</p>
          ) : (
            evidence.map((ev, i) => {
              const id = ev.item_id || ids[i] || ''
              const it = id ? itemsById[id] : undefined
              const editing = editId === (clinician ? 'c:' + id : id)
              return (
                <div key={id || String(i)} className="mem-evidence">
                  {editing ? (
                    <textarea
                      rows={3}
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                    />
                  ) : (
                    <p className="mem-evidence-text">{ev.text || it?.text || '—'}</p>
                  )}
                  <p className="mem-prov">
                    {ev.provenance_summary || it?.provenance?.summary || 'Data / conversation'}
                    {ev.data_window || it?.provenance?.data_window
                      ? ` · ${ev.data_window || it?.provenance?.data_window}`
                      : ''}
                  </p>
                  <p className="mem-prov">
                    Learned{' '}
                    {ev.learned_at
                      ? new Date(ev.learned_at).toLocaleDateString()
                      : it?.learned_at
                        ? new Date(it.learned_at).toLocaleDateString()
                        : '—'}
                    {' · '}
                    Reinforced{' '}
                    {ev.last_reinforced_at
                      ? new Date(ev.last_reinforced_at).toLocaleDateString()
                      : it?.last_reinforced_at
                        ? new Date(it.last_reinforced_at).toLocaleDateString()
                        : '—'}
                    {ev.type || it?.type ? ` · ${ev.type || it?.type}` : ''}
                  </p>
                  <div className="mem-item-actions">
                    {editing ? (
                      <>
                        <button
                          type="button"
                          className="btn sm primary"
                          disabled={busy}
                          onClick={() => {
                            if (clinician) onClinicianCorrect?.(id, editText)
                            else onCorrect?.(id, editText)
                            setEditId(null)
                            setEditText('')
                          }}
                        >
                          Save
                        </button>
                        <button
                          type="button"
                          className="btn sm ghost"
                          onClick={() => {
                            setEditId(null)
                            setEditText('')
                          }}
                        >
                          Cancel
                        </button>
                      </>
                    ) : clinician ? (
                      <>
                        <button
                          type="button"
                          className="btn sm ghost"
                          onClick={() => {
                            setEditId('c:' + id)
                            setEditText(ev.text || it?.text || '')
                          }}
                        >
                          Correct
                        </button>
                        {it && (it.status || 'active') === 'active' && !it.dismissed ? (
                          <button
                            type="button"
                            className="btn sm ghost"
                            onClick={() => onClinicianDismiss?.(id)}
                          >
                            Dismiss
                          </button>
                        ) : it ? (
                          <button
                            type="button"
                            className="btn sm ghost"
                            onClick={() => onClinicianRestore?.(id)}
                          >
                            Restore
                          </button>
                        ) : null}
                      </>
                    ) : id ? (
                      <>
                        <button
                          type="button"
                          className="btn sm ghost"
                          disabled={busy}
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            setEditId(id)
                            setEditText(ev.text || it?.text || '')
                          }}
                        >
                          Correct this
                        </button>
                        <button
                          type="button"
                          className="btn sm ghost"
                          disabled={busy}
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            onRemove?.(id)
                          }}
                        >
                          This isn&apos;t right
                        </button>
                        <button
                          type="button"
                          className="btn sm ghost"
                          disabled={busy}
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            onMute?.(id)
                          }}
                        >
                          Don&apos;t use this
                        </button>
                      </>
                    ) : null}
                  </div>
                </div>
              )
            })
          )}
        </div>
      ) : null}
    </div>
  )
}

function MemoryProposalCard({
  msg,
  onChange,
  onAccept,
  onCancel,
  onEdit,
  busy,
}: {
  msg: AdvisorMsg
  onChange: (next: MemoryProposal) => void
  onAccept: () => void
  onCancel: () => void
  onEdit: () => void
  busy?: boolean
}) {
  const p = msg.memoryProposal
  if (!p) return null
  if (msg.memoryUi === 'cancelled') return null
  if (msg.memoryUi === 'saved') {
    return (
      <div className="mem-proposal-card saved">
        <div className="mem-proposal-title">Profile updated</div>
        <p className="mem-proposal-note">Saved to your personalization memory.</p>
      </div>
    )
  }
  const editing = msg.memoryUi === 'editing'
  const set = (patch: Partial<MemoryProposal>) => onChange({ ...p, ...patch })
  return (
    <div className="mem-proposal-card">
      <div className="mem-proposal-title">Personalization update</div>
      <p className="mem-proposal-note">
        Not a clinical record. Nothing is written until you accept.
      </p>
      {editing ? (
        <div className="mem-proposal-edit">
          <label>
            Action
            <select
              value={p.action || 'add'}
              onChange={(e) => set({ action: e.target.value })}
              disabled={busy}
            >
              <option value="add">Add</option>
              <option value="edit">Edit</option>
              <option value="delete">Delete</option>
              <option value="set_learning">Learning on/off</option>
              <option value="clear">Clear all</option>
            </select>
          </label>
          {(p.action === 'add' || p.action === 'edit') && (
            <>
              <label>
                Type
                <select
                  value={p.item_type || 'stated'}
                  onChange={(e) => set({ item_type: e.target.value })}
                  disabled={busy}
                >
                  <option value="behavioral_profile">Behavioral profile</option>
                  <option value="preference">Preference</option>
                  <option value="stated">Stated context</option>
                  <option value="data_pattern">Data pattern</option>
                </select>
              </label>
              <label>
                Text
                <textarea
                  rows={3}
                  value={p.item_text || p.preview_text || ''}
                  onChange={(e) =>
                    set({ item_text: e.target.value, preview_text: e.target.value })
                  }
                  disabled={busy}
                />
              </label>
            </>
          )}
          {p.action === 'set_learning' && (
            <label className="mem-check">
              <input
                type="checkbox"
                checked={!!p.enabled}
                onChange={(e) => set({ enabled: e.target.checked })}
                disabled={busy}
              />
              Learning enabled
            </label>
          )}
        </div>
      ) : (
        <div className="mem-proposal-preview">
          <div>
            <span className="mem-pill">{p.action || 'update'}</span>
            {p.item_type ? <span className="mem-pill muted">{p.item_type}</span> : null}
          </div>
          <p>{p.preview_text || p.item_text || '—'}</p>
        </div>
      )}
      <div className="mem-proposal-actions">
        {!editing ? (
          <>
            <button type="button" className="btn sm primary" disabled={busy} onClick={onAccept}>
              Accept
            </button>
            <button type="button" className="btn sm ghost" disabled={busy} onClick={onEdit}>
              Edit
            </button>
            <button type="button" className="btn sm ghost" disabled={busy} onClick={onCancel}>
              Cancel
            </button>
          </>
        ) : (
          <>
            <button type="button" className="btn sm primary" disabled={busy} onClick={onAccept}>
              Save
            </button>
            <button type="button" className="btn sm ghost" disabled={busy} onClick={onCancel}>
              Cancel
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function resolveAnnTime(a: Annotation): string {
  const st = a.start_time || '00:00'
  const et = a.end_time || '24:00'
  if (a.start_date === a.end_date) return `${a.start_date} ${st} → ${et}`
  return `${a.start_date} ${st} → ${a.end_date} ${et}`
}

/** Convert clientX inside chart shell to point index along the x axis. */
function clientXToPointIndex(
  clientX: number,
  shellEl: HTMLElement,
  nPoints: number,
): number {
  if (nPoints <= 1) return 0
  const r = shellEl.getBoundingClientRect()
  const left = r.left + CHART_ML
  const width = Math.max(1, r.width - CHART_ML - CHART_MR)
  const x = clamp((clientX - left) / width, 0, 1)
  return clamp(Math.round(x * (nPoints - 1)), 0, nPoints - 1)
}

function pointIndexToLeftPct(i: number, nPoints: number): number {
  if (nPoints <= 1) return 0
  return (clamp(i, 0, nPoints - 1) / (nPoints - 1)) * 100
}

function normHeader(h: string): string {
  return h.trim().toLowerCase().replace(/\s+/g, ' ')
}

function parseCsvClient(text: string): { headers: string[]; rows: Record<string, string>[] } {
  let src = text
  if (src.charCodeAt(0) === 0xfeff) src = src.slice(1)
  const lines = src.split(/\r?\n/).filter((l) => l.trim().length > 0)
  if (lines.length < 2) {
    throw new Error('CSV needs a header row and at least one data row.')
  }
  // simple CSV split respecting quotes
  const splitLine = (line: string): string[] => {
    const out: string[] = []
    let cur = ''
    let inQ = false
    for (let i = 0; i < line.length; i++) {
      const ch = line[i]
      if (inQ) {
        if (ch === '"') {
          if (line[i + 1] === '"') {
            cur += '"'
            i++
          } else {
            inQ = false
          }
        } else {
          cur += ch
        }
      } else if (ch === '"') {
        inQ = true
      } else if (ch === ',' || ch === '\t' || ch === ';') {
        out.push(cur)
        cur = ''
      } else {
        cur += ch
      }
    }
    out.push(cur)
    return out.map((s) => s.trim())
  }
  const headers = splitLine(lines[0])
  if (headers.length < 2) {
    throw new Error('CSV header row looks empty. Expected columns such as date, Steps, Average HRV.')
  }
  const rows: Record<string, string>[] = []
  for (let li = 1; li < lines.length; li++) {
    const cols = splitLine(lines[li])
    if (cols.every((c) => !c)) continue
    const row: Record<string, string> = {}
    headers.forEach((h, i) => {
      row[h] = cols[i] ?? ''
    })
    rows.push(row)
  }
  if (!rows.length) throw new Error('CSV has a header but no data rows.')
  if (rows.length > 5000) throw new Error('File has more than 5,000 data rows. Please upload a smaller export.')
  return { headers, rows }
}

function parseJsonClient(text: string): { headers: string[]; rows: Record<string, unknown>[] } {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch (e) {
    const msg = e instanceof Error ? e.message : 'Invalid JSON'
    throw new Error(`Invalid JSON: ${msg}`)
  }
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>
    const key = ['rows', 'data', 'days', 'records', 'trends'].find((k) => Array.isArray(obj[k]))
    if (!key) {
      throw new Error(
        'JSON object must contain an array under one of: rows, data, days, records, trends — or be a top-level array.',
      )
    }
    data = obj[key]
  }
  if (!Array.isArray(data)) {
    throw new Error('JSON must be an array of day objects.')
  }
  if (!data.length) throw new Error('JSON array is empty.')
  if (!data.every((r) => r && typeof r === 'object' && !Array.isArray(r))) {
    throw new Error('JSON array items must be objects with column keys.')
  }
  const rows = data as Record<string, unknown>[]
  const headers: string[] = []
  const seen = new Set<string>()
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k)) {
        seen.add(k)
        headers.push(k)
      }
    }
  }
  return { headers, rows }
}

function validateClientSchema(headers: string[]): { dateCol: string; signals: string[] } {
  const norms = headers.map((h) => ({ raw: h, n: normHeader(h) }))
  const dateCol = norms.find((h) => DATE_ALIASES.some((a) => h.n === a || h.n.includes(a)))?.raw
  if (!dateCol) {
    throw new Error(
      `Missing required column: date. Found: ${headers.slice(0, 20).join(', ')}${headers.length > 20 ? '…' : ''}. Expected a date/day column.`,
    )
  }
  const signals = norms
    .filter((h) => SIGNAL_ALIASES.some((a) => h.n.includes(a)))
    .map((h) => h.raw)
  if (!signals.length) {
    throw new Error(
      `Could not find any activity or recovery columns. Need at least one of: Steps, Activity Burn, High/Medium Activity Time, Average MET, Average HRV, Resting Heart Rate, or Sleep Duration. Found: ${headers.slice(0, 25).join(', ')}${headers.length > 25 ? '…' : ''}.`,
    )
  }
  return { dateCol, signals }
}

function peekDateRange(
  rows: Record<string, unknown>[],
  dateCol: string,
): { start: string; end: string; count: number } {
  const dates: string[] = []
  for (const r of rows) {
    const v = r[dateCol]
    if (v == null || String(v).trim() === '') continue
    const s = String(v).trim().slice(0, 10)
    // accept ISO-ish
    if (/^\d{4}-\d{2}-\d{2}/.test(s) || /^\d{1,2}\/\d{1,2}\/\d{2,4}/.test(String(v))) {
      dates.push(s)
    } else {
      dates.push(String(v).slice(0, 16))
    }
  }
  if (!dates.length) {
    throw new Error(
      `Could not read any values from the date column “${dateCol}”. Expected formats like YYYY-MM-DD.`,
    )
  }
  const sorted = [...dates].sort()
  return { start: sorted[0], end: sorted[sorted.length - 1], count: dates.length }
}

// ---------- Icons (inline SVG, consistent set) ----------
function IconMoon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M21 14.5A8.5 8.5 0 1 1 9.5 3a7 7 0 0 0 11.5 11.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
function IconSun() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  )
}
function IconRefresh() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
function IconNote() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M8 4h8a2 2 0 0 1 2 2v14l-4-2-4 2-4-2-4 2V6a2 2 0 0 1 2-2Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path d="M9 9h6M9 13h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}
function IconUpload() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 16V6M8 9l4-4 4 4M5 18h14"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
function IconSearch() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.8" />
      <path d="M16 16l4 4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}
function IconClose() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}
/**
 * Minimal renderer for the Advisor's replies: paragraphs, bullet lists,
 * blockquotes (the engine's verbatim packaged text) and **bold** spans.
 * Deliberately not a general markdown engine — the Advisor emits plain prose.
 */
function AdvisorText({ text }: { text: string }) {
  const blocks: ReactNode[] = []
  const lines = text.split('\n')
  let para: string[] = []
  let bullets: string[] = []
  let quote: string[] = []

  const renderInline = (s: string, key: string): ReactNode => (
    <span key={key}>
      {s.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*)/g).map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
          return <strong key={i}>{part.slice(2, -2)}</strong>
        }
        if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
          return <code key={i}>{part.slice(1, -1)}</code>
        }
        if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
          return <em key={i}>{part.slice(1, -1)}</em>
        }
        return <span key={i}>{part}</span>
      })}
    </span>
  )

  const flushPara = () => {
    if (para.length) {
      blocks.push(<p key={`p${blocks.length}`}>{renderInline(para.join(' '), 'i')}</p>)
      para = []
    }
  }
  const flushBullets = () => {
    if (bullets.length) {
      blocks.push(
        <ul key={`u${blocks.length}`}>
          {bullets.map((b, i) => (
            <li key={i}>{renderInline(b, `b${i}`)}</li>
          ))}
        </ul>,
      )
      bullets = []
    }
  }
  const flushQuote = () => {
    if (quote.length) {
      blocks.push(
        <blockquote key={`q${blocks.length}`}>{renderInline(quote.join(' '), 'q')}</blockquote>,
      )
      quote = []
    }
  }

  for (const raw of lines) {
    const ln = raw.trim()
    if (!ln) {
      flushPara()
      flushBullets()
      flushQuote()
      continue
    }
    if (/^#{1,4}\s+/.test(ln)) {
      flushPara()
      flushBullets()
      flushQuote()
      blocks.push(
        <h4 key={`h${blocks.length}`} className="adv-h">
          {renderInline(ln.replace(/^#{1,4}\s+/, ''), 'h')}
        </h4>,
      )
    } else if (ln.startsWith('> ')) {
      flushPara()
      flushBullets()
      quote.push(ln.slice(2))
    } else if (/^\d+[.)]\s+/.test(ln)) {
      flushPara()
      flushQuote()
      bullets.push(ln.replace(/^\d+[.)]\s+/, ''))
    } else if (/^[-*]\s+/.test(ln)) {
      flushPara()
      flushQuote()
      bullets.push(ln.replace(/^[-*]\s+/, ''))
    } else {
      flushBullets()
      flushQuote()
      para.push(ln)
    }
  }
  flushPara()
  flushBullets()
  flushQuote()

  return <div className="adv-body">{blocks}</div>
}

/** Compact chip showing one of the patient's own axis values. */
function AxisPill({ axisKey, value }: { axisKey: string; value: number | null }) {
  const meta = AXIS_META[axisKey]
  if (!meta || value == null) return null
  return (
    <span className="adv-pill" style={{ borderColor: meta.color }}>
      <span className="adv-pill-dot" style={{ background: meta.color }} aria-hidden />
      {meta.name} {(value * 100).toFixed(0)}
    </span>
  )
}

/**
 * Dependency-free sparkline over the patient's own rolling score series. Kept
 * deliberately small and self-contained so richer in-chat charts can be dropped
 * in later without reworking the message layout.
 */
function Sparkline({ values }: { values: (number | null)[] }) {
  const present = values.filter((v): v is number => v != null)
  if (present.length < 3) return null
  const W = 104
  const H = 28
  const P = 3
  const min = Math.min(...present)
  const max = Math.max(...present)
  const span = max - min || 1
  const coords: string[] = []
  values.forEach((v, i) => {
    if (v == null) return
    const x = P + (i / Math.max(1, values.length - 1)) * (W - P * 2)
    const y = H - P - ((v - min) / span) * (H - P * 2)
    coords.push(`${x.toFixed(1)},${y.toFixed(1)}`)
  })
  const last = (coords[coords.length - 1] || '').split(',')
  const label = `Your rolling score over this window, ${min.toFixed(1)} to ${max.toFixed(1)}`
  return (
    <span className="adv-spark">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        <polyline
          points={coords.join(' ')}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="1.6"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {last.length === 2 ? <circle cx={last[0]} cy={last[1]} r="2.4" fill="var(--accent)" /> : null}
      </svg>
    </span>
  )
}

/**
 * Which of the patient's own numbers a reply refers to. This only decides which
 * chips/sparkline to surface from data already on screen — it never inspects,
 * filters or alters what the Advisor said.
 */
function advisorDataRefs(text: string): { axes: string[]; trend: boolean } {
  const axes = AXIS_WORDS.filter(([word]) => new RegExp(`\\b${word}\\b`, 'i').test(text)).map(
    ([, key]) => key,
  )
  return { axes, trend: TREND_RE.test(text) }
}

function AdvisorDataStrip({
  refs,
  axisScores,
  bap,
  trendValues,
}: {
  refs: { axes: string[]; trend: boolean }
  axisScores: Record<string, number | null> | null
  bap: number | null
  trendValues: (number | null)[]
}) {
  const pills = refs.axes.slice(0, 3)
  const showSpark = refs.trend && trendValues.filter((v) => v != null).length >= 3
  if (pills.length === 0 && !showSpark) return null
  return (
    <div className="adv-data">
      {showSpark && bap != null ? (
        <span className="adv-pill strong">Score {bap.toFixed(1)}</span>
      ) : null}
      {pills.map((k) => (
        <AxisPill key={k} axisKey={k} value={axisScores?.[k] ?? null} />
      ))}
      {showSpark ? <Sparkline values={trendValues} /> : null}
    </div>
  )
}

/** Reduced-motion preference — when set, replies appear whole instead of typing out. */
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(mq.matches)
    const onChange = () => setReduced(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])
  return reduced
}

function IconHistory() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1M3.5 5v4h4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M12 8v4.5l3 1.8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

function IconPlus() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
    </svg>
  )
}

function IconTrash() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function IconStop() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" />
    </svg>
  )
}

function IconChat() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M21 12a8 8 0 0 1-8 8H8l-4 3v-5.5A8 8 0 0 1 11 4h2a8 8 0 0 1 8 8Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path d="M9 11h6M9 14.5h3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  )
}
function IconSend() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 12 20 4l-7 16-2.2-6.2L4 12Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
    </svg>
  )
}
function IconFile() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M7 3h7l5 5v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path d="M14 3v5h5" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
    </svg>
  )
}

// ---------- Edge-aware tooltip ----------
function Tip({ tip, children }: { tip: string; children: ReactNode }) {
  const anchorRef = useRef<HTMLSpanElement>(null)
  const bubbleRef = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number }>({ left: 0, top: 0 })

  const place = useCallback(() => {
    const el = anchorRef.current
    const bubble = bubbleRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const pad = 8
    const maxW = Math.min(280, window.innerWidth - 16)
    const bh = bubble?.offsetHeight || Math.min(160, 24 + Math.ceil(tip.length / 42) * 16)
    const bw = bubble?.offsetWidth || maxW
    let left = r.left + r.width / 2 - bw / 2
    left = clamp(left, pad, window.innerWidth - bw - pad)
    let top = r.top - bh - 10
    if (top < pad) top = r.bottom + 10
    if (top + bh > window.innerHeight - pad) {
      top = clamp(r.top - bh - 10, pad, window.innerHeight - bh - pad)
    }
    setPos({ left, top })
  }, [tip])

  useLayoutEffect(() => {
    if (!open) return
    place()
    const onMove = () => place()
    window.addEventListener('scroll', onMove, true)
    window.addEventListener('resize', onMove)
    return () => {
      window.removeEventListener('scroll', onMove, true)
      window.removeEventListener('resize', onMove)
    }
  }, [open, place])

  return (
    <>
      <span
        ref={anchorRef}
        className="tip-anchor"
        tabIndex={0}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        {children}
      </span>
      {open
        ? createPortal(
            <div
              ref={bubbleRef}
              className="tip-portal"
              role="tooltip"
              style={{ left: pos.left, top: pos.top, maxWidth: Math.min(280, window.innerWidth - 16) }}
            >
              {tip}
            </div>,
            document.body,
          )
        : null}
    </>
  )
}

// ---------- Chart annotation card (hover peek / pinned) ----------
/** Presentation-only card for timeline markers. Edit/Delete call existing handlers. */
function ChartAnnCard({
  ann,
  anchorLeft,
  mode,
  shellRef,
  onEdit,
  onDelete,
  onDismiss,
  showActions,
}: {
  ann: Annotation
  anchorLeft: string
  mode: 'hover' | 'pinned'
  shellRef: RefObject<HTMLDivElement | null>
  onEdit: () => void
  onDelete: () => void
  onDismiss: () => void
  showActions: boolean
}) {
  const cardRef = useRef<HTMLDivElement>(null)
  const [shiftX, setShiftX] = useState(0)
  const [placeAbove, setPlaceAbove] = useState(true)

  useLayoutEffect(() => {
    const shell = shellRef.current
    const card = cardRef.current
    if (!shell || !card) return
    const sr = shell.getBoundingClientRect()
    const cr = card.getBoundingClientRect()
    const pad = 8
    let dx = 0
    if (cr.left < sr.left + pad) dx = sr.left + pad - cr.left
    if (cr.right > sr.right - pad) dx = sr.right - pad - cr.right
    setShiftX(dx)
    // Prefer above the lane; flip below if clipped at top of viewport
    const aboveTop = cr.top
    setPlaceAbove(aboveTop >= pad)
  }, [ann.id, anchorLeft, mode, shellRef])

  useEffect(() => {
    if (mode !== 'pinned') return
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node
      if (cardRef.current?.contains(t)) return
      if ((t as HTMLElement)?.closest?.('.chart-ann-pin')) return
      onDismiss()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onDismiss()
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [mode, onDismiss])

  const desc = (ann.text || '').trim()
  const preview =
    desc.length > 140 ? desc.slice(0, 137).trimEnd() + '…' : desc
  const via = ann.source === 'patient_via_advisor'

  return (
    <div
      ref={cardRef}
      className={`chart-ann-card ${mode} ${placeAbove ? 'above' : 'below'} tgt-${ann.target_type === 'score' ? 'score' : 'axis'}`}
      style={{
        left: anchorLeft,
        transform: `translate(calc(-50% + ${shiftX}px), 0)`,
      }}
      role={mode === 'pinned' ? 'dialog' : 'tooltip'}
      aria-label={ann.label || 'Annotation'}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="chart-ann-card-head">
        <span className="chart-ann-card-title">{ann.label || resolveAnnTarget(ann)}</span>
        {via ? (
          <span className="chart-ann-card-src" title="Created via chat">
            ✦ chat
          </span>
        ) : null}
        {mode === 'pinned' ? (
          <button
            type="button"
            className="chart-ann-card-x"
            aria-label="Dismiss"
            onClick={onDismiss}
          >
            ×
          </button>
        ) : null}
      </div>
      <div className="chart-ann-card-meta">
        <span>{resolveAnnTarget(ann)}</span>
        <span className="sep">·</span>
        <span>{resolveAnnTime(ann)}</span>
      </div>
      {preview ? <p className="chart-ann-card-desc">{preview}</p> : null}
      {showActions ? (
        <div className="chart-ann-card-actions">
          <button type="button" className="btn sm ghost" onClick={onEdit}>
            Edit
          </button>
          <button type="button" className="btn sm ghost danger" onClick={onDelete}>
            Delete
          </button>
        </div>
      ) : null}
    </div>
  )
}

// ---------- Dual range slider ----------
function DualRangeSlider({
  minTick,
  maxTick,
  lo,
  hi,
  onChange,
  formatLabel,
}: {
  minTick: number
  maxTick: number
  lo: number
  hi: number
  onChange: (lo: number, hi: number) => void
  formatLabel: (t: number) => string
}) {
  const trackRef = useRef<HTMLDivElement>(null)
  const drag = useRef<'lo' | 'hi' | null>(null)

  const span = Math.max(1, maxTick - minTick)
  const pctOf = (t: number) => ((t - minTick) / span) * 100

  const tickFromClientX = (clientX: number) => {
    const el = trackRef.current
    if (!el) return lo
    const r = el.getBoundingClientRect()
    const x = clamp((clientX - r.left) / Math.max(1, r.width), 0, 1)
    return Math.round(minTick + x * span)
  }

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (!drag.current) return
      const t = tickFromClientX(e.clientX)
      if (drag.current === 'lo') {
        const nextLo = clamp(t, minTick, hi - 1)
        onChange(nextLo, hi)
      } else {
        const nextHi = clamp(t, lo + 1, maxTick)
        onChange(lo, nextHi)
      }
    }
    const onUp = () => {
      drag.current = null
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  })

  return (
    <div className="range-wrap">
      <div className="range-track" ref={trackRef}>
        <div className="range-rail" />
        <div
          className="range-fill"
          style={{ left: `${pctOf(lo)}%`, width: `${pctOf(hi) - pctOf(lo)}%` }}
        />
        <div
          className="range-thumb"
          style={{ left: `${pctOf(lo)}%` }}
          role="slider"
          tabIndex={0}
          aria-label="Window start"
          aria-valuemin={minTick}
          aria-valuemax={hi - 1}
          aria-valuenow={lo}
          onPointerDown={(e) => {
            e.preventDefault()
            drag.current = 'lo'
            ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') onChange(clamp(lo - 1, minTick, hi - 1), hi)
            if (e.key === 'ArrowRight' || e.key === 'ArrowUp') onChange(clamp(lo + 1, minTick, hi - 1), hi)
          }}
        />
        <div
          className="range-thumb"
          style={{ left: `${pctOf(hi)}%` }}
          role="slider"
          tabIndex={0}
          aria-label="Window end"
          aria-valuemin={lo + 1}
          aria-valuemax={maxTick}
          aria-valuenow={hi}
          onPointerDown={(e) => {
            e.preventDefault()
            drag.current = 'hi'
            ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') onChange(lo, clamp(hi - 1, lo + 1, maxTick))
            if (e.key === 'ArrowRight' || e.key === 'ArrowUp') onChange(lo, clamp(hi + 1, lo + 1, maxTick))
          }}
        />
      </div>
      <div className="range-ticks">
        <span>{formatLabel(minTick)}</span>
        <span className="range-meta">
          {formatLabel(lo)} → {formatLabel(hi)}
        </span>
        <span>{formatLabel(maxTick)}</span>
      </div>
    </div>
  )
}

function ScoreRing({ value, color }: { value: number; color: string }) {
  const R = 74
  const C = 2 * Math.PI * R
  const frac = Math.max(0, Math.min(1, value / 100))
  return (
    <svg viewBox="0 0 180 180" className="ring" aria-label={`BAP score ${value.toFixed(1)}`}>
      <circle cx="90" cy="90" r={R} className="ring-track" strokeWidth="14" fill="none" />
      <circle
        cx="90"
        cy="90"
        r={R}
        stroke={color}
        strokeWidth="14"
        fill="none"
        strokeLinecap="round"
        strokeDasharray={C}
        strokeDashoffset={C * (1 - frac)}
        transform="rotate(-90 90 90)"
        style={{ transition: 'stroke-dashoffset 220ms cubic-bezier(0.2, 0.8, 0.2, 1)' }}
      />
      <text x="90" y="84" textAnchor="middle" className="ring-value">
        {value.toFixed(1)}
      </text>
      <text x="90" y="106" textAnchor="middle" className="ring-sub">
        out of 100
      </text>
    </svg>
  )
}

// ---------- Data drop box ----------
function DataDropBox({
  info,
  busy,
  error,
  onFile,
  onClear,
  compact,
}: {
  info: UploadInfo | null
  busy: boolean
  error: string | null
  onFile: (file: File) => void
  onClear: () => void
  compact?: boolean
}) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const onDragOver = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }
  const onDragLeave = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onFile(f)
  }
  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) onFile(f)
    e.target.value = ''
  }

  if (info && !busy) {
    return (
      <div className={`drop ${compact ? '' : ''}`} style={{ cursor: 'default' }}>
        <div className="drop-icon" aria-hidden>
          <IconFile />
        </div>
        <p className="drop-title">{info.fileName}</p>
        <p className="drop-hint">
          {info.rowCount} day{info.rowCount === 1 ? '' : 's'} · {info.dateStart} → {info.dateEnd}
        </p>
        <div className="drop-meta">
          <button
            type="button"
            className="btn sm"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
          >
            Replace file
          </button>
          <button type="button" className="btn sm ghost" onClick={onClear} disabled={busy}>
            Clear
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.json,text/csv,application/json"
          onChange={onChange}
          aria-label="Replace wearable data file"
        />
        {error ? <div className="drop-error" role="alert">{error}</div> : null}
      </div>
    )
  }

  return (
    <div
      className={`drop ${dragOver ? 'active' : ''} ${busy ? 'busy' : ''}`}
      onDragOver={onDragOver}
      onDragEnter={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      role="button"
      tabIndex={0}
      aria-label="Upload wearable data file"
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
      onClick={() => !busy && inputRef.current?.click()}
    >
      <div className="drop-icon" aria-hidden>
        <IconUpload />
      </div>
      <p className="drop-title">{busy ? 'Scoring your data…' : 'Drop wearable data here'}</p>
      <p className="drop-hint">
        {busy
          ? 'Parsing, validating, and running the BAP engine. This can take a minute for long exports.'
          : 'CSV or JSON · click to browse · Oura-style trends supported'}
      </p>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.json,text/csv,application/json"
        onChange={onChange}
        disabled={busy}
        aria-label="Choose wearable data file"
      />
      {error ? (
        <div
          className="drop-error"
          role="alert"
          onClick={(e) => e.stopPropagation()}
        >
          {error}
        </div>
      ) : null}
    </div>
  )
}

// ---------- Main ----------
function App() {
  const bundleArt = usePromptQlArtifact<Bundle>('bap_dashboard_bundle')
  const annArt = usePromptQlArtifact<AnnotationsDoc>('bap_dashboard_annotations')
  const advReplyArt = usePromptQlArtifact<AdvisorReplyDoc>('alyf_advisor_reply')
  // Purged, decrypted session view of chat transcripts ONLY (not clinical data).
  const advHistoryArt = usePromptQlArtifact<AdvisorHistoryView>('alyf_advisor_chat_history_view')
  const advMemoryArt = usePromptQlArtifact<MemoryViewDoc>('alyf_advisor_memory_view')
  // Portrait sidecar — refreshed even when the full memory view blob is sticky.
  const advPortraitArt = usePromptQlArtifact<{
    portrait?: MemoryPortrait
    learning_enabled?: boolean
    disclaimer?: string
    portrait_rev?: number
    updated_at_ms?: number
  }>('alyf_advisor_memory_portrait')
  // Items sidecar — authoritative item status when full view blob is sticky.
  const advMemoryItemsArt = usePromptQlArtifact<{
    items?: MemoryItem[]
    learning_enabled?: boolean
    updated_at_ms?: number
  }>('alyf_advisor_memory_items')
  const { run, isRunning } = usePromptQlExecuteProgram()

  const [dark, setDark] = useState(true)
  const [tab, setTab] = useState<TabId>('bap')
  const [loTick, setLoTick] = useState<number | null>(null)
  const [hiTick, setHiTick] = useState<number | null>(null)
  const [drillAxis, setDrillAxis] = useState<string | null>(null)
  const [seriesOn, setSeriesOn] = useState({
    BAP: true,
    Intensity: false,
    Recovery: false,
    Spread: false,
    Consistency: false,
  })

  // Annotations form
  const [annOpen, setAnnOpen] = useState(false)
  const [annTargetType, setAnnTargetType] = useState<AnnTargetType>('score')
  const [annAxisKey, setAnnAxisKey] = useState('C')
  const [annVariable, setAnnVariable] = useState<string>('')
  const [annLabel, setAnnLabel] = useState('')
  const [annText, setAnnText] = useState('')
  const [annStartDate, setAnnStartDate] = useState('')
  const [annEndDate, setAnnEndDate] = useState('')
  const [annStartTime, setAnnStartTime] = useState('00:00')
  const [annEndTime, setAnnEndTime] = useState('24:00')
  const [annStatus, setAnnStatus] = useState<string | null>(null)
  const [localAnns, setLocalAnns] = useState<Annotation[] | null>(null)
  const [expandedAnn, setExpandedAnn] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [dragMode, setDragMode] = useState(false)
  const [dragSel, setDragSel] = useState<{ a: number; b: number } | null>(null)
  const [hoverAnn, setHoverAnn] = useState<string | null>(null)
  /** Chart-pinned annotation card (independent of the side drawer list expand). */
  const [pinnedChartAnn, setPinnedChartAnn] = useState<string | null>(null)
  /** Expanded cluster id on the chart lane (zoom-aware group). */
  const [openAnnCluster, setOpenAnnCluster] = useState<string | null>(null)
  const chartShellRef = useRef<HTMLDivElement>(null)
  const dragActiveRef = useRef(false)

  // Alyf Advisor state. Conversations are session-scoped (the sandboxed iframe
  // has no storage); `advConvos` is the seam a persisted thread list drops into.
  const [advOpen, setAdvOpen] = useState(false)
  const [advConvos, setAdvConvos] = useState<AdvisorConvo[]>([])
  const [advActiveId, setAdvActiveId] = useState<string | null>(null)
  const [advInput, setAdvInput] = useState('')
  const [advBusy, setAdvBusy] = useState(false)
  const [advHistoryOpen, setAdvHistoryOpen] = useState(false)
  /** In-UI confirm dialog for delete-one / clear-all (no window.confirm in sandbox). */
  const [advConfirm, setAdvConfirm] = useState<null | { kind: 'one' | 'all'; id?: string }>(null)
  const [advHistoryReady, setAdvHistoryReady] = useState(false)
  const [advHistoryError, setAdvHistoryError] = useState<string | null>(null)
  const [advMemoryOpen, setAdvMemoryOpen] = useState(false)
  const [advMemoryError, setAdvMemoryError] = useState<string | null>(null)
  const [memClinicianOpen, setMemClinicianOpen] = useState(false)
  const [memEditId, setMemEditId] = useState<string | null>(null)
  /** Local overrides so mute/edit/delete update UI immediately even if view blob is sticky. */
  const [memItemOverrides, setMemItemOverrides] = useState<
    Record<string, Partial<MemoryItem> & { _removed?: boolean }>
  >({})
  const [memExpandedSec, setMemExpandedSec] = useState<Record<string, boolean>>({})
  const [memShowHistory, setMemShowHistory] = useState(false)
  const [memEditText, setMemEditText] = useState('')
  const [memConfirm, setMemConfirm] = useState<null | { kind: 'clear' | 'delete'; id?: string }>(null)
  const advPersistTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  /** True while a chat turn is in flight — history persist must wait. */
  const advBusyRef = useRef(false)
  /** Serializes history/memory background I/O so they never overlap each other or a turn. */
  const advBgLockRef = useRef(false)
  /** Latest convo list waiting to be persisted after the turn finishes. */
  const advPendingPersistRef = useRef<AdvisorConvo[] | null>(null)
  const advConvosRef = useRef<AdvisorConvo[]>([])
  const advHydratedRef = useRef(false)
  const [, setAdvTick] = useState(0) // re-render countdowns periodically
  // Review trace is internal QA only — never patient-facing chrome. Toggled with
  // the keyboard escape hatch below, off by default.
  const [advShowTrace, setAdvShowTrace] = useState(false)
  const [advStreamId, setAdvStreamId] = useState<string | null>(null)
  const [advStreamChars, setAdvStreamChars] = useState(0)
  const advScrollRef = useRef<HTMLDivElement>(null)
  const advInputRef = useRef<HTMLTextAreaElement>(null)
  const advStickRef = useRef(true)
  const advAbortRef = useRef<{ turnId: string; cancelled: boolean } | null>(null)
  const advReduceMotion = usePrefersReducedMotion()

  const advActive = advConvos.find((c) => c.id === advActiveId) || null
  const advMsgs = advActive ? advActive.msgs : EMPTY_ADV_MSGS
  // Latest reply artifact value, readable from inside the async turn handler.
  const advReplyRef = useRef<AdvisorReplyDoc | null>(null)

  // Upload state
  const [uploadBusy, setUploadBusy] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadInfo, setUploadInfo] = useState<UploadInfo | null>(null)
  const [forceEmpty, setForceEmpty] = useState(false)
  // When a new bundle arrives after upload, reset slider to full range
  const [sliderEpoch, setSliderEpoch] = useState(0)

  const bundle =
    !forceEmpty && bundleArt.status === 'ready' && bundleArt.data?.n_days
      ? bundleArt.data
      : null
  const advReply: AdvisorReplyDoc | null = (() => {
    if (advReplyArt.status !== 'ready' || advReplyArt.data == null) return null
    const raw: unknown = advReplyArt.data
    if (typeof raw === 'string') {
      try {
        return JSON.parse(raw) as AdvisorReplyDoc
      } catch {
        return null
      }
    }
    return raw as AdvisorReplyDoc
  })()
  // Stabilize artifact docs: parsing a JSON string every render creates a new
  // object identity and thrash-re-runs hydrate effects (messages vanish / history
  // panel glitches). Key off status + a stable serialized snapshot.
  const advHistoryRawKey =
    advHistoryArt.status === 'ready' && advHistoryArt.data != null
      ? typeof advHistoryArt.data === 'string'
        ? (advHistoryArt.data as string)
        : JSON.stringify(advHistoryArt.data)
      : ''
  const advHistoryDoc: AdvisorHistoryView | null = useMemo(() => {
    if (!advHistoryRawKey) return null
    try {
      const raw: unknown =
        advHistoryArt.status === 'ready' ? advHistoryArt.data : null
      if (raw == null) return null
      if (typeof raw === 'string') return JSON.parse(raw) as AdvisorHistoryView
      return raw as AdvisorHistoryView
    } catch {
      return null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [advHistoryRawKey, advHistoryArt.status])

  const advMemoryRawKey =
    advMemoryArt.status === 'ready' && advMemoryArt.data != null
      ? typeof advMemoryArt.data === 'string'
        ? (advMemoryArt.data as string)
        : JSON.stringify(advMemoryArt.data)
      : ''
  const advMemoryDoc: MemoryViewDoc | null = useMemo(() => {
    if (!advMemoryRawKey) return null
    try {
      const raw: unknown =
        advMemoryArt.status === 'ready' ? advMemoryArt.data : null
      if (raw == null) return null
      if (typeof raw === 'string') return JSON.parse(raw) as MemoryViewDoc
      return raw as MemoryViewDoc
    } catch {
      return null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [advMemoryRawKey, advMemoryArt.status])
  const itemsSideRawKey =
    advMemoryItemsArt.status === 'ready' && (advMemoryItemsArt as { data?: unknown }).data != null
      ? typeof (advMemoryItemsArt as { data?: unknown }).data === 'string'
        ? ((advMemoryItemsArt as { data: unknown }).data as string)
        : JSON.stringify((advMemoryItemsArt as { data: unknown }).data)
      : ''
  const itemsSide = useMemo(() => {
    if (!itemsSideRawKey) return null
    try {
      const raw = itemsSideRawKey.trim().replace(/\n\/\*[\s\S]*?\*\/\s*$/, '')
      return JSON.parse(raw) as {
        items?: MemoryItem[]
        updated_at_ms?: number
        learning_enabled?: boolean
      }
    } catch {
      return null
    }
  }, [itemsSideRawKey])
  const memoryItemsBase: MemoryItem[] = (() => {
    const fromView = Array.isArray(advMemoryDoc?.items)
      ? (advMemoryDoc!.items as MemoryItem[])
      : []
    const fromSide = Array.isArray(itemsSide?.items) ? (itemsSide!.items as MemoryItem[]) : []
    if (!fromSide.length) return fromView
    if (!fromView.length) return fromSide
    const sideTs = Number(itemsSide?.updated_at_ms || 0)
    const viewTs = Number(advMemoryDoc?.updated_at_ms || 0)
    // Prefer items sidecar when equal-or-newer (view blob is known sticky).
    return sideTs >= viewTs ? fromSide : fromView
  })()
  const memoryItems: MemoryItem[] = memoryItemsBase
    .map((it) => {
      const o = memItemOverrides[it.id]
      if (!o || o._removed) return it
      return { ...it, ...o }
    })
    .filter((it) => !memItemOverrides[it.id]?._removed)
  const memoryLearningOn =
    itemsSide?.learning_enabled !== undefined && itemsSide?.learning_enabled !== null
      ? !!itemsSide.learning_enabled
      : advMemoryDoc?.learning_enabled !== false
  const memoryActiveItems = memoryItems.filter(
    (it) => (it.status || 'active') === 'active' && !it.dismissed,
  )
  // Prefer fresher portrait sidecar (rev>=4 / newer stamp) over a sticky full view.
  const portraitSideRawKey =
    advPortraitArt.status === 'ready' && (advPortraitArt as { data?: unknown }).data != null
      ? typeof (advPortraitArt as { data?: unknown }).data === 'string'
        ? ((advPortraitArt as { data: unknown }).data as string)
        : JSON.stringify((advPortraitArt as { data: unknown }).data)
      : ''
  const portraitSide = useMemo(() => {
    if (!portraitSideRawKey) return null
    try {
      const raw = portraitSideRawKey.trim()
      const json = raw.startsWith('{') ? raw.replace(/\n\/\*[\s\S]*?\*\/\s*$/, '') : raw
      return JSON.parse(json) as {
        portrait?: MemoryPortrait
        updated_at_ms?: number
        portrait_rev?: number
        disclaimer?: string
      }
    } catch {
      return null
    }
  }, [portraitSideRawKey])
  const memoryPortrait: MemoryPortrait | null = (() => {
    const fromView = advMemoryDoc?.portrait || null
    const fromSide = portraitSide?.portrait || null
    if (!fromSide) return fromView
    if (!fromView) return fromSide
    const sideTs = Number(portraitSide?.updated_at_ms || 0)
    const viewTs = Number(advMemoryDoc?.updated_at_ms || 0)
    const sideRev = Number(portraitSide?.portrait_rev || 0)
    if (sideRev >= 4 || sideTs >= viewTs) return fromSide
    return fromView
  })()
  const memoryPortraitSections: MemoryPortraitSection[] = (() => {
    const secs = Array.isArray(memoryPortrait?.sections)
      ? (memoryPortrait!.sections as MemoryPortraitSection[])
      : []
    const activeIds = new Set(memoryActiveItems.map((it) => it.id))
    // Drop evidence the patient muted/deleted even if a sticky portrait blob still lists them.
    return secs
      .map((sec) => {
        const ids = (sec.item_ids || []).filter((id) => activeIds.has(id))
        const evidence = (sec.evidence || []).filter((ev) => {
          const id = ev.item_id || ''
          return id ? activeIds.has(id) : true
        })
        if (ids.length === 0 && evidence.length === 0) return null
        return { ...sec, item_ids: ids, evidence }
      })
      .filter(Boolean) as MemoryPortraitSection[]
  })()
  const memorySupersededItems = memoryItems.filter(
    (it) =>
      (it.status || '') === 'superseded' ||
      (it.status || '') === 'expired' ||
      (it.status || '') === 'dismissed',
  )
  // Annotations artifact is stored as a JSON text blob — parse stably (same pattern
  // as history/memory). Without this, deletes look gone in local state then
  // "come back" on reload because remote never applied the write.
  const annRawKey =
    annArt.status === 'ready' && (annArt as { data?: unknown }).data != null
      ? typeof (annArt as { data?: unknown }).data === 'string'
        ? ((annArt as { data: unknown }).data as string)
        : JSON.stringify((annArt as { data: unknown }).data)
      : ''
  const annDoc: AnnotationsDoc | null = useMemo(() => {
    if (!annRawKey) return null
    try {
      if (annRawKey.trim().startsWith('{') || annRawKey.trim().startsWith('[')) {
        // Prefer parsing the stable serialized key (works for JSON-string artifacts).
        const parsed = JSON.parse(annRawKey)
        if (parsed && typeof parsed === 'object' && Array.isArray((parsed as AnnotationsDoc).annotations)) {
          return parsed as AnnotationsDoc
        }
        // Sometimes the artifact data IS the annotations array
        if (Array.isArray(parsed)) {
          return { version: 2, annotations: parsed as Annotation[] }
        }
      }
      return null
    } catch {
      return null
    }
  }, [annRawKey])
  const remoteAnns: Annotation[] = useMemo(
    () =>
      Array.isArray(annDoc?.annotations) ? (annDoc!.annotations as Annotation[]) : [],
    [annDoc],
  )
  // Prefer confirmed local only while a save is in flight; otherwise store is SoT.
  const annotations = localAnns !== null ? localAnns : remoteAnns

  // Drop local override once remote store matches (create/edit/delete write-through).
  const remoteAnnIdsKey = useMemo(
    () =>
      remoteAnns
        .map((x) => x.id)
        .slice()
        .sort()
        .join('|'),
    [remoteAnns],
  )
  useEffect(() => {
    if (localAnns === null) return
    const a = localAnns
      .map((x) => x.id)
      .slice()
      .sort()
      .join('|')
    if (a === remoteAnnIdsKey) setLocalAnns(null)
  }, [localAnns, remoteAnnIdsKey])

  // Seed upload info from bundle metadata
  useEffect(() => {
    if (!bundle) return
    if (bundle.upload?.file_name) {
      setUploadInfo({
        fileName: bundle.upload.file_name,
        rowCount: bundle.upload.row_count || bundle.n_days,
        dateStart: bundle.upload.date_start || bundle.dates[0],
        dateEnd: bundle.upload.date_end || bundle.dates[bundle.n_days - 1],
      })
    } else if (bundle.source) {
      setUploadInfo({
        fileName: bundle.source,
        rowCount: bundle.n_days,
        dateStart: bundle.dates[0],
        dateEnd: bundle.dates[bundle.n_days - 1],
      })
    }
  }, [bundle])

  // Init dual slider to full range (and after upload epoch)
  useEffect(() => {
    if (bundle) {
      setLoTick(0)
      setHiTick(bundle.n_days * 2)
    }
  }, [bundle, sliderEpoch])

  // Seed annotation time fields from current window once
  useEffect(() => {
    if (!bundle || annStartDate) return
    setAnnStartDate(bundle.dates[0])
    setAnnEndDate(bundle.dates[bundle.n_days - 1])
  }, [bundle, annStartDate])

  const nDays = bundle?.n_days ?? 1
  const maxTick = nDays * 2
  const lo = loTick ?? 0
  const hi = hiTick ?? maxTick
  const { start: startIdx, end: endIdx } = ticksToDayRange(lo, hi, nDays)
  const windowKey = `${startIdx}_${endIdx}`
  const entry = bundle?.windows?.[windowKey] || null
  const windowDays = endIdx - startIdx + 1

  const pathKey = tab === 'bap' ? 'pure' : 'blend'
  const result: Result | null = entry ? entry[pathKey] : null
  const suggestions: SuggestionPack | null = entry
    ? tab === 'bap'
      ? entry.suggestions_pure
      : entry.suggestions_blend
    : null

  /**
   * Rolling history chart source — FIXED WIDTH (prefer 14d), independent of dual-handle
   * score window length. Filter points whose end_idx falls inside the selected date range.
   * Falls back to full_series 14d columns, then 1d, if trends missing.
   */
  const ROLLING_W = 14

  const chartPoints = useMemo(() => {
    if (!bundle)
      return [] as {
        date: string
        end_idx: number
        BAP: number | null
        I: number | null
        R: number | null
        Sp: number | null
        C: number | null
      }[]

    const rawTrend = bundle.trends?.[String(ROLLING_W)] || []
    if (rawTrend.length > 0) {
      return rawTrend
        .filter((p) => p.end_idx >= startIdx && p.end_idx <= endIdx)
        .map((p) => ({
          date: p.date,
          end_idx: p.end_idx,
          BAP: tab === 'bap' ? p.BAP_pure : p.BAP_blend,
          I: tab === 'bap' ? p.I_pure : p.I_blend,
          R: tab === 'bap' ? p.R_pure : p.R_blend,
          Sp: tab === 'bap' ? p.Sp_pure : p.Sp_blend,
          C: tab === 'bap' ? p.C_pure : p.C_blend,
        }))
    }

    const fs = bundle.full_series || []
    const mapped = fs
      .filter((p) => p.idx >= startIdx && p.idx <= endIdx)
      .map((p) => {
        const use14 = tab === 'bap' ? p.BAP_14d_pure : p.BAP_14d_blend
        const use1 = tab === 'bap' ? p.BAP_1d_pure : p.BAP_1d_blend
        return {
          date: p.date,
          end_idx: p.idx,
          BAP: use14 != null ? use14 : use1,
          I: tab === 'bap' ? p.I_1d_pure : p.I_1d_blend,
          R: tab === 'bap' ? p.R_1d_pure : p.R_1d_blend,
          Sp: tab === 'bap' ? p.Sp_1d_pure : p.Sp_1d_blend,
          C: tab === 'bap' ? p.C_1d_pure : p.C_1d_blend,
        }
      })
    return mapped.filter((p) => p.BAP != null)
  }, [bundle, startIdx, endIdx, tab])

  const band = scoreBand(result?.bap ?? 0)

  const radarData = useMemo(() => {
    if (!result) return []
    return (Object.keys(AXIS_META) as string[]).map((k) => {
      const v = result.axis_scores[k]
      return {
        axis: AXIS_META[k].name,
        score: v == null ? 0 : Math.round(v * 1000) / 10,
        available: result.detail?.axis_available?.[k] !== false && v != null,
      }
    })
  }, [result])

  const lineTheme = useMemo(
    () => ({
      text: { fill: dark ? '#cbd5e1' : '#334155', fontSize: 11 },
      grid: { line: { stroke: dark ? '#1e293b' : '#e2e8f0' } },
      axis: {
        ticks: { text: { fill: dark ? '#94a3b8' : '#64748b', fontSize: 10 } },
        legend: { text: { fill: dark ? '#cbd5e1' : '#334155' } },
      },
      crosshair: { line: { stroke: '#38bdf8' } },
      tooltip: {
        container: {
          background: dark ? '#0f172a' : '#fff',
          color: dark ? '#e2e8f0' : '#0f172a',
          fontSize: 12,
          borderRadius: 8,
          boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
        },
      },
    }),
    [dark],
  )

  const lineData = useMemo(() => {
    if (!chartPoints.length) return []
    const out: { id: string; color?: string; data: { x: string; y: number | null }[] }[] = []
    if (seriesOn.BAP) {
      out.push({
        id: 'BAP',
        data: chartPoints.map((p) => ({ x: p.date, y: p.BAP })),
      })
    }
    if (seriesOn.Intensity) {
      out.push({
        id: 'Intensity',
        data: chartPoints.map((p) => ({ x: p.date, y: p.I == null ? null : p.I * 100 })),
      })
    }
    if (seriesOn.Recovery) {
      out.push({
        id: 'Recovery',
        data: chartPoints.map((p) => ({ x: p.date, y: p.R == null ? null : p.R * 100 })),
      })
    }
    if (seriesOn.Spread) {
      out.push({
        id: 'Spread',
        data: chartPoints.map((p) => ({ x: p.date, y: p.Sp == null ? null : p.Sp * 100 })),
      })
    }
    if (seriesOn.Consistency) {
      out.push({
        id: 'Consistency',
        data: chartPoints.map((p) => ({ x: p.date, y: p.C == null ? null : p.C * 100 })),
      })
    }
    return out.filter((s) => s.data.length > 0)
  }, [chartPoints, seriesOn])

  /** Rolling score series for the in-chat sparkline (same points as the trend chart). */
  const advTrendValues = useMemo(() => chartPoints.map((p) => p.BAP), [chartPoints])

  const chartDateList = useMemo(() => chartPoints.map((p) => p.date), [chartPoints])
  const chartEndIdxList = useMemo(() => chartPoints.map((p) => p.end_idx), [chartPoints])

  const drillDaily = useMemo(() => {
    if (!bundle || !drillAxis || !entry) return []
    const ds = bundle.daily_series
    const dates = (ds.dates as string[]) || bundle.dates
    const slice = <T,>(arr: T[] | undefined): T[] => (arr || []).slice(startIdx, endIdx + 1)
    const fieldForAxis: Record<string, string[]> = {
      I: ['steps', 'calories_active', 'vigorous_intensity_seconds', 'moderate_intensity_seconds'],
      R: ['hrv_rmssd', 'resting_hr', 'breaths_per_min', 'temperature_delta'],
      Sp: ['inactivity_seconds', 'low_intensity_seconds'],
      C: ['steps', 'calories_active', 'sleep_duration_min', 'met_level'],
      V: [],
    }
    const fields = fieldForAxis[drillAxis] || []
    // Intensity drill: still show MET series when excluded from scoring
    const fieldsEff =
      drillAxis === 'I' &&
      metExcludedFromScore(bundle) &&
      !fields.includes('met_level') &&
      Array.isArray(ds.met_level)
        ? [...fields, 'met_level']
        : fields
    return fieldsEff
      .map((field) => {
        const vals = slice(ds[field] as (number | null)[])
        const xs = slice(dates)
        const isMetExcl = field === 'met_level' && metExcludedFromScore(bundle)
        return {
          id: isMetExcl ? metVarLabel(bundle) : humanVar(field),
          key: field,
          excludedFromScore: isMetExcl,
          tip: isMetExcl ? metVarTip(bundle) : undefined,
          data: xs.map((x, i) => ({ x, y: vals[i] == null ? null : Number(vals[i]) })),
        }
      })
      .filter((s) => s.data.some((d) => d.y != null))
  }, [bundle, drillAxis, entry, startIdx, endIdx])

  const axisVarOptions = useMemo(() => {
    const subs = result?.subscores?.[annAxisKey] || []
    return subs.map((s) => ({ value: s.name, label: humanVar(s.name) }))
  }, [result, annAxisKey])

  // ---- Annotation helpers ----
  const dateToIdx = useCallback(
    (date: string) => {
      if (!bundle) return 0
      const i = bundle.dates.indexOf(date)
      return i >= 0 ? i : 0
    },
    [bundle],
  )

  const applyManualTimesToIdx = useCallback(() => {
    if (!bundle) return { s: 0, e: nDays - 1 }
    let s = dateToIdx(annStartDate || bundle.dates[0])
    let e = dateToIdx(annEndDate || bundle.dates[nDays - 1])
    if (e < s) [s, e] = [e, s]
    return { s, e }
  }, [bundle, annStartDate, annEndDate, dateToIdx, nDays])

  const setAnnRangeFromIdx = (s: number, e: number) => {
    if (!bundle) return
    const a = clamp(Math.min(s, e), 0, nDays - 1)
    const b = clamp(Math.max(s, e), 0, nDays - 1)
    setAnnStartDate(bundle.dates[a])
    setAnnEndDate(bundle.dates[b])
    setAnnStartTime('00:00')
    setAnnEndTime('24:00')
  }

  const onChartPointerDown = (e: React.PointerEvent) => {
    if (!dragMode || !chartShellRef.current || chartPoints.length === 0) return
    e.preventDefault()
    e.stopPropagation()
    const pi = clientXToPointIndex(e.clientX, chartShellRef.current, chartPoints.length)
    dragActiveRef.current = true
    setDragSel({ a: pi, b: pi })
    ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
  }
  const onChartPointerMove = (e: React.PointerEvent) => {
    if (!dragMode || !dragActiveRef.current || !chartShellRef.current || chartPoints.length === 0) return
    e.preventDefault()
    const pi = clientXToPointIndex(e.clientX, chartShellRef.current, chartPoints.length)
    setDragSel((prev) => (prev ? { a: prev.a, b: pi } : { a: pi, b: pi }))
  }
  const onChartPointerUp = (e: React.PointerEvent) => {
    if (!dragMode || !dragActiveRef.current || !dragSel || chartPoints.length === 0) {
      dragActiveRef.current = false
      return
    }
    e.preventDefault()
    dragActiveRef.current = false
    const i0 = Math.min(dragSel.a, dragSel.b)
    const i1 = Math.max(dragSel.a, dragSel.b)
    const sIdx = chartEndIdxList[i0] ?? startIdx
    const eIdx = chartEndIdxList[i1] ?? endIdx
    setAnnRangeFromIdx(sIdx, eIdx)
    setDragSel(null)
    setDragMode(false)
    setAnnOpen(true)
    if (bundle) {
      setAnnStatus(
        `Time range set from chart: ${bundle.dates[clamp(sIdx, 0, nDays - 1)]} → ${bundle.dates[clamp(eIdx, 0, nDays - 1)]}`,
      )
    }
  }

  /** Geometry for an annotation on the current chart window (presentation only). */
  const markerStyleForAnn = useCallback((
    a: Annotation,
  ): {
    left: string
    width: string
    leftPct: number
    widthPct: number
    li: number
    ri: number
    isPoint: boolean
    anchorPct: number
  } | null => {
    if (!bundle || chartPoints.length === 0) return null
    const n = chartPoints.length
    let li = -1
    let ri = -1
    for (let i = 0; i < n; i++) {
      const ei = chartEndIdxList[i]
      if (ei >= a.start_idx && ei <= a.end_idx) {
        if (li < 0) li = i
        ri = i
      }
    }
    if (li < 0) {
      const di0 = chartDateList.indexOf(a.start_date)
      const di1 = chartDateList.indexOf(a.end_date)
      if (di0 < 0 && di1 < 0) return null
      li = di0 >= 0 ? di0 : di1
      ri = di1 >= 0 ? di1 : di0
    }
    if (ri < li) [li, ri] = [ri, li]
    const leftPct = pointIndexToLeftPct(li, n)
    const rightPct = pointIndexToLeftPct(ri, n)
    // Point-in-time: single day (or identical ends) → thin vertical marker.
    const isPoint =
      a.start_date === a.end_date ||
      a.start_idx === a.end_idx ||
      li === ri ||
      rightPct - leftPct < 0.8
    const widthPct = isPoint ? 0 : Math.max(0.8, rightPct - leftPct)
    const anchorPct = isPoint ? leftPct : (leftPct + rightPct) / 2
    const plot = `(100% - ${CHART_ML + CHART_MR}px)`
    return {
      left: isPoint
        ? `calc(${CHART_ML}px + ${plot} * ${leftPct / 100})`
        : `calc(${CHART_ML}px + ${plot} * ${leftPct / 100})`,
      width: isPoint
        ? '2px'
        : `calc(${plot} * ${widthPct / 100})`,
      leftPct,
      widthPct: isPoint ? 0 : widthPct,
      li,
      ri,
      isPoint,
      anchorPct,
    }
  }, [bundle, chartPoints, chartEndIdxList, chartDateList])

  /** Zoom-aware clusters: collapse markers that sit within ~28px at current chart width. */
  const chartAnnLayout = useMemo(() => {
    type Laid = {
      ann: Annotation
      st: NonNullable<ReturnType<typeof markerStyleForAnn>>
    }
    const laid: Laid[] = []
    for (const a of annotations) {
      const st = markerStyleForAnn(a)
      if (st) laid.push({ ann: a, st })
    }
    laid.sort((x, y) => x.st.anchorPct - y.st.anchorPct || x.ann.start_date.localeCompare(y.ann.start_date))
    const shellW = chartShellRef.current?.clientWidth || 640
    const plotW = Math.max(120, shellW - CHART_ML - CHART_MR)
    const threshPct = (28 / plotW) * 100
    type Cluster = {
      id: string
      items: Laid[]
      anchorPct: number
      leftPct: number
      rightPct: number
    }
    const clusters: Cluster[] = []
    for (const item of laid) {
      const last = clusters[clusters.length - 1]
      const a0 = item.st.anchorPct
      if (last && Math.abs(a0 - last.anchorPct) <= threshPct) {
        last.items.push(item)
        last.anchorPct =
          last.items.reduce((s, it) => s + it.st.anchorPct, 0) / last.items.length
        last.leftPct = Math.min(last.leftPct, item.st.leftPct)
        last.rightPct = Math.max(
          last.rightPct,
          item.st.isPoint ? item.st.leftPct : item.st.leftPct + item.st.widthPct,
        )
      } else {
        clusters.push({
          id: `c_${item.ann.id}`,
          items: [item],
          anchorPct: a0,
          leftPct: item.st.leftPct,
          rightPct: item.st.isPoint
            ? item.st.leftPct
            : item.st.leftPct + item.st.widthPct,
        })
      }
    }
    return { laid, clusters, plotW }
    // Geometry deps live on markerStyleForAnn; shell width is best-effort each render.
  }, [annotations, markerStyleForAnn])

  const dragSelStyle = useMemo(() => {
    if (!dragSel || chartPoints.length === 0) return null
    const n = chartPoints.length
    const li = Math.min(dragSel.a, dragSel.b)
    const ri = Math.max(dragSel.a, dragSel.b)
    const leftPct = pointIndexToLeftPct(li, n)
    const rightPct = pointIndexToLeftPct(ri, n)
    const widthPct = Math.max(1.2, rightPct - leftPct)
    return {
      left: `calc(${CHART_ML}px + (100% - ${CHART_ML + CHART_MR}px) * ${leftPct / 100})`,
      width: `calc((100% - ${CHART_ML + CHART_MR}px) * ${widthPct / 100})`,
    }
  }, [dragSel, chartPoints.length])

  const saveAnnotations = async (next: Annotation[]) => {
    // Optimistic UI only after we attempt write; roll back if store write fails
    // so delete+reload cannot resurrect items that never persisted.
    const previous = annotations
    setLocalAnns(next)
    setAnnStatus('Saving…')
    const payload = JSON.stringify({ version: 2, annotations: next })
    try {
      const stage = await run({
        description: 'Stage BAP dashboard annotations payload',
        packageName: 'bap_oura_dashboard',
        entrypoint: 'stage_annotations.py',
        arguments: [payload],
      })
      let resultRun = stage
      if (stage.outcome !== 'completed') {
        // Stage failed — try argv-direct save as fallback
        resultRun = await run({
          description: 'Save BAP dashboard annotations (direct)',
          packageName: 'bap_oura_dashboard',
          entrypoint: 'save_annotations.py',
          arguments: [payload],
        })
      } else {
        resultRun = await run({
          description: 'Save BAP dashboard annotations (from staged payload)',
          packageName: 'bap_oura_dashboard',
          entrypoint: 'save_annotations.py',
        })
      }
      if (resultRun.outcome === 'completed') {
        setAnnStatus('Saved')
        // Keep localAnns=next until remote artifact matches (avoids stale reload resurrection).
        try {
          await annArt.reload()
        } catch {
          /* local already correct */
        }
        setTimeout(() => setAnnStatus(null), 1600)
        return true
      }
      // Write failed — restore previous list and surface error
      setLocalAnns(previous)
      setAnnStatus(resultRun.error || 'Save failed — change not stored')
      return false
    } catch (e) {
      setLocalAnns(previous)
      setAnnStatus(e instanceof Error ? e.message : 'Save failed — change not stored')
      return false
    }
  }

  const buildAnnotationFromForm = (id?: string): Annotation | null => {
    const text = annText.trim()
    if (!text) {
      setAnnStatus('Enter annotation text first')
      return null
    }
    if (!bundle) return null
    const { s, e } = applyManualTimesToIdx()
    const label =
      annLabel.trim() ||
      (annTargetType === 'score'
        ? 'Score note'
        : annVariable
          ? humanVar(annVariable)
          : AXIS_META[annAxisKey]?.name || 'Axis note')
    const existing = id ? annotations.find((x) => x.id === id) : undefined
    return {
      id: id || uid(),
      created_at: existing?.created_at || new Date().toISOString(),
      updated_at: new Date().toISOString(),
      target_type: annTargetType,
      target_key: annTargetType === 'score' ? 'bap' : annAxisKey,
      axis_key: annTargetType === 'axis' ? annAxisKey : null,
      variable: annTargetType === 'axis' && annVariable ? annVariable : null,
      label,
      text,
      start_date: bundle.dates[s],
      end_date: bundle.dates[e],
      start_time: annStartTime || '00:00',
      end_time: annEndTime || '24:00',
      start_idx: s,
      end_idx: e,
      tab,
      source: existing?.source || 'manual',
    }
  }

  const addOrUpdateAnnotation = async () => {
    const draft = buildAnnotationFromForm(editingId || undefined)
    if (!draft) return
    let next: Annotation[]
    if (editingId) {
      next = annotations.map((a) => (a.id === editingId ? draft : a))
    } else {
      next = [...annotations, draft]
    }
    setAnnText('')
    setAnnLabel('')
    setEditingId(null)
    await saveAnnotations(next)
  }

  /**
   * Same creation path as the manual annotation UI — identical schema/store.
   * Used by the Advisor conversational preview after the patient confirms.
   */
  const createAnnotation = async (
    input: {
      target_type: AnnTargetType
      axis_key?: string | null
      variable?: string | null
      label: string
      text: string
      start_date: string
      end_date: string
      start_time?: string
      end_time?: string
      source?: 'manual' | 'patient_via_advisor'
    },
  ): Promise<Annotation | null> => {
    if (!bundle) return null
    const dates = bundle.dates || []
    if (!dates.length) return null
    let s = dates.indexOf(input.start_date)
    let e = dates.indexOf(input.end_date)
    // Snap to nearest available date if exact match missing.
    if (s < 0) {
      s = 0
      for (let i = 0; i < dates.length; i++) {
        if (dates[i] >= input.start_date) {
          s = i
          break
        }
        s = i
      }
    }
    if (e < 0) {
      e = dates.length - 1
      for (let i = dates.length - 1; i >= 0; i--) {
        if (dates[i] <= input.end_date) {
          e = i
          break
        }
        e = i
      }
    }
    if (e < s) [s, e] = [e, s]
    const target_type: AnnTargetType = input.target_type === 'axis' ? 'axis' : 'score'
    const axis_key =
      target_type === 'axis' ? input.axis_key || input.variable || 'C' : null
    const variable =
      target_type === 'axis' && input.variable ? input.variable : null
    const draft: Annotation = {
      id: uid(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      target_type,
      target_key: target_type === 'score' ? 'bap' : String(axis_key),
      axis_key,
      variable,
      label: (input.label || '').trim() || (target_type === 'score' ? 'Score note' : 'Axis note'),
      text: (input.text || '').trim() || input.label,
      start_date: dates[s],
      end_date: dates[e],
      start_time: input.start_time || '00:00',
      end_time: input.end_time || '24:00',
      start_idx: s,
      end_idx: e,
      tab,
      source: input.source || 'manual',
    }
    await saveAnnotations([...annotations, draft])
    setExpandedAnn(draft.id)
    return draft
  }

  const advUpdateProposal = (msgId: string, next: AnnotationProposal) => {
    setAdvConvos((prev) =>
      prev.map((c) => ({
        ...c,
        msgs: c.msgs.map((m) =>
          m.id === msgId ? { ...m, annotationProposal: next, annotationUi: 'editing' } : m,
        ),
      })),
    )
  }

  const advAcceptProposal = async (msg: AdvisorMsg) => {
    const p = msg.annotationProposal
    if (!p || !bundle || msg.annotationUi === 'saved') return
    const created = await createAnnotation({
      target_type: p.target_type === 'axis' ? 'axis' : 'score',
      axis_key: p.axis_key,
      variable: p.variable,
      label: p.title || 'Note',
      text: p.description || p.title || '',
      start_date: p.start_date || bundle.dates[0],
      end_date: p.end_date || bundle.dates[bundle.dates.length - 1],
      start_time: p.start_time || '00:00',
      end_time: p.end_time || '24:00',
      source: 'patient_via_advisor',
    })
    if (!created) return
    setAdvConvos((prev) =>
      prev.map((c) => ({
        ...c,
        msgs: c.msgs.map((m) =>
          m.id === msg.id
            ? {
                ...m,
                annotationUi: 'saved',
                annotationSavedId: created.id,
                content:
                  m.content +
                  `\n\nSaved. You’ll see it in the annotations panel and as a marker on the graph.`,
              }
            : m,
        ),
      })),
    )
  }

  const advCancelProposal = (msgId: string) => {
    setAdvConvos((prev) =>
      prev.map((c) => ({
        ...c,
        msgs: c.msgs.map((m) =>
          m.id === msgId
            ? {
                ...m,
                annotationUi: 'cancelled',
                annotationProposal: m.annotationProposal
                  ? { ...m.annotationProposal, status: 'proposal' }
                  : null,
              }
            : m,
        ),
      })),
    )
  }

  const startEdit = (a: Annotation) => {
    setEditingId(a.id)
    setExpandedAnn(a.id)
    setAnnTargetType(a.target_type === 'axis' ? 'axis' : 'score')
    setAnnAxisKey(a.axis_key || (a.target_type === 'axis' ? a.target_key : 'C'))
    setAnnVariable(a.variable || '')
    setAnnLabel(a.label || '')
    setAnnText(a.text || '')
    setAnnStartDate(a.start_date)
    setAnnEndDate(a.end_date)
    setAnnStartTime(a.start_time || '00:00')
    setAnnEndTime(a.end_time || '24:00')
    setAnnOpen(true)
  }

  const removeAnnotation = async (id: string) => {
    if (editingId === id) setEditingId(null)
    const next = annotations.filter((a) => a.id !== id)
    const ok = await saveAnnotations(next)
    if (!ok) {
      // saveAnnotations already restored previous + set error status
      return
    }
  }

  /**
   * Persist the durable transcript list (non-empty, non-expired only) to the
   * encrypted store. Never writes clinical artifacts. Fire-and-forget; errors
   * surface as a muted banner, not a crash. Never runs while a chat turn is
   * in flight — concurrent history runs were racing the Advisor and breaking
   * both the chat and the history tab.
   */
  const advPersistHistory = useCallback(
    async (list: AdvisorConvo[]) => {
      // Snapshot for a possible deferred retry if a turn is active.
      advPendingPersistRef.current = list
      // Never race history I/O with an Advisor turn OR memory pipeline.
      if (advBusyRef.current || advBgLockRef.current) {
        return
      }
      advBgLockRef.current = true
      const now = Date.now()
      // Never persist empty drafts — only real transcripts. (advPurgeExpired no longer
      // strips empty drafts so the UI can keep a blank "new chat" without wiping history.)
      const durable = advPurgeExpired(
        list
          .filter((c) => (c.msgs && c.msgs.length > 0))
          .map((c) => ({
            id: c.id,
            title: c.title,
            createdAt: c.createdAt,
            updatedAt: c.updatedAt,
            expiresAt: c.createdAt + ADV_RETENTION_MS,
            msgs: (c.msgs || [])
              .filter((m) => !m.failed)
              .map((m) => ({
                id: m.id,
                role: m.role,
                content: m.content,
                category: m.category,
                escalation: m.escalation,
              })),
          })),
        now,
      )
      try {
        const payload = JSON.stringify({ op: 'replace', conversations: durable })
        // Prefer argv-only save — avoids the stuck alyf_advisor_chat_history_pending blob.
        // Fall back to stage+save only if argv path errors for size.
        let saved = await run({
          description: 'Save Alyf Advisor chat history (encrypted, 7-day TTL)',
          packageName: 'bap_oura_dashboard',
          entrypoint: 'save_advisor_history.py',
          arguments: [payload],
        })
        if (saved.outcome !== 'completed') {
          // Fallback: stage on the new pending id, then save without argv.
          const staged = await run({
            description: 'Stage Alyf Advisor chat history (transcripts only)',
            packageName: 'bap_oura_dashboard',
            entrypoint: 'stage_advisor_history.py',
            arguments: [payload],
          })
          if (staged.outcome !== 'completed') {
            throw new Error(staged.error || saved.error || 'Could not save chat history')
          }
          saved = await run({
            description: 'Save Alyf Advisor chat history (encrypted, 7-day TTL)',
            packageName: 'bap_oura_dashboard',
            entrypoint: 'save_advisor_history.py',
          })
        }
        if (saved.outcome !== 'completed') {
          throw new Error(saved.error || 'Could not save chat history')
        }
        setAdvHistoryError(null)
        advPendingPersistRef.current = null
        // Skip view reload while user is mid-turn; hydrate can wait until idle.
        if (!advBusyRef.current) {
          try {
            await advHistoryArt.reload()
          } catch {
            /* view refresh is best-effort */
          }
        }
      } catch (e) {
        // Never crash the chat on history failures — muted banner only.
        setAdvHistoryError(
          e instanceof Error ? e.message : 'Could not save chat history',
        )
      } finally {
        advBgLockRef.current = false
      }
    },
    [run, advHistoryArt],
  )

  /** Debounced persist after local transcript changes (skipped while a turn runs). */
  const advSchedulePersist = useCallback(
    (list: AdvisorConvo[]) => {
      advPendingPersistRef.current = list
      if (advPersistTimer.current) clearTimeout(advPersistTimer.current)
      advPersistTimer.current = setTimeout(() => {
        if (advBusyRef.current || advBgLockRef.current) return
        // Prefer the latest in-memory snapshot so we don't write a stale list.
        const snap = advConvosRef.current.length ? advConvosRef.current : list
        void advPersistHistory(snap)
      }, 600)
    },
    [advPersistHistory],
  )

  /** Append messages to a conversation by id (the active thread may change mid-flight). */
  const advAppend = useCallback(
    (convoId: string, msgs: AdvisorMsg[]) => {
      setAdvConvos((prev) => {
        const now = Date.now()
        const exists = prev.some((c) => c.id === convoId)
        let next: AdvisorConvo[]
        if (!exists) {
          // Start-convo + append can race React state; never drop the message.
          const title =
            msgs[0]?.role === 'user' ? msgs[0].content.slice(0, 52) : 'New conversation'
          next = [
            {
              id: convoId,
              title: title || 'New conversation',
              createdAt: now,
              updatedAt: now,
              expiresAt: now + ADV_RETENTION_MS,
              msgs: [...msgs],
              readOnly: false,
            },
            ...prev.filter((c) => (c.msgs && c.msgs.length > 0) || c.id === convoId),
          ]
        } else {
          next = prev.map((c) =>
            c.id === convoId
              ? {
                  ...c,
                  msgs: [...c.msgs, ...msgs],
                  updatedAt: now,
                  // Expiry is fixed from createdAt — never extended by new messages.
                  expiresAt: c.createdAt + ADV_RETENTION_MS,
                  title:
                    (c.title === 'New conversation' || !c.title) && msgs[0]?.role === 'user'
                      ? msgs[0].content.slice(0, 52)
                      : c.title,
                  readOnly: false,
                }
              : c,
          )
        }
        advSchedulePersist(next)
        return next
      })
    },
    [advSchedulePersist],
  )

  /** Ensure there is a writable empty draft without touching the history panel. */
  const advEnsureDraft = useCallback((): string => {
    // Reuse an existing empty writable draft when possible (no flicker / no history close).
    const existing = advConvosRef.current.find(
      (c) => !c.readOnly && (!c.msgs || c.msgs.length === 0),
    )
    if (existing) {
      if (advActiveId !== existing.id) setAdvActiveId(existing.id)
      return existing.id
    }
    const id = uid()
    const now = Date.now()
    setAdvConvos((prev) => {
      // Keep durable history + at most one empty draft.
      const kept = prev.filter(
        (c) => (c.msgs && c.msgs.length > 0) || c.id === id || c.readOnly,
      )
      return [
        {
          id,
          title: 'New conversation',
          createdAt: now,
          updatedAt: now,
          expiresAt: now + ADV_RETENTION_MS,
          msgs: [],
          readOnly: false,
        },
        ...kept.filter((c) => c.id !== id),
      ]
    })
    setAdvActiveId(id)
    return id
  }, [advActiveId])

  /** Explicit "New chat" — fresh draft and close history so the composer is front-and-center. */
  const advStartConvo = useCallback((): string => {
    const id = uid()
    const now = Date.now()
    setAdvConvos((prev) => {
      // Drop other empty drafts; keep durable history intact.
      const kept = prev.filter((c) => (c.msgs && c.msgs.length > 0) || c.id === id)
      return [
        {
          id,
          title: 'New conversation',
          createdAt: now,
          updatedAt: now,
          expiresAt: now + ADV_RETENTION_MS,
          msgs: [],
          readOnly: false,
        },
        ...kept.filter((c) => c.id !== id),
      ]
    })
    setAdvActiveId(id)
    setAdvHistoryOpen(false)
    setAdvConfirm(null)
    return id
  }, [])

  /** Open a past conversation read-only from the history panel. */
  const advOpenFromHistory = useCallback((id: string) => {
    if (advAbortRef.current) advAbortRef.current.cancelled = true
    setAdvStreamId(null)
    setAdvBusy(false)
    setAdvActiveId(id)
    setAdvConvos((prev) =>
      prev.map((c) => (c.id === id ? { ...c, readOnly: true } : c)),
    )
    setAdvHistoryOpen(false)
    setAdvConfirm(null)
    advStickRef.current = true
  }, [])

  /**
   * Hard-delete one conversation via op=delete on the encrypted store (same
   * store the loader reads). Await + confirm BEFORE treating UI as final.
   * Debounced replace-persist was the resurrection bug (fire-and-forget).
   */
  const advDeleteConvo = useCallback(
    async (id: string) => {
      const previous = advConvosRef.current
      setAdvConvos((prev) => prev.filter((c) => c.id !== id))
      setAdvActiveId((cur) => (cur === id ? null : cur))
      setAdvConfirm(null)
      setAdvHistoryError(null)
      try {
        for (let i = 0; i < 50 && (advBusyRef.current || advBgLockRef.current); i++) {
          await new Promise((r) => setTimeout(r, 100))
        }
        if (advBusyRef.current || advBgLockRef.current) {
          throw new Error('Could not delete — Advisor is busy. Try again in a moment.')
        }
        advBgLockRef.current = true
        const payload = JSON.stringify({ op: 'delete', id })
        let saved = await run({
          description: 'Delete Alyf Advisor conversation (encrypted store)',
          packageName: 'bap_oura_dashboard',
          entrypoint: 'save_advisor_history.py',
          arguments: [payload],
        })
        if (saved.outcome !== 'completed') {
          const staged = await run({
            description: 'Stage history delete',
            packageName: 'bap_oura_dashboard',
            entrypoint: 'stage_advisor_history.py',
            arguments: [payload],
          })
          if (staged.outcome !== 'completed') {
            throw new Error(staged.error || saved.error || 'Delete failed')
          }
          saved = await run({
            description: 'Delete Alyf Advisor conversation (encrypted store)',
            packageName: 'bap_oura_dashboard',
            entrypoint: 'save_advisor_history.py',
          })
        }
        if (saved.outcome !== 'completed') {
          throw new Error(saved.error || 'Delete failed')
        }
        advPendingPersistRef.current = null
        try {
          await advHistoryArt.reload()
        } catch {
          /* store already updated */
        }
      } catch (e) {
        setAdvConvos(previous)
        setAdvHistoryError(e instanceof Error ? e.message : 'Could not delete conversation')
      } finally {
        advBgLockRef.current = false
      }
    },
    [run, advHistoryArt],
  )

  /** Hard clear-all via op=clear on the same encrypted store. */
  const advClearAllHistory = useCallback(async () => {
    const previous = advConvosRef.current
    setAdvConvos([])
    setAdvActiveId(null)
    setAdvConfirm(null)
    setAdvHistoryError(null)
    try {
      for (let i = 0; i < 50 && (advBusyRef.current || advBgLockRef.current); i++) {
        await new Promise((r) => setTimeout(r, 100))
      }
      if (advBusyRef.current || advBgLockRef.current) {
        throw new Error('Could not clear history — Advisor is busy. Try again in a moment.')
      }
      advBgLockRef.current = true
      const payload = JSON.stringify({ op: 'clear' })
      let saved = await run({
        description: 'Clear all Alyf Advisor chat history (encrypted store)',
        packageName: 'bap_oura_dashboard',
        entrypoint: 'save_advisor_history.py',
        arguments: [payload],
      })
      if (saved.outcome !== 'completed') {
        const staged = await run({
          description: 'Stage history clear',
          packageName: 'bap_oura_dashboard',
          entrypoint: 'stage_advisor_history.py',
          arguments: [payload],
        })
        if (staged.outcome !== 'completed') {
          throw new Error(staged.error || saved.error || 'Clear failed')
        }
        saved = await run({
          description: 'Clear all Alyf Advisor chat history (encrypted store)',
          packageName: 'bap_oura_dashboard',
          entrypoint: 'save_advisor_history.py',
        })
      }
      if (saved.outcome !== 'completed') {
        throw new Error(saved.error || 'Clear failed')
      }
      advPendingPersistRef.current = null
      try {
        await advHistoryArt.reload()
      } catch {
        /* ok */
      }
    } catch (e) {
      setAdvConvos(previous)
      setAdvHistoryError(e instanceof Error ? e.message : 'Could not clear chat history')
    } finally {
      advBgLockRef.current = false
    }
  }, [run, advHistoryArt])

  /**
   * Alyf Advisor turn. Stages the payload (patient window + conversation history)
   * as an artifact, then runs the advisor program, which applies the guardrail
   * layer and answers from the research corpus + this patient's engine output.
   * The engine, the prompt, the retrieval and the guardrails are untouched here —
   * this path only ferries the question out and the reply back.
   *
   * Reliability: the reply is resolved by polling the reply artifact until the
   * doc carrying THIS turn's id lands (a single reload can legitimately observe
   * the previous turn's doc), with a hard timeout, and `advBusy` is always
   * cleared in `finally` so the composer can never latch shut.
   */
  const runMemoryOp = useCallback(
    async (payload: Record<string, unknown>, description: string) => {
      if (advBusyRef.current) {
        setAdvMemoryError('Wait for the Advisor to finish answering first.')
        return false
      }
      if (advBgLockRef.current) {
        setAdvMemoryError('Another save is still finishing — try again in a moment.')
        return false
      }
      const op = String(payload.op || '')
      const iid = payload.id != null ? String(payload.id) : ''
      // Optimistic UI — sticky content-addressed view blobs otherwise make mute/edit look dead.
      if (iid && (op === 'mute' || op === 'delete' || op === 'clinician_dismiss')) {
        setMemItemOverrides((prev) => ({
          ...prev,
          [iid]:
            op === 'delete'
              ? { _removed: true, status: 'deleted' }
              : { status: 'dismissed', dismissed: true },
        }))
      } else if (iid && op === 'edit' && typeof payload.text === 'string') {
        setMemItemOverrides((prev) => ({
          ...prev,
          [iid]: { text: String(payload.text), status: 'active', dismissed: false },
        }))
      } else if (iid && op === 'clinician_restore') {
        setMemItemOverrides((prev) => ({
          ...prev,
          [iid]: { status: 'active', dismissed: false },
        }))
      } else if (op === 'clear') {
        setMemItemOverrides((prev) => {
          const next = { ...prev }
          for (const it of memoryItemsBase) {
            next[it.id] = { _removed: true, status: 'deleted' }
          }
          return next
        })
      }
      advBgLockRef.current = true
      try {
        const body = JSON.stringify(payload)
        const saved = await run({
          description,
          packageName: 'bap_oura_dashboard',
          entrypoint: 'save_advisor_memory.py',
          arguments: [body],
        })
        if (saved.outcome !== 'completed') {
          throw new Error(saved.error || 'Memory save failed')
        }
        setAdvMemoryError(null)
        try {
          await advMemoryArt.reload()
        } catch {
          /* best-effort */
        }
        try {
          await advPortraitArt.reload()
        } catch {
          /* best-effort — portrait sidecar */
        }
        try {
          await advMemoryItemsArt.reload()
        } catch {
          /* best-effort — items sidecar */
        }
        setMemItemOverrides({})
        return true
      } catch (e) {
        setMemItemOverrides({})
        setAdvMemoryError(e instanceof Error ? e.message : 'Memory save failed')
        return false
      } finally {
        advBgLockRef.current = false
      }
    },
    [run, advMemoryArt, advPortraitArt, advMemoryItemsArt, memoryItemsBase],
  )

  const runMemoryPipeline = useCallback(
    async (msgs: AdvisorMsg[]) => {
      if (!memoryLearningOn) return
      if (advBusyRef.current || advBgLockRef.current) return
      const slim = (msgs || [])
        .filter((m) => !m.failed)
        .slice(-16)
        .map((m) => ({ role: m.role, content: m.content }))
      if (!slim.some((m) => m.role === 'user')) return
      advBgLockRef.current = true
      try {
        await run({
          description: 'Alyf Advisor — update personalization memory',
          packageName: 'bap_oura_dashboard',
          entrypoint: 'save_advisor_memory.py',
          arguments: [
            JSON.stringify({
              op: 'pipeline',
              messages: slim,
              path: pathKey,
              start_idx: startIdx,
              end_idx: endIdx,
              run_data_pass: true,
            }),
          ],
        })
        try {
          await advMemoryArt.reload()
        } catch {
          /* ignore */
        }
      } catch {
        /* never break chat */
      } finally {
        advBgLockRef.current = false
      }
    },
    [memoryLearningOn, run, advMemoryArt, pathKey, startIdx, endIdx],
  )

  const applyMemoryProposal = async (proposal: MemoryProposal) => {
    const action = proposal.action || 'none'
    if (action === 'set_learning') {
      return runMemoryOp(
        { op: 'set_learning', enabled: !!proposal.enabled },
        'Set Alyf personalization learning',
      )
    }
    if (action === 'clear') {
      return runMemoryOp({ op: 'clear' }, 'Clear Alyf personalization memory')
    }
    if (action === 'delete' && proposal.item_id) {
      return runMemoryOp(
        { op: 'delete', id: proposal.item_id },
        'Delete personalization memory item',
      )
    }
    if (action === 'edit' && proposal.item_id && proposal.item_text) {
      return runMemoryOp(
        { op: 'edit', id: proposal.item_id, text: proposal.item_text },
        'Edit personalization memory item',
      )
    }
    if (action === 'add' && proposal.item_text) {
      return runMemoryOp(
        {
          op: 'add',
          type: proposal.item_type || 'stated',
          text: proposal.item_text,
          provenance_summary: 'Confirmed by patient in chat',
        },
        'Add personalization memory item',
      )
    }
    return false
  }

  const advUpdateMemoryProposal = (msgId: string, next: MemoryProposal) => {
    setAdvConvos((prev) =>
      prev.map((c) =>
        c.id !== advActiveId
          ? c
          : {
              ...c,
              msgs: c.msgs.map((m) =>
                m.id === msgId ? { ...m, memoryProposal: next } : m,
              ),
            },
      ),
    )
  }

  const advSetMemoryUi = (msgId: string, ui: AdvisorMsg['memoryUi']) => {
    setAdvConvos((prev) =>
      prev.map((c) =>
        c.id !== advActiveId
          ? c
          : {
              ...c,
              msgs: c.msgs.map((m) => (m.id === msgId ? { ...m, memoryUi: ui } : m)),
            },
      ),
    )
  }

  const advAcceptMemoryProposal = async (msgId: string) => {
    const msg = advMsgs.find((m) => m.id === msgId)
    const proposal = msg?.memoryProposal
    if (!proposal) return
    const ok = await applyMemoryProposal(proposal)
    if (ok) advSetMemoryUi(msgId, 'saved')
  }

  const advCancelMemoryProposal = (msgId: string) => {
    advSetMemoryUi(msgId, 'cancelled')
  }

  const sendAdvisorTurn = async (rawText: string) => {
    const text = rawText.trim()
    if (!text || advBusy || !bundle) return

    // Past transcripts opened from history are read-only — start a fresh writable chat.
    const activeIsReadOnly = !!(
      advActive?.readOnly ||
      (advActiveId && advConvos.find((c) => c.id === advActiveId)?.readOnly)
    )
    const convoId = activeIsReadOnly
      ? advStartConvo()
      : advActiveId ?? advEnsureDraft()
    const turnId = uid()
    // Fresh chats start empty; for a live thread use the in-memory messages.
    const priorMsgs = activeIsReadOnly
      ? []
      : advConvos.find((c) => c.id === convoId)?.msgs ?? []

    // Instant echo: the question is on screen and the draft cleared before any run.
    advAppend(convoId, [{ id: `u_${turnId}`, role: 'user', content: text }])
    setAdvInput('')
    setAdvBusy(true)
    advBusyRef.current = true
    // Cancel any pending history write — history never races the Advisor turn.
    if (advPersistTimer.current) {
      clearTimeout(advPersistTimer.current)
      advPersistTimer.current = null
    }
    advStickRef.current = true
    advAbortRef.current = { turnId, cancelled: false }

    const failWith = (message: string) => {
      advAppend(convoId, [
        { id: `e_${turnId}`, role: 'assistant', content: message, failed: true, retryText: text },
      ])
    }

    const normalizeReply = (raw: unknown): AdvisorReplyDoc | null => {
      if (!raw) return null
      if (typeof raw === 'string') {
        try {
          const parsed = JSON.parse(raw)
          return parsed && typeof parsed === 'object' ? (parsed as AdvisorReplyDoc) : null
        } catch {
          return null
        }
      }
      if (typeof raw === 'object') return raw as AdvisorReplyDoc
      return null
    }

    try {
      // Only the tail of the thread is forwarded — the whole conversation stays
      // on screen, but the payload has to fit in a single argv string.
      const history = priorMsgs
        .filter((m) => !m.failed)
        .slice(-ADV_HISTORY_TURNS * 2)
        .map((m) => ({
          role: m.role,
          content:
            m.content.length > ADV_HISTORY_CHARS
              ? `${m.content.slice(0, ADV_HISTORY_CHARS)}…`
              : m.content,
        }))

      const payload = JSON.stringify({
        turn_id: turnId,
        message: text,
        history,
        path: pathKey,
        start_idx: startIdx,
        end_idx: endIdx,
      })
      // Single program run (latency pack): payload via argv — alyf_advisor also
      // mirrors to alyf_advisor_turn_pending. stage_advisor_turn kept for legacy.
      const answered = await run({
        description: 'Alyf Advisor — answer from this patient’s BAP data and research',
        packageName: 'bap_oura_dashboard',
        entrypoint: 'alyf_advisor.py',
        arguments: [payload],
      })
      if (advAbortRef.current?.cancelled) return
      if (answered.outcome !== 'completed') {
        throw new Error(answered.error || 'The Advisor could not answer that')
      }

      // Resolve the reply for THIS turn. A single reload can legitimately still
      // observe the previous turn's doc, so poll until the ids match. The latest
      // artifact value is read off a ref, which an effect keeps in sync — that
      // works whether or not `reload()` resolves with the new value.
      // Also accept a value returned directly from reload() when the hook provides it.
      const deadline = Date.now() + ADV_REPLY_TIMEOUT_MS
      let doc: AdvisorReplyDoc | null = null
      let delay = 120
      while (Date.now() < deadline) {
        let reloaded: unknown = null
        try {
          reloaded = await advReplyArt.reload()
        } catch {
          // A failed refetch is not fatal — keep polling until the deadline.
        }
        // Prefer reload() return value if it looks like data; else ref synced by effect.
        const fromReload = normalizeReply(reloaded)
        const fromData =
          advReplyArt.status === 'ready' ? normalizeReply(advReplyArt.data) : null
        const fromRef = normalizeReply(advReplyRef.current)
        for (const candidate of [fromReload, fromData, fromRef]) {
          if (candidate && candidate.turn_id === turnId && candidate.reply) {
            doc = candidate
            break
          }
        }
        if (doc) break
        await new Promise((r) => setTimeout(r, delay))
        if (advAbortRef.current?.cancelled) return
        delay = Math.min(700, Math.round(delay * 1.3))
      }
      if (advAbortRef.current?.cancelled) return

      if (!doc) {
        failWith('That took longer than expected and I didn’t get an answer back.')
        return
      }

      const msgId = `a_${turnId}`
      const proposal = doc.annotation_proposal || null
      const hasProposal =
        !!proposal &&
        proposal.status === 'proposal' &&
        !proposal.needs_clarification &&
        !!proposal.start_date &&
        !!proposal.end_date
      advAppend(convoId, [
        {
          id: msgId,
          role: 'assistant',
          content: doc.reply,
          category: doc.category,
          grounding: doc.grounding_verdict,
          stages: doc.stages,
          sources: doc.sources_used,
          escalation: doc.symptom_escalation,
          annotationProposal: hasProposal ? proposal : proposal?.needs_clarification ? proposal : null,
          annotationUi: hasProposal ? 'pending' : undefined,
          memoryProposal: doc.memory_proposal || null,
          memoryUi:
            doc.memory_proposal &&
            doc.memory_proposal.status === 'proposal' &&
            !doc.memory_proposal.needs_clarification
              ? 'pending'
              : undefined,
          // Suppress exploratory chips after emergency / hard out-of-scope turns.
          followUps:
            doc.symptom_escalation || doc.category === 'symptom'
              ? []
              : Array.isArray(doc.follow_ups)
                ? doc.follow_ups.filter((q) => typeof q === 'string' && q.trim()).slice(0, 4)
                : [],
        },
      ])
      // Stream the arrived reply in. Reduced motion shows it whole.
      if (!advReduceMotion) {
        setAdvStreamChars(0)
        setAdvStreamId(msgId)
      }
    } catch (e) {
      if (!advAbortRef.current?.cancelled) {
        failWith(e instanceof Error ? e.message : 'Something went wrong getting that answer.')
      }
    } finally {
      // Unconditional: the input re-enables after every turn, success or not.
      if (advAbortRef.current?.turnId === turnId) advAbortRef.current = null
      advBusyRef.current = false
      setAdvBusy(false)
      // Defer background I/O until React has flushed the assistant append into
      // advConvosRef — prevents history saves that drop the just-arrived reply.
      if (advPersistTimer.current) clearTimeout(advPersistTimer.current)
      advPersistTimer.current = setTimeout(() => {
        if (advBusyRef.current) return
        const snap = advConvosRef.current
        if (snap.length) advPendingPersistRef.current = snap
        const pending = advPendingPersistRef.current
        if (pending) void advPersistHistory(pending)
        // Memory pipeline only after history has a chance to start; lock serializes.
        setTimeout(() => {
          if (advBusyRef.current) return
          const msgs =
            advConvosRef.current.find((c) => c.id === convoId)?.msgs || []
          void runMemoryPipeline(msgs)
        }, 500)
      }, 450)
    }
  }

  // Mirror the reply artifact into a ref so the turn handler's polling loop can
  // read the newest value without re-subscribing to React state.
  useEffect(() => {
    advReplyRef.current = advReply
  }, [advReply])

  useEffect(() => {
    advConvosRef.current = advConvos
  }, [advConvos])

  /** Stop waiting on the in-flight turn without wedging the composer. */
  const advStop = () => {
    if (advAbortRef.current) advAbortRef.current.cancelled = true
    setAdvStreamId(null)
    advBusyRef.current = false
    setAdvBusy(false)
    advInputRef.current?.focus()
    const pending = advPendingPersistRef.current
    if (pending) {
      if (advPersistTimer.current) clearTimeout(advPersistTimer.current)
      advPersistTimer.current = setTimeout(() => {
        void advPersistHistory(pending)
      }, 400)
    }
  }

  const advRetry = (msg: AdvisorMsg) => {
    if (!msg.retryText) return
    setAdvConvos((prev) =>
      prev.map((c) => ({ ...c, msgs: c.msgs.filter((m) => m.id !== msg.id) })),
    )
    void sendAdvisorTurn(msg.retryText)
  }

  // Type-out: advance a character budget on an animation frame while streaming.
  useEffect(() => {
    if (!advStreamId) return
    const target = advMsgs.find((m) => m.id === advStreamId)
    if (!target) {
      setAdvStreamId(null)
      return
    }
    const total = target.content.length
    const started = performance.now()
    let raf = 0
    const step = (now: number) => {
      const shown = Math.min(total, Math.round(((now - started) / 1000) * STREAM_CPS))
      setAdvStreamChars(shown)
      if (shown < total) {
        raf = requestAnimationFrame(step)
      } else {
        setAdvStreamId(null)
      }
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [advStreamId, advMsgs])

  /**
   * Sticky autoscroll: follow new content, but disengage the moment the user
   * scrolls up to read, and re-engage when they come back to the bottom.
   */
  const onAdvScroll = () => {
    const el = advScrollRef.current
    if (!el) return
    advStickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48
  }

  useEffect(() => {
    const el = advScrollRef.current
    if (el && advOpen && advStickRef.current) el.scrollTop = el.scrollHeight
  }, [advMsgs, advBusy, advOpen, advStreamChars])

  // Load history then memory serially on drawer open — never in parallel with each
  // other or with a turn (parallel run() was glitching history + dropping turns).
  useEffect(() => {
    if (!advOpen) return
    let cancelled = false
    ;(async () => {
      // Wait until any in-flight turn finishes before loading.
      const waitIdle = async () => {
        for (let i = 0; i < 80 && (advBusyRef.current || advBgLockRef.current); i++) {
          await new Promise((r) => setTimeout(r, 100))
          if (cancelled) return
        }
      }
      await waitIdle()
      if (cancelled) return
      advBgLockRef.current = true
      try {
        try {
          const loaded = await run({
            description: 'Load Alyf Advisor chat history (purge expired transcripts)',
            packageName: 'bap_oura_dashboard',
            entrypoint: 'load_advisor_history.py',
          })
          if (cancelled) return
          if (loaded.outcome !== 'completed') {
            setAdvHistoryError(loaded.error || 'Could not load chat history')
          } else {
            setAdvHistoryError(null)
          }
          try {
            await advHistoryArt.reload()
          } catch {
            /* ignore */
          }
        } catch (e) {
          if (!cancelled) {
            setAdvHistoryError(e instanceof Error ? e.message : 'Could not load chat history')
          }
        } finally {
          if (!cancelled) setAdvHistoryReady(true)
        }
        if (cancelled || advBusyRef.current) return
        try {
          await run({
            description: 'Load Alyf personalization memory',
            packageName: 'bap_oura_dashboard',
            entrypoint: 'load_advisor_memory.py',
          })
          if (!cancelled) {
            try {
              await advMemoryArt.reload()
            } catch {
              /* ignore */
            }
          }
        } catch {
          /* non-fatal */
        }
      } finally {
        advBgLockRef.current = false
      }
    })()
    return () => {
      cancelled = true
    }
    // Only when the drawer opens — not on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [advOpen])

  // Hydrate React state from the session-view artifact (purged transcripts only).
  useEffect(() => {
    if (!advHistoryDoc) return
    // Never thrash live chat while a turn is answering — that was wiping prompts.
    if (advBusyRef.current) return
    const incoming = (advHistoryDoc.conversations || [])
      .map((c) =>
        advWithExpiry({
          id: String(c.id),
          title: c.title || 'Conversation',
          createdAt: Number(c.createdAt) || Date.now(),
          updatedAt: Number(c.updatedAt) || Number(c.createdAt) || Date.now(),
          expiresAt: Number(c.expiresAt) || undefined,
          msgs: Array.isArray(c.msgs) ? c.msgs : [],
          readOnly: true,
        }),
      )
      .filter((c) => c.msgs.length > 0)
    const purged = advPurgeExpired(incoming)
    // Merge: keep any in-flight local draft (empty or actively messaging) that
    // isn't in the store yet; prefer local non-readOnly active thread content.
    // Never replace a live (writable) thread with a thinner store snapshot.
    setAdvConvos((prev) => {
      // Always keep local writable threads (empty new-chat drafts + live msgs).
      const localLive = prev.filter((c) => !c.readOnly)
      const localIds = new Set(localLive.map((c) => c.id))

      if (!advHydratedRef.current) {
        advHydratedRef.current = true
        const fromStore = purged.filter((c) => !localIds.has(c.id))
        // Store history + local draft so history is available BEFORE first send.
        return [...localLive, ...fromStore].sort(
          (a, b) => (b.updatedAt || 0) - (a.updatedAt || 0),
        )
      }
      // Store is SoT for history after first hydrate. Live writable threads win.
      // Do NOT keep longer local readOnly copies when store dropped them (delete resurrection).
      const map = new Map<string, AdvisorConvo>()
      for (const c of purged) map.set(c.id, { ...c, readOnly: true })
      for (const c of prev) {
        if (!c.readOnly) {
          map.set(c.id, c)
        }
      }
      return Array.from(map.values()).sort(
        (a, b) => (b.updatedAt || 0) - (a.updatedAt || 0),
      )
    })
  }, [advHistoryDoc, advActiveId])

  // Refresh "deletes in N days" labels about once a minute while history is open.
  useEffect(() => {
    if (!advOpen || !advHistoryOpen) return
    const t = setInterval(() => setAdvTick((n) => n + 1), 60_000)
    return () => clearInterval(t)
  }, [advOpen, advHistoryOpen])

  // Client-side TTL sweep whenever the drawer is open (storage-layer purge also
  // runs on load/save; this keeps the UI honest without waiting for a round-trip).
  useEffect(() => {
    if (!advOpen) return
    const sweep = () => {
      setAdvConvos((prev) => {
        const next = advPurgeExpired(prev)
        if (next.length !== prev.length) {
          // Active id may have been purged.
          setAdvActiveId((cur) => (cur && !next.some((c) => c.id === cur) ? null : cur))
          advSchedulePersist(next)
        }
        return next.length === prev.length ? prev : next
      })
    }
    sweep()
    const t = setInterval(sweep, 5 * 60_000)
    return () => clearInterval(t)
  }, [advOpen, advSchedulePersist])

  // Focus the composer when the drawer opens, and seed a draft WITHOUT closing history.
  // (advStartConvo closes the history panel — that made history unreachable on a fresh chat.)
  useEffect(() => {
    if (!advOpen) return
    if (!advActiveId) advEnsureDraft()
    const t = setTimeout(() => {
      if (!advActive?.readOnly) advInputRef.current?.focus()
    }, 60)
    return () => clearTimeout(t)
  }, [advOpen, advActiveId, advEnsureDraft, advActive?.readOnly])

  // Internal QA escape hatch for the review trace (Shift+Alt+T). Not patient chrome.
  useEffect(() => {
    if (!advOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.shiftKey && e.altKey && (e.key === 'T' || e.key === 't')) {
        e.preventDefault()
        setAdvShowTrace((v) => !v)
      } else if (e.key === 'Escape') {
        setAdvOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [advOpen])

  const handleRefresh = async () => {
    setUploadError(null)
    const resultRun = await run({
      description: 'Rebuild the BAP dashboard free-range data bundle from Oura trends',
      packageName: 'bap_oura_dashboard',
      entrypoint: 'build_dashboard_bundle.py',
    })
    if (resultRun.outcome === 'completed') {
      bundleArt.reload()
      annArt.reload()
      setLocalAnns(null)
      setForceEmpty(false)
      setSliderEpoch((e) => e + 1)
    } else {
      setUploadError(resultRun.error || 'Refresh failed')
    }
  }

  const handleUploadFile = async (file: File) => {
    setUploadError(null)
    setUploadBusy(true)
    try {
      const name = file.name || 'upload.csv'
      const lower = name.toLowerCase()
      if (!lower.endsWith('.csv') && !lower.endsWith('.json') && !file.type.includes('csv') && !file.type.includes('json') && file.type !== 'text/plain') {
        throw new Error('Unsupported file type. Please upload a .csv or .json wearable export.')
      }
      if (file.size > 8 * 1024 * 1024) {
        throw new Error('File is larger than 8 MB. Please upload a smaller export.')
      }
      const text = await file.text()
      if (!text.trim()) throw new Error('The file is empty.')

      // Client-side parse + validate for specific errors
      let headers: string[]
      let rows: Record<string, unknown>[]
      const trimmed = text.trimStart()
      const looksJson =
        lower.endsWith('.json') || trimmed.startsWith('[') || trimmed.startsWith('{')
      if (looksJson) {
        const parsed = parseJsonClient(text)
        headers = parsed.headers
        rows = parsed.rows
      } else {
        const parsed = parseCsvClient(text)
        headers = parsed.headers
        rows = parsed.rows
      }

      const { dateCol } = validateClientSchema(headers)
      const range = peekDateRange(rows, dateCol)

      // Optimistic meta
      setUploadInfo({
        fileName: name,
        rowCount: range.count,
        dateStart: range.start,
        dateEnd: range.end,
      })

      const payload = JSON.stringify({
        file_name: name,
        content: text,
        subject: 'Uploaded subject',
        profile_label: 'Uploaded wearable data',
      })

      // Stage via artifact (avoid huge argv when possible — stage_upload accepts argv fallback)
      const stage = await run({
        description: 'Stage wearable file for BAP scoring',
        packageName: 'bap_oura_dashboard',
        entrypoint: 'stage_upload.py',
        arguments: [payload],
      })
      if (stage.outcome !== 'completed') {
        throw new Error(
          stage.error ||
            'Could not stage the upload payload. Try a shorter export if the file is very large.',
        )
      }

      const ingest = await run({
        description: 'Validate wearable upload and rebuild BAP dashboard bundle',
        packageName: 'bap_oura_dashboard',
        entrypoint: 'ingest_user_data.py',
      })
      if (ingest.outcome !== 'completed') {
        throw new Error(ingest.error || 'Scoring failed after upload.')
      }

      setForceEmpty(false)
      bundleArt.reload()
      annArt.reload()
      setLocalAnns(null)
      setSliderEpoch((e) => e + 1)
      setLoTick(null) // force re-init to full range
      setHiTick(null)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setUploadError(msg)
    } finally {
      setUploadBusy(false)
    }
  }

  const handleClearUpload = () => {
    setForceEmpty(true)
    setUploadInfo(null)
    setUploadError(null)
    setLoTick(null)
    setHiTick(null)
    setDrillAxis(null)
  }

  const onRangeChange = (nextLo: number, nextHi: number) => {
    setLoTick(nextLo)
    setHiTick(nextHi)
  }

  // ---------- Shell chrome (always available) ----------
  const chrome = (body: ReactNode) => (
    <div className={`app ${dark ? 'dark' : ''}`}>
      <div className="shell">
        <header className="topbar">
          <div className="brand">
            <h1>
              <Tip tip={HEADER_TIPS.bap}>Balanced Activity Profile</Tip>
            </h1>
            <p className="subtitle">
              Clinical activity status · dual-anchor scoring · personal trend view
            </p>
          </div>
          <div className="actions">
            <button
              type="button"
              className="btn icon ghost"
              aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
              onClick={() => setDark((v) => !v)}
            >
              {dark ? <IconSun /> : <IconMoon />}
            </button>
            {bundle ? (
              <>
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => setAdvOpen(true)}
                  aria-label="Ask the Alyf Advisor about this score"
                >
                  <IconChat />
                  Ask the Advisor
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => setAnnOpen(true)}
                  aria-label="Open annotations"
                >
                  <IconNote />
                  Notes
                  {annotations.length > 0 ? (
                    <span className="badge-count">{annotations.length}</span>
                  ) : null}
                </button>
                <button
                  type="button"
                  className="btn primary"
                  onClick={handleRefresh}
                  disabled={isRunning || uploadBusy}
                >
                  <IconRefresh />
                  {isRunning || uploadBusy ? 'Working…' : 'Refresh'}
                </button>
              </>
            ) : null}
          </div>
        </header>
        {body}
      </div>
    </div>
  )

  // Loading skeleton
  if (bundleArt.status === 'loading' && !forceEmpty) {
    return chrome(
      <div className="state" aria-busy="true" aria-live="polite">
        <div className="state-card" style={{ alignItems: 'stretch' }}>
          <div className="skeleton skel-ring" style={{ alignSelf: 'center' }} />
          <div className="skeleton skel-row" style={{ width: '60%' }} />
          <div className="skeleton skel-row" style={{ width: '80%' }} />
          <div className="skeleton skel-block" />
          <p className="muted" style={{ textAlign: 'center', margin: 0 }}>
            Loading interactive BAP dashboard…
          </p>
        </div>
      </div>,
    )
  }

  // Empty state (no data / cleared / hard error without prior bind)
  if (forceEmpty || bundleArt.status === 'error' || !bundle) {
    const hardErr =
      !forceEmpty && bundleArt.status === 'error'
        ? `Could not load the existing data bundle. ${bundleArt.error || ''} You can still upload a new file.`
        : null
    return chrome(
      <div className="state">
        <div className="state-card">
          <h2 className="state-title">Score a wearable export</h2>
          <p className="state-copy">
            Drop a CSV or JSON file to run the Balanced Activity Profile engine and explore the
            full interactive dashboard.
          </p>
          <div style={{ width: '100%' }}>
            <Tip tip={HEADER_TIPS.dropbox}>
              <DataDropBox
                info={null}
                busy={uploadBusy}
                error={uploadError || hardErr}
                onFile={handleUploadFile}
                onClear={handleClearUpload}
              />
            </Tip>
          </div>
          {!forceEmpty && bundleArt.status !== 'error' ? (
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => {
                setForceEmpty(false)
                bundleArt.reload()
              }}
            >
              Reload last scored data
            </button>
          ) : null}
        </div>
      </div>,
    )
  }

  if (!result || !entry) {
    return chrome(
      <div className="state">
        <div className="state-card">
          <div className="skeleton skel-block" />
          <p className="muted" style={{ margin: 0 }}>
            Preparing window scores…
          </p>
        </div>
      </div>,
    )
  }

  const bap = result.bap ?? 0
  const d = result.detail
  const conditions = bundle.conditions_flag || []
  const displayHours = Math.max(12, windowDays * 24)
  const axisKeys = ['I', 'R', 'V', 'Sp', 'C'] as const
  // Shared sufficiency check (same rule as Alyf Advisor / axis_availability.py):
  // engine detail.axis_available[k] !== false AND axis_scores[k] != null.
  const axisAvailMap: Record<string, boolean> = Object.fromEntries(
    axisKeys.map((k) => [
      k,
      result.detail?.axis_available?.[k] !== false && result.axis_scores?.[k] != null,
    ]),
  )
  const unavailableAxisKeys = axisKeys.filter((k) => !axisAvailMap[k])
  const availableAxisKeys = axisKeys.filter((k) => axisAvailMap[k])
  const isPartialBap = unavailableAxisKeys.length > 0
  const bapDisplay = result.bap == null ? null : bap

  const prevPoint = chartPoints.length >= 2 ? chartPoints[chartPoints.length - 2] : null
  const prevBap = prevPoint?.BAP ?? null
  const delta = prevBap != null && result.bap != null ? result.bap - prevBap : null
  const chartEmpty = lineData.length === 0 || chartPoints.length === 0

  const sexLabel =
    bundle.sex === 'M' ? 'male' : bundle.sex === 'F' ? 'female' : bundle.sex

  return chrome(
    <>
      {/* Controls: tabs + drop + slider */}
      <section className="card">
        <div className="controls">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="tabs" role="tablist" aria-label="Score views">
              <button
                role="tab"
                aria-selected={tab === 'bap'}
                className={`tab ${tab === 'bap' ? 'active' : ''}`}
                onClick={() => setTab('bap')}
              >
                <Tip tip={HEADER_TIPS.pure_tab}>Balanced Activity Profile</Tip>
              </button>
              <button
                role="tab"
                aria-selected={tab === 'trend'}
                className={`tab ${tab === 'trend' ? 'active' : ''}`}
                onClick={() => setTab('trend')}
              >
                <Tip tip={HEADER_TIPS.trend_tab}>Trend Analysis</Tip>
              </button>
            </div>
            <Tip tip={HEADER_TIPS.dropbox}>
              <DataDropBox
                info={uploadInfo}
                busy={uploadBusy}
                error={uploadError}
                onFile={handleUploadFile}
                onClear={handleClearUpload}
                compact
              />
            </Tip>
          </div>

          <div>
            <div className="range-head">
              <h2 style={{ margin: 0, fontSize: 'var(--fs-md)' }}>
                <Tip tip={HEADER_TIPS.slider}>Analysis time window</Tip>
              </h2>
              <div className="range-val">
                {windowDays === 1 ? '1 day' : `${windowDays} days`} ({displayHours} hours) · 12-hour
                steps
              </div>
            </div>
            <DualRangeSlider
              minTick={0}
              maxTick={maxTick}
              lo={lo}
              hi={hi}
              onChange={onRangeChange}
              formatLabel={(t) => formatTickLabel(t, bundle.dates)}
            />
            <div className="range-inputs">
              <label>
                Start (day)
                <select
                  value={startIdx}
                  onChange={(e) => {
                    const s = Number(e.target.value)
                    const eIdx = Math.max(s, endIdx)
                    const t = dayRangeToTicks(s, eIdx, nDays)
                    setLoTick(t.lo)
                    setHiTick(t.hi)
                  }}
                >
                  {bundle.dates.map((d0, i) => (
                    <option key={d0} value={i} disabled={i > endIdx}>
                      {d0}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                End (day)
                <select
                  value={endIdx}
                  onChange={(e) => {
                    const eIdx = Number(e.target.value)
                    const s = Math.min(startIdx, eIdx)
                    const t = dayRangeToTicks(s, eIdx, nDays)
                    setLoTick(t.lo)
                    setHiTick(t.hi)
                  }}
                >
                  {bundle.dates.map((d0, i) => (
                    <option key={d0} value={i} disabled={i < startIdx}>
                      {d0}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p className="muted" style={{ margin: '8px 0 0' }}>
              {bundle.subject} · {bundle.profile_label} · age {bundle.age}, {sexLabel} ·{' '}
              {entry.date_start} → {entry.date_end} ·{' '}
              {tab === 'bap' ? 'population clinical view' : 'personal trend view'}
            </p>
          </div>
        </div>
      </section>

      <div className="grid">
        {/* Hero score + suggestions */}
        <section className="card span-7">
          <div className="row-between">
            <h2 style={{ margin: 0 }}>
              {tab === 'bap' ? (
                <Tip tip={HEADER_TIPS.pure_tab}>Clinical population score</Tip>
              ) : (
                <Tip tip={HEADER_TIPS.trend_tab}>Personal baseline blend score</Tip>
              )}
            </h2>
            <span className="pill">
              {tab === 'bap' ? 'Pure population' : 'Blend with personal history'}
            </span>
          </div>

          <div className="hero" style={{ marginTop: 12 }}>
            <ScoreRing value={bapDisplay == null ? 0 : bapDisplay} color={band.color} />
            <div style={{ minWidth: 0, width: '100%' }}>
              <div className="row-between">
                <span className="band" style={{ color: band.color }}>
                  {band.label}
                </span>
                {isPartialBap ? (
                  <Tip
                    tip={`This BAP uses only ${availableAxisKeys
                      .map((k) => AXIS_META[k]?.name || k)
                      .join(', ')}. Unavailable: ${unavailableAxisKeys
                      .map((k) => AXIS_META[k]?.name || k)
                      .join(', ')}. Missing axes are omitted (not scored as 0).`}
                  >
                    <span className="flag warn" style={{ marginLeft: 8 }}>
                      Partial · {availableAxisKeys.length}/5 axes
                    </span>
                  </Tip>
                ) : null}
                {delta != null && (
                  <span className={`delta ${delta >= 0 ? 'up' : 'down'}`}>
                    {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(2)} vs prior rolling point
                  </span>
                )}
              </div>
              <div className="hero-stats">
                <div className="stat">
                  <label>
                    <Tip tip={HEADER_TIPS.confidence}>Confidence</Tip>
                  </label>
                  <div className="stat-val">{pct(result.confidence, 0)}</div>
                  <div className="bar">
                    <span style={{ width: pct(result.confidence), background: 'var(--accent)' }} />
                  </div>
                </div>
                <div className="stat">
                  <label>
                    <Tip tip={HEADER_TIPS.reliability}>Reliability factor</Tip>
                  </label>
                  <div className="stat-val">{fmt(d.reliability_factor)}</div>
                </div>
                <div className="stat">
                  <label>
                    <Tip tip={HEADER_TIPS.scenario}>Spread scenario</Tip>
                  </label>
                  <div className="stat-val">
                    {d.scenario === 'A'
                      ? 'Daily aggregates'
                      : d.scenario === 'B'
                        ? 'Intraday bouts'
                        : d.scenario}
                  </div>
                </div>
                <div className="stat">
                  <label>
                    <Tip tip={HEADER_TIPS.couch}>Active-then-sit penalty</Tip>
                  </label>
                  <div className="stat-val">{fmt(d.P_couch, 3)}</div>
                </div>
                <div className="stat">
                  <label>
                    <Tip tip={HEADER_TIPS.window}>Window length</Tip>
                  </label>
                  <div className="stat-val">{Math.round(d.W_hours || displayHours)} hours</div>
                </div>
                <div className="stat">
                  <label>
                    <Tip tip="Population intensity backbone used inside the composite before balance adjustments.">
                      Intensity backbone
                    </Tip>
                  </label>
                  <div className="stat-val">{fmt(d.S_I_pop, 3)}</div>
                </div>
              </div>

              <div className="flags">
                {result.flags.length === 0 ? (
                  <span className="flag none">No scoring flags</span>
                ) : (
                  result.flags.map((f) => (
                    <Tip key={f} tip={FLAG_INFO[f]?.why || f}>
                      <span className="flag">{FLAG_INFO[f]?.label || humanVar(f)}</span>
                    </Tip>
                  ))
                )}
                {(['I', 'R', 'V', 'Sp', 'C'] as const).map((ak) => {
                  const axisOk =
                    result.detail?.axis_available?.[ak] !== false &&
                    result.axis_scores?.[ak] != null
                  if (axisOk) return null
                  const name = AXIS_META[ak]?.name || ak
                  const tips: Record<string, string> = {
                    I: 'Not enough intensity inputs in this window to compute Intensity.',
                    R: 'Not enough recovery inputs (HRV, resting heart rate, etc.) in this window.',
                    V: 'No workout-type or heart-rate zone data in this export, so Variety has no inputs and its weight is shared across the other axes.',
                    Sp: 'Not enough spread / activity-distribution inputs in this window to compute Spread.',
                    C: 'Not enough day-to-day steadiness history in this window to compute Consistency.',
                  }
                  return (
                    <Tip key={`axis-unavail-${ak}`} tip={tips[ak] || `${name} could not be scored in this window.`}>
                      <span className="flag warn">{name} unavailable</span>
                    </Tip>
                  )
                })}
                {conditions.length === 0 ? (
                  <Tip tip={HEADER_TIPS.conditions}>
                    <span className="flag none">No conditions flagged</span>
                  </Tip>
                ) : (
                  conditions.map((c) => (
                    <Tip key={c} tip={HEADER_TIPS.conditions}>
                      <span className="flag cond">Condition: {humanVar(String(c))}</span>
                    </Tip>
                  ))
                )}
                <Tip tip="Medical conditions are display-only flags. They do not modify the numeric score in this dashboard.">
                  <span className="flag none">Score not condition-adjusted</span>
                </Tip>
              </div>
            </div>
          </div>

          <div className="suggest-box">
            <h3 style={{ marginTop: 0 }}>
              <Tip tip={HEADER_TIPS.suggestions}>Predictions and recommendations</Tip>
            </h3>
            <p className="suggest-packaged">{suggestions?.packaged}</p>
            {suggestions?.why_line ? <p className="suggest-why">{suggestions.why_line}</p> : null}
            {suggestions?.predictions?.length ? (
              <ul className="pred-list">
                {suggestions.predictions.map((p, i) => (
                  <li key={i} className="pred-item">
                    <span className="pred-pts">+{fmt(p.predicted_bap_gain_points, 1)} pts</span>
                    <span>{p.action}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            {suggestions?.gates_fired?.length ? (
              <div className="flags" style={{ marginTop: 8 }}>
                {suggestions.gates_fired.map((g) => (
                  <span key={g} className="flag warn">
                    Gate: {humanVar(g)}
                  </span>
                ))}
              </div>
            ) : null}
            {suggestions?.standing_frame ? (
              <p className="frame-note">{suggestions.standing_frame}</p>
            ) : null}
          </div>
        </section>

        {/* Radar */}
        <section className="card span-5">
          <h2>
            <Tip tip="Five-axis radar of the current window. Each spoke is an axis score from 0 to 100.">
              Five-axis profile
            </Tip>
          </h2>
          <SaveAsImage fileName="bap-radar.png">
            <div className="chart">
              <ResponsiveRadar
                data={radarData}
                keys={['score']}
                indexBy="axis"
                maxValue={100}
                margin={{ top: 36, right: 48, bottom: 36, left: 48 }}
                gridShape="circular"
                gridLabelOffset={14}
                dotSize={8}
                dotColor={{ theme: 'background' }}
                dotBorderWidth={2}
                colors={['#38bdf8']}
                fillOpacity={0.35}
                blendMode="normal"
                motionConfig="gentle"
                theme={lineTheme}
              />
            </div>
          </SaveAsImage>
        </section>

        {/* Axes + drill-down */}
        <section className="card span-12">
          <div className="row-between">
            <h2 style={{ margin: 0 }}>
              <Tip tip="High-level axis scores. Use the magnifying glass on an axis to inspect its underlying variables and daily signals.">
                Axes
              </Tip>
            </h2>
            {drillAxis ? (
              <button type="button" className="btn sm" onClick={() => setDrillAxis(null)}>
                Close drill-down
              </button>
            ) : null}
          </div>
          <div className="axis-grid" style={{ marginTop: 8 }}>
            {axisKeys.map((k) => {
              const meta = AXIS_META[k]
              const score = result.axis_scores[k]
              const available = axisAvailMap[k] === true
              const active = drillAxis === k
              return (
                <div key={k} className={`axis-card ${available ? '' : 'dropped'}`}>
                  <div className="axis-top">
                    <div>
                      <div className="axis-name">
                        <Tip tip={meta.tip}>{meta.name}</Tip>
                        {!available ? (
                          <span className="tag" style={{ marginLeft: 6 }}>
                            unavailable
                          </span>
                        ) : null}
                      </div>
                      <div
                        className="axis-score"
                        style={{ color: available ? meta.color : 'var(--muted)' }}
                      >
                        {available ? fmt((score as number) * 100, 1) : '—'}
                      </div>
                    </div>
                    <button
                      type="button"
                      className={`mag ${active ? 'active' : ''}`}
                      title={`Inspect ${meta.name}`}
                      aria-label={`Drill down into ${meta.name}`}
                      onClick={() => setDrillAxis(active ? null : k)}
                    >
                      <IconSearch />
                    </button>
                  </div>
                  <div className="axis-bar">
                    <span
                      style={{
                        width: available
                          ? `${Math.max(0, Math.min(100, (score as number) * 100))}%`
                          : '0%',
                        background: meta.color,
                      }}
                    />
                  </div>
                  <div className="muted" style={{ marginTop: 6 }}>
                    Weight {fmt((d.inter_weights?.[k] ?? 0) * 100, 0)}%
                    {!available ? ' · unavailable in this data' : ''}
                  </div>
                </div>
              )
            })}
          </div>

          {drillAxis ? (
            <div className="drill">
              <div className="row-between">
                <h3 style={{ margin: 0 }}>
                  <Tip tip={AXIS_META[drillAxis].tip}>{AXIS_META[drillAxis].name} detail</Tip>
                </h3>
                <span className="muted">
                  One axis at a time · {entry.date_start} → {entry.date_end}
                </span>
              </div>

              <div className="scroll-x" style={{ marginTop: 8 }}>
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>
                        <Tip tip="Underlying measurement feeding this axis.">Variable</Tip>
                      </th>
                      <th>
                        <Tip tip="Whether this variable had usable data in the window.">Present</Tip>
                      </th>
                      <th>
                        <Tip tip="Final variable score from 0 to 1 after population and optional personal blend.">
                          Score
                        </Tip>
                      </th>
                      <th>
                        <Tip tip="Population-only score before personal blend.">Population score</Tip>
                      </th>
                      <th>
                        <Tip tip="Personal score versus your own baseline. Empty when no baseline applies.">
                          Personal score
                        </Tip>
                      </th>
                      <th>
                        <Tip tip="How strongly personal history is mixed in (0 means pure population).">
                          Personal mix
                        </Tip>
                      </th>
                      <th>
                        <Tip tip="Relative importance of this variable inside the axis.">Weight</Tip>
                      </th>
                      <th>
                        <Tip tip="Raw wearable statistic used as input.">Raw value</Tip>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {(result.subscores?.[drillAxis] || []).map((s) => (
                      <tr key={s.name} className={s.present ? '' : 'absent'}>
                        <td>
                          <Tip tip={VAR_TIPS[s.name] || `Input variable ${humanVar(s.name)}.`}>
                            {humanVar(s.name)}
                          </Tip>
                          {s.suppressed ? <span className="tag">suppressed</span> : null}
                        </td>
                        <td>{s.present ? 'Yes' : 'No'}</td>
                        <td>{fmt(s.s, 3)}</td>
                        <td>{fmt(s.s_pop, 3)}</td>
                        <td>{fmt(s.s_per, 3)}</td>
                        <td>{fmt(s.alpha, 2)}</td>
                        <td>{fmt((s.w || 0) * 100, 0)}%</td>
                        <td>
                          {typeof s.raw === 'number'
                            ? fmt(s.raw, 2)
                            : s.raw == null
                              ? '—'
                              : String(s.raw)}
                        </td>
                      </tr>
                    ))}
                    {drillAxis === 'I' && metExcludedFromScore(bundle) ? (
                      <tr key="met_level_display_only" className="absent">
                        <td>
                          <Tip tip={metVarTip(bundle)}>
                            {metVarLabel(bundle)}
                          </Tip>
                          <span className="tag">not used in score</span>
                        </td>
                        <td>Display only</td>
                        <td>—</td>
                        <td>—</td>
                        <td>—</td>
                        <td>—</td>
                        <td>—</td>
                        <td>
                          {(() => {
                            const ds = bundle?.daily_series?.met_level as
                              | (number | null)[]
                              | undefined
                            if (!ds || !ds.length) return '—'
                            const slice = ds.slice(startIdx, endIdx + 1).filter((v) => v != null) as number[]
                            if (!slice.length) return '—'
                            const avg = slice.reduce((a, b) => a + b, 0) / slice.length
                            return fmt(avg, 2)
                          })()}
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>

              {drillDaily.length > 0 ? (
                <>
                  <h3>
                    <Tip tip="Daily wearable signals for this axis inside the selected window.">
                      Daily signals
                    </Tip>
                  </h3>
                  <SaveAsImage fileName={`bap-axis-${drillAxis}.png`}>
                    <div className="chart lg">
                      <ResponsiveLine
                        data={drillDaily}
                        margin={{ top: 20, right: 20, bottom: 48, left: 48 }}
                        xScale={{ type: 'point' }}
                        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
                        axisBottom={{
                          tickRotation: -45,
                          format: (v: string) => String(v).slice(5),
                          tickValues:
                            drillDaily[0] && drillDaily[0].data.length > 12
                              ? drillDaily[0].data
                                  .filter((_, i) => i % Math.ceil(drillDaily[0].data.length / 8) === 0)
                                  .map((d) => d.x)
                              : undefined,
                        }}
                        axisLeft={{ tickValues: 5 }}
                        enablePoints={false}
                        enableGridX={false}
                        useMesh
                        theme={lineTheme}
                        legends={[
                          {
                            anchor: 'top-left',
                            direction: 'row',
                            translateY: -12,
                            itemWidth: 120,
                            itemHeight: 16,
                            symbolSize: 8,
                          },
                        ]}
                      />
                    </div>
                  </SaveAsImage>
                </>
              ) : (
                <p className="muted" style={{ marginTop: 8 }}>
                  No daily series available for this axis in the current export.
                </p>
              )}
            </div>
          ) : null}
        </section>

        {/* Trend chart with annotation markers + drag */}
        <section className="card span-12">
          <div className="row-between">
            <h2 style={{ margin: 0 }}>
              <Tip tip={HEADER_TIPS.rolling}>Rolling history (14-day BAP)</Tip>
            </h2>
            <div className="actions">
              <button
                type="button"
                className={`btn sm ${dragMode ? 'primary' : ''}`}
                onClick={() => {
                  setDragMode((v) => !v)
                  setDragSel(null)
                  dragActiveRef.current = false
                  if (!dragMode) setAnnOpen(true)
                }}
              >
                {dragMode ? 'Dragging time range…' : 'Drag time on chart'}
              </button>
              <button type="button" className="btn sm ghost" onClick={() => setAnnOpen(true)}>
                <IconNote /> Annotations
              </button>
              <button
                type="button"
                className="btn sm ghost"
                onClick={() => setMemClinicianOpen(true)}
                title="Patient personalization profile. Not a clinical record."
              >
                Memory
              </button>
            </div>
          </div>
          <p className="muted" style={{ margin: '4px 0 8px' }}>
            Shows a trailing 14-day score for each day inside the selected analysis window (
            {chartPoints.length} point{chartPoints.length === 1 ? '' : 's'}). Drag mode sets an
            annotation’s date range.
          </p>
          {dragMode ? (
            <div className="drag-hint">
              Click and drag across the chart to set the annotation start and end dates. Release to
              apply. You can also type dates in the annotation panel.
            </div>
          ) : null}
          <div className="toggles">
            {(
              [
                ['BAP', 'BAP'],
                ['Intensity', 'Intensity'],
                ['Recovery', 'Recovery'],
                ['Spread', 'Spread'],
                ['Consistency', 'Consistency'],
              ] as const
            ).map(([key, label]) => (
              <label key={key} className="toggle">
                <input
                  type="checkbox"
                  checked={seriesOn[key]}
                  onChange={() => setSeriesOn((s) => ({ ...s, [key]: !s[key] }))}
                />
                <Tip tip={`Show or hide the ${label} series on the rolling chart.`}>{label}</Tip>
              </label>
            ))}
          </div>
          {chartEmpty ? (
            <p className="muted" style={{ padding: '24px 0' }}>
              Not enough days in this analysis window for a 14-day rolling series. Widen the dual
              handles (need ends from day 14 onward) or pick a longer range.
            </p>
          ) : (
            <SaveAsImage fileName="bap-trend.png">
              <div className="chart-shell" ref={chartShellRef}>
                <div className="chart lg">
                  <ResponsiveLine
                    data={lineData}
                    margin={{ top: CHART_MT, right: CHART_MR, bottom: CHART_MB, left: CHART_ML }}
                    xScale={{ type: 'point' }}
                    yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false }}
                    axisBottom={{
                      tickRotation: -45,
                      format: (v: string) => String(v).slice(5),
                      tickValues:
                        chartDateList.length > 12
                          ? chartDateList.filter(
                              (_, i) => i % Math.ceil(chartDateList.length / 8) === 0,
                            )
                          : undefined,
                    }}
                    axisLeft={{ tickValues: 6 }}
                    colors={(s: { id: string | number }) => {
                      const id = String(s.id)
                      if (id === 'BAP') return '#38bdf8'
                      const key = Object.keys(AXIS_META).find((k) => AXIS_META[k].name === id)
                      return key ? AXIS_META[key].color : '#94a3b8'
                    }}
                    enablePoints={chartPoints.length <= 2}
                    pointSize={6}
                    enableGridX={false}
                    useMesh={!dragMode}
                    isInteractive={!dragMode}
                    theme={lineTheme}
                    legends={[
                      {
                        anchor: 'top-left',
                        direction: 'row',
                        translateY: -10,
                        itemWidth: 90,
                        itemHeight: 16,
                        symbolSize: 10,
                      },
                    ]}
                  />
                </div>
                {/* Annotation overlay: bands/lines in plot + icon lane at bottom axis.
                    Labels live in hover/pin cards only — never on-chart text. */}
                <div className="chart-ann-layer" aria-label="Timeline annotations">
                  {chartAnnLayout.clusters.map((cl) => {
                    const multi = cl.items.length > 1
                    const plot = `(100% - ${CHART_ML + CHART_MR}px)`
                    const anchorLeft = `calc(${CHART_ML}px + ${plot} * ${cl.anchorPct / 100})`
                    const single = !multi ? cl.items[0] : null
                    const isPoint = single ? single.st.isPoint : false
                    const hot =
                      multi
                        ? cl.items.some(
                            (it) =>
                              hoverAnn === it.ann.id ||
                              pinnedChartAnn === it.ann.id ||
                              expandedAnn === it.ann.id,
                          ) || openAnnCluster === cl.id
                        : !!(
                            single &&
                            (hoverAnn === single.ann.id ||
                              pinnedChartAnn === single.ann.id ||
                              expandedAnn === single.ann.id)
                          )
                    const targetKind = single
                      ? single.ann.target_type === 'score'
                        ? 'score'
                        : 'axis'
                      : 'mix'
                    const viaAdvisor = single
                      ? single.ann.source === 'patient_via_advisor'
                      : cl.items.some((it) => it.ann.source === 'patient_via_advisor')

                    return (
                      <div key={cl.id} className="chart-ann-group">
                        {/* Range bands (only non-point members) — soft fill, stronger edges */}
                        {cl.items.map(({ ann, st }) =>
                          st.isPoint ? null : (
                            <div
                              key={`band_${ann.id}`}
                              className={`chart-ann-band ${hot ? 'hot' : ''} tgt-${ann.target_type === 'score' ? 'score' : 'axis'}`}
                              style={{
                                left: st.left,
                                width: st.width,
                                top: CHART_MT,
                                bottom: CHART_MB,
                              }}
                              aria-hidden
                            />
                          ),
                        )}
                        {/* Point marker lines */}
                        {cl.items.map(({ ann, st }) =>
                          st.isPoint ? (
                            <div
                              key={`line_${ann.id}`}
                              className={`chart-ann-line ${hot ? 'hot' : ''} tgt-${ann.target_type === 'score' ? 'score' : 'axis'}`}
                              style={{
                                left: st.left,
                                top: CHART_MT,
                                bottom: CHART_MB,
                              }}
                              aria-hidden
                            />
                          ) : null,
                        )}
                        {/* Bottom-lane hit target + icon / cluster count */}
                        <button
                          type="button"
                          className={`chart-ann-pin ${multi ? 'cluster' : isPoint ? 'point' : 'range'} tgt-${targetKind} ${hot ? 'hot' : ''} ${viaAdvisor ? 'via-advisor' : ''}`}
                          style={{ left: anchorLeft }}
                          aria-label={
                            multi
                              ? `${cl.items.length} annotations near this time. Activate to list.`
                              : `${single!.ann.label || 'Annotation'}: ${resolveAnnTarget(single!.ann)}. ${resolveAnnTime(single!.ann)}`
                          }
                          tabIndex={0}
                          onMouseEnter={() => {
                            if (!multi) setHoverAnn(single!.ann.id)
                            else setHoverAnn(cl.items[0].ann.id)
                          }}
                          onMouseLeave={() => {
                            if (!pinnedChartAnn && openAnnCluster !== cl.id) setHoverAnn(null)
                          }}
                          onFocus={() => {
                            if (!multi) setHoverAnn(single!.ann.id)
                          }}
                          onBlur={() => {
                            if (!pinnedChartAnn && openAnnCluster !== cl.id) setHoverAnn(null)
                          }}
                          onClick={(e) => {
                            e.stopPropagation()
                            if (multi) {
                              setOpenAnnCluster((cur) => (cur === cl.id ? null : cl.id))
                              setPinnedChartAnn(null)
                              return
                            }
                            setOpenAnnCluster(null)
                            setPinnedChartAnn((cur) =>
                              cur === single!.ann.id ? null : single!.ann.id,
                            )
                            setExpandedAnn(single!.ann.id)
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              ;(e.currentTarget as HTMLButtonElement).click()
                            }
                            if (e.key === 'Escape') {
                              setPinnedChartAnn(null)
                              setOpenAnnCluster(null)
                              setHoverAnn(null)
                            }
                          }}
                        >
                          {multi ? (
                            <span className="chart-ann-count">{cl.items.length}</span>
                          ) : (
                            <span className="chart-ann-dot" aria-hidden>
                              {viaAdvisor ? (
                                <span className="chart-ann-src" title="From chat">
                                  ✦
                                </span>
                              ) : null}
                            </span>
                          )}
                        </button>

                        {/* Hover peek (single) — not when pinned */}
                        {!multi &&
                        single &&
                        hoverAnn === single.ann.id &&
                        pinnedChartAnn !== single.ann.id ? (
                          <ChartAnnCard
                            ann={single.ann}
                            anchorLeft={anchorLeft}
                            mode="hover"
                            shellRef={chartShellRef}
                            onEdit={() => {
                              startEdit(single.ann)
                              setAnnOpen(true)
                              setPinnedChartAnn(null)
                            }}
                            onDelete={() => void removeAnnotation(single.ann.id)}
                            onDismiss={() => setHoverAnn(null)}
                            showActions={false}
                          />
                        ) : null}

                        {/* Pinned card (single) */}
                        {!multi && single && pinnedChartAnn === single.ann.id ? (
                          <ChartAnnCard
                            ann={single.ann}
                            anchorLeft={anchorLeft}
                            mode="pinned"
                            shellRef={chartShellRef}
                            onEdit={() => {
                              startEdit(single.ann)
                              setAnnOpen(true)
                            }}
                            onDelete={() => {
                              void removeAnnotation(single.ann.id)
                              setPinnedChartAnn(null)
                            }}
                            onDismiss={() => setPinnedChartAnn(null)}
                            showActions
                          />
                        ) : null}

                        {/* Cluster expand list */}
                        {multi && openAnnCluster === cl.id ? (
                          <div
                            className="chart-ann-cluster-pop"
                            style={{ left: anchorLeft }}
                            role="listbox"
                            aria-label="Annotations in this cluster"
                          >
                            {cl.items.map(({ ann }) => (
                              <button
                                key={ann.id}
                                type="button"
                                className="chart-ann-cluster-item"
                                role="option"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setOpenAnnCluster(null)
                                  setPinnedChartAnn(ann.id)
                                  setExpandedAnn(ann.id)
                                  setHoverAnn(ann.id)
                                }}
                              >
                                <span className="chart-ann-cluster-title">
                                  {ann.label || resolveAnnTarget(ann)}
                                </span>
                                <span className="chart-ann-cluster-meta">
                                  {resolveAnnTarget(ann)} · {ann.start_date}
                                  {ann.start_date !== ann.end_date ? ` → ${ann.end_date}` : ''}
                                </span>
                              </button>
                            ))}
                            <button
                              type="button"
                              className="chart-ann-cluster-dismiss"
                              onClick={() => setOpenAnnCluster(null)}
                            >
                              Close
                            </button>
                          </div>
                        ) : null}
                      </div>
                    )
                  })}
                </div>
                {dragSelStyle ? (
                  <div
                    className="chart-sel"
                    style={{
                      ...dragSelStyle,
                      top: CHART_MT,
                      bottom: CHART_MB,
                    }}
                  />
                ) : null}
                <div
                  className={`chart-drag-layer ${dragMode ? '' : 'off'}`}
                  onPointerDown={onChartPointerDown}
                  onPointerMove={onChartPointerMove}
                  onPointerUp={onChartPointerUp}
                  onPointerCancel={() => {
                    dragActiveRef.current = false
                    setDragSel(null)
                  }}
                />
              </div>
            </SaveAsImage>
          )}
        </section>

        {/* Coverage */}
        <section className="card span-12">
          <h2>
            <Tip tip="How much of each axis could be scored from available wearable inputs, and the resulting inter-axis weights after unavailable axes drop out.">
              Input coverage and axis weights
            </Tip>
          </h2>
          <div className="axis-grid">
            {axisKeys.map((k) => {
              const meta = AXIS_META[k]
              const subs = result.subscores?.[k] || []
              const present = subs.filter((s) => s.present).length
              const total = subs.length
              const w = d.inter_weights?.[k]
              const avail = result.detail?.axis_available?.[k] !== false
              return (
                <div key={k} className="axis-card">
                  <div className="axis-name">
                    <Tip tip={meta.tip}>{meta.name}</Tip>
                  </div>
                  <div className="muted" style={{ marginTop: 4 }}>
                    {present} of {total} variables present
                  </div>
                  <div className="axis-bar">
                    <span
                      style={{
                        width: total ? `${(present / total) * 100}%` : '0%',
                        background: meta.color,
                      }}
                    />
                  </div>
                  <div className="muted" style={{ marginTop: 6 }}>
                    {avail ? `Weight ${fmt((w ?? 0) * 100, 1)}%` : 'Dropped · weight redistributed'}
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      </div>

      <footer className="foot">
        {bundle.subject} · interactive BAP dashboard v4 · free-range dual-handle window (12h steps)
        · data drop box · rolling history = fixed 14-day trailing series filtered to the selected
        dates · dual-anchor engine · age {bundle.age} {bundle.sex} · condition personalization off
        (flags only) · annotations: score/axis with optional variable · edge-aware tooltips.
      </footer>

      {/* Alyf Advisor chat drawer — conversational explainer on the score view */}
      {advOpen
        ? createPortal(
            <>
              <div className="ann-backdrop" onClick={() => setAdvOpen(false)} />
              <aside
                className="ann-drawer adv-drawer"
                role="dialog"
                aria-modal="true"
                aria-label="Alyf Advisor"
              >
                <div className="adv-head">
                  <div className="adv-head-id">
                    <span className="adv-avatar" aria-hidden>
                      <IconChat />
                    </span>
                    <span className="adv-head-name">Alyf Advisor</span>
                  </div>
                  <div className="adv-head-actions">
                    <button
                      type="button"
                      className="btn icon ghost"
                      aria-label="Conversation history"
                      aria-expanded={advHistoryOpen}
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        // History panel should always toggle — never blocked by busy
                        // or by whether the current chat is empty / brand new.
                        setAdvMemoryOpen(false)
                        setAdvConfirm(null)
                        setAdvHistoryOpen((v) => {
                          const next = !v
                          // Opening history on a fresh chat: if load hasn't finished,
                          // keep the panel open on "Loading…" (ready flag handles UI).
                          return next
                        })
                      }}
                      title="Conversation history"
                    >
                      <IconHistory />
                    </button>
                    <button
                      type="button"
                      className="btn sm ghost"
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        setAdvHistoryOpen(false)
                        setAdvMemoryOpen(true)
                      }}
                      title="What Alyf has learned about you"
                    >
                      Profile
                    </button>
                    <button
                      type="button"
                      className="btn icon ghost"
                      aria-label="New chat"
                      onClick={() => {
                        advStop()
                        advStartConvo()
                      }}
                    >
                      <IconPlus />
                    </button>
                    <button
                      type="button"
                      className="btn icon ghost"
                      aria-label="Close the Alyf Advisor"
                      onClick={() => setAdvOpen(false)}
                    >
                      <IconClose />
                    </button>
                  </div>
                </div>

                {advHistoryOpen ? (
                  <div className="adv-history">
                    <div className="adv-history-policy">
                      <p>
                        Chat transcripts are retained for <strong>{ADV_RETENTION_DAYS} days</strong>{' '}
                        from when each conversation starts, then permanently deleted. A
                        personalization profile is retained longer and can be viewed, edited, or
                        cleared at any time. Transcripts are stored separately from BAP scores and
                        wearable data.
                      </p>
                      {advHistoryError ? (
                        <p className="adv-history-error" role="status">
                          {advHistoryError}
                        </p>
                      ) : null}
                    </div>
                    {!advHistoryReady ? (
                      <p className="adv-history-empty">Loading history…</p>
                    ) : advConvos.filter((c) => c.msgs.length > 0).length === 0 ? (
                      <div className="adv-history-empty-block">
                        <p className="adv-history-empty">No past conversations yet.</p>
                        <p className="adv-history-empty-hint">
                          When you chat with the Advisor, threads show up here for a week.
                        </p>
                      </div>
                    ) : (
                      <>
                        <ul>
                          {advConvos
                            .filter((c) => c.msgs.length > 0)
                            .map((c) => (
                              <li key={c.id} className="adv-history-row">
                                <button
                                  type="button"
                                  className={`adv-history-item${c.id === advActiveId ? ' active' : ''}`}
                                  onClick={() => advOpenFromHistory(c.id)}
                                >
                                  <span className="adv-history-title">{advConvoPreview(c)}</span>
                                  <span className="adv-history-meta">
                                    <span className="adv-history-when">
                                      {advFormatWhen(c.updatedAt || c.createdAt)}
                                    </span>
                                    <span className="adv-history-ttl">
                                      {advDeletesInLabel(c.expiresAt || c.createdAt + ADV_RETENTION_MS)}
                                    </span>
                                  </span>
                                </button>
                                <button
                                  type="button"
                                  className="adv-history-del btn icon ghost"
                                  aria-label="Delete conversation"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    setAdvConfirm({ kind: 'one', id: c.id })
                                  }}
                                >
                                  <IconTrash />
                                </button>
                              </li>
                            ))}
                        </ul>
                        <div className="adv-history-foot">
                          <button
                            type="button"
                            className="adv-history-clear"
                            onClick={() => setAdvConfirm({ kind: 'all' })}
                          >
                            Clear all history now
                          </button>
                        </div>
                      </>
                    )}
                    {advConfirm ? (
                      <div className="adv-confirm" role="alertdialog" aria-modal="true">
                        <p className="adv-confirm-text">
                          {advConfirm.kind === 'all'
                            ? 'Delete all Advisor chat history now? This cannot be undone. Your BAP scores and wearable data are not affected.'
                            : 'Delete this conversation permanently? This cannot be undone.'}
                        </p>
                        <div className="adv-confirm-actions">
                          <button
                            type="button"
                            className="btn ghost"
                            onClick={() => setAdvConfirm(null)}
                          >
                            Cancel
                          </button>
                          <button
                            type="button"
                            className="btn danger"
                            onClick={() => {
                              if (advConfirm.kind === 'all') void advClearAllHistory()
                              else if (advConfirm.id) void advDeleteConvo(advConfirm.id)
                            }}
                          >
                            {advConfirm.kind === 'all' ? 'Clear all' : 'Delete'}
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}

                <div className="adv-scroll" ref={advScrollRef} onScroll={onAdvScroll}>
                  {advMsgs.length === 0 ? (
                    <div className="adv-intro">
                      <span className="adv-intro-mark" aria-hidden>
                        <IconChat />
                      </span>
                      <p className="adv-intro-line">
                        Ask me anything about your activity score.
                      </p>
                      <div className="adv-starters">
                        {ADVISOR_STARTERS.map((s) => (
                          <button
                            key={s}
                            type="button"
                            className="adv-starter"
                            onClick={() => sendAdvisorTurn(s)}
                            disabled={advBusy || !bundle}
                          >
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {advMsgs.map((m) => {
                    const streaming = m.id === advStreamId
                    const shown = streaming ? m.content.slice(0, advStreamChars) : m.content
                    const refs =
                      m.role === 'assistant' && !m.failed && !streaming
                        ? advisorDataRefs(m.content)
                        : { axes: [], trend: false }
                    return (
                      <div
                        key={m.id}
                        className={`adv-msg ${m.role}${m.escalation ? ' escalate' : ''}${
                          m.failed ? ' failed' : ''
                        }`}
                      >
                        <div className="adv-bubble">
                          <AdvisorText text={shown} />
                          {streaming ? <span className="adv-caret" aria-hidden /> : null}
                        </div>
                        {m.role === 'assistant' && !m.failed ? (
                          <AdvisorDataStrip
                            refs={refs}
                            axisScores={result?.axis_scores ?? null}
                            bap={result?.bap ?? null}
                            trendValues={advTrendValues}
                          />
                        ) : null}
                        {/* Independent footer: annotationDraft + suggestions are separate fields/regions. */}
                        {m.role === 'assistant' && !m.failed && !streaming && !m.escalation
                          ? (() => {
                              const hasAnnDraft = !!(
                                m.annotationProposal &&
                                (m.annotationUi === 'pending' ||
                                  m.annotationUi === 'editing' ||
                                  m.annotationUi === 'saved' ||
                                  m.annotationUi === 'cancelled')
                              )
                              const isLatest = m.id === advMsgs[advMsgs.length - 1]?.id
                              const rawFollow =
                                isLatest && Array.isArray(m.followUps) ? m.followUps : []
                              // Dedupe: no "add to timeline / annotation" chips when a draft is already shown.
                              const filteredFollow = rawFollow.filter((q) => {
                                if (!hasAnnDraft) return true
                                const low = (q || '').toLowerCase()
                                return !(
                                  low.includes('annotation') ||
                                  low.includes('timeline') ||
                                  low.includes('log this') ||
                                  low.includes('note this') ||
                                  (low.includes('add') && low.includes('note'))
                                )
                              })
                              // Cap at 2 chips when a draft shares the footer; else normal 2–4.
                              const followShow = hasAnnDraft
                                ? filteredFollow.slice(0, 2)
                                : filteredFollow.slice(0, 4)
                              const showFollow = followShow.length > 0
                              if (!hasAnnDraft && !showFollow) return null
                              return (
                                <div className="adv-msg-footer">
                                  {hasAnnDraft ? (
                                    <div className="adv-footer-ann">
                                      <AnnotationProposalCard
                                        msg={m}
                                        dates={bundle?.dates || []}
                                        compact={showFollow && m.annotationUi !== 'editing'}
                                        varOptionsFor={(axisKey) => {
                                          const subs = result?.subscores?.[axisKey] || []
                                          return subs.map((s) => ({
                                            value: s.name,
                                            label: humanVar(s.name),
                                          }))
                                        }}
                                        onChange={(next) => advUpdateProposal(m.id, next)}
                                        onAccept={() => void advAcceptProposal(m)}
                                        onCancel={() => advCancelProposal(m.id)}
                                        busy={advBusy || annStatus === 'Saving…'}
                                      />
                                    </div>
                                  ) : null}
                                  {showFollow ? (
                                    <div
                                      className={`adv-footer-followups${hasAnnDraft ? ' with-ann' : ''}`}
                                      aria-label="Suggested follow-up questions"
                                    >
                                      {followShow.map((q) => (
                                        <button
                                          key={q}
                                          type="button"
                                          className="adv-starter"
                                          onClick={() => void sendAdvisorTurn(q)}
                                          disabled={
                                            advBusy || !bundle || !!advActive?.readOnly
                                          }
                                        >
                                          {q}
                                        </button>
                                      ))}
                                    </div>
                                  ) : null}
                                </div>
                              )
                            })()
                          : null}
                        {m.role === 'assistant' &&
                        !m.failed &&
                        !streaming &&
                        m.memoryProposal &&
                        m.memoryUi !== 'cancelled' ? (
                          <MemoryProposalCard
                            msg={m}
                            busy={advBusy}
                            onChange={(next) => advUpdateMemoryProposal(m.id, next)}
                            onAccept={() => void advAcceptMemoryProposal(m.id)}
                            onCancel={() => advCancelMemoryProposal(m.id)}
                            onEdit={() => advSetMemoryUi(m.id, 'editing')}
                          />
                        ) : null}
                        {m.failed ? (
                          <button
                            type="button"
                            className="adv-retry"
                            onClick={() => advRetry(m)}
                            disabled={advBusy}
                          >
                            Try again
                          </button>
                        ) : null}
                        {advShowTrace && m.role === 'assistant' && !m.failed ? (
                          <details className="adv-sources">
                            <summary>
                              {m.category
                                ? ADVISOR_ROUTE_LABELS[m.category] || m.category
                                : 'trace'}
                              {m.grounding ? ` · grounding: ${m.grounding}` : ''}
                              {m.sources?.length ? ` · ${m.sources.length} passages` : ''}
                            </summary>
                            {m.stages?.length ? (
                              <p className="adv-trace-stages">{m.stages.join(' → ')}</p>
                            ) : null}
                            {m.sources?.length ? (
                              <ul>
                                {m.sources.map((s) => (
                                  <li key={s.id}>
                                    <span className="adv-src-doc">{s.doc}</span>
                                    <span className="muted"> · {s.section}</span>
                                    {s.provisional ? (
                                      <span className="adv-chip caution">provisional</span>
                                    ) : null}
                                  </li>
                                ))}
                              </ul>
                            ) : null}
                          </details>
                        ) : null}
                      </div>
                    )
                  })}

                  {advBusy && !advStreamId ? (
                    <div className="adv-msg assistant" aria-live="polite">
                      <div className="adv-bubble nascent">
                        <span className="sr-only">Advisor is writing a reply</span>
                        <div className="adv-typing" aria-hidden>
                          <span />
                          <span />
                          <span />
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="adv-composer">
                  {advActive?.readOnly ? (
                    <div className="adv-readonly-banner">
                      <p>Viewing a past conversation (read-only).</p>
                      <button
                        type="button"
                        className="btn ghost"
                        onClick={() => {
                          advStop()
                          advStartConvo()
                        }}
                      >
                        New chat
                      </button>
                    </div>
                  ) : null}
                  <div className="adv-input-row">
                    <textarea
                      ref={advInputRef}
                      className="adv-input"
                      value={advInput}
                      onChange={(e) => {
                        setAdvInput(e.target.value)
                        const el = e.target as HTMLTextAreaElement
                        el.style.height = 'auto'
                        el.style.height = `${Math.min(132, el.scrollHeight)}px`
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          if (!advActive?.readOnly) void sendAdvisorTurn(advInput)
                        }
                      }}
                      placeholder={
                        advActive?.readOnly
                          ? 'This conversation is read-only — start a new chat to ask more'
                          : bundle
                            ? 'Ask about your score…'
                            : 'Load score data first'
                      }
                      rows={1}
                      disabled={!bundle || !!advActive?.readOnly}
                      aria-label="Ask the Alyf Advisor a question"
                    />
                    {advBusy ? (
                      <button
                        type="button"
                        className="adv-send stop"
                        onClick={advStop}
                        aria-label="Stop generating"
                      >
                        <IconStop />
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="adv-send"
                        onClick={() => void sendAdvisorTurn(advInput)}
                        disabled={!advInput.trim() || !bundle || !!advActive?.readOnly}
                        aria-label="Send message"
                      >
                        <IconSend />
                      </button>
                    )}
                  </div>
                  <p className="adv-disclaimer">
                    Alyf Advisor explains your activity score and does not provide medical
                    advice. Please consult your care team about any health decisions. Chat
                    transcripts are deleted after {ADV_RETENTION_DAYS} days. A personalization
                    profile is retained longer and can be viewed, edited, or deleted at any time.
                  </p>
                </div>
              </aside>
            </>,
            document.body,
          )
        : null}

      {/* Patient personalization memory drawer */}
      {advMemoryOpen
        ? createPortal(
            <>
              <div
                className="ann-backdrop"
                onClick={() => setAdvMemoryOpen(false)}
                aria-hidden
              />
              <aside
                className="ann-drawer mem-drawer"
                role="dialog"
                aria-modal="true"
                aria-label="What Alyf has learned about you"
              >
                <div className="ann-drawer-head">
                  <div>
                    <h2>What Alyf has learned about you</h2>
                    <p className="drawer-sub">
                      A plain-language portrait of who you are in your data — not a medical
                      record, diagnosis, or how you feel.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() => setAdvMemoryOpen(false)}
                    aria-label="Close"
                  >
                    ×
                  </button>
                </div>
                <div className="mem-body">
                  {advMemoryError ? <div className="banner warn">{advMemoryError}</div> : null}
                  <p className="mem-retention">
                    {advMemoryDoc?.retention_note ||
                      'Chat transcripts delete weekly; this profile can persist until you clear it. Situational events (exams, travel) belong on your annotation timeline.'}
                  </p>
                  <label className="mem-toggle">
                    <input
                      type="checkbox"
                      checked={memoryLearningOn}
                      onChange={(e) =>
                        void runMemoryOp(
                          { op: 'set_learning', enabled: e.target.checked },
                          'Toggle personalization learning',
                        )
                      }
                    />
                    <span>
                      {`Learning: ${memoryLearningOn ? 'on' : 'off'}. When off, no new traits are formed or used.`}
                    </span>
                  </label>
                  {memoryActiveItems.length === 0 && memoryPortraitSections.length === 0 ? (
                    <div className="mem-empty">
                      Nothing in your portrait yet. As Alyf answers from your data, durable
                      patterns about who you are appear here. One-off periods such as finals or
                      travel are recorded on the timeline as annotations instead.
                    </div>
                  ) : (
                    <>
                      {memoryPortrait?.summary || memoryPortraitSections.length > 0 ? (
                        <div className="mem-portrait">
                          <div className="mem-portrait-label">Your portrait</div>
                          {memoryPortrait?.summary ? (
                            <p className="mem-portrait-summary">{memoryPortrait.summary}</p>
                          ) : null}
                          <p className="mem-portrait-note">
                            {memoryPortrait?.disclaimer ||
                              'These are observations about activity patterns and preferences. They are not clinical conclusions.'}
                          </p>
                        </div>
                      ) : null}
                      <div className="mem-sections">
                        {memoryPortraitSections.map((sec) => (
                          <PortraitSectionCard
                            key={sec.key}
                            section={sec}
                            expanded={!!memExpandedSec[sec.key]}
                            onToggle={() =>
                              setMemExpandedSec((prev) => ({
                                ...prev,
                                [sec.key]: !prev[sec.key],
                              }))
                            }
                            itemsById={Object.fromEntries(memoryItems.map((it) => [it.id, it]))}
                            editId={memEditId}
                            editText={memEditText}
                            setEditId={setMemEditId}
                            setEditText={setMemEditText}
                            busy={isRunning}
                            onCorrect={(id, t) => {
                              if (!id || !t.trim()) return
                              void runMemoryOp({ op: 'edit', id, text: t }, 'Correct memory item')
                            }}
                            onRemove={(id) => {
                              if (!id) return
                              setMemConfirm({ kind: 'delete', id })
                            }}
                            onMute={(id) => {
                              if (!id) return
                              void runMemoryOp({ op: 'mute', id }, 'Mute memory item')
                            }}
                          />
                        ))}
                      </div>
                      {memoryPortraitSections.length === 0 && memoryActiveItems.length > 0 ? (
                        <ul className="mem-list">
                          {memoryActiveItems.map((it) => (
                            <li key={it.id} className="mem-item">
                              <p>{it.text}</p>
                              <div className="mem-item-actions">
                                <button
                                  type="button"
                                  className="btn sm ghost"
                                  onClick={() => {
                                    setMemEditId(it.id)
                                    setMemEditText(it.text)
                                  }}
                                >
                                  Correct this
                                </button>
                                <button
                                  type="button"
                                  className="btn sm ghost"
                                  onClick={() => setMemConfirm({ kind: 'delete', id: it.id })}
                                >
                                  This isn&apos;t right
                                </button>
                                <button
                                  type="button"
                                  className="btn sm ghost"
                                  onClick={() =>
                                    void runMemoryOp({ op: 'mute', id: it.id }, 'Mute memory item')
                                  }
                                >
                                  Don&apos;t use this
                                </button>
                              </div>
                            </li>
                          ))}
                        </ul>
                      ) : null}
                      {memorySupersededItems.length > 0 ? (
                        <div className="mem-history-wrap">
                          <button
                            type="button"
                            className="btn sm ghost"
                            onClick={() => setMemShowHistory((v) => !v)}
                          >
                            {memShowHistory ? 'Hide' : 'Show'} earlier / unused notes (
                            {memorySupersededItems.length})
                          </button>
                          {memShowHistory ? (
                            <ul className="mem-list dim-list">
                              {memorySupersededItems.map((it) => (
                                <li key={it.id} className="mem-item dim">
                                  <div className="mem-item-top">
                                    <span className="mem-pill muted">{it.status || 'prior'}</span>
                                  </div>
                                  <p>{it.text}</p>
                                </li>
                              ))}
                            </ul>
                          ) : null}
                        </div>
                      ) : null}
                    </>
                  )}
                  <div className="mem-footer">
                    <button
                      type="button"
                      className="btn sm danger"
                      disabled={memoryActiveItems.length === 0}
                      onClick={() => setMemConfirm({ kind: 'clear' })}
                    >
                      Clear all
                    </button>
                  </div>
                  {memConfirm ? (
                    <div className="mem-confirm">
                      <p>
                        {memConfirm.kind === 'clear'
                          ? 'Clear your whole personalization profile? Chat history, annotations, and BAP scores are not affected.'
                          : 'Remove this from your portrait?'}
                      </p>
                      <div className="mem-proposal-actions">
                        <button
                          type="button"
                          className="btn sm primary"
                          onClick={() => {
                            if (memConfirm.kind === 'clear') {
                              void runMemoryOp({ op: 'clear' }, 'Clear all memory')
                            } else if (memConfirm.id) {
                              void runMemoryOp(
                                { op: 'delete', id: memConfirm.id },
                                'Delete memory item',
                              )
                            }
                            setMemConfirm(null)
                          }}
                        >
                          Confirm
                        </button>
                        <button
                          type="button"
                          className="btn sm ghost"
                          onClick={() => setMemConfirm(null)}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>
              </aside>
            </>,
            document.body,
          )
        : null}

      {/* Clinician personalization profile */}
      {memClinicianOpen
        ? createPortal(
            <>
              <div
                className="ann-backdrop"
                onClick={() => setMemClinicianOpen(false)}
                aria-hidden
              />
              <aside
                className="ann-drawer mem-drawer clinician"
                role="dialog"
                aria-modal="true"
                aria-label="Patient personalization profile"
              >
                <div className="ann-drawer-head">
                  <div>
                    <h2>Personalization profile</h2>
                    <p className="drawer-sub">
                      Not a clinical record or diagnosis. This is the same cohesive portrait the
                      patient sees. Expand any section for provenance. Flag, dismiss, or correct
                      items below.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() => setMemClinicianOpen(false)}
                    aria-label="Close"
                  >
                    ×
                  </button>
                </div>
                <div className="mem-body">
                  <p className="mem-retention">
                    {`Learning: ${memoryLearningOn ? 'enabled' : 'paused'}.`} · {memoryActiveItems.length}{' '}
                    active · {memoryItems.length} stored (incl. history)
                  </p>
                  <p className="mem-portrait-note">
                    Personalization context only — not part of the medical record.
                  </p>
                  {memoryItems.length === 0 ? (
                    <div className="mem-empty">No personalization items for this patient.</div>
                  ) : (
                    <>
                      {memoryPortrait?.summary ? (
                        <div className="mem-portrait">
                          <div className="mem-portrait-label">Portrait</div>
                          <p className="mem-portrait-summary">{memoryPortrait.summary}</p>
                        </div>
                      ) : null}
                      <div className="mem-sections">
                        {memoryPortraitSections.map((sec) => (
                          <PortraitSectionCard
                            key={'c-' + sec.key}
                            section={sec}
                            expanded={!!memExpandedSec['c:' + sec.key]}
                            onToggle={() =>
                              setMemExpandedSec((prev) => ({
                                ...prev,
                                ['c:' + sec.key]: !prev['c:' + sec.key],
                              }))
                            }
                            itemsById={Object.fromEntries(memoryItems.map((it) => [it.id, it]))}
                            editId={memEditId}
                            editText={memEditText}
                            setEditId={setMemEditId}
                            setEditText={setMemEditText}
                            clinician
                            onClinicianCorrect={(id, t) => {
                              void runMemoryOp(
                                {
                                  op: 'clinician_correct',
                                  id,
                                  text: t,
                                  who: 'clinician',
                                  note: 'Corrected in admin panel',
                                },
                                'Clinician correct memory',
                              )
                            }}
                            onClinicianDismiss={(id) => {
                              void runMemoryOp(
                                {
                                  op: 'clinician_dismiss',
                                  id,
                                  who: 'clinician',
                                  note: 'Dismissed in admin panel',
                                },
                                'Clinician dismiss memory',
                              )
                            }}
                            onClinicianRestore={(id) => {
                              void runMemoryOp(
                                {
                                  op: 'clinician_restore',
                                  id,
                                  who: 'clinician',
                                },
                                'Clinician restore memory',
                              )
                            }}
                          />
                        ))}
                      </div>
                      {memoryPortraitSections.length === 0 ? (
                        <ul className="mem-list">
                          {memoryItems.map((it) => (
                            <li
                              key={it.id}
                              className={
                                'mem-item' +
                                ((it.status || '') !== 'active' || it.dismissed ? ' dim' : '')
                              }
                            >
                              <div className="mem-item-top">
                                <span className="mem-pill">{it.type}</span>
                                <span className="mem-pill muted">{it.status || 'active'}</span>
                              </div>
                              <p>{it.text}</p>
                              {it.provenance?.summary ? (
                                <p className="mem-prov">Provenance: {it.provenance.summary}</p>
                              ) : null}
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </>
                  )}
                </div>
              </aside>
            </>,
            document.body,
          )
        : null}

      {/* Annotations side drawer */}
      {annOpen
        ? createPortal(
            <>
              <div
                className="ann-backdrop"
                onClick={() => setAnnOpen(false)}
                aria-hidden
              />
              <aside
                className="ann-drawer"
                role="dialog"
                aria-modal="true"
                aria-label="Administrator annotations"
              >
                <div className="ann-drawer-head">
                  <h2>
                    <Tip tip={HEADER_TIPS.annotations}>Administrator annotations</Tip>
                  </h2>
                  <button
                    type="button"
                    className="btn icon ghost"
                    aria-label="Close annotations"
                    onClick={() => setAnnOpen(false)}
                  >
                    <IconClose />
                  </button>
                </div>
                <div className="ann-drawer-body">
                  <p className="muted" style={{ margin: 0 }}>
                    Attach a note to the overall score or an axis (optionally one variable). Set the
                    time range by typing dates or dragging on the chart.
                  </p>

                  <div className="ann-form">
                    <div className="ann-mode">
                      <button
                        type="button"
                        className={`btn sm ${annTargetType === 'score' ? 'active' : ''}`}
                        onClick={() => setAnnTargetType('score')}
                      >
                        Score
                      </button>
                      <button
                        type="button"
                        className={`btn sm ${annTargetType === 'axis' ? 'active' : ''}`}
                        onClick={() => setAnnTargetType('axis')}
                      >
                        Axis
                      </button>
                      <button
                        type="button"
                        className={`btn sm ${dragMode ? 'primary' : ''}`}
                        onClick={() => {
                          setDragMode((v) => !v)
                          setDragSel(null)
                          dragActiveRef.current = false
                        }}
                      >
                        {dragMode ? 'Drag active' : 'Drag time on chart'}
                      </button>
                    </div>

                    {annTargetType === 'axis' ? (
                      <>
                        <label>
                          Axis
                          <select
                            value={annAxisKey}
                            onChange={(e) => {
                              setAnnAxisKey(e.target.value)
                              setAnnVariable('')
                            }}
                          >
                            {Object.entries(AXIS_META).map(([k, v]) => (
                              <option key={k} value={k}>
                                {v.name}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Variable (optional narrowing)
                          <select
                            value={annVariable}
                            onChange={(e) => setAnnVariable(e.target.value)}
                          >
                            <option value="">Whole axis</option>
                            {axisVarOptions.map((o) => (
                              <option key={o.value} value={o.value}>
                                {o.label}
                              </option>
                            ))}
                          </select>
                        </label>
                      </>
                    ) : (
                      <div className="muted">Target: Overall BAP score</div>
                    )}

                    <div className="row2">
                      <label>
                        Start date
                        <input
                          type="date"
                          value={annStartDate}
                          onChange={(e) => setAnnStartDate(e.target.value)}
                        />
                      </label>
                      <label>
                        End date
                        <input
                          type="date"
                          value={annEndDate}
                          onChange={(e) => setAnnEndDate(e.target.value)}
                        />
                      </label>
                    </div>
                    <div className="row2">
                      <label>
                        Start time
                        <select
                          value={annStartTime}
                          onChange={(e) => setAnnStartTime(e.target.value)}
                        >
                          <option value="00:00">00:00</option>
                          <option value="12:00">12:00</option>
                        </select>
                      </label>
                      <label>
                        End time
                        <select value={annEndTime} onChange={(e) => setAnnEndTime(e.target.value)}>
                          <option value="12:00">12:00</option>
                          <option value="24:00">24:00</option>
                        </select>
                      </label>
                    </div>

                    <label>
                      Short label
                      <input
                        value={annLabel}
                        onChange={(e) => setAnnLabel(e.target.value)}
                        placeholder="e.g. Travel week"
                      />
                    </label>
                    <label>
                      Note
                      <textarea
                        value={annText}
                        onChange={(e) => setAnnText(e.target.value)}
                        placeholder="Explain the outlying circumstance…"
                      />
                    </label>

                    <div className="muted" style={{ fontSize: 11 }}>
                      Will save as:{' '}
                      <strong>
                        {annTargetType === 'score'
                          ? 'Overall BAP score'
                          : annVariable
                            ? `${AXIS_META[annAxisKey]?.name} · ${humanVar(annVariable)}`
                            : `${AXIS_META[annAxisKey]?.name} axis`}
                      </strong>
                      {' · '}
                      {annStartDate || '—'} {annStartTime} → {annEndDate || '—'} {annEndTime}
                    </div>

                    <div className="ann-actions">
                      <button
                        type="button"
                        className="btn primary sm"
                        onClick={addOrUpdateAnnotation}
                        disabled={isRunning}
                      >
                        {isRunning ? 'Saving…' : editingId ? 'Update note' : 'Add note'}
                      </button>
                      {editingId ? (
                        <button
                          type="button"
                          className="btn sm"
                          onClick={() => {
                            setEditingId(null)
                            setAnnText('')
                            setAnnLabel('')
                          }}
                        >
                          Cancel edit
                        </button>
                      ) : null}
                      {annStatus ? <span className="muted">{annStatus}</span> : null}
                    </div>
                  </div>

                  <h3 style={{ margin: '8px 0 0' }}>Saved notes</h3>
                  {annotations.length === 0 ? (
                    <p className="ann-empty">
                      No annotations yet. Add one above or drag a range on the chart.
                    </p>
                  ) : (
                    <div className="ann-list">
                      {annotations.map((a) => {
                        const open = expandedAnn === a.id
                        return (
                          <div key={a.id} className="ann-item">
                            <button
                              type="button"
                              className="ann-summary"
                              onClick={() => setExpandedAnn(open ? null : a.id)}
                              onMouseEnter={() => setHoverAnn(a.id)}
                              onMouseLeave={() => setHoverAnn(null)}
                            >
                              <div className="ann-title">
                                {a.label || 'Untitled'} {open ? '▾' : '▸'}
                              </div>
                              <div className="ann-target">
                                {resolveAnnTarget(a)}
                                <br />
                                {resolveAnnTime(a)}
                              </div>
                              {!open ? <div className="ann-preview">{a.text}</div> : null}
                            </button>
                            {open ? (
                              <div className="ann-body">
                                <div className="ann-text">{a.text}</div>
                                <div className="ann-meta">
                                  <span>
                                    Created {new Date(a.created_at).toLocaleString()}
                                    {a.updated_at
                                      ? ` · Updated ${new Date(a.updated_at).toLocaleString()}`
                                      : ''}
                                  </span>
                                </div>
                                <div className="ann-actions" style={{ marginTop: 8 }}>
                                  <button type="button" className="btn sm" onClick={() => startEdit(a)}>
                                    Edit
                                  </button>
                                  <button
                                    type="button"
                                    className="btn sm danger"
                                    onClick={() => removeAnnotation(a.id)}
                                  >
                                    Delete
                                  </button>
                                </div>
                              </div>
                            ) : null}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              </aside>
            </>,
            document.body,
          )
        : null}
    </>,
  )
}

export default App