# Balanced Activity Profile (BAP) Clinical Scoring Engine

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-51%2F51%20Passing-brightgreen.svg)](tests/)
[![TypeScript](https://img.shields.io/badge/TypeScript-React-blue.svg)](web/)

> An interpretable **0–100 clinical activity scoring engine**, statistical **dual-anchor baseline model**, and clinician-facing interactive dashboard with conversational health intelligence (**Alyf Advisor**). Developed for [Alyf Healthcare Platform](https://alyf.health) to bridge wearable physiological data and clinical consultations.

---

## 1. Executive Summary & Clinical Intent

Most consumer fitness scores (e.g., Whoop Strain, Garmin Body Battery, Apple Activity Rings) are **gamified coaching metrics**: they push users toward continuous improvement, penalize rest, or measure acute cardiovascular load without regard for physiological balance.

**BAP (Balanced Activity Profile)** is engineered from the ground up as a **diagnostic and clinical status score**:
- **Objective:** Enable physicians and patients to immediately assess an individual's physical activity balance, recovery adequacy, and circadian consistency over any arbitrary time window (from 1 day up to 200+ days).
- **Scale:** An intuitive **0–100** score where 50 represents average healthy population balance, >75 represents optimal clinical vitality, and <40 flags clinical sedentary or dysregulated behavior.
- **Device-Agnostic:** Ingests normalized wearable telemetry (METs, steps, active calories, sleep duration, resting heart rate, HRV) from Oura Ring, Apple Health, Whoop, and clinical accelerometers.

```
       ┌────────────────────────────────────────────────────────┐
       │             Raw Wearables Telemetry Ingest             │
       │    (Oura Ring Trends, Apple Health, Clinical METs)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             5-Axis Multi-Dimensional Scoring           │
       │   Intensity • Recovery • Consistency • Variety • Spread │
       └───────────────────────────┬────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌─────────────────────────┐   ┌──────────────────────────┐
       │ Population Anchor       │   │ Personal Dual-Anchor     │
       │ Calibrated vs NHANES /  │   │ Historical Baseline with │
       │ Healthy Reference Norms │   │ √ Law & Honesty Clamp    │
       └────────────┬────────────┘   └─────────────┬────────────┘
                    └──────────────┬──────────────┘
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │              Master BAP Composite (0–100)              │
       │      + Qualitative Stage 5 Suggestion Engine           │
       │      + Alyf Advisor Conversational Health Intelligence │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation & Normalization Law

### The Five Physiological Axes
A healthy lifestyle is multi-dimensional. High cardiovascular intensity cannot compensate for sleep deprivation or irregular circadian timing. BAP evaluates five orthogonal axes:

| Axis | Physiological Focus | Key Telemetry Signals | Target Distribution |
| :--- | :--- | :--- | :--- |
| **Intensity** | Cardiovascular load & metabolic expenditure | Active calories, MET minutes (>3 METs, >6 METs), step volume | Half-normal / Log-logistic |
| **Recovery** | Autonomic restoration & sleep architecture | Total sleep, REM/Deep ratio, Rest Heart Rate drop, overnight HRV (rMSSD) | Sigmoidal saturation |
| **Consistency**| Circadian stability & habit regularity | Inter-day variance of activity onset, wake time, and bedtime | Exponential decay on $\sigma^2$ |
| **Variety** | Musculoskeletal & metabolic diversification | Distribution across cross-training modalities (aerobic, resistance, mobility) | Shannon entropy / Gini index |
| **Spread** | Inactivity fragmentation (anti-sedentariness) | Hourly active bouts, uninterrupted sedentary blocks, active hour ratio | Beta distribution |

### Dual-Anchor Normalization Law
When evaluating chronic health, evaluating a patient strictly against the population risks demotivating those recovering from injury or illness, while evaluating them strictly against their own baseline creates baseline drift (a chronically sedentary person scoring 90 because they did not change).

BAP implements a **Dual-Anchor Blend**:
$$	ext{BAP}_{	ext{final}} = w_{	ext{pop}} \cdot S_{	ext{pop}} + w_{	ext{base}} \cdot S_{	ext{personal}}$$

Where personal baseline adherence incorporates:
1. **Concave Square-Root Maintenance Law ($\sqrt{\cdot}$):** Diminishing returns on excessive volume, rewarding steady physiological maintenance over boom-and-bust cycles:
   $$S_{	ext{personal}} = 100 \cdot \sqrt{rac{X_{	ext{current}}}{X_{	ext{baseline}}}}$$
2. **Honesty Clamp ($\pm 0.12$):** Bounded deviation prevents short-term volatility or wearable sensor dropouts from corrupting the longitudinal score.

---

## 3. Repository Architecture

```text
bap-clinical-engine/
├── engine/                      # Core Algorithmic & Statistical Engine
│   ├── bapFINAL.py              # Master BAP computation (5 axes, dual-anchor, √ curve)
│   ├── suggestion_engine.py     # Stage 5 clinical suggestion & habit recommendation logic
│   └── config.py                # Reference population anchors and clinical cutoffs
│
├── tests/                       # Comprehensive Validation & Regression Suite
│   ├── test_bapFINAL.py         # 51/51 automated regression tests across golden patient profiles
│   └── fixtures/                # Deterministic synthetic & clinical edge-case profiles
│
├── ingest/                      # Wearable ETL & Telemetry Ingestion
│   └── oura_ingest.py           # Oura Ring trends CSV parser, multi-window slicing, MET resolver
│
├── advisor/                     # Alyf Advisor Conversational Health Intelligence
│   ├── alyf_advisor.py          # Latency-optimized LLM agent grounded on clinical telemetry
│   ├── query_layer.py           # O(1) multi-window evaluation & time-window resolution
│   ├── guardrails.py            # Safety gates ensuring lifestyle coaching vs medical diagnosis boundaries
│   └── memory.py                # Clinician-facing longitudinal notes & timeline annotations
│
├── web/                         # Interactive Clinician & Patient Dashboard
│   ├── src/
│   │   ├── BapDashboard.tsx     # Full React application (Recharts, interactive timeline, multi-window)
│   │   └── BapDashboard.css     # Dark mode tokens, clinical status badges, responsive styling
│   └── package.json
│
└── docs/                        # Specifications, Clinical Briefs & Formulas
    ├── BAP_Master_Spec.md       # Master clinical formula specification
    └── CLINICAL_SIGN_OFF.md     # Clinical sign-off checklist and safety audit
```

---

## 4. Key Engineering & Data Science Highlights

1. **Precompute-on-Ingest Engine:**
   - Rather than calculating expensive rolling statistics on the fly during UI navigation or conversational queries, the pipeline scores all inclusive sub-windows ($O(N^2)$ rolling spans from 1d to 60d) during ingestion.
   - All interactive UI sliders and LLM conversational tool calls resolve via $O(1)$ dictionary lookups.

2. **51/51 Deterministic Golden Profile Tests:**
   - Includes full regression testing across four distinct clinical golden patient profiles:
     - **Sophia:** Active endurance runner with variable sleep and recovery deficits.
     - **Priya:** Highly consistent desk professional with low spread (prolonged sedentary blocks).
     - **Ethan:** High variance, boom-and-bust weekend warrior.
     - **Michael:** Cardiac rehabilitation patient monitored for recovery bounds and intensity limits.

3. **Alyf Advisor (Clinician AI Copilot):**
   - Grounded conversational analyst that ingests the precomputed score bundle.
   - Enforces strict clinical safety guardrails: answers doctor and patient questions on physiological trends while refusing out-of-scope medical diagnoses and prescribing habits within clinical boundaries.

---

## 5. Quickstart & Installation

### Python Engine
```bash
# Clone the repository
git clone https://github.com/shreychettiar/bap-clinical-engine.git
cd bap-clinical-engine

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the test suite
python tests/test_bapFINAL.py
```

### Clinician Dashboard UI
```bash
cd web
npm install
npm run dev
```

---

## 6. Author & License

- **Author:** [Shrey Chettiar](https://github.com/shreychettiar)
- **Role:** Health Informatics & Algorithmic Lead for BAP @ Alyf
- **License:** MIT License
