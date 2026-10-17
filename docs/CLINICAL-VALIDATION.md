# Solace-AI Clinical Validation

> ℹ️ **Strongest doc, two attribution fixes (flagged 2026-07-21).** The clinical thresholds here are verified in code. Two corrections: the H-13 contextual crisis regexes live in the **therapy** service (`therapy_service/src/domain/service.py:333`), not the safety Layer-1 detector; and "ESCALATE blocks insight generation" references code that doesn't exist (ESCALATE is a label with no gating). Accuracy items REV-25/26/27/28/29/30 + A4 are now **code-reconciled to the documented claims** (2026-07-30) and await clinician sign-off — see **§10** below.

> **Audience**: Licensed clinicians evaluating the MVP.
> **Purpose**: Explicitly state what Solace-AI implements per clinical
> standards, where it deviates, and where clinician oversight is required.

This document is the single source of truth for the published papers,
clinical guidelines, and diagnostic manuals that Solace-AI's reasoning
and scoring code trace back to. It is referenced from inline code
comments via docstring citations.

---

## 1. Questionnaire scoring

### 1.1 PHQ-9 (Depression)

- **Citation**: Kroenke K, Spitzer RL, Williams JBW. 2001. "The PHQ-9:
  Validity of a brief depression severity measure." *JAMA* 16(9): 606-613.
- **Implemented thresholds** (match paper exactly):
  - 0-4: Minimal / None
  - 5-9: Mild
  - 10-14: Moderate
  - 15-19: Moderately Severe
  - 20-27: Severe
- **Per-item score scale**: 0-3 (Not at all / Several days / More than
  half the days / Nearly every day).
- **Clinical action mapping**:
  - Minimal: watchful waiting, repeat assessment
  - Mild: monitoring, behavioural activation
  - Moderate: guided CBT
  - Moderately Severe: active treatment, consider medication consult
  - Severe: immediate clinical attention
- **Source**:
  [`services/diagnosis_service/src/domain/severity.py`][severity].

### 1.2 GAD-7 (Generalized Anxiety)

- **Citation**: Spitzer RL, Kroenke K, Williams JBW, Lowe B. 2006.
  "A brief measure for assessing generalized anxiety disorder: the
  GAD-7." *Arch Intern Med* 166(10): 1092-1097.
- **Implemented thresholds** (match paper exactly):
  - 0-4: Minimal
  - 5-9: Mild
  - 10-14: Moderate
  - 15-21: Severe
- **Per-item score scale**: 0-3.

### 1.3 PCL-5 (PTSD) — **documented deviation**

- **Standard citation**: Weathers FW, Litz BT, Keane TM, Palmieri PA,
  Marx BP, Schnurr PP. 2013. "The PTSD Checklist for DSM-5 (PCL-5)."
  US National Center for PTSD. The standard 20-item instrument has a
  probable-PTSD diagnostic cutoff of 31-33 (max 80).
- **This service uses a 10-item abbreviated screener** with max 40
  instead of 80. To preserve the same clinical sensitivity as the
  full instrument we **halve every threshold**:
  - 16 (was 31): SEVERE / probable PTSD
  - 11 (was 22): MODERATE
  - 9 (was 17): MILD / clinically notable
  - <9: MINIMAL / below threshold
- **Deviation rationale**: The full PCL-5 is too long for AI-mediated
  screening. The abbreviated form keeps the five DSM-5 PTSD criterion
  clusters (B-F). A clinician-administered full PCL-5 is still
  required for a probable-PTSD determination.
- **Risk**: results from the screener may slightly differ from the
  standard; should not be reported as a standalone PCL-5 score.
- **Source**:
  [`services/diagnosis_service/src/domain/value_objects.py`][value_objects]
  (`from_pcl5`) and
  [`services/diagnosis_service/src/domain/severity.py`][severity]
  (`_interpret_pcl5`).

### 1.4 PHQ-15 (Somatic Symptoms)

- **Citation**: Kroenke K, Spitzer RL, Williams JBW. 2002. "The PHQ-15:
  Validity of a new measure for evaluating the severity of somatic
  symptoms." *Psychosom Med* 64(2): 258-266.
- **Implemented thresholds**: 0-4 Minimal / 5-9 Mild / 10-14 Moderate /
  15+ Severe.

---

## 2. Stepped care and treatment response

- **Citation**: NICE Clinical Guideline 90 (2009, updated 2022).
  "Depression in adults: recognition and management."
- **Stepped-care levels**: classified by PHQ-9 score at baseline,
  evaluated for treatment response after 4-6 weeks per UK NICE
  recommendation.
- **Response classification** (each versus baseline PHQ-9):
  - REMISSION: current PHQ-9 ≤ 4 (C-16: this check runs FIRST so a
    patient who drops to Minimal is not misclassified as RESPONDING
    by percentage-based rules)
  - RESPONDING: ≥50% reduction
  - PARTIAL: 25-49% reduction
  - NON_RESPONDER: <25% reduction
  - DETERIORATING: any upward movement
- **Source**:
  [`services/therapy_service/src/domain/treatment_planner.py`][treatment_planner].

---

## 3. Safety and crisis detection

### 3.1 Four-layer crisis detection (Solace-AI specific)

- **Layer 1 — Keyword / regex**: deterministic pattern match for
  canonical crisis phrases.
- **Layer 2 — Sentiment / risk scoring**: hopelessness, despair,
  emotional valence.
- **Layer 3 — Temporal pattern**: trajectory monitoring (M-03:
  includes CRITICAL-level users in the deteriorating check; a worsening
  CRITICAL-level trajectory fires a `TrajectoryAlertEvent`).
- **Layer 4 — LLM assessor**: contextual judgement for nuanced intent,
  sarcasm, passive vs active ideation.

### 3.2 Crisis-keyword contextualisation (H-13)

The Layer-1 detector distinguishes "harm reduction strategies" (benign
therapy language) from "harm myself" (self-directed crisis) via
contextual patterns:

- `\bharm\s+(myself|herself|himself|themselves|me)\b`
- `\b(in\s+danger|dangerous\s+to)\b`

Bare occurrences of `harm` or `danger` in non-self-directed contexts
do not trigger escalation.

### 3.3 Protective factors (H-06)

Protective factors (social support, treatment engagement, coping
skills, positive outlook, family connection) reduce the numeric risk
score by up to 15% of their weighted strength. **Invariant**: the
crisis-level category is NEVER downgraded from CRITICAL to a lower
level by protective factors. A user reporting suicidal ideation plus
strong supports is still escalated — evidence base is too weak to
"talk ourselves out" of a flagged CRITICAL.

### 3.4 Self-harm guideline

- **Citation**: NICE Clinical Guideline 133 (2011). "Self-harm:
  longer-term management."

---

## 4. Differential diagnosis

### 4.1 DSM-5-TR criteria

All ICD-10 ↔ DSM-5 mappings trace to:
American Psychiatric Association. 2022. *Diagnostic and Statistical
Manual of Mental Disorders, Fifth Edition, Text Revision (DSM-5-TR)*.

### 4.2 AMIE-style 4-step chain-of-reasoning

Solace-AI's diagnosis chain is inspired by Google's AMIE (Tu et al.
2024, *Nature*): Analyze → Hypothesize → Challenge → Synthesize.

### 4.3 Devil's Advocate anti-sycophancy (H-07)

- **Citation**: Croskerry P. 2003. "The importance of cognitive errors
  in diagnosis and strategies to minimize them." *Ann Emerg Med.*
- **Implementation**: step 3 challenges each hypothesis independently;
  step 4 applies the per-hypothesis confidence adjustment (H-07), not
  a total-chain adjustment that would over-penalise uncontested
  hypotheses.

### 4.4 Bayesian calibration (H-08)

Step 4 passes the actual symptoms extracted in step 1 to the
calibrator (pre-H-08 bug passed an empty list). Calibration combines
prior prevalence, likelihood (criteria coverage + specificity +
exclusion), and sample-consistency across N LLM samples.

### 4.5 Confidence thresholds (C-15)

Unified across the service:

| Confidence | Range | Action |
|-----------|-------|--------|
| HIGH | ≥ 0.70 | proceed with recommendations |
| MEDIUM | 0.50-0.70 | request more information |
| LOW | 0.30-0.50 | flag for clinician review |
| **ESCALATE** | < 0.30 | do not generate insight; escalate |

---

## 5. Memory and forgetting

### 5.1 Ebbinghaus decay

- **Citation**: Ebbinghaus H. 1885 (1913 English translation). *Memory:
  A Contribution to Experimental Psychology*.
- **Implementation**:
  retention_strength = stability × exp(−λ × t) where λ is a
  category-specific rate and stability tracks reinforcement
  separately from current retention (C-20 double-compound bug fixed).

### 5.2 Crisis-content retention override

Safety-flagged content (`retention_category=PERMANENT`) never decays,
regardless of time elapsed. Clinical audit trail for disclosure
accounting must remain queryable indefinitely.

---

## 6. Therapeutic modalities

Solace-AI implements six evidence-based modalities:

| Modality | Primary citation |
|---------|------------------|
| CBT | Beck 1976; Beck & Dozois 2011 |
| DBT | Linehan 1993; 2015 |
| ACT | Hayes, Strosahl, Wilson 1999; 2011 |
| MI | Miller & Rollnick 1991; 2013 |
| Mindfulness | Kabat-Zinn 1990 |
| SFBT | de Shazer 1985; Berg & Dolan 2001 |

Technique selection uses the spec-weighted formula:
`0.4 × clinical + 0.3 × personal + 0.2 × context + 0.1 × history`.

---

## 7. Personality model

- **Big Five / OCEAN**: Costa & McCrae 1992. *NEO PI-R Professional
  Manual.*
- **Ensemble weights** (H-22): RoBERTa-fine-tuned 0.5 + LLM zero-shot
  0.3 + LIWC features 0.2.
- **MoEL empathy**: Lin, Madotto, Shin, Xu, Fung 2019. "MoEL: Mixture
  of Empathetic Listeners." *EMNLP.*

---

## 8. Known limitations and clinician-oversight requirements

- **Not a diagnostic device**: Solace-AI is a screening and supportive
  companion. All differential diagnoses are provisional; definitive
  diagnosis requires a licensed clinician.
- **PCL-5 screener only**: the 10-item abbreviated form (§1.3) cannot
  replace a clinician-administered full PCL-5 for a probable-PTSD
  determination.
- **Crisis escalation is mandatory**: HIGH and CRITICAL assessments
  auto-escalate to the on-call clinician pool; the system does NOT
  wait for user consent before alerting.
- **Not FDA-cleared**: no investigational device exemption claimed.
  Out-of-scope per MVP.
- **No published RCT**: out-of-scope per MVP; planned post-MVP.

---

## 9. Deviations summary

| ID | Deviation | Rationale |
|----|-----------|-----------|
| PCL-5 (§1.3) | 10-item halved screener | brevity for AI-mediated use; full PCL-5 required for formal dx |
| ESCALATE tier (§4.5) | New confidence tier below LOW | prevent silent insight generation on dangerously uncertain cases |
| Protective-factor floor (§3.3) | Never downgrade CRITICAL | patient safety conservatism |

---

## 10. Clinical-accuracy reconciliation (REV-25..30, A4) — code now matches claim, **pending D2 clinician sign-off**

2026-07-30 (Stage 1 · A3/A4): the six clinical-accuracy drifts flagged in the
header note were reconciled — the **code was changed to match the documented
clinical claim** (each via a failing Prove-It test first, then to green). Every
item below is a patient-facing clinical behavior change and **requires licensed-
clinician sign-off (D2)** before launch. Each links its Prove-It test.

| REV | Claim | Code before | Resolution (code now) | Clinician decision needed |
|-----|-------|-------------|-----------------------|---------------------------|
| **REV-25** | CRITICAL crisis threshold = 0.85; detection latency observable; Layer-1 keyword gate < 10 ms | Threshold 0.9 (0.85–0.899 scored HIGH, not CRITICAL); `detection_time_ms` hard-coded 0; no latency SLO | `critical_threshold` 0.9 → **0.85** (`value_objects.py` + runtime `crisis_detector.py`); lone non-negated critical keyword now floors to **CRITICAL** (`escalate_immediately`, was HIGH/`intervene`); `detection_time_ms` = real `perf_counter` ms; Layer-1 gate SLO test (measured median ~0.07 ms) | Confirm the more-sensitive threshold + lone-keyword→CRITICAL escalation match clinical intent and on-call load. `test_rev25_reconcile.py` |
| **REV-26** | Devil's-Advocate detects 6 cognitive biases | Only 4 emitted; `base_rate_neglect` + `attribution_error` were dead metadata (no inputs to detect them) | Removed the 2 un-emittable biases so **advertised = emittable (4)**; docstring notes any re-add must ship with emission logic | Confirm base-rate-neglect / attribution-error are not required for MVP, or request real detectors. `test_rev26_reconcile.py` |
| **REV-27** | Bayesian calibration N=3 samples; `< 60% agreement → Uncertain` label | N=5; no Uncertain label | `calibration_samples` 5 → **3**; added `Uncertain` (<0.60) / Moderate / High agreement label on `CalibrationResult`. **2026-08-24 follow-up: true N-sample sampling now wired** (see §10.1) — the agreement signal is no longer a consistency-score proxy | Confirm N=3 vs 5 and the 0.60/0.80 band boundaries. `test_rev27_reconcile.py`, `test_rev27_nsample.py` |
| **REV-28** | Corrective RAG: `< 3 relevant docs → degraded/structured error` | SUCCESS returned with 1–2 relevant docs (FAILED only on zero) | Added `min_relevant_documents` (default 3, env-configurable) + `INSUFFICIENT_CONTEXT` status; <3 relevant → degraded result with only the relevant subset | Confirm the default minimum of 3 and that the degraded result should surface the partial docs. `test_rev28_reconcile.py` |
| **REV-29** | Diagnoses / clinical memories **never decay** | `clinical` got only a 0.5 decay *modifier* and still decayed; `clinical`/`diagnosis` categories fell back to MEDIUM_TERM | `NEVER_DECAY_CATEGORIES = {permanent, clinical, diagnosis}` → zero decay in both DecayManager and the SQL batch path | Confirm which retention categories count as never-decay. `test_rev29_a4_reconcile.py` |
| **A4** | SQL batch decay uses the **same per-category rates** as the in-process DecayManager, referenced from last access | SQL applied a flat ~0.05/day to all records (~2.5× drift from short-term 0.15) | SQL now builds a per-category `CASE` from one shared rate table (short 0.15 / medium 0.05 / long 0.02 /day; never-decay = 0), keyed off `accessed_at` | Confirm the per-category half-lives. `test_rev29_a4_reconcile.py` |
| **REV-30** | Step-4 calibration integrates the **current message's** symptoms | Current-turn symptoms merged into `session.symptoms` only *after* the chain, so step-4 saw prior-turn symptoms only | Current-turn (step-1) symptoms merged into a deduped working set fed to step-4 calibration before it runs (session state untouched → no double-count) | Confirm the corrected current-turn weighting/timing matches intended diagnostic reasoning. `test_rev30_reconcile.py` |

### 10.1 REV-27 follow-up — true N-sample calibration (2026-08-24, **pending D2 clinician sign-off**)

**Patient-facing calibration behaviour change.** REV-27 had landed the N=3 /
`Uncertain (<0.60)` label, but the agreement signal was a single-shot
**consistency-score PROXY** — no independent samples were ever drawn. This
follow-up wires **genuine N-sample calibration** in
[`confidence.py`](../services/diagnosis_service/src/domain/confidence.py):

- The calibrator now draws **N independent estimates** (N =
  `calibration_samples`, **default 3**, unchanged) via an **injectable,
  seedable sampler** (`ConsistencySampler`), and derives the agreement signal
  from their **distribution** rather than a single value.
- **How agreement is derived:** `agreement = mean × (1 − min(1, std × K))`
  where `mean`/`std` are over the N drawn estimates and `K =
  sample_spread_penalty` (default 2.0). Tightly-clustered high samples →
  high agreement (**High** band); wide disagreement (large `std`) drives
  agreement down into the **Uncertain** band. The result carries the full
  distribution (`sample_count`, `sample_mean`, `sample_spread`,
  `sample_agreement`).
- **Cited thresholds are unchanged:** confidence tiers (HIGH ≥ 0.70 / MEDIUM /
  LOW / ESCALATE < 0.30) and the agreement bands (`Uncertain` < 0.60 /
  Moderate 0.60–0.80 / High > 0.80) are preserved exactly.
- **OFF / inert by default:** with no sampler injected the calibrator uses a
  **degenerate sampler** (N identical copies → `std = 0`), so `agreement ==
  mean ==` the legacy consistency proxy. Existing behaviour and the degenerate
  N=1 case are byte-for-byte backward-compatible; `DiagnosisService` continues
  to construct the calibrator with the default (inert) sampler. A real
  stochastic path is opt-in by injecting `SeededGaussianSampler` (deterministic
  under a fixed seed for test/audit replay).

Clinician decision needed: confirm the spread-penalty formulation (`K = 2.0`)
and that deriving `Uncertain` from sample **disagreement** (not just a low
point-estimate) matches clinical intent, and approve enabling a live
stochastic sampler in production. Prove-It: `test_rev27_nsample.py`.

---

[severity]: ../services/diagnosis_service/src/domain/severity.py
[value_objects]: ../services/diagnosis_service/src/domain/value_objects.py
[treatment_planner]: ../services/therapy_service/src/domain/treatment_planner.py
