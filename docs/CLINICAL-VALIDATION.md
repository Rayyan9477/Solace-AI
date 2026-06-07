# Solace-AI Clinical Validation

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

[severity]: ../services/diagnosis_service/src/domain/severity.py
[value_objects]: ../services/diagnosis_service/src/domain/value_objects.py
[treatment_planner]: ../services/therapy_service/src/domain/treatment_planner.py
