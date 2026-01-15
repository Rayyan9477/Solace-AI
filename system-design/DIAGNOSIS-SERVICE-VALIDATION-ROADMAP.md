# Diagnosis Service - Clinical Validation Roadmap

**Document Version**: 1.0
**Date**: January 2026
**Status**: Roadmap for Achieving FDA Clearance

---

## Executive Summary

This document outlines the clinical validation roadmap to achieve FDA clearance for the Solace-AI Diagnosis Service as a Clinical Decision Support System (CDSS) for mental health diagnosis.

**Current Status**: ✅ World-class technical implementation complete, ⚠️ Clinical validation needed

**Timeline to FDA Clearance**: 18-24 months
**Estimated Budget**: $350K-$500K
**Target Outcome**: FDA 510(k) clearance as Class II CDSS

---

## Validation Phases Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                  CLINICAL VALIDATION ROADMAP                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: Retrospective Validation (Months 1-3)                     │
│  ├─ Dataset: MIMIC-IV mental health subset (500-1000 cases)         │
│  ├─ Metrics: Top-10 accuracy, sensitivity, specificity              │
│  ├─ Goal: Baseline performance benchmarking                         │
│  └─ Deliverable: Technical report                                   │
│                                                                      │
│  Phase 2: Prospective Study (Months 4-12)                           │
│  ├─ Setting: Hospital/clinic partnership                            │
│  ├─ Participants: 100-300 patients                                  │
│  ├─ Design: Randomized vs board-certified psychiatrists             │
│  ├─ Goal: Match AMIE's 59% top-10 accuracy                         │
│  └─ Deliverable: Peer-reviewed publication                         │
│                                                                      │
│  Phase 3: External Validation (Months 13-18)                        │
│  ├─ Sites: 3-5 independent clinical sites                           │
│  ├─ Purpose: Generalizability assessment                            │
│  ├─ Goal: Multi-site performance validation                         │
│  └─ Deliverable: FDA submission package                            │
│                                                                      │
│  Phase 4: FDA Submission (Months 19-24)                             │
│  ├─ Pathway: 510(k) clearance for Class II CDSS                    │
│  ├─ Documents: Clinical evaluation, risk management, validation     │
│  ├─ Goal: FDA clearance                                            │
│  └─ Deliverable: Cleared CDSS for clinical use                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Retrospective Validation (Months 1-3)

### Objectives
- Establish baseline diagnostic accuracy on publicly available datasets
- Benchmark against AMIE's published performance (59% top-10 accuracy)
- Identify performance strengths and weaknesses by disorder type
- Document methodology following STARD-AI guidelines

### Study Design

**Dataset**: MIMIC-IV Mental Health Subset
- **Source**: MIT Critical Data, publicly available
- **Size**: 500-1000 historical patient records
- **Selection Criteria**: Patients with documented mental health diagnoses (ICD-10 codes)
- **Exclusions**: Incomplete records, non-psychiatric diagnoses

**Comparator**: Board-certified psychiatrist gold standard diagnoses (from medical records)

**Evaluation Metrics**:
- Top-1, Top-3, Top-10 diagnostic accuracy
- Sensitivity and specificity by disorder (depression, anxiety, PTSD, etc.)
- F1-score, AUC-ROC
- Diagnostic agreement (Cohen's κ)

### Success Criteria
- ✅ Top-10 accuracy ≥50% (aspirational: ≥59% to match AMIE)
- ✅ Sensitivity ≥80% for major disorders (depression, anxiety)
- ✅ Specificity ≥80% for major disorders
- ✅ Substantial inter-rater agreement (κ ≥0.75)

### Deliverables
1. Technical Report with Performance Benchmarks
2. STARD-AI Compliance Checklist
3. Analysis of Performance by Disorder Type
4. Comparison with AMIE Benchmarks

### Resources Required
- 1 ML Engineer (60% FTE for 3 months)
- 1 Clinical Consultant (20% FTE for 3 months)
- 1 Statistician (20% FTE for 2 months)
- MIMIC-IV dataset access (free with credentialed researcher status)

### Budget Estimate: $40K-$60K

---

## Phase 2: Prospective Clinical Study (Months 4-12)

### Objectives
- Validate diagnostic accuracy in real-world clinical setting
- Demonstrate non-inferiority to board-certified psychiatrists
- Assess anti-sycophancy mechanism effectiveness
- Measure patient satisfaction and clinical utility
- Generate peer-reviewed publication

### Study Design

**Study Type**: Prospective, randomized, blinded comparison study

**Setting**: Hospital or academic medical center psychiatric clinic

**Sample Size**:
- **Target**: 300 patients (powered for 80% to detect 10% difference)
- **Minimum**: 100 patients (pilot study)
- **Recruitment**: Consecutive patients seeking mental health evaluation

**Inclusion Criteria**:
- Age 18-65 years
- Seeking mental health evaluation
- English-speaking
- Able to provide informed consent
- No acute safety concerns requiring immediate intervention

**Exclusion Criteria**:
- Active psychosis or mania
- Acute suicidality requiring immediate hospitalization
- Cognitive impairment preventing informed consent
- Non-English speaking (for initial study)

**Study Arms**:
1. **Intervention Group**: Diagnosis Service assessment + psychiatrist review
2. **Control Group**: Standard psychiatrist assessment only

**Blinding**:
- Patients blinded to assignment
- Independent evaluator blinded to assessment source
- Outcome assessors blinded

### Outcome Measures

**Primary Outcome**:
- Diagnostic accuracy (top-10) compared to independent expert panel consensus diagnosis

**Secondary Outcomes**:
- Top-1 and Top-3 diagnostic accuracy
- Sensitivity and specificity by disorder
- Time to diagnosis
- Patient satisfaction (5-point Likert scale)
- Clinician satisfaction with CDSS recommendations
- Number of cognitive biases detected by Devil's Advocate
- Confidence calibration accuracy

### Evaluation Process

**Gold Standard**: Independent expert panel (3 board-certified psychiatrists)
- Blind review of all case information
- Consensus diagnosis via modified Delphi method
- Inter-rater reliability assessment

**Assessment Flow**:
```
Patient Enrollment
    ↓
Randomization
    ↓
┌────────────────────┬────────────────────┐
│  Intervention Arm  │   Control Arm      │
├────────────────────┼────────────────────┤
│ 1. Diagnosis       │ 1. Standard        │
│    Service         │    Psychiatrist    │
│    Assessment      │    Assessment      │
│ 2. Psychiatrist    │                    │
│    Review          │                    │
└────────────────────┴────────────────────┘
    ↓                        ↓
    └────────────────────────┘
              ↓
    Independent Expert Panel
       (3 Psychiatrists)
              ↓
       Gold Standard Diagnosis
              ↓
    Performance Analysis
```

### IRB Requirements

**Institutional Review Board Approval**:
- Protocol submission and approval
- Informed consent documents
- Data safety monitoring plan
- Adverse event reporting procedures
- Privacy/HIPAA compliance plan

**Ethical Considerations**:
- All patients receive standard clinical care
- CDSS recommendations reviewed by qualified psychiatrist
- Safety monitoring for deterioration or crisis
- Right to withdraw at any time
- Data de-identification and security

### Data Collection

**Case Report Forms**:
- Demographics (age, gender, race/ethnicity, education)
- Presenting symptoms (using Diagnosis Service symptom extraction)
- Medical and psychiatric history
- Current medications
- PHQ-9, GAD-7, PCL-5 scores
- Diagnosis Service differential diagnosis output
- Psychiatrist diagnosis and confidence
- Expert panel consensus diagnosis
- Patient satisfaction survey
- Time metrics (assessment duration)

**Data Quality**:
- Electronic data capture system
- Real-time validation rules
- Regular data quality audits
- Source document verification (10% sample)

### Statistical Analysis Plan

**Sample Size Calculation**:
- Null hypothesis: Diagnosis Service accuracy = 50%
- Alternative hypothesis: Accuracy ≥59% (AMIE benchmark)
- Power: 80%
- Alpha: 0.05 (two-tailed)
- Required sample: N = 250-300

**Primary Analysis**:
- Top-10 diagnostic accuracy with 95% confidence interval
- Non-inferiority test vs psychiatrist gold standard
- Sensitivity and specificity by disorder
- Cohen's κ for diagnostic agreement

**Secondary Analysis**:
- Subgroup analysis by disorder type
- Bias detection effectiveness
- Patient satisfaction comparison
- Time efficiency analysis

### Publication Strategy

**Target Journal**: Nature Digital Medicine, JAMA Psychiatry, or The Lancet Psychiatry

**Manuscript Outline**:
1. **Introduction**: Mental health diagnosis challenges, AI opportunity, study objectives
2. **Methods**: Study design, participants, interventions, outcomes, statistical analysis
3. **Results**: Patient characteristics, diagnostic accuracy, secondary outcomes
4. **Discussion**: Interpretation, clinical implications, anti-sycophancy effectiveness, limitations
5. **Conclusion**: Clinical utility, future directions

**Author Team**:
- Principal Investigator (psychiatrist)
- Co-investigators (2-3 clinicians)
- Technical lead (AI system developer)
- Statistician
- Clinical research coordinators

### Success Criteria
- ✅ Top-10 diagnostic accuracy ≥59% (match AMIE)
- ✅ Non-inferior to psychiatrist assessment (within 5%)
- ✅ Patient satisfaction ≥4.0/5.0
- ✅ Manuscript accepted in peer-reviewed journal (impact factor ≥10)

### Deliverables
1. IRB-Approved Study Protocol
2. Complete Dataset (de-identified)
3. Statistical Analysis Report
4. Peer-Reviewed Publication
5. Clinical Study Report for FDA Submission

### Resources Required
- **Clinical Team**:
  - 1 Principal Investigator (psychiatrist, 20% FTE)
  - 2-3 Co-investigators (psychiatrists, 10% FTE each)
  - 3 Expert panel psychiatrists (consultant basis)
  - 1 Research Coordinator (100% FTE)

- **Technical Team**:
  - 1 ML Engineer (50% FTE)
  - 1 Data Engineer (30% FTE)
  - 1 Statistician (30% FTE)

- **Administrative**:
  - IRB submissions and monitoring
  - Participant recruitment and consent
  - Data management
  - Regulatory compliance

### Budget Estimate: $200K-$300K
- Clinical site fees: $100K-$150K
- Personnel costs: $80K-$120K
- Data management system: $10K-$15K
- IRB fees: $5K-$10K
- Publication costs: $5K

---

## Phase 3: External Validation (Months 13-18)

### Objectives
- Demonstrate generalizability across clinical settings
- Validate performance in diverse patient populations
- Assess real-world clinical integration
- Generate evidence for FDA submission

### Study Design

**Study Type**: Multi-site external validation study

**Sites**: 3-5 independent clinical sites
- Academic medical center
- Community mental health clinic
- Private practice group
- Veterans Affairs (VA) hospital (if feasible)
- Rural/underserved setting

**Sample Size**: 100-200 patients per site (total 300-1000)

**Design**: Same as Phase 2 prospective study
- Randomized comparison
- Blinded evaluation
- Expert panel consensus gold standard

### Site Selection Criteria
- Geographic diversity (different regions)
- Population diversity (urban/rural, demographics)
- Setting diversity (academic/community/private)
- Sufficient patient volume
- Research infrastructure and IRB capability

### Evaluation Metrics
- Same as Phase 2
- Additional: Performance variation across sites
- Subgroup analysis by site characteristics

### Success Criteria
- ✅ Consistent performance across sites (≤10% variation in top-10 accuracy)
- ✅ No statistically significant performance degradation in any subgroup
- ✅ Generalizability to diverse populations demonstrated

### Deliverables
1. Multi-Site Validation Report
2. Subgroup Analysis by Site and Population
3. Real-World Clinical Integration Report
4. FDA Clinical Evaluation Report

### Resources Required
- Site recruitment and contracting
- Coordinating center for data harmonization
- Multi-site IRB or individual site IRBs
- Statistician for multi-level modeling

### Budget Estimate: $150K-$200K

---

## Phase 4: FDA Submission (Months 19-24)

### Objectives
- Achieve FDA clearance as Class II CDSS
- Establish post-market surveillance plan
- Enable clinical deployment

### Regulatory Pathway

**Device Classification**: Class II Medical Device (Clinical Decision Support System)

**Clearance Pathway**: 510(k) Premarket Notification
- Predicate device identification
- Substantial equivalence demonstration
- Clinical performance data

**Alternative**: De Novo classification (if no suitable predicate)

### FDA Submission Timeline

```
Month 19-20: Pre-Submission Preparation
    ├─ Q-Submission for FDA feedback
    ├─ Pre-Submission meeting request
    ├─ Predicate device search
    └─ Document preparation

Month 20-21: Pre-Submission Meeting
    ├─ Meeting with FDA reviewers
    ├─ Receive feedback on approach
    ├─ Clarify regulatory requirements
    └─ Finalize submission strategy

Month 21-23: 510(k) Submission Package
    ├─ Device description and intended use
    ├─ Substantial equivalence discussion
    ├─ Clinical evaluation report
    ├─ Software description document
    ├─ Risk management file (ISO 14971)
    ├─ Cybersecurity documentation
    ├─ Labeling and instructions for use
    └─ Submit to FDA

Month 23-24: FDA Review
    ├─ Substantive review by FDA
    ├─ Additional information requests
    ├─ Deficiency letter response
    └─ Clearance decision
```

### Key Submission Documents

**1. Device Description**
- Indications for use
- Device description (hardware/software)
- Intended patient population
- Clinical environment and use case
- User qualifications (psychiatrist review required)

**2. Substantial Equivalence**
- Predicate device identification
- Comparison of technological characteristics
- Performance comparison
- Justification of substantial equivalence

**3. Clinical Evaluation Report**
- Summary of clinical studies (Phases 2-3)
- Diagnostic accuracy data
- Safety profile
- Clinical utility evidence
- STARD-AI compliant reporting

**4. Software Documentation**
- Software description document (SDD)
- Software development lifecycle (SDLC)
- Verification and validation (V&V)
- Risk analysis (ISO 14971)
- Cybersecurity risk assessment

**5. Risk Management**
- Hazard analysis
- Risk estimation
- Risk controls
- Residual risk evaluation
- Post-market surveillance plan

**6. Labeling**
- Indications for use
- Contraindications
- Warnings and precautions
- Instructions for use
- Technical specifications
- Limitations of the device

### Predetermined Change Control Plan (PCCP)

**Required for Adaptive AI Systems**:
- Pre-specified changes to algorithm
- Performance monitoring specifications
- Change authorization procedures
- Re-validation trigger thresholds
- Documentation and notification requirements

**Our PCCP Approach**:
- Model retraining frequency: Quarterly
- Performance degradation threshold: 5% drop in accuracy
- Change approval: Internal review board + FDA notification
- Re-validation: Annual performance audit

### Post-Market Surveillance

**Real-World Performance Monitoring**:
- Continuous accuracy monitoring
- Adverse event reporting
- Bias drift detection
- User feedback collection
- Annual performance reports to FDA

**Metrics Tracked**:
- Diagnostic accuracy in clinical use
- False positive/negative rates
- Safety incidents
- User satisfaction
- Clinical outcomes (if available)

### Success Criteria
- ✅ FDA 510(k) clearance granted
- ✅ Post-market surveillance plan approved
- ✅ Cleared for clinical marketing and use

### Deliverables
1. Complete 510(k) Submission Package
2. FDA Clearance Letter
3. Post-Market Surveillance Plan
4. Labeling and Instructions for Use
5. Quality System Documentation

### Resources Required
- Regulatory Affairs Consultant (100% FTE for 6 months)
- Technical team for documentation
- Clinical team for clinical evaluation
- Quality assurance specialist
- Legal review

### Budget Estimate: $100K-$150K
- Regulatory consultant: $60K-$100K
- FDA user fees: $15K-$20K
- Legal review: $10K-$15K
- Documentation and preparation: $15K-$20K

---

## Total Program Summary

### Timeline
- **Phase 1**: Months 1-3 (Retrospective validation)
- **Phase 2**: Months 4-12 (Prospective clinical study)
- **Phase 3**: Months 13-18 (External validation)
- **Phase 4**: Months 19-24 (FDA submission and clearance)

**Total Duration**: 24 months (2 years)

### Budget
- **Phase 1**: $40K-$60K
- **Phase 2**: $200K-$300K
- **Phase 3**: $150K-$200K
- **Phase 4**: $100K-$150K

**Total Budget**: $490K-$710K (midpoint: ~$600K)

### Key Milestones

```
┌─────────────────────────────────────────────────────────────┐
│  Month 3:  Retrospective validation complete                │
│  Month 6:  Prospective study enrollment 50% complete        │
│  Month 12: Prospective study complete, manuscript submitted │
│  Month 15: External validation enrollment complete          │
│  Month 18: External validation complete                     │
│  Month 21: 510(k) submission to FDA                        │
│  Month 24: FDA clearance granted ✅                         │
└─────────────────────────────────────────────────────────────┘
```

### Risk Mitigation

**Risk 1: Recruitment challenges**
- Mitigation: Multiple clinical sites, patient incentives, streamlined consent

**Risk 2: Performance below target**
- Mitigation: Pilot study first, model refinement between phases

**Risk 3: FDA regulatory changes**
- Mitigation: Early engagement with FDA, flexible submission strategy

**Risk 4: Competitive landscape changes**
- Mitigation: Focus on unique anti-sycophancy features, rapid execution

**Risk 5: Budget overruns**
- Mitigation: Phased approach, contingency budget (20% reserve)

---

## Next Steps (Immediate Actions)

### Month 1-2: Foundation
1. ✅ Secure funding commitment
2. ✅ Engage regulatory consultant
3. ✅ Identify clinical site partners
4. ✅ Access MIMIC-IV dataset
5. ✅ Assemble study team

### Month 2-3: Retrospective Validation
1. ✅ Run Diagnosis Service on MIMIC-IV test set
2. ✅ Calculate baseline accuracy metrics
3. ✅ Complete STARD-AI documentation
4. ✅ Publish technical report

### Month 3-4: Prospective Study Preparation
1. ✅ Finalize study protocol
2. ✅ Submit IRB application
3. ✅ Develop data collection systems
4. ✅ Train clinical research staff

**Critical Path Item**: IRB approval typically takes 4-8 weeks, plan accordingly

---

## Conclusion

This validation roadmap provides a clear path to FDA clearance for the Solace-AI Diagnosis Service within 24 months. The phased approach balances scientific rigor with practical execution, building evidence from retrospective analysis through prospective validation to multi-site external validation.

**Key Success Factors**:
1. Strong clinical partnerships for patient recruitment
2. Rigorous study design following STARD-AI guidelines
3. Early and frequent FDA engagement
4. Adequate budget and timeline contingency
5. Focus on unique anti-sycophancy value proposition

**Expected Outcome**: FDA-cleared Clinical Decision Support System for mental health diagnosis with peer-reviewed publication demonstrating world-class diagnostic accuracy and unique bias mitigation capabilities.

---

**Document Status**: ✅ Complete
**Last Updated**: January 2026
**Owner**: Clinical Development Team
**Next Review**: Quarterly progress updates
