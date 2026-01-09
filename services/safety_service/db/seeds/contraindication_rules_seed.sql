-- Solace-AI Safety Service - Contraindication Rules Seed Data
-- Initial clinical rules for therapeutic technique contraindications
-- Version: 1.0.0
-- Date: 2026-01-08

-- Clear existing data (for development/testing only)
-- TRUNCATE contraindication_rules CASCADE;

-- ============================================================================
-- ABSOLUTE CONTRAINDICATIONS (Never use - dangerous)
-- ============================================================================

-- Exposure Therapy + Active Psychosis
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'EXPOSURE_THERAPY',
    'ACTIVE_PSYCHOSIS',
    'ABSOLUTE',
    1.0,
    'Exposure therapy during active psychosis can worsen symptoms and cause severe distress',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EXPOSURE_THERAPY' AND condition = 'ACTIVE_PSYCHOSIS'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'GROUNDING_TECHNIQUES', 0 FROM rule
UNION ALL
SELECT id, 'PROGRESSIVE_MUSCLE_RELAXATION', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EXPOSURE_THERAPY' AND condition = 'ACTIVE_PSYCHOSIS'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Stabilization of psychotic symptoms', 0 FROM rule
UNION ALL
SELECT id, 'Medication compliance', 1 FROM rule;

-- Exposure Therapy + Acute Crisis
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'EXPOSURE_THERAPY',
    'ACUTE_CRISIS',
    'ABSOLUTE',
    0.95,
    'Exposure therapy during crisis can overwhelm coping capacity and escalate risk',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EXPOSURE_THERAPY' AND condition = 'ACUTE_CRISIS'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'GROUNDING_TECHNIQUES', 0 FROM rule
UNION ALL
SELECT id, 'DBT_DISTRESS_TOLERANCE', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EXPOSURE_THERAPY' AND condition = 'ACUTE_CRISIS'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Crisis resolution', 0 FROM rule
UNION ALL
SELECT id, 'Safety plan in place', 1 FROM rule;

-- EMDR + Dissociative Disorder
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'EMDR',
    'DISSOCIATIVE_DISORDER',
    'ABSOLUTE',
    0.95,
    'EMDR can trigger severe dissociation in patients with dissociative disorders',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EMDR' AND condition = 'DISSOCIATIVE_DISORDER'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'SOMATIC_EXPERIENCING', 0 FROM rule
UNION ALL
SELECT id, 'GROUNDING_TECHNIQUES', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EMDR' AND condition = 'DISSOCIATIVE_DISORDER'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Dissociation management skills', 0 FROM rule
UNION ALL
SELECT id, 'Specialized therapist', 1 FROM rule;

-- ============================================================================
-- RELATIVE CONTRAINDICATIONS (Use with caution)
-- ============================================================================

-- Cognitive Restructuring + Severe Depression
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'COGNITIVE_RESTRUCTURING',
    'SEVERE_DEPRESSION',
    'RELATIVE',
    0.7,
    'Cognitive restructuring requires cognitive capacity that may be impaired in severe depression',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'COGNITIVE_RESTRUCTURING' AND condition = 'SEVERE_DEPRESSION'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'BEHAVIORAL_ACTIVATION', 0 FROM rule
UNION ALL
SELECT id, 'GROUNDING_TECHNIQUES', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'COGNITIVE_RESTRUCTURING' AND condition = 'SEVERE_DEPRESSION'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Baseline cognitive functioning', 0 FROM rule
UNION ALL
SELECT id, 'Moderate energy levels', 1 FROM rule;

-- Mindfulness Meditation + Severe PTSD
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'MINDFULNESS_MEDITATION',
    'SEVERE_PTSD',
    'RELATIVE',
    0.65,
    'Mindfulness can trigger flashbacks in severe PTSD without proper grounding skills',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'MINDFULNESS_MEDITATION' AND condition = 'SEVERE_PTSD'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'GROUNDING_TECHNIQUES', 0 FROM rule
UNION ALL
SELECT id, 'PROGRESSIVE_MUSCLE_RELAXATION', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'MINDFULNESS_MEDITATION' AND condition = 'SEVERE_PTSD'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Grounding skills mastery', 0 FROM rule
UNION ALL
SELECT id, 'Safety cues established', 1 FROM rule;

-- Emotion Regulation + Active Substance Use
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'EMOTION_REGULATION',
    'ACTIVE_SUBSTANCE_USE',
    'RELATIVE',
    0.65,
    'Substance use impairs emotional regulation capacity and skill application',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EMOTION_REGULATION' AND condition = 'ACTIVE_SUBSTANCE_USE'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'DBT_DISTRESS_TOLERANCE', 0 FROM rule
UNION ALL
SELECT id, 'GROUNDING_TECHNIQUES', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'EMOTION_REGULATION' AND condition = 'ACTIVE_SUBSTANCE_USE'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Sobriety or stable substance use', 0 FROM rule
UNION ALL
SELECT id, 'Addiction treatment engagement', 1 FROM rule;

-- ============================================================================
-- TECHNIQUE-SPECIFIC PREREQUISITES
-- ============================================================================

-- DBT Diary Card + Acute Crisis
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'DBT_DIARY_CARD',
    'ACUTE_CRISIS',
    'TECHNIQUE_SPECIFIC',
    0.6,
    'Diary cards require emotional stability and are typically introduced after crisis skills',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'DBT_DIARY_CARD' AND condition = 'ACUTE_CRISIS'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'DBT_DISTRESS_TOLERANCE', 0 FROM rule
UNION ALL
SELECT id, 'GROUNDING_TECHNIQUES', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'DBT_DIARY_CARD' AND condition = 'ACUTE_CRISIS'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Distress tolerance skills', 0 FROM rule
UNION ALL
SELECT id, 'Crisis skills mastery', 1 FROM rule
UNION ALL
SELECT id, 'Stable emotional baseline', 2 FROM rule;

-- ============================================================================
-- TIMING CONTRAINDICATIONS
-- ============================================================================

-- Interpersonal Effectiveness + Acute Mania
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'INTERPERSONAL_EFFECTIVENESS',
    'ACUTE_MANIA',
    'TIMING',
    0.7,
    'Interpersonal skills training requires stable mood state for effective learning',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'INTERPERSONAL_EFFECTIVENESS' AND condition = 'ACUTE_MANIA'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'GROUNDING_TECHNIQUES', 0 FROM rule
UNION ALL
SELECT id, 'DBT_DISTRESS_TOLERANCE', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'INTERPERSONAL_EFFECTIVENESS' AND condition = 'ACUTE_MANIA'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Mood stabilization', 0 FROM rule
UNION ALL
SELECT id, 'Medication compliance', 1 FROM rule;

-- ============================================================================
-- SEVERITY-BASED CONTRAINDICATIONS
-- ============================================================================

-- Behavioral Activation + Suicidal Ideation
INSERT INTO contraindication_rules (technique, condition, contraindication_type, severity, rationale, created_by)
VALUES (
    'BEHAVIORAL_ACTIVATION',
    'SUICIDAL_IDEATION',
    'SEVERITY',
    0.75,
    'Behavioral activation must be carefully modified when suicidal ideation is present',
    'system_seed'
) ON CONFLICT (technique, condition) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'BEHAVIORAL_ACTIVATION' AND condition = 'SUICIDAL_IDEATION'
)
INSERT INTO rule_alternatives (rule_id, alternative_technique, display_order)
SELECT id, 'DBT_DISTRESS_TOLERANCE', 0 FROM rule
UNION ALL
SELECT id, 'GROUNDING_TECHNIQUES', 1 FROM rule
ON CONFLICT (rule_id, alternative_technique) DO NOTHING;

WITH rule AS (
    SELECT id FROM contraindication_rules
    WHERE technique = 'BEHAVIORAL_ACTIVATION' AND condition = 'SUICIDAL_IDEATION'
)
INSERT INTO rule_prerequisites (rule_id, prerequisite, display_order)
SELECT id, 'Safety plan in place', 0 FROM rule
UNION ALL
SELECT id, 'Active safety monitoring', 1 FROM rule
UNION ALL
SELECT id, 'Clinician supervision', 2 FROM rule;

-- Verify inserted data
SELECT
    r.technique,
    r.condition,
    r.contraindication_type,
    r.severity,
    COUNT(DISTINCT a.id) as alternative_count,
    COUNT(DISTINCT p.id) as prerequisite_count
FROM contraindication_rules r
LEFT JOIN rule_alternatives a ON r.id = a.rule_id
LEFT JOIN rule_prerequisites p ON r.id = p.rule_id
GROUP BY r.id
ORDER BY r.contraindication_type, r.severity DESC;
