-- Solace-AI Safety Service - Contraindication Rules Database Schema
-- PostgreSQL schema for storing therapeutic technique contraindication rules
-- Version: 1.0.0
-- Date: 2026-01-08

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types for type safety
CREATE TYPE contraindication_type AS ENUM (
    'ABSOLUTE',           -- Never use - dangerous
    'RELATIVE',           -- Use with caution - monitor closely
    'TECHNIQUE_SPECIFIC', -- Requires prerequisites
    'TIMING',             -- Inappropriate timing
    'SEVERITY'            -- Inappropriate for severity level
);

CREATE TYPE therapy_technique AS ENUM (
    'EXPOSURE_THERAPY',
    'COGNITIVE_RESTRUCTURING',
    'BEHAVIORAL_ACTIVATION',
    'MINDFULNESS_MEDITATION',
    'DBT_DIARY_CARD',
    'DBT_DISTRESS_TOLERANCE',
    'EMOTION_REGULATION',
    'INTERPERSONAL_EFFECTIVENESS',
    'SOMATIC_EXPERIENCING',
    'EMDR',
    'ACCEPTANCE_COMMITMENT',
    'PROGRESSIVE_MUSCLE_RELAXATION',
    'GROUNDING_TECHNIQUES'
);

CREATE TYPE mental_health_condition AS ENUM (
    'ACTIVE_PSYCHOSIS',
    'SEVERE_DEPRESSION',
    'ACTIVE_SUBSTANCE_USE',
    'DISSOCIATIVE_DISORDER',
    'ACUTE_MANIA',
    'SEVERE_PTSD',
    'PERSONALITY_DISORDER',
    'EATING_DISORDER',
    'ACUTE_CRISIS',
    'SUICIDAL_IDEATION'
);

-- Main contraindication rules table
CREATE TABLE IF NOT EXISTS contraindication_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    technique therapy_technique NOT NULL,
    condition mental_health_condition NOT NULL,
    contraindication_type contraindication_type NOT NULL,
    severity DECIMAL(3, 2) NOT NULL CHECK (severity >= 0 AND severity <= 1),
    rationale TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    version INTEGER NOT NULL DEFAULT 1,

    -- Ensure unique technique-condition pairs
    CONSTRAINT unique_technique_condition UNIQUE (technique, condition),

    -- Add index for faster lookups
    CONSTRAINT chk_severity_range CHECK (severity BETWEEN 0 AND 1)
);

-- Alternative techniques table (one-to-many)
CREATE TABLE IF NOT EXISTS rule_alternatives (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id UUID NOT NULL,
    alternative_technique therapy_technique NOT NULL,
    display_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_rule_alternatives_rule
        FOREIGN KEY (rule_id)
        REFERENCES contraindication_rules(id)
        ON DELETE CASCADE,

    -- Ensure no duplicate alternatives for the same rule
    CONSTRAINT unique_rule_alternative UNIQUE (rule_id, alternative_technique)
);

-- Prerequisites table (one-to-many)
CREATE TABLE IF NOT EXISTS rule_prerequisites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id UUID NOT NULL,
    prerequisite TEXT NOT NULL,
    display_order INTEGER NOT NULL DEFAULT 0,
    is_required BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_rule_prerequisites_rule
        FOREIGN KEY (rule_id)
        REFERENCES contraindication_rules(id)
        ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_rules_technique ON contraindication_rules(technique);
CREATE INDEX IF NOT EXISTS idx_rules_condition ON contraindication_rules(condition);
CREATE INDEX IF NOT EXISTS idx_rules_type ON contraindication_rules(contraindication_type);
CREATE INDEX IF NOT EXISTS idx_rules_severity ON contraindication_rules(severity);
CREATE INDEX IF NOT EXISTS idx_rules_active ON contraindication_rules(is_active);
CREATE INDEX IF NOT EXISTS idx_rules_created_at ON contraindication_rules(created_at);

CREATE INDEX IF NOT EXISTS idx_alternatives_rule_id ON rule_alternatives(rule_id);
CREATE INDEX IF NOT EXISTS idx_prerequisites_rule_id ON rule_prerequisites(rule_id);

-- Create trigger for automatic updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.version = OLD.version + 1;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_contraindication_rules_updated_at
    BEFORE UPDATE ON contraindication_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create audit log table for rule changes
CREATE TABLE IF NOT EXISTS contraindication_rule_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id UUID NOT NULL,
    operation VARCHAR(10) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(255),
    changed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    change_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_rule_id ON contraindication_rule_audit(rule_id);
CREATE INDEX IF NOT EXISTS idx_audit_changed_at ON contraindication_rule_audit(changed_at);

-- Create audit trigger
CREATE OR REPLACE FUNCTION audit_contraindication_rule_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'DELETE') THEN
        INSERT INTO contraindication_rule_audit (rule_id, operation, old_values, changed_by)
        VALUES (OLD.id, 'DELETE', row_to_json(OLD)::jsonb, OLD.updated_by);
        RETURN OLD;
    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO contraindication_rule_audit (rule_id, operation, old_values, new_values, changed_by)
        VALUES (NEW.id, 'UPDATE', row_to_json(OLD)::jsonb, row_to_json(NEW)::jsonb, NEW.updated_by);
        RETURN NEW;
    ELSIF (TG_OP = 'INSERT') THEN
        INSERT INTO contraindication_rule_audit (rule_id, operation, new_values, changed_by)
        VALUES (NEW.id, 'INSERT', row_to_json(NEW)::jsonb, NEW.created_by);
        RETURN NEW;
    END IF;
END;
$$ language 'plpgsql';

CREATE TRIGGER audit_contraindication_rules
    AFTER INSERT OR UPDATE OR DELETE ON contraindication_rules
    FOR EACH ROW
    EXECUTE FUNCTION audit_contraindication_rule_changes();

-- Grant permissions (adjust based on your database user)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO safety_service_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO safety_service_user;

-- Comments for documentation
COMMENT ON TABLE contraindication_rules IS 'Clinical rules for therapeutic technique contraindications';
COMMENT ON COLUMN contraindication_rules.technique IS 'Therapeutic technique being evaluated';
COMMENT ON COLUMN contraindication_rules.condition IS 'Mental health condition that contraindicates the technique';
COMMENT ON COLUMN contraindication_rules.contraindication_type IS 'Severity type of contraindication (ABSOLUTE, RELATIVE, etc.)';
COMMENT ON COLUMN contraindication_rules.severity IS 'Severity score from 0.0 to 1.0';
COMMENT ON COLUMN contraindication_rules.rationale IS 'Clinical reasoning for the contraindication';
COMMENT ON COLUMN contraindication_rules.is_active IS 'Whether the rule is currently active';
COMMENT ON COLUMN contraindication_rules.version IS 'Version number for optimistic locking';

COMMENT ON TABLE rule_alternatives IS 'Alternative techniques to use instead of contraindicated technique';
COMMENT ON TABLE rule_prerequisites IS 'Prerequisites required before using the technique with the condition';
COMMENT ON TABLE contraindication_rule_audit IS 'Audit log of all changes to contraindication rules';
