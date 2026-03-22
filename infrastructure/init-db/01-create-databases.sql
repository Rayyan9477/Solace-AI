-- Solace-AI: Create per-service databases
-- This script runs automatically on first PostgreSQL startup via docker-entrypoint-initdb.d

CREATE DATABASE solace_users;
CREATE DATABASE solace_safety;
CREATE DATABASE solace_diagnosis;
CREATE DATABASE solace_therapy;
CREATE DATABASE solace_personality;
CREATE DATABASE solace_memory;
CREATE DATABASE solace_notifications;
CREATE DATABASE solace_analytics;
CREATE DATABASE solace_config;

-- Grant the default solace user full access to each database
DO $$
DECLARE
    db TEXT;
BEGIN
    FOREACH db IN ARRAY ARRAY[
        'solace_users', 'solace_safety', 'solace_diagnosis',
        'solace_therapy', 'solace_personality', 'solace_memory',
        'solace_notifications', 'solace_analytics', 'solace_config'
    ]
    LOOP
        EXECUTE format('GRANT ALL PRIVILEGES ON DATABASE %I TO solace', db);
    END LOOP;
END
$$;
