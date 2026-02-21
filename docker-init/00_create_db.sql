-- Ensure required databases exist at startup.
-- Note: POSTGRES_DB is not a server GUC, so current_setting('POSTGRES_DB') is usually NULL.
DO $$
DECLARE
    db_name text;
BEGIN
    FOREACH db_name IN ARRAY ARRAY['anko', 'admin'] LOOP
        IF NOT EXISTS (SELECT FROM pg_database WHERE datname = db_name) THEN
            EXECUTE format('CREATE DATABASE %I', db_name);
        END IF;
    END LOOP;
END
$$ LANGUAGE plpgsql;
