-- Ensure target database exists at startup, using POSTGRES_DB env
DO $$
DECLARE
    target_db text := current_setting('POSTGRES_DB', true);
BEGIN
    IF target_db IS NULL OR target_db = '' THEN
        target_db := 'anko';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = target_db) THEN
        EXECUTE format('CREATE DATABASE %I', target_db);
    END IF;
END
$$ LANGUAGE plpgsql;
