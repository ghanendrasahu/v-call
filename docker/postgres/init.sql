-- ──────────────────────────────────────────────────────────────────────
-- Bootstrap DB extensions used by the app (pgcrypto for gen_random_uuid).
-- Runs once on first `docker compose up` via docker-entrypoint-initdb.d.
-- ──────────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Per-tenant schema support is handled at the ORM level with row-level
-- tenant_id isolation; this file is a hook for any infra-level setup.
