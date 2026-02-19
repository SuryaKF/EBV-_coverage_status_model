-- Core schema for ETL, registry, and inference logging

CREATE TABLE IF NOT EXISTS raw_coverage_data (
    id BIGSERIAL PRIMARY KEY,
    batch_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    payer_name TEXT,
    state_name TEXT,
    acronym TEXT,
    expansion TEXT,
    explanation TEXT,
    coverage_status TEXT,
    row_hash TEXT NOT NULL,
    raw_payload JSONB NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_raw_batch_hash
    ON raw_coverage_data (batch_id, row_hash);

CREATE TABLE IF NOT EXISTS stg_coverage_data (
    id BIGSERIAL PRIMARY KEY,
    raw_id BIGINT NOT NULL REFERENCES raw_coverage_data(id),
    batch_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    payer_name TEXT NOT NULL,
    state_name TEXT NOT NULL,
    acronym TEXT NOT NULL,
    expansion TEXT NOT NULL,
    explanation TEXT NOT NULL,
    coverage_status TEXT NOT NULL,
    row_hash TEXT NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_stg_batch_hash
    ON stg_coverage_data (batch_id, row_hash);

CREATE TABLE IF NOT EXISTS rejected_coverage_data (
    id BIGSERIAL PRIMARY KEY,
    batch_id TEXT NOT NULL,
    source_name TEXT,
    raw_id BIGINT,
    row_hash TEXT,
    raw_payload JSONB,
    reject_reason TEXT NOT NULL,
    rejected_at TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_reject_batch_hash_reason
    ON rejected_coverage_data (batch_id, row_hash, reject_reason);

CREATE TABLE IF NOT EXISTS curated_coverage_data (
    id BIGSERIAL PRIMARY KEY,
    stg_id BIGINT NOT NULL REFERENCES stg_coverage_data(id),
    batch_id TEXT NOT NULL,
    payer_name TEXT NOT NULL,
    state_name TEXT NOT NULL,
    acronym TEXT NOT NULL,
    expansion TEXT NOT NULL,
    explanation TEXT NOT NULL,
    coverage_status TEXT NOT NULL,
    row_hash TEXT NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_curated_batch_hash
    ON curated_coverage_data (batch_id, row_hash);

CREATE TABLE IF NOT EXISTS dataset_registry (
    snapshot_id TEXT PRIMARY KEY,
    snapshot_table TEXT NOT NULL UNIQUE,
    row_count INTEGER NOT NULL,
    feature_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY,
    artifact_uri TEXT NOT NULL,
    snapshot_id TEXT NOT NULL REFERENCES dataset_registry(snapshot_id),
    feature_version TEXT NOT NULL,
    metrics_json JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('staging', 'approved', 'archived')),
    created_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    approved_by TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_approved_singleton
    ON model_registry ((status))
    WHERE status = 'approved';

CREATE TABLE IF NOT EXISTS prediction_log (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    payer_name TEXT NOT NULL,
    state_name TEXT NOT NULL,
    acronym TEXT NOT NULL,
    expansion TEXT,
    explanation TEXT,
    prediction TEXT NOT NULL,
    confidence_json JSONB,
    model_version TEXT NOT NULL REFERENCES model_registry(model_version),
    latency_ms INTEGER NOT NULL,
    predicted_at TIMESTAMPTZ NOT NULL
);
