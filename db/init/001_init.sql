CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS source_metadata (
    id uuid PRIMARY KEY,
    name text NOT NULL,
    platform_type text NOT NULL,
    url text,
    default_language text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document (
    id uuid PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES source_metadata(id),
    original_text text NOT NULL,
    hash text,
    language text,
    status text NOT NULL DEFAULT 'pending',
    received_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_version (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name text NOT NULL,
    version text NOT NULL DEFAULT 'unknown',
    task_type text NOT NULL,
    provider text NOT NULL,
    artifact_uri text,
    registered_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (model_name, version, task_type, provider)
);

CREATE TABLE IF NOT EXISTS analysis_run (
    id text PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES document(id),
    sentiment_model_id uuid REFERENCES model_version(id),
    ner_model_id uuid REFERENCES model_version(id),
    summary_model_id uuid REFERENCES model_version(id),
    run_status text NOT NULL,
    trigger_type text NOT NULL,
    prompt_version text,
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    error_message text
);

CREATE TABLE IF NOT EXISTS analysis_result (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_run_id text NOT NULL UNIQUE REFERENCES analysis_run(id) ON DELETE CASCADE,
    sentiment_score real NOT NULL,
    sentiment_label text NOT NULL,
    entities jsonb NOT NULL DEFAULT '[]'::jsonb,
    summary text,
    processing_ms integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_document_source_id
    ON document(source_id);

CREATE INDEX IF NOT EXISTS idx_document_status
    ON document(status);

CREATE INDEX IF NOT EXISTS idx_analysis_run_document_id
    ON analysis_run(document_id);

CREATE INDEX IF NOT EXISTS idx_analysis_run_status
    ON analysis_run(run_status);

CREATE INDEX IF NOT EXISTS idx_analysis_result_sentiment_label
    ON analysis_result(sentiment_label);

CREATE INDEX IF NOT EXISTS idx_analysis_result_entities_gin
    ON analysis_result USING gin (entities);
