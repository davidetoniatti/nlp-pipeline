```mermaid
erDiagram
    SOURCE_METADATA {
        uuid id PK
        text name
        text platform_type
        text url
        text default_language
        timestamptz created_at
    }

    DOCUMENT {
        uuid id PK
        uuid source_id FK
        text original_text
        text hash
        text language
        text status
        timestamptz received_at
        timestamptz created_at
    }

    MODEL_VERSION {
        uuid id PK
        text model_name
        text version
        text revision
        text tokenizer_name
        text task_type
        text provider
        text artifact_uri
        jsonb metadata
        timestamptz registered_at
    }

    ANALYSIS_RUN {
        uuid id PK
        uuid document_id FK
        uuid sentiment_model_id FK
        uuid ner_model_id FK
        uuid summary_model_id FK
        text run_status
        text trigger_type
        text prompt_version
        text prompt_hash
        text service_git_sha
        timestamptz started_at
        timestamptz completed_at
        timestamptz created_at
        text error_message
    }

    ANALYSIS_RESULT {
        uuid id PK
        uuid analysis_run_id FK
        real sentiment_score
        text sentiment_label
        jsonb entities
        text summary
        integer processing_ms
        timestamptz created_at
    }

    SOURCE_METADATA ||--o{ DOCUMENT : produces
    DOCUMENT ||--o{ ANALYSIS_RUN : has
    ANALYSIS_RUN ||--o| ANALYSIS_RESULT : outputs

    MODEL_VERSION o|--o{ ANALYSIS_RUN : sentiment_model
    MODEL_VERSION o|--o{ ANALYSIS_RUN : ner_model
    MODEL_VERSION o|--o{ ANALYSIS_RUN : summary_model
```

### Note
- `analysis_result.analysis_run_id` è `UNIQUE` e referenzia `analysis_run(id)` con `ON DELETE CASCADE`.
- `document.hash` ha un indice univoco parziale: unico solo quando non è `NULL`.
- `model_version` ha un vincolo `UNIQUE (model_name, version, revision, task_type, provider)`.
- I `DEFAULT` SQL e gli indici non sono rappresentati direttamente nel diagramma ER Mermaid.
